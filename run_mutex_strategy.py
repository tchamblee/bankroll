import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import config
from genome import Strategy
from backtest import BacktestEngine
from trade_simulator import TradeSimulator

def load_all_candidates():
    candidates = []
    print(f"Loading candidates from all horizons...")
    
    for h in config.PREDICTION_HORIZONS:
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}_top5_unique.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if data:
            # LOAD ALL CLUSTER CHAMPIONS (Not just the top 1)
            for d in data:
                try:
                    s = Strategy.from_dict(d)
                    s.horizon = h
                    s.training_id = d.get('training_id', 'legacy') # Capture ID
                    
                    # Load metrics for ranking
                    # 'test_sortino' might be missing in portfolio files, use 'fitness' (Validation Score)
                    s.sortino = d.get('test_sortino', d.get('fitness', 0))
                    s.robust = d.get('robust_score', 0)
                    
                    # Add to candidates (Trusting upstream selection, will Re-Validate later)
                    candidates.append(s)

                except Exception as e:
                    # print(f"Error loading {h}: {e}")
                    pass
            
    # Sort candidates globally by Sortino (Priority)
    candidates.sort(key=lambda x: x.sortino, reverse=True)
    
    # Print candidates per horizon
    from collections import Counter
    counts = Counter([c.horizon for c in candidates])
    print(f"  Loaded {len(candidates)} total candidates. Counts per Horizon: {dict(counts)}")
    return candidates

def filter_global_correlation(candidates, backtester, threshold=0.7):
    print(f"\nRunning Global Correlation Filter (Threshold: {threshold})...")
    
    print("  Generating signal matrix...")
    backtester.ensure_context(candidates)
    sig_matrix = backtester.generate_signal_matrix(candidates)
    
    selected = []
    selected_indices = []
    
    for i, candidate in enumerate(candidates):
        is_unique = True
        current_sig = sig_matrix[:, i]
        
        # Check against already selected
        for existing_idx in selected_indices:
            existing_sig = sig_matrix[:, existing_idx]
            
            # Fast correlation
            if np.std(current_sig) == 0 or np.std(existing_sig) == 0:
                corr = 0
            else:
                corr = np.corrcoef(current_sig, existing_sig)[0, 1]
                
            if abs(corr) > threshold:
                is_unique = False
                break
        
        if is_unique:
            selected.append(candidate)
            selected_indices.append(i)
            
    print(f"  Selected {len(selected)} globally unique strategies.")
    return selected, sig_matrix[:, selected_indices]

from numba import jit

@jit(nopython=True)
def _jit_simulate_mutex_custom(sig_matrix, prices, highs, lows, atr, hours, weekdays, horizons, sl_mults, tp_mults, lot_size, spread_pct, comm_pct, account_size, end_hour):
    n_bars, n_strats = sig_matrix.shape
    net_returns = np.zeros(n_bars, dtype=np.float64)
    trades_count = 0
    
    # State
    position = 0.0
    entry_price = 0.0
    entry_idx = 0
    entry_atr = 0.0
    
    current_horizon = 0
    current_sl_mult = 0.0
    current_tp_mult = 0.0
    
    prev_pos = 0.0
    
    for i in range(n_bars):
        # 1. Check Barriers / Exits
        exit_trade = False
        barrier_price = 0.0
        
        # Force Close (Time/EOD)
        if hours[i] >= end_hour or weekdays[i] >= 5:
            exit_trade = True
            
        elif position != 0:
            # Time Limit
            if (i - entry_idx) >= current_horizon:
                exit_trade = True
            
            # SL/TP
            if not exit_trade:
                h_prev = highs[i-1] if i > 0 else prices[i]
                l_prev = lows[i-1] if i > 0 else prices[i]
                
                sl_dist = entry_atr * current_sl_mult
                tp_dist = entry_atr * current_tp_mult
                
                if position > 0:
                    if current_sl_mult > 0 and l_prev <= (entry_price - sl_dist):
                        exit_trade = True
                        barrier_price = entry_price - sl_dist
                    elif current_tp_mult > 0 and h_prev >= (entry_price + tp_dist):
                        exit_trade = True
                        barrier_price = entry_price + tp_dist
                else:
                    if current_sl_mult > 0 and h_prev >= (entry_price + sl_dist):
                        exit_trade = True
                        barrier_price = entry_price + sl_dist
                    elif current_tp_mult > 0 and l_prev <= (entry_price - tp_dist):
                        exit_trade = True
                        barrier_price = entry_price - tp_dist
                        
        if exit_trade:
            position = 0.0
            
        # 2. Check Entries (Priority Mux)
        if position == 0.0 and not (hours[i] >= end_hour or weekdays[i] >= 5):
            for s_idx in range(n_strats):
                sig = sig_matrix[i, s_idx]
                if sig != 0:
                    position = float(sig)
                    entry_price = prices[i]
                    entry_idx = i
                    entry_atr = atr[i]
                    
                    # Capture Params from active strategy
                    current_horizon = horizons[s_idx]
                    current_sl_mult = sl_mults[s_idx]
                    current_tp_mult = tp_mults[s_idx]
                    break 
                    
        # 3. PnL Calculation
        curr_price = prices[i]
        if barrier_price != 0.0:
            curr_price = barrier_price
            
        if i > 0:
            price_change = curr_price - prices[i-1]
            gross_pnl = prev_pos * lot_size * price_change
            
            pos_change = abs(position - prev_pos)
            cost = pos_change * lot_size * curr_price * (0.5 * spread_pct + comm_pct)
            
            net_rets = (gross_pnl - cost) / account_size
            net_returns[i] = net_rets
            
            if pos_change > 0:
                trades_count += 1
        else:
            # First bar
            pos_change = abs(position - 0.0)
            cost = pos_change * lot_size * curr_price * (0.5 * spread_pct + comm_pct)
            net_returns[i] = -cost / account_size
            if pos_change > 0:
                trades_count += 1
                
        prev_pos = position
        
    return net_returns, trades_count

def simulate_mutex_portfolio(backtester, unique_strats, sig_matrix, prices, highs, lows, atr, times):
    # Extract Params
    horizons = np.array([s.horizon for s in unique_strats], dtype=np.int64)
    sl_mults = np.array([getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for s in unique_strats], dtype=np.float64)
    tp_mults = np.array([getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for s in unique_strats], dtype=np.float64)
    
    # Time Features
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values.astype(np.int8)
        weekdays = times.dt.dayofweek.values.astype(np.int8)
    else:
        dt_idx = pd.to_datetime(times)
        hours = dt_idx.hour.values.astype(np.int8)
        weekdays = dt_idx.dayofweek.values.astype(np.int8)
        
    return _jit_simulate_mutex_custom(
        sig_matrix.astype(np.float64),
        prices.astype(np.float64),
        highs.astype(np.float64),
        lows.astype(np.float64),
        atr.astype(np.float64),
        hours,
        weekdays,
        horizons,
        sl_mults,
        tp_mults,
        config.STANDARD_LOT_SIZE,
        config.SPREAD_BPS / 10000.0,
        config.COST_BPS / 10000.0,
        config.ACCOUNT_SIZE,
        config.TRADING_END_HOUR
    )

def run_mutex_backtest():
    print("\n" + "="*80)
    print("🔒 MUTEX STRATEGY EXECUTION (Single Slot) 🔒")
    print("="*80 + "\n")
    
    # 1. Setup
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # Initialize Simulator (reusing backtester configs)
    simulator = TradeSimulator(
        prices=backtester.open_vec.flatten(),
        times=backtester.times_vec,
        spread_bps=backtester.spread_bps,
        cost_bps=backtester.effective_cost_bps,
        lot_size=backtester.standard_lot,
        account_size=backtester.account_size
    )
    
    # 2. Selection
    candidates = load_all_candidates()
    
    # --- RE-VALIDATION STEP ---
    print(f"\nRe-Validating {len(candidates)} candidates under strict Safe Entry rules...")
    if not candidates:
        print("❌ No candidates to validate.")
        return

    # Generate signals for all (with new Safe Entry logic)
    full_sig_matrix = backtester.generate_signal_matrix(candidates)
    
    # Shift for Next-Open Execution
    full_sig_matrix = np.vstack([np.zeros((1, full_sig_matrix.shape[1]), dtype=full_sig_matrix.dtype), full_sig_matrix[:-1]])
    
    # Slice for OOS (Test) Data ONLY
    split_idx = backtester.val_idx
    val_sig_matrix = full_sig_matrix[split_idx:]

    print(f"  Validating on OOS data (Indices {split_idx} to {len(df)})...")
    
    # Batch Simulate
    val_rets, val_trades = backtester.run_simulation_batch(
        val_sig_matrix, 
        candidates, 
        backtester.open_vec[split_idx:], 
        backtester.times_vec.iloc[split_idx:] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[split_idx:],
        highs=backtester.high_vec[split_idx:],
        lows=backtester.low_vec[split_idx:],
        atr=backtester.atr_vec[split_idx:]
    )
    
    valid_candidates = []
    total_dropped = 0
    
    for i, strat in enumerate(candidates):
        total_pnl = np.sum(val_rets[:, i])
        n_trades = val_trades[i]
        expectancy = total_pnl / n_trades if n_trades > 0 else 0

        if total_pnl > 0 and expectancy > 0.0005:
            # Update internal metrics to reflect REALITY
            avg_ret = np.mean(val_rets[:, i])
            downside = np.sqrt(np.mean(np.minimum(val_rets[:, i], 0)**2)) + 1e-9
            real_sortino = (avg_ret / downside) * np.sqrt(backtester.annualization_factor)
            
            strat.sortino = real_sortino
            valid_candidates.append(strat)
        else:
            total_dropped += 1
            
    print(f"  Dropped {total_dropped} strategies failing new safety rules.")
    print(f"  Retained {len(valid_candidates)} profitable strategies.")
    
    candidates = valid_candidates
    
    unique_strats, sig_matrix = filter_global_correlation(candidates, backtester, threshold=0.5)

    if not unique_strats:
        print("❌ No unique strategies found.")
        return

    print("\nSelected Portfolio (Ranked Priority):")
    mutex_data = []
    for i, s in enumerate(unique_strats):
        print(f"  {i+1}. {s.name} (H{s.horizon}) | ID: {getattr(s, 'training_id', 'legacy')} | Sortino: {s.sortino:.2f}")
        s_dict = s.to_dict()
        s_dict['horizon'] = s.horizon
        s_dict['training_id'] = getattr(s, 'training_id', 'legacy')
        mutex_data.append(s_dict)

    mutex_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
    with open(mutex_path, 'w') as f:
        json.dump(mutex_data, f, indent=4)
    print(f"💾 Saved Mutex Portfolio to {mutex_path}")

    # 3. Execution (Raw Composite Signal)
    # --- FIX: LOOKAHEAD BIAS (Next Open Execution) ---
    # Shift signals forward by 1
    sig_matrix = np.vstack([np.zeros((1, sig_matrix.shape[1]), dtype=sig_matrix.dtype), sig_matrix[:-1]])
    
    # --- STRICT OOS REPORTING ---
    oos_start = backtester.val_idx
    print(f"\nCalculating Performance on OOS Data ONLY (Index {oos_start}+)...")
    
    # Prepare OOS Data Slices
    sig_matrix_oos = sig_matrix[oos_start:]
    open_oos = backtester.open_vec[oos_start:]
    highs_oos = backtester.high_vec[oos_start:]
    lows_oos = backtester.low_vec[oos_start:]
    atr_oos = backtester.atr_vec[oos_start:]
    times_oos = backtester.times_vec.iloc[oos_start:] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[oos_start:]
    
    # Run Mutex Simulator on OOS Slice
    net_rets, trades_count = simulate_mutex_portfolio(
        backtester, 
        unique_strats, 
        sig_matrix_oos,
        open_oos,
        highs_oos,
        lows_oos,
        atr_oos,
        times_oos
    )
    
    # Performance Metrics
    results = []
    
    # 1. Mutex Portfolio Metrics
    valid_ret = net_rets
    valid_cum_ret = np.cumsum(valid_ret)
    
    # Max Drawdown Calculation
    running_max = np.maximum.accumulate(valid_cum_ret)
    running_max = np.maximum(0, running_max)
    drawdown = valid_cum_ret - running_max
    max_dd = np.min(drawdown)
    
    total_ret_pct = np.sum(valid_ret)
    total_profit = total_ret_pct * backtester.account_size
    
    avg_ret = np.mean(valid_ret)
    downside_sq = np.mean(np.minimum(valid_ret, 0)**2)
    downside = np.sqrt(downside_sq) + 1e-9
    
    sortino = (avg_ret / downside) * np.sqrt(backtester.annualization_factor)
    sharpe = (np.mean(valid_ret) / np.std(valid_ret)) * np.sqrt(backtester.annualization_factor)
    
    results.append({
        'Name': 'MUTEX PORTFOLIO',
        'TrainID': 'N/A',
        'Profit': total_profit,
        'Return%': total_ret_pct * 100,
        'MaxDD%': max_dd * 100,
        'Sortino': sortino,
        'Sharpe': sharpe,
        'Trades': int(trades_count) 
    })
    
    # 2. Individual Strategy Metrics
    print("\nBenchmarking Individual Strategies (OOS Only)...")
    
    for i, strat in enumerate(unique_strats):
        # Slice OOS Signals
        single_sig_matrix_oos = sig_matrix_oos[:, [i]]
        
        s_net_rets_batch, s_trades_batch = backtester.run_simulation_batch(
            single_sig_matrix_oos, 
            [strat], 
            open_oos, 
            times_oos, 
            time_limit=strat.horizon,
            highs=highs_oos,
            lows=lows_oos,
            atr=atr_oos
        )
        
        s_net_rets = s_net_rets_batch[:, 0]
        s_trades = s_trades_batch[0]
        
        # Metrics
        s_valid_ret = s_net_rets
        s_valid_cum_ret = np.cumsum(s_valid_ret)
        
        s_running_max = np.maximum(0, np.maximum.accumulate(s_valid_cum_ret))
        s_drawdown = s_valid_cum_ret - s_running_max
        s_max_dd = np.min(s_drawdown)
        
        s_profit = np.sum(s_valid_ret) * backtester.account_size
        s_ret_pct = np.sum(s_valid_ret)
        s_avg = np.mean(s_valid_ret)
        s_downside = np.sqrt(np.mean(np.minimum(s_valid_ret, 0)**2)) + 1e-9
        s_sortino = (s_avg / s_downside) * np.sqrt(backtester.annualization_factor)
        s_sharpe = (np.mean(s_valid_ret) / np.std(s_valid_ret)) * np.sqrt(backtester.annualization_factor)
        
        results.append({
            'Name': f"{strat.name} (H{strat.horizon})",
            'TrainID': getattr(strat, 'training_id', 'legacy'),
            'Profit': s_profit,
            'Return%': s_ret_pct * 100,
            'MaxDD%': s_max_dd * 100,
            'Sortino': s_sortino,
            'Sharpe': s_sharpe,
            'Trades': int(s_trades)
        })
        
    # Print Table
    results.sort(key=lambda x: x['Sortino'], reverse=True)
    
    print("\n" + "="*110)
    print("🏆 PERFORMANCE LEADERBOARD (Mutex vs. Components)")
    print("="*110)
    print(f"{ 'Name':<30} | {'TrainID':<8} | {'Profit ($)':<12} | {'Ret %':<8} | {'MaxDD %':<8} | {'Sortino':<8} | {'Sharpe':<8} | {'Trades':<6}")
    print("-" * 110)
    
    for r in results:
        print(f"{r['Name']:<30} | {r['TrainID']:<8} | {r['Profit']:>12,.2f} | {r['Return%']:>8.2f} | {r['MaxDD%']:>8.2f} | {r['Sortino']:>8.2f} | {r['Sharpe']:>8.2f} | {r['Trades']:>6}")
    print("-" * 110)    
    
    # Mutex Curve
    mutex_curve = valid_cum_ret * backtester.account_size
    
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(mutex_curve)), mutex_curve, label='Mutex Portfolio', color='purple', linewidth=2)
    plt.title(f"Mutex Portfolio Performance\nReturn: {total_ret_pct*100:.2f}% | Sortino: {sortino:.2f} | Trades: {int(trades_count)}")
    plt.ylabel("Cumulative Profit ($)")
    plt.xlabel("Bars")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = os.path.join(config.DIRS['PLOTS_DIR'], "mutex_performance.png")
    plt.savefig(out_path)
    print(f"\n📸 Saved Chart to {out_path}")

if __name__ == "__main__":
    run_mutex_backtest()
