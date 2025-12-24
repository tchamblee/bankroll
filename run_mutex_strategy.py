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
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_portfolio_{h}.json")
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
def _jit_simulate_priority_portfolio(sig_matrix, warmup):
    n_bars, n_strats = sig_matrix.shape
    final_pos = np.zeros(n_bars)
    
    # Simulation Loop: Stateless Priority Mux
    # At every bar, the highest ranked strategy (lowest index) 
    # that has a non-zero signal gets to drive the car.
    
    for t in range(warmup, n_bars):
        # Default to flat
        chosen_sig = 0.0
        
        for i in range(n_strats):
            sig = sig_matrix[t, i]
            if sig != 0:
                chosen_sig = float(sig)
                break # Found the highest priority active signal
        
        final_pos[t] = chosen_sig
                
    return final_pos

def simulate_mutex_portfolio(backtester, unique_strats, sig_matrix):
    warmup = 3200
    # Use the new Priority Preemption logic
    return _jit_simulate_priority_portfolio(sig_matrix.astype(np.float64), warmup)

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
    val_sig_matrix = backtester.generate_signal_matrix(candidates)
    
    # Shift for Next-Open Execution
    val_sig_matrix = np.vstack([np.zeros((1, val_sig_matrix.shape[1]), dtype=val_sig_matrix.dtype), val_sig_matrix[:-1]])
    
    # Batch Simulate
    val_rets, val_trades = backtester.run_simulation_batch(
        val_sig_matrix, 
        candidates, 
        backtester.open_vec, 
        backtester.times_vec,
        highs=backtester.high_vec,
        lows=backtester.low_vec,
        atr=backtester.atr_vec
    )
    
    valid_candidates = []
    total_dropped = 0
    
    for i, strat in enumerate(candidates):
        total_pnl = np.sum(val_rets[:, i])
        if total_pnl > 0:
            # Update internal metrics to reflect REALITY
            strat.sortino = 1.0 # Placeholder, just needs to be positive to pass sort
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
    for i, s in enumerate(unique_strats):
        print(f"  {i+1}. {s.name} (H{s.horizon}) | ID: {getattr(s, 'training_id', 'legacy')} | Sortino: {s.sortino:.2f}")

    # 3. Execution (Raw Composite Signal)
    # --- FIX: LOOKAHEAD BIAS (Next Open Execution) ---
    # Shift signals forward by 1
    sig_matrix = np.vstack([np.zeros((1, sig_matrix.shape[1]), dtype=sig_matrix.dtype), sig_matrix[:-1]])
    
    final_pos = simulate_mutex_portfolio(backtester, unique_strats, sig_matrix)
    warmup = 3200

    # Fast Simulation
    # TODO: Add Stop Loss from gene if supported?
    net_rets, trades_count = simulator.simulate_fast(
        final_pos, 
        take_profit_pct=config.DEFAULT_TAKE_PROFIT, 
        time_limit_bars=config.DEFAULT_TIME_LIMIT,
        highs=backtester.high_vec,
        lows=backtester.low_vec,
        atr=backtester.atr_vec
    )
    
    # Performance Metrics
    results = []
    
    # 1. Mutex Portfolio Metrics
    valid_ret = net_rets[warmup:]
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
    print("\nBenchmarking Individual Strategies...")
    for i, strat in enumerate(unique_strats):
        # Use centralized batch simulation to ensure consistency with Report (Lookahead Filter)
        single_sig_matrix = sig_matrix[:, [i]]
        
        s_net_rets_batch, s_trades_batch = backtester.run_simulation_batch(
            single_sig_matrix, 
            [strat], 
            backtester.open_vec, 
            backtester.times_vec, 
            time_limit=strat.horizon,
            highs=backtester.high_vec,
            lows=backtester.low_vec,
            atr=backtester.atr_vec
        )
        
        s_net_rets = s_net_rets_batch[:, 0]
        s_trades = s_trades_batch[0]
        
        # Metrics
        s_valid_ret = s_net_rets[warmup:]
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
