import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import config
from strategy_genome import Strategy
from backtest_engine import BacktestEngine
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
            # STRICT SELECTION: Only take the #1 Top Dog
            d = data[0]
            try:
                s = Strategy.from_dict(d)
                s.horizon = h
                s.training_id = d.get('training_id', 'legacy') # Capture ID
                
                # Load metrics for ranking
                metrics = d.get('metrics', {})
                s.sortino = metrics.get('sortino_oos', 0)
                s.robust = metrics.get('robust_return', 0)
                
                # Filter: Must be PROFITABLE on the FULL dataset
                # Since Mutex runs on the full history, we can't include strategies that lose money overall.
                total_ret = metrics.get('full_return', 0) 
                if s.sortino > 0 and total_ret > 0:
                    candidates.append(s)
                else:
                    print(f"  Skipping Unprofitable Strategy: {s.name} (H{h}) | Sortino: {s.sortino:.2f} | Ret: {total_ret:.4f}")

            except Exception as e:
                # print(f"Error loading {h}: {e}")
                pass
            
    # Sort candidates globally by Sortino (Priority)
    candidates.sort(key=lambda x: x.sortino, reverse=True)
    print(f"  Loaded {len(candidates)} total candidates.")
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
def _jit_simulate_mutex_portfolio(sig_matrix, warmup):
    n_bars, n_strats = sig_matrix.shape
    final_pos = np.zeros(n_bars)
    active_strat_idx = -1 
    current_lots = 0.0
    
    # Simulation Loop
    for t in range(warmup, n_bars):
        if active_strat_idx == -1:
            # IDLE
            for i in range(n_strats):
                sig = sig_matrix[t, i]
                if sig != 0:
                    active_strat_idx = i
                    current_lots = float(sig)
                    final_pos[t] = current_lots
                    break
        else:
            # LOCKED
            sig = sig_matrix[t, active_strat_idx]
            if sig == 0:
                final_pos[t] = 0.0
                active_strat_idx = -1
                current_lots = 0.0
            else:
                # We stay in the current strategy even if it changes sign (as per original logic)
                # But original logic had: elif np.sign(sig) != np.sign(current_lots):
                # Actually original logic always assigned final_pos[t] = sig
                # So it was just:
                final_pos[t] = float(sig)
                current_lots = float(sig)
                
    return final_pos

def simulate_mutex_portfolio(backtester, unique_strats, sig_matrix):
    warmup = 3200
    # Ensure sig_matrix is float64 for numba if needed, or int is fine if it matches
    return _jit_simulate_mutex_portfolio(sig_matrix.astype(np.float64), warmup)

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
        prices=backtester.close_vec.flatten(),
        times=backtester.times_vec,
        spread_bps=backtester.spread_bps,
        cost_bps=backtester.effective_cost_bps,
        lot_size=backtester.standard_lot,
        account_size=backtester.account_size
    )
    
    # 2. Selection
    candidates = load_all_candidates()
    unique_strats, sig_matrix = filter_global_correlation(candidates, backtester, threshold=0.5)
    
    if not unique_strats:
        print("❌ No unique strategies found.")
        return

    print("\nSelected Portfolio (Ranked Priority):")
    for i, s in enumerate(unique_strats):
        print(f"  {i+1}. {s.name} (H{s.horizon}) | ID: {getattr(s, 'training_id', 'legacy')} | Sortino: {s.sortino:.2f}")

    # --- PRE-FILTER MUTEX INPUTS (Safe Entry) ---
    print("  Applying Safe Entry Filters to Mutex Inputs...")
    times = backtester.times_vec
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values
        weekdays = times.dt.dayofweek.values
    else:
        dt_idx = pd.to_datetime(times)
        hours = dt_idx.hour.values
        weekdays = dt_idx.dayofweek.values
        
    for i, strat in enumerate(unique_strats):
        est_duration_hours = (strat.horizon * 2.0) / 60.0
        cutoff_hour = config.TRADING_END_HOUR - est_duration_hours
        safe_mask = (hours < cutoff_hour) & (weekdays < 5)
        sig_matrix[:, i] = sig_matrix[:, i] * safe_mask

    # 3. Execution (Raw Composite Signal)
    final_pos = simulate_mutex_portfolio(backtester, unique_strats, sig_matrix)
    warmup = 3200

    # 4. Calculate PnL (Using TradeSimulator)
    # Apply standard barriers (120 bars default, SL 0.5%)
    net_rets, trades_count = simulator.simulate_fast(final_pos, take_profit_pct=0.015, time_limit_bars=120)
    
    # --- Comparative Analysis ---
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
        'Trades': int(trades_count) # Might be off due to warmup, but close enough
    })
    
    # 2. Individual Strategy Metrics
    print("\nBenchmarking Individual Strategies...")
    for i, strat in enumerate(unique_strats):
        # Use centralized batch simulation to ensure consistency with Report (Lookahead Filter)
        single_sig_matrix = sig_matrix[:, [i]]
        
        s_net_rets_batch, s_trades_batch = backtester.run_simulation_batch(
            single_sig_matrix, 
            [strat], 
            backtester.close_vec, 
            backtester.times_vec, 
            time_limit=strat.horizon
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
    # 5. Visualize
    # Need cumulative PnL series for plot
    # Re-simulating Mutex is cheap, but we have metrics. 
    # Just reconstructcumsum.
    
    # Mutex Curve
    # We computed valid_cum_ret (percentage). 
    # Profit Curve = valid_cum_ret * account_size.
    # Start at 0.
    mutex_curve = valid_cum_ret * backtester.account_size
    
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(mutex_curve)), mutex_curve, label='Mutex Portfolio', color='purple', linewidth=2)
    plt.title(f"Mutex Portfolio Performance\nReturn: {results[0]['Return%']:.2f}% | Sortino: {results[0]['Sortino']:.2f} | Trades: {results[0]['Trades']}")
    plt.ylabel("Cumulative Profit ($)")
    plt.xlabel("Bars")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = os.path.join(config.DIRS['PLOTS_DIR'], "mutex_performance.png")
    plt.savefig(out_path)
    print(f"\n📸 Saved Chart to {out_path}")

if __name__ == "__main__":
    run_mutex_backtest()