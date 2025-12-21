import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import config
from strategy_genome import Strategy
from backtest_engine import BacktestEngine

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
                # Load metrics for ranking
                metrics = d.get('metrics', {})
                s.sortino = metrics.get('sortino_oos', 0)
                s.robust = metrics.get('robust_return', 0)
                candidates.append(s)
            except: pass
            
    # Sort candidates globally by Sortino (Priority)
    candidates.sort(key=lambda x: x.sortino, reverse=True)
    print(f"  Loaded {len(candidates)} total candidates.")
    return candidates

def filter_global_correlation(candidates, backtester, threshold=0.7):
    print(f"\nRunning Global Correlation Filter (Threshold: {threshold})...")
    
    # Generate signals for all (expensive but necessary)
    # We use a large chunk of data or the full set
    # Using full set ensures validity
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

def simulate_mutex_portfolio(backtester, unique_strats, sig_matrix):
    n_bars = sig_matrix.shape[0]
    warmup = 3200
    
    # Vectors
    final_pos = np.zeros(n_bars)
    active_strat_idx = -1 
    current_lots = 0
    
    # Simulation Loop
    for t in range(warmup, n_bars):
        if active_strat_idx == -1:
            # IDLE
            for i in range(len(unique_strats)):
                sig = sig_matrix[t, i]
                if sig != 0:
                    active_strat_idx = i
                    current_lots = sig
                    final_pos[t] = current_lots
                    break
        else:
            # LOCKED
            sig = sig_matrix[t, active_strat_idx]
            if sig == 0:
                final_pos[t] = 0
                active_strat_idx = -1
                current_lots = 0
            elif np.sign(sig) != np.sign(current_lots):
                final_pos[t] = sig
                current_lots = sig
            else:
                final_pos[t] = sig
                current_lots = sig
                
    return final_pos

def run_mutex_backtest():
    print("\n" + "="*80)
    print("🔒 MUTEX STRATEGY EXECUTION (Single Slot) 🔒")
    print("="*80 + "\n")
    
    # 1. Setup
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df, annualization_factor=181440)
    
    # 2. Selection
    candidates = load_all_candidates()
    unique_strats, sig_matrix = filter_global_correlation(candidates, backtester, threshold=0.6)
    
    if not unique_strats:
        print("❌ No unique strategies found.")
        return

    print("\nSelected Portfolio (Ranked Priority):")
    for i, s in enumerate(unique_strats):
        print(f"  {i+1}. {s.name} (H{s.horizon}) | Sortino: {s.sortino:.2f}")

    # 3. Execution
    final_pos = simulate_mutex_portfolio(backtester, unique_strats, sig_matrix)
    warmup = 3200

    # 4. Calculate PnL (Vectorized on final_pos)
    # Shift pos to align with returns
    pos_shifted = np.roll(final_pos, 1)
    pos_shifted[0] = 0
    
    rets = backtester.returns_vec.flatten()
    prices = backtester.close_vec.flatten()
    
    # Gross
    gross_pnl = pos_shifted * backtester.standard_lot * prices * rets
    
    # Costs
    lot_change = np.abs(np.diff(final_pos, prepend=0))
    costs = lot_change * backtester.standard_lot * prices * backtester.total_cost_pct
    
    net_pnl = gross_pnl - costs
    
    # --- Comparative Analysis ---
    results = []
    
    # 1. Mutex Portfolio Metrics
    valid_pnl = net_pnl[warmup:]
    valid_ret = valid_pnl / backtester.account_size
    valid_cum_ret = np.cumsum(valid_ret)
    
    # Max Drawdown Calculation
    running_max = np.maximum.accumulate(valid_cum_ret)
    running_max = np.maximum(0, running_max)
    drawdown = valid_cum_ret - running_max
    max_dd = np.min(drawdown)
    
    total_profit = np.sum(valid_pnl)
    total_ret_pct = total_profit / backtester.account_size
    avg_ret = np.mean(valid_ret)
    
    # Correct Downside Deviation (RMSE of negative returns)
    downside_sq = np.mean(np.minimum(valid_ret, 0)**2)
    downside = np.sqrt(downside_sq) + 1e-9
    
    sortino = (avg_ret / downside) * np.sqrt(backtester.annualization_factor)
    sharpe = (np.mean(valid_ret) / np.std(valid_ret)) * np.sqrt(backtester.annualization_factor)
    trades = np.sum(lot_change[warmup:])
    
    results.append({
        'Name': 'MUTEX PORTFOLIO',
        'Profit': total_profit,
        'Return%': total_ret_pct * 100,
        'MaxDD%': max_dd * 100,
        'Sortino': sortino,
        'Sharpe': sharpe,
        'Trades': int(trades)
    })
    
    # 2. Individual Strategy Metrics
    print("\nBenchmarking Individual Strategies...")
    for i, strat in enumerate(unique_strats):
        # Generate single signal vector
        # We already have sig_matrix[:, i]
        sig = sig_matrix[:, i]
        
        # Calculate PnL for this specific strategy
        s_pos = np.roll(sig, 1)
        s_pos[0] = 0
        
        s_gross = s_pos * backtester.standard_lot * prices * rets
        s_change = np.abs(np.diff(sig, prepend=0))
        s_costs = s_change * backtester.standard_lot * prices * backtester.total_cost_pct
        s_net = s_gross - s_costs
        
        # Metrics
        s_valid = s_net[warmup:]
        s_valid_ret = s_valid / backtester.account_size
        s_valid_cum_ret = np.cumsum(s_valid_ret)
        
        # Drawdown
        s_running_max = np.maximum(0, np.maximum.accumulate(s_valid_cum_ret))
        s_drawdown = s_valid_cum_ret - s_running_max
        s_max_dd = np.min(s_drawdown)
        
        s_profit = np.sum(s_valid)
        s_ret_pct = s_profit / backtester.account_size
        s_avg = np.mean(s_valid_ret)
        s_downside = np.sqrt(np.mean(np.minimum(s_valid_ret, 0)**2)) + 1e-9
        s_sortino = (s_avg / s_downside) * np.sqrt(backtester.annualization_factor)
        s_sharpe = (np.mean(s_valid_ret) / np.std(s_valid_ret)) * np.sqrt(backtester.annualization_factor)
        s_trades = np.sum(s_change[warmup:])
        
        results.append({
            'Name': f"{strat.name} (H{strat.horizon})",
            'Profit': s_profit,
            'Return%': s_ret_pct * 100,
            'MaxDD%': s_max_dd * 100,
            'Sortino': s_sortino,
            'Sharpe': s_sharpe,
            'Trades': int(s_trades)
        })
        
    # Print Table
    results.sort(key=lambda x: x['Sortino'], reverse=True)
    
    print("\n" + "="*95)
    print("🏆 PERFORMANCE LEADERBOARD (Mutex vs. Components)")
    print("="*95)
    print(f"{ 'Name':<30} | {'Profit ($)':<12} | {'Ret %':<8} | {'MaxDD %':<8} | {'Sortino':<8} | {'Sharpe':<8} | {'Trades':<6}")
    print("-" * 95)
    
    for r in results:
        print(f"{r['Name']:<30} | {r['Profit']:>12,.2f} | {r['Return%']:>8.2f} | {r['MaxDD%']:>8.2f} | {r['Sortino']:>8.2f} | {r['Sharpe']:>8.2f} | {r['Trades']:>6}")
    print("-" * 95)
    
    # 5. Visualize
    cum_pnl = np.cumsum(net_pnl)
    valid_cum_pnl = cum_pnl[warmup:] - cum_pnl[warmup] # Start from 0
    
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(valid_cum_pnl)), valid_cum_pnl, label='Mutex Portfolio', color='purple', linewidth=2)
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
