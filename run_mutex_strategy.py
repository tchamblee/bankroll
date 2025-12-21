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
                # print(f"    Skipping {candidate.name} (H{candidate.horizon}): Correlated with {selected[selected_indices.index(existing_idx)].name} ({corr:.2f})")
                break
        
        if is_unique:
            selected.append(candidate)
            selected_indices.append(i)
            
    print(f"  Selected {len(selected)} globally unique strategies.")
    return selected, sig_matrix[:, selected_indices]

def run_mutex_backtest():
    print("\n" + "="*80)
    print("🔒 MUTEX STRATEGY EXECUTION (Single Slot) 🔒")
    print("="*80 + "\n")
    
    # 1. Setup
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df)
    
    # 2. Selection
    candidates = load_all_candidates()
    unique_strats, sig_matrix = filter_global_correlation(candidates, backtester, threshold=0.6) # Strict global filter
    
    if not unique_strats:
        print("❌ No unique strategies found.")
        return

    print("\nSelected Portfolio (Ranked Priority):")
    for i, s in enumerate(unique_strats):
        print(f"  {i+1}. {s.name} (H{s.horizon}) | Sortino: {s.sortino:.2f}")

    # 3. Execution Simulation
    n_bars = sig_matrix.shape[0]
    warmup = 3200
    
    # Vectors
    final_pos = np.zeros(n_bars)
    active_strat_idx = -1 # None
    trade_log = [] # (Start, End, StratName, PnL) 
    
    # Trade State
    entry_price = 0.0
    entry_idx = 0
    current_lots = 0
    
    prices = backtester.close_vec.flatten()
    # We iterate bar by bar (vectorizing mutex logic is hard due to path dependence)
    # Optimization: We can jump? No, lock can release any time.
    
    print("\nRunning Simulation...")
    for t in range(warmup, n_bars):
        
        if active_strat_idx == -1:
            # --- IDLE STATE: Look for Entry ---
            for i in range(len(unique_strats)):
                sig = sig_matrix[t, i]
                if sig != 0:
                    # LOCK
                    active_strat_idx = i
                    current_lots = sig # 1 to 3
                    final_pos[t] = current_lots
                    entry_idx = t
                    entry_price = prices[t]
                    break
        else:
            # --- LOCKED STATE: Follow Active Strategy ---
            # Get signal from the owner
            sig = sig_matrix[t, active_strat_idx]
            
            if sig == 0:
                # EXIT
                # Position closed at close of T
                final_pos[t] = 0
                active_strat_idx = -1
                current_lots = 0
            elif np.sign(sig) != np.sign(current_lots):
                # FLIP
                # Old pos closes, new opens
                final_pos[t] = sig
                current_lots = sig
            else:
                # HOLD / RESIZE
                final_pos[t] = sig
                current_lots = sig

    # 4. Calculate PnL (Vectorized on final_pos)
    # Shift pos to align with returns
    pos_shifted = np.roll(final_pos, 1)
    pos_shifted[0] = 0
    
    rets = backtester.returns_vec.flatten()
    
    # Gross
    gross_pnl = pos_shifted * backtester.standard_lot * prices * rets
    
    # Costs
    lot_change = np.abs(np.diff(final_pos, prepend=0))
    costs = lot_change * backtester.standard_lot * prices * backtester.total_cost_pct
    
    net_pnl = gross_pnl - costs
    
    # Stats
    total_profit = np.sum(net_pnl)
    total_ret_pct = total_profit / backtester.account_size
    
    # Trades count
    trades = np.sum(lot_change[warmup:])
    
    # Sortino
    valid_pnl = net_pnl[warmup:]
    avg_ret = np.mean(valid_pnl / backtester.account_size)
    downside = np.std(np.minimum(valid_pnl / backtester.account_size, 0)) + 1e-9
    sortino = (avg_ret / downside) * np.sqrt(backtester.annualization_factor)
    
    print("\n" + "="*40)
    print("🔒 MUTEX RESULTS")
    print("="*40)
    print(f"Total Profit:    ${total_profit:,.2f}")
    print(f"Return on Acct:  {total_ret_pct*100:.2f}%")
    print(f"Sortino Ratio:   {sortino:.2f}")
    print(f"Total Trades:    {int(trades)}")
    
    # 5. Visualize
    cum_pnl = np.cumsum(net_pnl)
    valid_cum_pnl = cum_pnl[warmup:] - cum_pnl[warmup] # Start from 0
    
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(valid_cum_pnl)), valid_cum_pnl, label='Mutex Portfolio', color='purple', linewidth=2)
    plt.title(f"Mutex Portfolio Performance\nReturn: {total_ret_pct*100:.2f}% | Sortino: {sortino:.2f} | Trades: {int(trades)}")
    plt.ylabel("Cumulative Profit ($)")
    plt.xlabel("Bars")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = os.path.join(config.DIRS['PLOTS_DIR'], "mutex_performance.png")
    plt.savefig(out_path)
    print(f"\n📸 Saved Chart to {out_path}")

if __name__ == "__main__":
    run_mutex_backtest()
