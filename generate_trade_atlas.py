import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import config
from strategy_genome import Strategy
from backtest_engine import BacktestEngine
from run_mutex_strategy import load_all_candidates, filter_global_correlation, simulate_mutex_portfolio

def load_and_rank_strategies(horizon):
    # Prefer the Top 5 Unique file which contains pre-calculated metrics from report_top_strategies.py
    file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}_top5_unique.json")
    if not os.path.exists(file_path):
        # Fallback to Top 10 if Top 5 Unique doesn't exist (legacy support)
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}_top10.json")
    
    if not os.path.exists(file_path):
        # Final Fallback to raw file (metrics might be missing/defaulted to -999)
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")
        
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    strategies = []
    for d in data:
        try:
            strat = Strategy.from_dict(d)
            # Hydrate metrics if available, or assume they are in the dict
            # The JSON saved by report_top_strategies.py has a 'metrics' key
            metrics = d.get('metrics', {})
            strat.robust_return = metrics.get('robust_return', -999)
            strat.full_return = metrics.get('full_return', -999)
            strat.val_return = metrics.get('val_return', -999)
            strat.test_return = metrics.get('test_return', -999)
            strat.sortino_oos = metrics.get('sortino_oos', -999)
            strat.generation_found = d.get('generation', '?')
            strategies.append(strat)
        except Exception as e:
            pass
            
    # Sort: Primary = Robust%, Secondary = Ret%(Full)
    strategies.sort(key=lambda x: (x.robust_return, x.full_return), reverse=True)
    return strategies

def extract_trades(signals, prices, times, lot_size, cost_pct):
    trades = []
    
    # Position state
    in_trade = False
    trade_start_idx = 0
    trade_type = None # 'Long' or 'Short'
    trade_events = [] # list of (idx, lot_change, current_lots, price, type)
    trade_pnl_accum = 0.0
    
    # Iterate
    # signals[i] is position at close of bar i.
    # We enter at close of i (price[i]).
    # PnL starts accruing from i+1.
    
    n = len(signals)
    
    for i in range(1, n):
        prev_pos = signals[i-1]
        curr_pos = signals[i]
        price = prices[i]
        prev_price = prices[i-1]
        
        # 1. Accrue PnL for holding from i-1 to i
        if prev_pos != 0:
            gross_pnl = prev_pos * lot_size * (price - prev_price)
            # No cost for holding
            trade_pnl_accum += gross_pnl
        
        # 2. Check for Position Change (Transactions)
        if curr_pos != prev_pos:
            change = curr_pos - prev_pos
            cost = abs(change) * lot_size * price * cost_pct
            
            # If we were in a trade, deduct cost from that trade
            if in_trade:
                trade_pnl_accum -= cost
            
            # Determine Action
            action = ""
            if prev_pos == 0 and curr_pos != 0: # OPEN
                in_trade = True
                trade_start_idx = i
                trade_type = 'Long' if curr_pos > 0 else 'Short'
                trade_pnl_accum = -cost # Start with transaction cost
                action = "ENTRY"
                trade_events.append({'idx': i, 'lots': curr_pos, 'price': price, 'action': action, 'change': change})
                
            elif prev_pos != 0 and curr_pos == 0: # CLOSE
                action = "EXIT"
                trade_events.append({'idx': i, 'lots': 0, 'price': price, 'action': action, 'change': change})
                
                # Close Trade
                trades.append({
                    'start_idx': trade_start_idx,
                    'end_idx': i,
                    'type': trade_type,
                    'events': trade_events,
                    'pnl': trade_pnl_accum
                })
                in_trade = False
                trade_events = []
                trade_pnl_accum = 0.0
                
            elif prev_pos != 0 and curr_pos != 0: 
                # FLIP or RESIZE
                if np.sign(curr_pos) != np.sign(prev_pos): # FLIP
                    # Close current
                    trade_events.append({'idx': i, 'lots': 0, 'price': price, 'action': "FLIP_EXIT", 'change': -prev_pos})
                    trades.append({
                        'start_idx': trade_start_idx,
                        'end_idx': i,
                        'type': trade_type,
                        'events': trade_events,
                        'pnl': trade_pnl_accum
                    })
                    
                    # Open new
                    in_trade = True
                    trade_start_idx = i
                    trade_type = 'Long' if curr_pos > 0 else 'Short'
                    # Cost was fully deducted above? 
                    # change = curr - prev. 
                    # Cost = abs(curr - prev) * ...
                    # This cost covers closing prev and opening curr.
                    # We assigned it all to the previous trade above. 
                    # Ideally split it? Or just let previous trade take the hit. 
                    # Let's reset PnL for new trade to 0 (free entry? No).
                    # Simplified: Flip cost belongs to the "exit" of the old trade effectively. 
                    trade_pnl_accum = 0.0 
                    trade_events = [{'idx': i, 'lots': curr_pos, 'price': price, 'action': "FLIP_ENTRY", 'change': curr_pos}]
                    
                else: # RESIZE (Add or Reduce)
                    action = "ADD" if abs(curr_pos) > abs(prev_pos) else "REDUCE"
                    trade_events.append({'idx': i, 'lots': curr_pos, 'price': price, 'action': action, 'change': change})

    # Close open trade at end
    if in_trade:
        trades.append({
            'start_idx': trade_start_idx,
            'end_idx': n-1,
            'type': trade_type,
            'events': trade_events,
            'pnl': trade_pnl_accum,
            'status': 'Open'
        })
        
    return trades

def plot_trade(trade, prices, times, output_dir, trade_id, horizon):
    # Context window
    context = 20 # bars
    start = max(0, trade['start_idx'] - context)
    end = min(len(prices), trade['end_idx'] + context)
    
    section_times = times[start:end]
    section_prices = prices[start:end]
    
    plt.figure(figsize=(12, 6))
    
    # 1. Plot Price Line
    plt.plot(section_times, section_prices, color='black', alpha=0.6, linewidth=1.5, label='EUR/USD')
    
    # 2. Shade Active Region
    # Create mask for active period
    active_mask = (section_times.index >= trade['start_idx']) & (section_times.index <= trade['end_idx'])
    
    if trade['type'] == 'Long':
        plt.fill_between(section_times, section_prices, min(section_prices)*0.99, where=active_mask, color='green', alpha=0.1)
    else:
        plt.fill_between(section_times, section_prices, max(section_prices)*1.01, where=active_mask, color='red', alpha=0.1)

    # 3. Mark Events
    for event in trade['events']:
        idx = event['idx']
        if idx < start or idx >= end: continue
        
        # Get time/price for this specific index (using integer indexing on sliced array is tricky, use original df)
        # Use boolean mask on section
        # Optimization: Map global idx to section offset
        local_idx = idx - start
        if local_idx < 0 or local_idx >= len(section_times): continue
        
        t = section_times.iloc[local_idx]
        p = section_prices[local_idx]
        
        action = event['action']
        lots = event['lots']
        
        if "ENTRY" in action:
            marker = '^' if trade['type'] == 'Long' else 'v'
            color = 'green' if trade['type'] == 'Long' else 'red'
            plt.scatter(t, p, color=color, marker=marker, s=100, zorder=5)
            plt.annotate(f"{action}\n{abs(lots)}L", (t, p), xytext=(0, 20 if trade['type']=='Long' else -20), 
                         textcoords='offset points', ha='center', color=color, fontweight='bold')
            
        elif "EXIT" in action:
            plt.scatter(t, p, color='blue', marker='x', s=100, zorder=5)
            plt.annotate(f"{action}", (t, p), xytext=(0, 10), textcoords='offset points', ha='center', color='blue')
            
        elif "ADD" in action or "REDUCE" in action:
            marker = 'o'
            plt.scatter(t, p, color='orange', marker=marker, s=60, zorder=4)
            plt.annotate(f"{action}\n->{abs(lots)}L", (t, p), xytext=(0, 15), textcoords='offset points', ha='center', fontsize=8)

    # 4. Title & PnL
    pnl_color = 'green' if trade['pnl'] > 0 else 'red'
    plt.title(f"Horizon {horizon} | Trade #{trade_id} | Type: {trade['type']} | PnL: ${trade['pnl']:.2f}", fontsize=14, color='black')
    
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Save
    fname = f"trade_{trade_id:04d}_{trade['type']}.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def main():
    print("======================================================")
    print("🌍 GENERATING TRADE ATLAS (Consolidated Analysis) 🌍")
    print("======================================================\n")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    # Load Full Data
    print("Loading Market Data...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df)
    # Ensure times are available (assuming 'time_start' column exists)
    if 'time_start' in df.columns:
        times = df['time_start']
    else:
        times = pd.Series(df.index)

    # Output Root
    atlas_root = os.path.join(config.DIRS['OUTPUT_DIR'], "trade_atlas")
    if not os.path.exists(atlas_root): os.makedirs(atlas_root)

    for h in config.PREDICTION_HORIZONS:
        print(f"\nAnalyzing Horizon {h}...")
        
        # 1. Sort & Select
        strategies = load_and_rank_strategies(h)
        if not strategies:
            print(f"  No strategies found for Horizon {h}")
            continue
            
        # Print Top 10 Table
        print(f"  🏆 Top 10 Strategies (Sorted by Robust% > Full%):")
        print(f"  {'Rank':<4} {'Name':<15} {'Robust%':<10} {'Full%':<10} {'Val%':<10} {'Test%':<10} {'Sortino':<10}")
        print("-" * 80)
        
        for i, s in enumerate(strategies[:10]):
            print(f"  {i+1:<4} {s.name:<15} {s.robust_return*100:6.2f}%   {s.full_return*100:6.2f}%   {s.val_return*100:6.2f}%   {s.test_return*100:6.2f}%   {s.sortino_oos:6.2f}")
            
        top_strat = strategies[0]
        print(f"\n  👑 Selected Champion: {top_strat.name} (Robust: {top_strat.robust_return*100:.2f}%)")
        
        # 2. Generate Signals (Full History)
        print("  Generating Signals...")
        # Need to re-init context? BacktestEngine does it?
        # Re-ensure context for this specific strategy
        backtester.ensure_context([top_strat])
        signals = top_strat.generate_signal(backtester.context, cache={})
        
        # 3. Extract Trades
        # Need to know warmup to skip plotting garbage?
        # Yes, skip first 3200 bars as per previous findings.
        warmup = 3200
        
        # Signals vector includes warmup, we pass it all but tell extractor to ignore trades starting before warmup?
        # Actually, extractor iterates. We can just filter the result list.
        
        trades = extract_trades(signals, backtester.close_vec.flatten(), times, backtester.standard_lot, backtester.total_cost_pct)
        
        valid_trades = [t for t in trades if t['start_idx'] >= warmup]
        print(f"  Found {len(valid_trades)} valid trades (post-warmup). Generating plots...")
        
        # 4. Generate Plots
        h_dir = os.path.join(atlas_root, f"horizon_{h}")
        if not os.path.exists(h_dir): os.makedirs(h_dir)
        
        # Clear old plots
        import shutil
        shutil.rmtree(h_dir)
        os.makedirs(h_dir)
        
        for i, trade in enumerate(valid_trades):
            plot_trade(trade, backtester.close_vec.flatten(), times, h_dir, i+1, h)
            if (i+1) % 50 == 0:
                print(f"    ... Plotted {i+1}/{len(valid_trades)} trades")
                
        print(f"  ✅ Saved {len(valid_trades)} plots to {h_dir}")

if __name__ == "__main__":
    main()
