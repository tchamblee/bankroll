import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import shutil
import config
from strategy_genome import Strategy
from backtest_engine import BacktestEngine
from trade_simulator import TradeSimulator

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

def plot_trade(trade, prices, times, output_dir, trade_id, horizon):
    # Adapter for Trade object
    start_idx = trade.entry_idx
    end_idx = trade.exit_idx if trade.exit_idx is not None else len(prices) - 1
    
    trade_type = "Long" if trade.direction > 0 else "Short"
    
    # Context window
    context = 20 # bars
    start = max(0, start_idx - context)
    end = min(len(prices), end_idx + context)
    
    # Handle Series vs Array
    if hasattr(times, 'iloc'):
        section_times = times.iloc[start:end]
    else:
        section_times = times[start:end]
        
    section_prices = prices[start:end]
    
    plt.figure(figsize=(12, 6))
    
    # 1. Plot Price Line
    plt.plot(section_times, section_prices, color='black', alpha=0.6, linewidth=1.5, label='EUR/USD')
    
    # 2. Shade Active Region
    # Create mask for active period
    # Need to match index if times is Series
    if hasattr(section_times, 'index'):
        active_mask = (section_times.index >= start_idx) & (section_times.index <= end_idx)
    else:
        # Assuming times is list-like aligned with prices
        # Create a boolean mask manually
        active_mask = np.zeros(len(section_times), dtype=bool)
        # local indices
        s_loc = max(0, start_idx - start)
        e_loc = min(len(section_times), end_idx - start + 1)
        active_mask[s_loc:e_loc] = True
    
    if trade_type == 'Long':
        plt.fill_between(section_times, section_prices, min(section_prices)*0.99, where=active_mask, color='green', alpha=0.1)
    else:
        plt.fill_between(section_times, section_prices, max(section_prices)*1.01, where=active_mask, color='red', alpha=0.1)

    # 3. Mark Events
    for event in trade.events:
        idx = event['idx'] # Changed from get('idx') since Trade uses dataclass/list of dicts
        if idx < start or idx >= end: continue
        
        local_idx = idx - start
        if local_idx < 0 or local_idx >= len(section_times): continue
        
        if hasattr(section_times, 'iloc'):
            t = section_times.iloc[local_idx]
        else:
            t = section_times[local_idx]
            
        p = section_prices[local_idx]
        
        action = event['action']
        size = event.get('size', 0)
        
        if "ENTRY" in action:
            marker = '^' if trade_type == 'Long' else 'v'
            color = 'green' if trade_type == 'Long' else 'red'
            plt.scatter(t, p, color=color, marker=marker, s=100, zorder=5)
            plt.annotate(f"{action}\n{size}L", (t, p), xytext=(0, 20 if trade_type=='Long' else -20), 
                         textcoords='offset points', ha='center', color=color, fontweight='bold')
            
        elif "EXIT" in action or "SL" in action or "TP" in action or "TIME" in action:
            plt.scatter(t, p, color='blue', marker='x', s=100, zorder=5)
            plt.annotate(f"{action}", (t, p), xytext=(0, 10), textcoords='offset points', ha='center', color='blue')
            
    # 4. Title & PnL
    pnl_color = 'green' if trade.pnl > 0 else 'red'
    plt.title(f"Horizon {horizon} | Trade #{trade_id} | Type: {trade_type} | PnL: ${trade.pnl:.2f}", fontsize=14, color='black')
    
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Save
    fname = f"trade_{trade_id:04d}_{trade_type}.png"
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
    
    if 'time_start' in df.columns:
        times = df['time_start']
    else:
        times = pd.Series(df.index)

    # Initialize Simulator
    # Reuse cost model from backtester
    simulator = TradeSimulator(
        prices=backtester.close_vec.flatten(),
        times=times,
        spread_bps=1.0, # Default per backtester config
        cost_bps=0.5
    )

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
            
        # Print Top 10
        print(f"  🏆 Top 10 Strategies:")
        for i, s in enumerate(strategies[:10]):
            print(f"  {i+1:<4} {s.name:<15} {s.robust_return*100:6.2f}%")
            
        top_strat = strategies[0]
        print(f"\n  👑 Selected Champion: {top_strat.name}")
        
        # 2. Generate Signals
        print("  Generating Signals...")
        backtester.ensure_context([top_strat])
        signals = top_strat.generate_signal(backtester.context, cache={})
        
        # 3. Simulate Trades
        # Barriers: Time = Horizon, SL = 0.5%
        print(f"  Simulating with TimeLimit={h}, SL=0.5%...")
        trades, _ = simulator.simulate(signals, stop_loss_pct=0.005, time_limit_bars=h)
        
        warmup = 3200
        valid_trades = [t for t in trades if t.entry_idx >= warmup]
        print(f"  Found {len(valid_trades)} valid trades. Generating plots...")
        
        # 4. Generate Plots
        h_dir = os.path.join(atlas_root, f"horizon_{h}")
        if not os.path.exists(h_dir): os.makedirs(h_dir)
        else:
            shutil.rmtree(h_dir)
            os.makedirs(h_dir)
        
        for i, trade in enumerate(valid_trades):
            plot_trade(trade, backtester.close_vec.flatten(), times, h_dir, i+1, h)
            if (i+1) % 50 == 0:
                print(f"    ... Plotted {i+1}/{len(valid_trades)} trades")
                
        print(f"  ✅ Saved {len(valid_trades)} plots to {h_dir}")

if __name__ == "__main__":
    main()