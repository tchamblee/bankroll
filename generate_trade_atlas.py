import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import shutil
import config
from genome import Strategy
from backtest import BacktestEngine
from trade_simulator import TradeSimulator

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
        idx = event['idx']
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
    print("🌍 GENERATING TRADE ATLAS (Mutex Portfolio) 🌍")
    print("======================================================\n")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    portfolio_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
    if not os.path.exists(portfolio_path):
         print("❌ Mutex portfolio not found. Run 'run_mutex_strategy.py' first.")
         return
    
    with open(portfolio_path, 'r') as f:
         portfolio_data = json.load(f)
         
    strategies = [Strategy.from_dict(d) for d in portfolio_data]
    # Re-attach horizon info if missing in Strategy object (it's not a standard field)
    for s, d in zip(strategies, portfolio_data):
        s.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)

    # Load Full Data
    print("Loading Market Data...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df)
    
    if 'time_start' in df.columns:
        times = df['time_start']
    else:
        times = pd.Series(df.index)

    # Initialize Simulator
    simulator = TradeSimulator(
        prices=backtester.close_vec.flatten(),
        times=times,
        spread_bps=config.SPREAD_BPS, 
        cost_bps=config.COST_BPS
    )

    # Output Root
    atlas_root = os.path.join(config.DIRS['OUTPUT_DIR'], "trade_atlas")
    if not os.path.exists(atlas_root): os.makedirs(atlas_root)
    
    # Clean previous atlas
    # shutil.rmtree(atlas_root)
    # os.makedirs(atlas_root)

    print(f"Generating Atlas for {len(strategies)} Strategies in Mutex Portfolio...")
    print(f"NOTE: Only plotting trades in OOS period (Start Index: {backtester.val_idx})")

    for i, strat in enumerate(strategies):
        h = getattr(strat, 'horizon', 120)
        print(f"\n[{i+1}/{len(strategies)}] Analyzing {strat.name} (H{h})...")
        
        # 2. Generate Signals
        signals = backtester.generate_signal_matrix([strat], horizon=h).flatten()
        
        # 3. Simulate Trades
        sl_pct = getattr(strat, 'stop_loss_pct', config.DEFAULT_STOP_LOSS)
        tp_pct = getattr(strat, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT)
        
        trades, _ = simulator.simulate(signals, stop_loss_pct=sl_pct, take_profit_pct=tp_pct, time_limit_bars=h)
        
        # FILTER: OOS Only
        oos_start = backtester.val_idx
        valid_trades = [t for t in trades if t.entry_idx >= oos_start]
        
        print(f"  Found {len(trades)} total trades. {len(valid_trades)} in OOS period.")
        
        if not valid_trades:
            print("  No OOS trades to plot.")
            continue

        # 4. Generate Plots
        strat_dir = os.path.join(atlas_root, f"{strat.name}_H{h}")
        if not os.path.exists(strat_dir): os.makedirs(strat_dir)
        else:
            shutil.rmtree(strat_dir)
            os.makedirs(strat_dir)
        
        # Limit to first 100 trades to prevent excessive rendering
        plot_limit = 100
        for j, trade in enumerate(valid_trades[:plot_limit]):
            plot_trade(trade, backtester.close_vec.flatten(), times, strat_dir, j+1, h)
            if (j+1) % 50 == 0:
                print(f"    ... Plotted {j+1}/{min(len(valid_trades), plot_limit)} trades")
                
        print(f"  ✅ Saved {min(len(valid_trades), plot_limit)} plots to {strat_dir}")

if __name__ == "__main__":
    main()
