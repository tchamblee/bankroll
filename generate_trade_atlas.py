import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import shutil
import argparse
import config
from genome import Strategy
from backtest import BacktestEngine
from trade_simulator import TradeSimulator
from backtest.utils import find_strategy_in_files

def plot_trade(trade, prices, times, output_dir, trade_id, horizon):
    # Adapter for Trade object
    start_idx = trade.entry_idx
    end_idx = trade.exit_idx if trade.exit_idx is not None else len(prices) - 1
    
    trade_type = "Long" if trade.direction > 0 else "Short"
    
    # Context window
    context = 30 # bars
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
    if hasattr(section_times, 'index'):
        active_mask = (section_times.index >= start_idx) & (section_times.index <= end_idx)
    else:
        active_mask = np.zeros(len(section_times), dtype=bool)
        s_loc = max(0, start_idx - start)
        e_loc = min(len(section_times), end_idx - start + 1)
        active_mask[s_loc:e_loc] = True
    
    # Calculate dynamic Y-limits
    p_min = section_prices.min()
    p_max = section_prices.max()
    p_range = p_max - p_min
    if p_range == 0: p_range = p_min * 0.001
    
    y_bottom = p_min - (p_range * 0.15) 
    y_top = p_max + (p_range * 0.15)
    
    plt.ylim(y_bottom, y_top)
    
    if trade_type == 'Long':
        plt.fill_between(section_times, section_prices, y_bottom, where=active_mask, color='green', alpha=0.1)
    else:
        plt.fill_between(section_times, section_prices, y_top, where=active_mask, color='red', alpha=0.1)

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
            plt.scatter(t, p, color=color, marker=marker, s=120, zorder=5)
            plt.annotate(f"{action}\n{size}L", (t, p), xytext=(0, 25 if trade_type=='Long' else -25), 
                         textcoords='offset points', ha='center', color=color, fontweight='bold', fontsize=9)
            
        elif any(x in action for x in ["EXIT", "SL", "TP", "TIME"]):
            plt.scatter(t, p, color='blue', marker='x', s=100, zorder=5)
            plt.annotate(f"{action}", (t, p), xytext=(0, 15), textcoords='offset points', ha='center', color='blue', fontsize=8)
            
    # 4. Title & PnL
    pnl_text = f"${trade.pnl:.2f}"
    pnl_c = 'green' if trade.pnl > 0 else 'red'
    
    plt.title(f"{trade_type} Trade #{trade_id} | PnL: {pnl_text} | Reason: {trade.exit_reason}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    fname = f"trade_{trade_id:04d}_{trade_type}_{trade.exit_reason}.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutex", action="store_true", help="Generate Atlas for Mutex Portfolio")
    parser.add_argument("--strategy", type=str, help="Specific strategy name to analyze")
    parser.add_argument("--horizon", type=int, default=120, help="Horizon override for single strategy")
    args = parser.parse_args()

    print("🌍 GENERATING TRADE ATLAS 🌍")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    strategies = []
    
    # Mode 1: Mutex Portfolio
    if args.mutex:
        portfolio_path = config.MUTEX_PORTFOLIO_FILE
        if os.path.exists(portfolio_path):
            print(f"Loading Mutex Portfolio: {portfolio_path}")
            with open(portfolio_path, 'r') as f:
                 portfolio_data = json.load(f)
            strategies = [Strategy.from_dict(d) for d in portfolio_data]
            for s, d in zip(strategies, portfolio_data):
                s.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)
    
    # Mode 2: Single Strategy
    elif args.strategy:
        print(f"Searching for strategy: {args.strategy}")
        s_data = find_strategy_in_files(args.strategy)
        if s_data:
            s = Strategy.from_dict(s_data)
            s.horizon = args.horizon
            # Try to get horizon from data if available
            if 'horizon' in s_data: s.horizon = s_data['horizon']
            strategies = [s]
            print(f"✅ Found {s.name} (Horizon: {s.horizon})")
        else:
            print(f"❌ Strategy {args.strategy} not found.")
            return

    else:
        # Default behavior: Try Mutex
        portfolio_path = config.MUTEX_PORTFOLIO_FILE
        if os.path.exists(portfolio_path):
            print(f"No arguments provided. Defaulting to Mutex Portfolio: {portfolio_path}")
            with open(portfolio_path, 'r') as f:
                 portfolio_data = json.load(f)
            strategies = [Strategy.from_dict(d) for d in portfolio_data]
            for s, d in zip(strategies, portfolio_data):
                s.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)
        else:
            print("Please specify --mutex or --strategy <name>")
            return

    # Load Full Data
    print("Loading Market Data...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df)
    
    if 'time_start' in df.columns:
        times = df['time_start']
    else:
        times = pd.Series(df.index)

    simulator = TradeSimulator(
        prices=backtester.close_vec.flatten(),
        times=times,
        spread_bps=config.SPREAD_BPS, 
        cost_bps=config.COST_BPS
    )

    atlas_root = os.path.join(config.DIRS['OUTPUT_DIR'], "trade_atlas")
    if not os.path.exists(atlas_root): os.makedirs(atlas_root)

    print(f"Generating Atlas for {len(strategies)} Strategies...")
    print(f"NOTE: Only plotting trades in OOS period (Index >= {backtester.val_idx})")

    for i, strat in enumerate(strategies):
        h = getattr(strat, 'horizon', 120) 
        print(f"\n[{i+1}/{len(strategies)}] Analyzing {strat.name} (H{h})...")
        
        # Signals
        raw_signals = backtester.generate_signal_matrix([strat], horizon=h).flatten()
        # Shift signals for Next-Open Execution logic
        signals = np.roll(raw_signals, 1)
        signals[0] = 0
        
        sl_pct = getattr(strat, 'stop_loss_pct', config.DEFAULT_STOP_LOSS)
        tp_pct = getattr(strat, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT)
        
        trades, _ = simulator.simulate(
            signals, 
            stop_loss_pct=sl_pct, 
            take_profit_pct=tp_pct, 
            time_limit_bars=h,
            highs=backtester.high_vec.flatten(),
            lows=backtester.low_vec.flatten(),
            atr=backtester.atr_vec.flatten()
        )
        
        oos_start = backtester.val_idx
        valid_trades = [t for t in trades if t.entry_idx >= oos_start]
        
        print(f"  Found {len(trades)} total trades. {len(valid_trades)} in OOS period.")
        
        if not valid_trades:
            print("  No OOS trades to plot.")
            continue

        strat_dir = os.path.join(atlas_root, f"{strat.name}_H{h}")
        if os.path.exists(strat_dir): shutil.rmtree(strat_dir)
        os.makedirs(strat_dir)
        
        plot_limit = 50
        print(f"  Plotting first {plot_limit} trades...")
        for j, trade in enumerate(valid_trades[:plot_limit]):
            plot_trade(trade, backtester.close_vec.flatten(), times, strat_dir, j+1, h)
                
        print(f"  ✅ Saved plots to {strat_dir}")

if __name__ == "__main__":
    main()
