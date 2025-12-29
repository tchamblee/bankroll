#!/usr/bin/env python3
import sys
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from tabulate import tabulate
import config
from backtest.engine import BacktestEngine
from genome import Strategy

def find_strategy(name):
    """
    Scans all relevant JSON files in output/strategies/ for the given strategy name.
    """
    search_files = [
        "output/strategies/candidates.json",
        "output/strategies/mutex_portfolio.json",
        "output/strategies/found_strategies.json",
    ]
    # Add all apex files
    search_files.extend(glob.glob("output/strategies/apex_strategies_*.json"))
    
    # Deduplicate paths while preserving order
    seen = set()
    ordered_files = []
    for f in search_files:
        if f not in seen:
            ordered_files.append(f)
            seen.add(f)
            
    for fpath in ordered_files:
        if not os.path.exists(fpath): continue
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict): data = [data] # Handle single object
                
                for s_dict in data:
                    if s_dict.get('name') == name:
                        print(f"✅ Found strategy in {fpath}")
                        strat = Strategy.from_dict(s_dict)
                        # Hydrate extra fields
                        strat.horizon = s_dict.get('horizon', 60) # Default if missing
                        
                        # Hydrate metrics if available
                        metrics = s_dict.get('metrics', {})
                        strat.sortino_oos = metrics.get('sortino_oos', s_dict.get('test_sortino', 0.0))
                        strat.robust_return = metrics.get('robust_return', s_dict.get('test_return', 0.0))
                        
                        return strat
        except Exception as e:
            # print(f"⚠️ Error reading {fpath}: {e}") # specific error
            pass
            
    return None

def calc_metrics(returns_vec, trades_count):
    if len(returns_vec) == 0: return {}
    
    total_ret = np.sum(returns_vec)
    
    # Annualized Return
    mean_ret = np.mean(returns_vec)
    ann_ret = mean_ret * config.ANNUALIZATION_FACTOR
    
    # Sortino Logic (Matched to BacktestEngine)
    # Downside Deviation = RMS of returns < 0
    downside_returns = returns_vec.copy()
    downside_returns[downside_returns > 0] = 0
    downside_sq = downside_returns ** 2
    downside_dev = np.sqrt(np.mean(downside_sq))
    
    # Avoid div by zero
    if downside_dev < 1e-9:
        sortino = 0.0
    else:
        sortino = (mean_ret * config.ANNUALIZATION_FACTOR) / (downside_dev * np.sqrt(config.ANNUALIZATION_FACTOR))
    
    # Drawdown
    cum_ret = np.cumsum(returns_vec)
    if len(cum_ret) > 0:
        peak = np.maximum.accumulate(cum_ret)
        dd = cum_ret - peak
        max_dd = np.min(dd)
    else:
        max_dd = 0.0
        
    return {
        "trades": int(trades_count),
        "total_return": total_ret,
        "ann_return": ann_ret,
        "sortino": sortino,
        "max_drawdown": max_dd
    }

def main():
    parser = argparse.ArgumentParser(description="Audit a strategy on OOS data.")
    parser.add_argument("name", type=str, help="Name of the strategy to audit")
    args = parser.parse_args()
    
    print(f"🔍 Searching for strategy: {args.name}...")
    strategy = find_strategy(args.name)
    
    if not strategy:
        print(f"❌ Strategy '{args.name}' not found in output/strategies/.")
        sys.exit(1)
        
    print(f"📉 Loading Market Data...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found. Run generate_features.py first.")
        sys.exit(1)
        
    bars_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    print(f"⚙️ Initializing Backtest Engine...")
    engine = BacktestEngine(bars_df)
    
    print(f"🚀 Simulating OOS Performance (Test Set Start Idx: {engine.val_idx})...")
    
    # 1. Generate Signals (Full History)
    print("  - Generating Signals...")
    signals = engine.generate_signal_matrix([strategy]) # (N, 1)
    
    # CRITICAL: Shift signals by 1 to enforce Next-Open execution (Lag 1)
    # Signal[t] (calculated at Close) must execute at Open[t+1]
    signals = np.roll(signals, 1, axis=0)
    signals[0] = 0 # First bar has no prior signal
    
    # 2. Slice for OOS
    oos_slice = slice(engine.val_idx, None)
    
    signals_oos = signals[oos_slice]
    prices_oos = engine.close_vec[oos_slice]
    times_oos = engine.times_vec[oos_slice]
    highs_oos = engine.high_vec[oos_slice]
    lows_oos = engine.low_vec[oos_slice]
    atr_oos = engine.atr_vec[oos_slice] if engine.atr_vec is not None else None
    
    # 3. Prepare Sub-Signals for Breakdown
    # Ensure we copy correctly
    signals_long = signals_oos.copy()
    signals_long[signals_long < 0] = 0
    
    signals_short = signals_oos.copy()
    signals_short[signals_short > 0] = 0
    
    # 4. Run Simulations
    print("  - Running Trade Simulator (Combined)...")
    ret_combined, trd_combined = engine.run_simulation_batch(
        signals_oos, [strategy], prices_oos, times_oos, 
        time_limit=strategy.horizon, highs=highs_oos, lows=lows_oos, atr=atr_oos
    )
    
    print("  - Running Trade Simulator (Long Only)...")
    ret_long, trd_long = engine.run_simulation_batch(
        signals_long, [strategy], prices_oos, times_oos, 
        time_limit=strategy.horizon, highs=highs_oos, lows=lows_oos, atr=atr_oos
    )
    
    print("  - Running Trade Simulator (Short Only)...")
    ret_short, trd_short = engine.run_simulation_batch(
        signals_short, [strategy], prices_oos, times_oos, 
        time_limit=strategy.horizon, highs=highs_oos, lows=lows_oos, atr=atr_oos
    )
    
    # 5. Calculate Metrics
    m_comb = calc_metrics(ret_combined[:, 0], trd_combined[0])
    m_long = calc_metrics(ret_long[:, 0], trd_long[0])
    m_short = calc_metrics(ret_short[:, 0], trd_short[0])
    
    # 6. Print Report
    print("\n" + "="*50)
    print(f"📊 AUDIT REPORT: {args.name}")
    print("="*50)
    print(f"Horizon:     {strategy.horizon} bars")
    print(f"Stop Loss:   {strategy.stop_loss_pct}%")
    print(f"Take Profit: {strategy.take_profit_pct}%")
    print("-" * 50)
    
    headers = ["Metric", "Total", "Longs", "Shorts"]
    table = [
        ["Trades", m_comb.get('trades', 0), m_long.get('trades', 0), m_short.get('trades', 0)],
        ["Total Return", f"{m_comb.get('total_return', 0):.2%}", f"{m_long.get('total_return', 0):.2%}", f"{m_short.get('total_return', 0):.2%}"],
        ["Ann. Return", f"{m_comb.get('ann_return', 0):.2%}", f"{m_long.get('ann_return', 0):.2%}", f"{m_short.get('ann_return', 0):.2%}"],
        ["Sortino Ratio", f"{m_comb.get('sortino', 0):.2f}", f"{m_long.get('sortino', 0):.2f}", f"{m_short.get('sortino', 0):.2f}"],
        ["Max Drawdown", f"{m_comb.get('max_drawdown', 0):.2%}", f"{m_long.get('max_drawdown', 0):.2%}", f"{m_short.get('max_drawdown', 0):.2%}"],
    ]
    
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
    
    print("\n✅ Audit Complete.")

if __name__ == "__main__":
    main()
