import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import config
from strategy_genome import Strategy
from backtest_engine import BacktestEngine

def load_ensemble_strategies():
    strategies = []
    print(f"Loading Top 5 Unique Strategies from all horizons...")
    
    for h in config.PREDICTION_HORIZONS:
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}_top5_unique.json")
        if not os.path.exists(file_path):
            print(f"  ⚠️  Missing file for Horizon {h}")
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        count = 0
        for d in data:
            try:
                s = Strategy.from_dict(d)
                s.horizon = h # Tag it
                strategies.append(s)
                count += 1
            except: pass
        print(f"  Loaded {count} strategies from Horizon {h}")
        
    return strategies

def run_ensemble():
    print("\n" + "="*80)
    print("🧠 THE BRAIN: ENSEMBLE STRATEGY EXECUTION 🧠")
    print("="*80 + "\n")
    
    # 1. Load Data & Strategies
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df)
    
    strategies = load_ensemble_strategies()
    if not strategies:
        print("❌ No strategies loaded.")
        return
        
    print(f"\nrunning backtest for {len(strategies)} combined strategies...")
    
    # 2. Generate Signals
    # This matrix is (N_Bars, N_Strategies)
    # Values are -1, 0, 1 (assuming strategies output unit direction, or scaled)
    # Strategy.generate_signal outputs lot-scaled signal if gene has 'lots', 
    # but standard system uses -1/0/1 * LotSize. 
    # Let's assume they output -1, 0, 1.
    
    # Ensure context
    backtester.ensure_context(strategies)
    raw_signals = backtester.generate_signal_matrix(strategies)
    
    # 3. Aggregate (The Brain)
    # Sum of votes
    net_votes = np.sum(raw_signals, axis=1)
    
    # Debug Vote Distribution
    print(f"\n📊 Vote Distribution Stats:")
    print(f"   Mean: {np.mean(net_votes):.4f}")
    print(f"   Std:  {np.std(net_votes):.4f}")
    print(f"   Min/Max: {np.min(net_votes)} / {np.max(net_votes)}")
    print(f"   Zero Count: {np.sum(net_votes == 0)} ({np.sum(net_votes == 0)/len(net_votes)*100:.1f}%)")
    print(f"   Active Count (|v|>=1): {np.sum(np.abs(net_votes) >= 1)}")
    
    # Map to Lots (1-3)
    # Adjusted Thresholds: 1->1, 3->2, 6->3
    final_lots = np.zeros(len(net_votes))
    
    # Longs
    final_lots[net_votes >= 1] = 1
    final_lots[net_votes >= 3] = 2
    final_lots[net_votes >= 6] = 3
    
    # Shorts
    final_lots[net_votes <= -1] = -1
    final_lots[net_votes <= -3] = -2
    final_lots[net_votes <= -6] = -3
    
    # 4. Calculate Performance
    # Warmup exclusion (3200 bars)
    warmup = 3200
    
    prices = backtester.close_vec.flatten()
    rets = backtester.returns_vec.flatten()
    
    # Shift signals for PnL (Open position at T, Realize return at T+1)
    pos = np.roll(final_lots, 1)
    pos[0] = 0
    
    # Gross PnL
    # PnL = Pos * LotSize * Return (approx) 
    # More accurate: Pos * LotSize * (Price_change / Price_prev) * Price_prev = Pos * LotSize * Price_change
    gross_pnl_dollar = pos * backtester.standard_lot * rets * prices 
    # Wait, rets is log ret ~ pct change. 
    # Position Notional = Lots * 100k * Price
    # PnL = Notional * Ret
    gross_pnl_dollar = pos * backtester.standard_lot * prices * rets # close enough for vector calc
    
    # Costs
    lot_change = np.abs(np.diff(final_lots, prepend=0))
    # Cost = Change * 100k * Price * Cost_Pct
    costs_dollar = lot_change * backtester.standard_lot * prices * backtester.total_cost_pct
    
    net_pnl_dollar = gross_pnl_dollar - costs_dollar
    
    # Cumulative PnL
    cum_pnl = np.cumsum(net_pnl_dollar)
    
    # Metrics (Post-Warmup)
    valid_pnl = net_pnl_dollar[warmup:]
    valid_cum_pnl = np.cumsum(valid_pnl)
    
    total_profit = np.sum(valid_pnl)
    total_ret_pct = total_profit / backtester.account_size
    
    # Sortino
    avg_ret = np.mean(valid_pnl / backtester.account_size)
    downside = np.std(np.minimum(valid_pnl / backtester.account_size, 0)) + 1e-9
    sortino = (avg_ret / downside) * np.sqrt(backtester.annualization_factor)
    
    sharpe = (np.mean(valid_pnl) / np.std(valid_pnl)) * np.sqrt(backtester.annualization_factor)
    
    trades = np.sum(lot_change[warmup:])
    
    print("\n" + "="*40)
    print("🚀 ENSEMBLE RESULTS (Post-Warmup)")
    print("="*40)
    print(f"Total Profit:    ${total_profit:,.2f}")
    print(f"Return on Acct:  {total_ret_pct*100:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Sortino Ratio:   {sortino:.2f}")
    print(f"Total Trades:    {int(trades)}")
    print(f"Max Drawdown:    TBD") # TODO
    
    # 5. Visualize
    plt.figure(figsize=(15, 8))
    
    # Plot Ensemble
    # We plot starting from 0 for the valid period
    times = df['time_start'].values[warmup:]
    
    plt.plot(range(len(valid_cum_pnl)), valid_cum_pnl, label='Ensemble (The Brain)', color='gold', linewidth=2)
    
    plt.title(f"The Brain: Ensemble Strategy Performance\nReturn: {total_ret_pct*100:.2f}% | Sortino: {sortino:.2f} | Trades: {int(trades)}")
    plt.ylabel("Cumulative Profit ($)")
    plt.xlabel("Bars (Post-Warmup)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = os.path.join(config.DIRS['PLOTS_DIR'], "ensemble_performance.png")
    plt.savefig(out_path)
    print(f"\n📸 Saved Ensemble Chart to {out_path}")

if __name__ == "__main__":
    run_ensemble()
