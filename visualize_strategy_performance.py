import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import re
from feature_engine import FeatureEngine
from strategy_genome import Strategy, Gene
from backtest_engine import BacktestEngine
import config

def parse_gene_string(gene_str):
    # Format: feature operator threshold
    # e.g., "frac_diff_02 > 0.45"
    match = re.match(r"(.+)\s+([<>=]+)\s+([-\d\.]+)", gene_str)
    if match:
        return Gene(match.group(1), match.group(2), float(match.group(3)))
    return None

def reconstruct_strategy(strat_dict):
    # The 'logic' string in apex_strategies.json is: "[Name] LONG:(...) | SHORT:(...)"
    # This parser is a bit brittle, better to rely on the Gene objects if we saved them raw.
    # But apex_strategies.json saved the __repr__ string.
    # Let's try to parse the string.
    
    logic = strat_dict['logic']
    name = strat_dict['name']
    
    # Extract Long and Short blocks
    # Format: [Name] LONG:(g1 AND g2) | SHORT:(g3 AND g4)
    try:
        long_part = logic.split("LONG:(")[1].split(")")[0]
        short_part = logic.split("SHORT:(")[1].split(")")[0]
        
        long_genes = []
        if long_part != "None":
            for g_str in long_part.split(" AND "):
                g = parse_gene_string(g_str)
                if g: long_genes.append(g)
                
        short_genes = []
        if short_part != "None":
            for g_str in short_part.split(" AND "):
                g = parse_gene_string(g_str)
                if g: short_genes.append(g)
                
        return Strategy(name=name, long_genes=long_genes, short_genes=short_genes)
    except Exception as e:
        print(f"Error parsing strategy {name}: {e}")
        return None

def plot_performance(engine, strategies):
    print("Running Backtest for Visualization...")
    # Use the backtest engine to get equity curves
    # We need to access the raw signal matrix logic manually to get full curve
    # evaluate_population splits by set, we want the FULL curve
    
    backtester = BacktestEngine(engine.bars, cost_bps=0.2)
    full_signal_matrix = backtester.generate_signal_matrix(strategies)
    
    returns_vec = backtester.returns_vec
    
    # Calculate PnL
    signals_shifted = np.roll(full_signal_matrix, 1, axis=0)
    signals_shifted[0, :] = 0
    
    strat_returns = signals_shifted * returns_vec
    
    # Costs
    trades = np.abs(np.diff(full_signal_matrix, axis=0, prepend=0))
    costs = trades * backtester.cost_pct
    net_returns = strat_returns - costs
    
    cumulative = np.cumsum(net_returns, axis=0)
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Plot Strategies
    for i, strat in enumerate(strategies):
        plt.plot(cumulative[:, i], label=f"{strat.name} (Sharpe: {strategies[i].test_sharpe:.3f})")
        
    # Plot Asset (Buy & Hold) for comparison
    asset_cum = np.cumsum(returns_vec)
    plt.plot(asset_cum, label="Buy & Hold (EUR/USD)", color='black', alpha=0.3, linestyle='--')
    
    # Shade Regions
    train_end = backtester.train_idx
    val_end = backtester.val_idx
    
    # Get Y-limits to shade full height
    ymin, ymax = plt.ylim()
    
    plt.axvspan(0, train_end, color='green', alpha=0.05, label="Train")
    plt.axvspan(train_end, val_end, color='yellow', alpha=0.05, label="Validation")
    plt.axvspan(val_end, len(cumulative), color='red', alpha=0.05, label="Test (OOS)")
    
    plt.title("Apex Strategies Performance (OOS Cliff Test)", fontsize=16)
    plt.ylabel("Cumulative Log Return")
    plt.xlabel("Bar Index")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(config.DIRS['PLOTS_DIR'], "strategy_performance.png")
    plt.savefig(output_path)
    print(f"📸 Saved Performance Chart to {output_path}")

class MockEngine:
    def __init__(self, df):
        self.bars = df

if __name__ == "__main__":
    
    # Load Data
    print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found. Run generate_features.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    engine = MockEngine(df)
    
    # Load Apex Strategies
    # Look for all horizon files
    import glob
    apex_files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], "apex_strategies_*.json"))
    
    all_strategies_data = []
    for fpath in apex_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                all_strategies_data.extend(data)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
    strategies = []
    for d in all_strategies_data:
        s = reconstruct_strategy(d)
        if s:
            s.test_sharpe = d.get('test_sharpe', 0.0) # inject score for legend
            strategies.append(s)
            
    if not strategies:
        print("No strategies found to visualize.")
        sys.exit(0)
            
    plot_performance(engine, strategies)
