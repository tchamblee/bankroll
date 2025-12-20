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
    
    output_path = "data/strategy_performance.png"
    plt.savefig(output_path)
    print(f"📸 Saved Performance Chart to {output_path}")

if __name__ == "__main__":
    # Load Data
    engine = FeatureEngine(config.DIRS['DATA_RAW_TICKS'])
    df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    engine.create_volume_bars(df, volume_threshold=250)
    
    # Need to load survivors to know what features to calc? 
    # Or just calc everything. Let's calc everything to be safe.
    engine.add_features_to_bars(windows=[50, 100, 200, 400])
    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_delta_features(lookback=10)
    engine.add_delta_features(lookback=50)
    
    # Load Apex Strategies
    with open("data/apex_strategies.json", "r") as f:
        apex_data = json.load(f)
        
    strategies = []
    for d in apex_data:
        s = reconstruct_strategy(d)
        if s:
            s.test_sharpe = d['test_sharpe'] # inject score for legend
            strategies.append(s)
            
    plot_performance(engine, strategies)
