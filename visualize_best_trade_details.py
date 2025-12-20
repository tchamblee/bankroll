import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import re
from strategy_genome import Strategy, StaticGene, RelationalGene, DeltaGene, ZScoreGene, TimeGene, ConsecutiveGene
from backtest_engine import BacktestEngine
import config

def parse_gene_string(gene_str):
    match = re.match(r"(.+)\s+([<>=!]+)\s+(.+)", gene_str)
    if not match: return None
    left, op, right = match.groups()
    left, op, right = left.strip(), op.strip(), right.strip()
    
    if left.startswith("Consecutive("):
        direction = left.split("(")[1].split(")")[0]
        return ConsecutiveGene(direction, op, int(re.sub(r"[^0-9]", "", right)))
    if left.startswith("Time("):
        mode = left.split("(")[1].split(")")[0]
        return TimeGene(mode, op, int(re.sub(r"[^0-9]", "", right)))
    if left.startswith("Z("):
        try:
            inner = left.split("Z(")[1].split(")")[0]
            feature, window = [x.strip() for x in inner.split(",")]
            if right.endswith('σ'): right = right[:-1]
            return ZScoreGene(feature, op, float(right), int(window))
        except: return None
    if left.startswith("Delta("):
        try:
            inner = left.split("Delta(")[1].split(")")[0]
            feature, lookback = [x.strip() for x in inner.split(",")]
            return DeltaGene(feature, op, float(right), int(lookback))
        except: return None
    try:
        threshold = float(right)
        return StaticGene(left, op, threshold)
    except ValueError:
        return RelationalGene(left, op, right)

def reconstruct_strategy(strat_dict):
    logic = strat_dict['logic']
    name = strat_dict['name']
    try:
        match = re.search(r"\]\[(.*?)\]", logic)
        logic_type = match.group(1) if match else "AND"
        min_con = int(logic_type.split("(")[1].split(")")[0]) if logic_type.startswith("VOTE(") else None
        sep = " + " if logic_type == "VOTE" else " AND "
        parts = logic.split(" | ")
        if len(parts) != 2: return None
        long_block, short_block = parts[0], parts[1]
        
        def extract_content(block, tag):
            if f"{tag}:(" in block:
                start = block.find(f"{tag}:(") + len(tag) + 2
                return block[start:block.rfind(")")]
            return "None"

        long_part = extract_content(long_block, "LONG")
        short_part = extract_content(short_block, "SHORT")
        
        l_genes = [parse_gene_string(g) for g in long_part.split(sep) if g != "None"]
        s_genes = [parse_gene_string(g) for g in short_part.split(sep) if g != "None"]
        return Strategy(name=name, long_genes=[g for g in l_genes if g], short_genes=[g for g in s_genes if g], min_concordance=min_con)
    except Exception as e:
        print(f"Error parsing {name}: {e}")
        return None

def visualize_best_strategy():
    # 1. Load Data
    print(f"Loading Feature Matrix...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # 2. Find Best Strategy
    import glob
    apex_files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], "apex_strategies_*.json"))
    if not apex_files:
        print("❌ No Apex strategies found. Run pipeline first.")
        return
        
    all_strats = []
    for f in apex_files:
        with open(f, 'r') as f_in:
            all_strats.extend(json.load(f_in))
            
    if not all_strats: return
    
    # Sort by test_sortino
    all_strats.sort(key=lambda x: x.get('test_sortino', -999), reverse=True)
    best_data = all_strats[0]
    print(f"🏆 Visualizing Best Strategy: {best_data['name']} (Sortino: {best_data['test_sortino']:.2f})")
    
    strat = reconstruct_strategy(best_data)
    if not strat: return
    
    # 3. Generate Signals on FULL data
    backtester = BacktestEngine(df)
    signals = strat.generate_signal(backtester.context)
    
    # 4. Prepare Plotting Data
    # Focus on the Test set for clarity, or full if requested. 
    # "Visualize every single trade" -> Full might be too crowded. 
    # Let's do the Test set first as it's the OOS proof.
    start_idx = backtester.val_idx
    prices = df['close'].values[start_idx:]
    sig = signals[start_idx:]
    times = df.index[start_idx:] # Assuming index is time or bar id
    
    plt.figure(figsize=(20, 10))
    plt.plot(times, prices, color='black', alpha=0.4, label='EUR/USD Price')
    
    # Detect Changes
    # Delta Signal
    diff = np.diff(sig, prepend=0)
    
    # Positions
    # Long Entry/Upsize: diff > 0 and sig > 0
    # Long Exit/Downsize: diff < 0 and sig >= 0
    # Short Entry/Upsize: diff < 0 and sig < 0
    # Short Exit/Downsize: diff > 0 and sig <= 0
    
    for i in range(len(sig)):
        t = times[i]
        p = prices[i]
        d = diff[i]
        s = sig[i]
        
        if d > 0: # Upsize Long or Exit Short
            if s > 0: # Long Entry / Upsize
                plt.scatter(t, p, color='green', marker='^', s=50*abs(d), alpha=0.8)
                plt.annotate(f"+{d}L", (t, p), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green')
            elif s == 0: # Full Short Exit
                plt.scatter(t, p, color='blue', marker='x', s=100, alpha=0.6)
                plt.annotate("EXIT", (t, p), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
        elif d < 0: # Upsize Short or Exit Long
            if s < 0: # Short Entry / Upsize
                plt.scatter(t, p, color='red', marker='v', s=50*abs(d), alpha=0.8)
                plt.annotate(f"{d}L", (t, p), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')
            elif s == 0: # Full Long Exit
                plt.scatter(t, p, color='blue', marker='x', s=100, alpha=0.6)
                plt.annotate("EXIT", (t, p), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='blue')

    plt.title(f"Detailed Trade Log: {best_data['name']} (OOS Period)\n{best_data['logic']}", fontsize=14)
    plt.ylabel("Price")
    plt.xlabel("Bar Index")
    plt.grid(True, alpha=0.2)
    
    # Add secondary axis for cumulative PnL
    # ... (omitted for focus on entry/exit)
    
    out_path = os.path.join(config.DIRS['PLOTS_DIR'], "best_strategy_trades.png")
    plt.savefig(out_path)
    print(f"📸 Saved Trade Visualization to {out_path}")

if __name__ == "__main__":
    visualize_best_strategy()
