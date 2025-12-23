import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import re
from feature_engine import FeatureEngine
from genome import Strategy, RelationalGene, DeltaGene, ZScoreGene, TimeGene, ConsecutiveGene
from backtest import BacktestEngine
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
        float(right) # Check if it is a number
        return None # StaticGene is deprecated
    except ValueError:
        return RelationalGene(left, op, right)

def reconstruct_strategy(strat_dict):
    # 1. New JSON Format (Native Serialization)
    if 'long_genes' in strat_dict:
        try:
            return Strategy.from_dict(strat_dict)
        except Exception as e:
            print(f"Error hydrating strategy {strat_dict.get('name')}: {e}")
            return None

    # 2. Legacy String Format (Regex Parsing)
    if 'logic' not in strat_dict: return None
    
    logic = strat_dict['logic']
    name = strat_dict['name']
    try:
        match = re.search(r"(])[(](.*?)[)]", logic)
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

def plot_performance(engine, strategies):
    backtester = BacktestEngine(engine.bars, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # 1. Generate Full Signal Matrix
    full_signal_matrix = backtester.generate_signal_matrix(strategies)
    
    # --- FIX: LOOKAHEAD BIAS (Next Open Execution) ---
    # Shift signals forward by 1
    full_signal_matrix = np.vstack([np.zeros((1, full_signal_matrix.shape[1]), dtype=full_signal_matrix.dtype), full_signal_matrix[:-1]])
    # Use Open prices
    prices = backtester.open_vec.flatten()
    
    times = backtester.times_vec
    
    # 2. Run Simulation via BacktestEngine's wrapper (consistent logic)
    print(f"Running Full Simulation for Plotting (SL={config.DEFAULT_STOP_LOSS*100}%, TimeLimit={config.DEFAULT_TIME_LIMIT})")
    net_returns, trades_count = backtester.run_simulation_batch(
        full_signal_matrix, 
        strategies,
        prices, 
        times, 
        time_limit=config.DEFAULT_TIME_LIMIT
    )
    
    # 3. Compute Cumulative Equity Curve
    cumulative = np.cumsum(net_returns, axis=0)
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    print(f"\n{'Strategy':<30} | {'Sortino':<8} | {'Total Ret %':<12} | {'Trades':<8}")
    print("-" * 75)
    
    for i, strat in enumerate(strategies):
        total_ret_pct = cumulative[-1, i]
        n_trades = trades_count[i]
        
        # We need Sortino for table. Use evaluation on test set for accuracy.
        # Can extract from returns vec manually to avoid re-running backtester
        test_start = backtester.val_idx
        test_rets = net_returns[test_start:, i]
        avg = np.mean(test_rets)
        downside = np.std(np.minimum(test_rets, 0)) + 1e-9
        sortino = (avg / downside) * np.sqrt(config.ANNUALIZATION_FACTOR)
        
        print(f"{strat.name:<30} | {sortino:<8.2f} | {total_ret_pct*100:<12.2f}% | {int(n_trades):<8}")
        
        label = f"{strat.name} (Sortino: {sortino:.1f} | Ret: {total_ret_pct*100:.1f}% | Tr: {int(n_trades)})"
        plt.plot(cumulative[:, i], label=label)
        
    # Plot Buy & Hold
    returns_vec = backtester.returns_vec.flatten()
    asset_cum = np.cumsum(returns_vec)
    plt.plot(asset_cum, label="Market Bench (Log Ret)", color='black', alpha=0.3, linestyle='--')
    
    # Shade Regions
    plt.axvspan(0, backtester.train_idx, color='green', alpha=0.05, label="Train")
    plt.axvspan(backtester.train_idx, backtester.val_idx, color='yellow', alpha=0.05, label="Val")
    plt.axvspan(backtester.val_idx, len(cumulative), color='red', alpha=0.05, label="Test (OOS)")
    
    plt.title(f"Apex Trading Performance: ${int(config.ACCOUNT_SIZE/1000)}k Account, {config.MIN_LOTS}-{config.MAX_LOTS} Lots (Tiered Position Sizing)", fontsize=16)
    plt.ylabel("Cumulative Account Return (%)")
    plt.xlabel("Bar Index")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(config.DIRS['PLOTS_DIR'], exist_ok=True)
    output_path = os.path.join(config.DIRS['PLOTS_DIR'], "strategy_performance.png")
    plt.savefig(output_path)
    print(f"📸 Saved Performance Chart to {output_path}")

class MockEngine:
    def __init__(self, df):
        self.bars = df

def filter_top_strategies(engine, strategies, top_n=20, chunk_size=1000):
    backtester = BacktestEngine(engine.bars, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    all_results = []
    
    # Chunk Processing
    total_chunks = (len(strategies) + chunk_size - 1) // chunk_size
    for i in range(0, len(strategies), chunk_size):
        chunk = strategies[i:i+chunk_size]
        
        try:
            results_df = backtester.evaluate_population(chunk, set_type='test')
            
            for idx, row in results_df.iterrows():
                all_results.append({
                    'strategy': chunk[idx],
                    'sortino': row['sortino'],
                    'sharpe': row['sharpe'],
                    'total_return': row['total_return']
                })
        except Exception as e:
            print(f"     ⚠️ Error in batch: {e}")
        
        backtester.reset_jit_context()
        
    sorted_results = sorted(all_results, key=lambda x: x['sortino'], reverse=True)
    top_performers = [x['strategy'] for x in sorted_results[:top_n]]
    
    return top_performers

if __name__ == "__main__":
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    engine = MockEngine(df)
    
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
    seen = set()
    for d in all_strategies_data:
        s = reconstruct_strategy(d)
        if s and str(s) not in seen:
            strategies.append(s)
            seen.add(str(s))
            
    if not strategies:
        print("No strategies found.")
        sys.exit(0)
    
    top_strategies = filter_top_strategies(engine, strategies, top_n=20)
    
    if top_strategies:
        plot_performance(engine, top_strategies)
    else:
        print("No viable strategies found after filtering.")