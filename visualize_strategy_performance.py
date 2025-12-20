import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import re
from feature_engine import FeatureEngine
from strategy_genome import Strategy, StaticGene, RelationalGene, DeltaGene, ZScoreGene, TimeGene, ConsecutiveGene
from backtest_engine import BacktestEngine
import config
from validate_features import triple_barrier_labels

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
        sep = f" {logic_type} "
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
    print("Running Backtest for Visualization...")
    
    # Setup Aligned Backtester
    backtester = BacktestEngine(engine.bars, cost_bps=0.5, annualization_factor=181440)
    
    # Calculate Prediction Sortino and Get Series
    # We need to compute labels. We'll use H=60 as default for visualization.
    df_labels = engine.bars.copy()
    df_labels['target_return'] = triple_barrier_labels(df_labels, lookahead=60, pt_sl_multiple=2.0)
    backtester_pred = BacktestEngine(df_labels, cost_bps=0.5, target_col='target_return', annualization_factor=181440)
    
    # Get Metrics AND Returns Series
    pred_results, pred_net_returns = backtester_pred.evaluate_population(strategies, set_type='test', prediction_mode=True, return_series=True)

    # Calculate Cumulative Returns for Prediction Mode
    cumulative = np.cumsum(pred_net_returns, axis=0)
    
    # Calc Trade Sortino for comparison (table only)
    trade_results = backtester.evaluate_population(strategies, set_type='test', prediction_mode=False)
    returns_vec = backtester.returns_vec

    # Plotting
    plt.figure(figsize=(15, 8))
    
    print(f"\n{'Strategy':<30} | {'Srt(Pred)':<8} | {'Ret(Pred)':<8} | {'Trades':<8} | {'Status':<12}")
    print("-" * 90)
    
    for i, strat in enumerate(strategies):
        # Use trades count from the Prediction calculation (matches training log)
        # Trades in prediction mode are just non-zero signals because we pay cost on entry? 
        # Actually backtester calculates cost on turnover.
        # But for 'Trades' count display, let's use the one that matches logic.
        
        # We need the trade count from the prediction execution
        # In prediction mode, 'signals' was used directly.
        # Let's trust the 'trades' column from the results df
        n_trades = pred_results.iloc[i]['trades']
        if n_trades == 0: continue
            
        full_ret = pred_results.iloc[i]['total_return']
        s_pred = pred_results.iloc[i]['sortino']
        
        print(f"{strat.name:<30} | {s_pred:<8.2f} | {full_ret:<8.4f} | {int(n_trades):<8} | {'PROFITABLE' if full_ret > 0 else 'LOSS'}")
        
        label = f"{strat.name} (Sortino: {s_pred:.1f} | Ret: {full_ret:.2%} | Trades: {int(n_trades)})"
        plt.plot(cumulative[:, i], label=label)
        
    # Plot Asset (Buy & Hold) for comparison - Normalized to start at 0
    asset_cum = np.cumsum(returns_vec[backtester_pred.val_idx:])
    plt.plot(asset_cum, label="Buy & Hold (EUR/USD)", color='black', alpha=0.3, linestyle='--')
    
    # Shade Regions (Only Test Set is relevant here as we sliced net_returns for 'test')
    # cumulative is length of test set.
    
    plt.title("Apex Strategies: Prediction Performance (Triple Barrier Execution)", fontsize=16)
    plt.ylabel("Cumulative Log Return")
    plt.xlabel("Bar Index")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(config.DIRS['PLOTS_DIR'], "strategy_performance.png")
    plt.savefig(output_path)
    print(f"📸 Saved Performance Chart to {output_path}")

class MockEngine:
    def __init__(self, df):
        self.bars = df

if __name__ == "__main__":
    print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
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
                all_strategies_data.extend(json.load(f))
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
        print("No strategies found to visualize.")
        sys.exit(0)
            
    plot_performance(engine, strategies)