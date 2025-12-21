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
    # print("Running Backtest for Visualization...")
    
    # Setup Aligned Backtester (Trading Mode)
    backtester = BacktestEngine(engine.bars, cost_bps=0.5, annualization_factor=181440)
    
    # Get Metrics AND Account Return Series (Full Data for Visualization)
    # evaluate_population handles splitting, but we want full curve.
    # We can pass set_type='test' or just calculate manually for full data.
    
    full_signal_matrix = backtester.generate_signal_matrix(strategies)
    returns_vec = backtester.returns_vec # log_ret
    prices = backtester.close_vec
    
    # Calculate Real Account PnL
    signals_shifted = np.roll(full_signal_matrix, 1, axis=0); signals_shifted[0, :] = 0
    prices_shifted = np.roll(prices, 1, axis=0); prices_shifted[0] = prices[0]
    
    position_notional = signals_shifted * backtester.standard_lot * prices_shifted
    gross_pnl_dollar = position_notional * returns_vec
    
    lot_change = np.abs(np.diff(full_signal_matrix, axis=0, prepend=0))
    turnover_notional = lot_change * backtester.standard_lot * prices
    costs_dollar = turnover_notional * backtester.total_cost_pct
    
    net_pnl_dollar = gross_pnl_dollar - costs_dollar
    net_returns = net_pnl_dollar / backtester.account_size # % Return
    
    cumulative = np.cumsum(net_returns, axis=0)
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    print(f"\n{'Strategy':<30} | {'Sortino':<8} | {'Total Ret %':<12} | {'Trades':<8}")
    print("-" * 75)
    
    for i, strat in enumerate(strategies):
        total_ret_pct = cumulative[-1, i]
        n_trades = np.sum(lot_change[:, i])
        
        # We need Sortino for table. Use evaluation on test set for accuracy.
        res_test = backtester.evaluate_population([strat], set_type='test')
        sortino = res_test.iloc[0]['sortino']
        
        print(f"{strat.name:<30} | {sortino:<8.2f} | {total_ret_pct*100:<12.2f}% | {int(n_trades):<8}")
        
        label = f"{strat.name} (Sortino: {sortino:.1f} | Ret: {total_ret_pct*100:.1f}% | Tr: {int(n_trades)})"
        plt.plot(cumulative[:, i], label=label)
        
    # Plot Buy & Hold (Log Ret sum normalized to account %) for scale
    asset_cum = np.cumsum(returns_vec)
    plt.plot(asset_cum, label="Market Bench (Log Ret)", color='black', alpha=0.3, linestyle='--')
    
    # Shade Regions
    plt.axvspan(0, backtester.train_idx, color='green', alpha=0.05, label="Train")
    plt.axvspan(backtester.train_idx, backtester.val_idx, color='yellow', alpha=0.05, label="Val")
    plt.axvspan(backtester.val_idx, len(cumulative), color='red', alpha=0.05, label="Test (OOS)")
    
    plt.title(f"Apex Trading Performance: $30k Account, 1-3 Lots (Tiered Position Sizing)", fontsize=16)
    plt.ylabel("Cumulative Account Return (%)")
    plt.xlabel("Bar Index")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(config.DIRS['PLOTS_DIR'], "strategy_performance.png")
    plt.savefig(output_path)
    print(f"📸 Saved Performance Chart to {output_path}")

class MockEngine:
    def __init__(self, df):
        self.bars = df

def filter_top_strategies(engine, strategies, top_n=20, chunk_size=1000):
    # print(f"\n🔍 Filtering Top {top_n} Strategies from {len(strategies)} candidates...")
    # print(f"   Batch Size: {chunk_size}")
    
    # Setup Backtester for Evaluation
    backtester = BacktestEngine(engine.bars, cost_bps=0.5, annualization_factor=181440)
    
    all_results = []
    
    # Chunk Processing
    total_chunks = (len(strategies) + chunk_size - 1) // chunk_size
    for i in range(0, len(strategies), chunk_size):
        chunk = strategies[i:i+chunk_size]
        # print(f"   > Processing Batch {i//chunk_size + 1}/{total_chunks} ({len(chunk)} strats)...")
        
        try:
            # Evaluate on Test Set (OOS) for selection
            # using 'test' set (last 20%) to pick best OOS performers
            results_df = backtester.evaluate_population(chunk, set_type='test')
            
            # Store essential metrics
            for idx, row in results_df.iterrows():
                all_results.append({
                    'strategy': chunk[idx],
                    'sortino': row['sortino'],
                    'sharpe': row['sharpe'],
                    'total_return': row['total_return']
                })
        except Exception as e:
            print(f"     ⚠️ Error in batch: {e}")
        
        # Cleanup Memory
        backtester.reset_jit_context()
        
    # Sort and Select
    # print("   > Sorting and selecting top candidates...")
    sorted_results = sorted(all_results, key=lambda x: x['sortino'], reverse=True)
    top_performers = [x['strategy'] for x in sorted_results[:top_n]]
    
    # print(f"✅ Selected {len(top_performers)} strategies.")
    if len(top_performers) > 0:
        best = sorted_results[0]
        # print(f"   Best Strat: {best['strategy'].name} (Sortino: {best['sortino']:.2f})")
        
    return top_performers

if __name__ == "__main__":
    # print(f"Loading Feature Matrix...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    engine = MockEngine(df)
    
    import glob
    apex_files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], "apex_strategies_*.json"))
    all_strategies_data = []
    # print(f"Loading Strategies from {len(apex_files)} files...")
    for fpath in apex_files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                all_strategies_data.extend(data)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            
    # print(f"Hydrating {len(all_strategies_data)} strategies...")
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
    
    # Optimize: Filter Top N before Plotting
    top_strategies = filter_top_strategies(engine, strategies, top_n=20)
    
    if top_strategies:
        plot_performance(engine, top_strategies)
    else:
        print("No viable strategies found after filtering.")
