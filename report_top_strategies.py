import json
import os
import re
import sys
import pandas as pd
import numpy as np
import config
from collections import Counter
from backtest_engine import BacktestEngine
from strategy_genome import Strategy, StaticGene, RelationalGene, DeltaGene, ZScoreGene, TimeGene, ConsecutiveGene

class MockEngine:
    def __init__(self, df):
        self.bars = df

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

def parse_genes_from_logic(logic_str):
    genes = []
    try:
        match = re.search(r"(])[(](.*?)[)]", logic_str)
        logic_type = match.group(1) if match else "AND"
        sep = " + " if logic_type == "VOTE" else " AND "
        parts = logic_str.split(" | ")
        if len(parts) < 2: return []
        for block, tag in [(parts[0], "LONG"), (parts[1], "SHORT")]:
            if f"{tag}:(" in block:
                start = block.find(f"{tag}:(") + len(tag) + 2
                content = block[start:block.rfind(")")]
                if content != "None":
                    for g in content.split(sep):
                        m = re.match(r"Delta\[(.*?),", g) or re.match(r"Z\[(.*?),", g)
                        if m: genes.append(m.group(1))
                        else:
                            tokens = g.split(' ')
                            if tokens: genes.append(tokens[0])
    except: pass
    return genes

def main():
    print("\n" + "="*120)
    print("🏆 APEX STRATEGY REPORT (REAL-MONEY TRADING MODE) 🏆")
    print(f"Account: ${config.ACCOUNT_SIZE:,.0f} | Lots: {config.MIN_LOTS}-{config.MAX_LOTS} | Cost: 0.5 bps + Spread")
    print("="*120 + "\n")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    base_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(base_df, cost_bps=0.5, annualization_factor=181440)
    
    horizons = config.PREDICTION_HORIZONS
    global_gene_counts = Counter()
    
    for h in horizons:
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}.json")
        if not os.path.exists(file_path): continue
            
        print(f"\n--- Horizon: {h} Bars ---")
        try:
            with open(file_path, 'r') as f:
                strategies_data = json.load(f)
            
            strategies = []
            for d in strategies_data:
                s = reconstruct_strategy(d)
                if s: strategies.append(s)
            
            if not strategies: continue

            # Evaluate on Test Set
            res_df = backtester.evaluate_population(strategies, set_type='test')
            
            # Evaluate on Full History for context
            full_signals = backtester.generate_signal_matrix(strategies)
            prices = backtester.close_vec
            rets = backtester.returns_vec
            
            sigs_shft = np.roll(full_signals, 1, axis=0); sigs_shft[0,:]=0
            prcs_shft = np.roll(prices, 1, axis=0); prcs_shft[0]=prices[0]
            
            pos_notional = sigs_shft * backtester.standard_lot * prcs_shft
            gross_pnl = pos_notional * rets
            
            lot_change = np.abs(np.diff(full_signals, axis=0, prepend=0))
            costs = lot_change * backtester.standard_lot * prices * backtester.total_cost_pct
            
            full_net_returns = (gross_pnl - costs) / backtester.account_size
            full_rets_pct = np.sum(full_net_returns, axis=0)

            df_data = []
            horizon_genes = []
            for i, strat in enumerate(strategies):
                row = res_df.iloc[i]
                if row['trades'] == 0 and full_rets_pct[i] == 0: continue
                
                genes = parse_genes_from_logic(strategies_data[i]['logic'])
                horizon_genes.extend(genes)
                global_gene_counts.update(genes)
                
                df_data.append({
                    'Name': strat.name,
                    'Sortino(OOS)': row['sortino'],
                    'Ret%(OOS)': row['total_return'] * 100,
                    'Ret%(Full)': full_rets_pct[i] * 100,
                    'Trades(OOS)': int(row['trades']),
                    'Stability': row['stability'],
                    'Status': 'PROFITABLE' if row['total_return'] > 0 else 'LOSS',
                    'Genes': ", ".join(genes[:2]) + ("..." if len(genes)>2 else "")
                })
            
            if df_data:
                df = pd.DataFrame(df_data).sort_values(by='Sortino(OOS)', ascending=False)
                for col in ['Sortino(OOS)', 'Ret%(OOS)', 'Ret%(Full)', 'Stability']:
                    df[col] = df[col].apply(lambda x: f"{x:.2f}")
                print(df.to_string(index=False))
            else:
                print("  No active strategies found.")
            
            print("-" * 120)
            print(f"  Dominant Genes (H{h}): " + ", ".join([f"{g}:{c}" for g,c in Counter(horizon_genes).most_common(5)]))
            
        except Exception as e:
            print(f"  Error processing horizon {h}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*120)
    print("🧬 GLOBAL DOMINANT GENES 🧬")
    if global_gene_counts:
        for i, (g, c) in enumerate(global_gene_counts.most_common(20), 1):
            print(f"{i:2d}. {g:<40} ({c})")

if __name__ == "__main__":
    main()
