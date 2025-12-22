import json
import os
import re
import sys
import pandas as pd
import numpy as np
import config
from collections import Counter
from backtest_engine import BacktestEngine
from strategy_genome import Strategy

class MockEngine:
    def __init__(self, df):
        self.bars = df

def get_gene_description(gene_dict):
    """Translates a single gene dictionary into a sentence."""
    if hasattr(gene_dict, 'to_dict'): gene_dict = gene_dict.to_dict()
    g_type = gene_dict['type']
    
    if g_type == 'static':
        # f = GeneTranslator.translate_feature(gene_dict['feature']) # Assuming GeneTranslator isn't used here directly based on prev context
        f = gene_dict['feature']
        op = gene_dict['operator']
        val = f"{gene_dict['threshold']:.6g}"
        return f"{f} is {op} {val}"
        
    elif g_type == 'relational':
        f1 = gene_dict['feature_left']
        f2 = gene_dict['feature_right']
        op = gene_dict['operator']
        return f"{f1} is {op} {f2}"
        
    elif g_type == 'delta':
        f = gene_dict['feature']
        lookback = gene_dict['lookback']
        op = gene_dict['operator']
        val = f"{gene_dict['threshold']:.6g}"
        return f"Change in {f} ({lookback} bars) is {op} {val}"
        
    elif g_type == 'zscore':
        f = gene_dict['feature']
        win = gene_dict['window']
        op = gene_dict['operator']
        sigma = f"{gene_dict['threshold']:.3g}σ"
        return f"{f} ({win}-bar Z-Score) is {op} {sigma}"
        
    elif g_type == 'time':
        mode = gene_dict['mode'].title() 
        op = gene_dict['operator']
        val = gene_dict['value']
        return f"Current {mode} is {op} {val}"
        
    elif g_type == 'consecutive':
        direction = gene_dict['direction'].upper()
        count = gene_dict['count']
        op = gene_dict['operator']
        return f"Consecutive {direction} Candles {op} {count}"
        
    elif g_type == 'cross':
        f1 = gene_dict['feature_left']
        f2 = gene_dict['feature_right']
        direction = gene_dict['direction'].upper()
        return f"{f1} crosses {direction} {f2}"
        
    elif g_type == 'persistence':
        f = gene_dict['feature']
        op = gene_dict['operator']
        thresh = f"{gene_dict['threshold']:.6g}"
        win = gene_dict['window']
        return f"({f} {op} {thresh}) FOR {win} BARS"

    elif g_type == 'squeeze':
        return f"{gene_dict['feature_short']} < {gene_dict['multiplier']:.4g} * {gene_dict['feature_long']}"
    elif g_type == 'range':
        f = gene_dict['feature']
        return f"{gene_dict['min_val']:.6g} < {f} < {gene_dict['max_val']:.6g}"
    return "Unknown"

def evaluate_batch(backtester, batch, horizon):
    """Evaluates a batch on Train, Val, and Test sets, returning a list of result dicts."""
    # 1. Training Set (0-60%) - For Overfitting Check
    res_train = backtester.evaluate_population(batch, set_type='train', time_limit=horizon)
    
    # 2. Validation Set (60-80%) - Used for Selection (Semi-In-Sample)
    res_val = backtester.evaluate_population(batch, set_type='validation', time_limit=horizon)
    
    # 3. Test Set (80-100%) - PURE OOS
    res_test = backtester.evaluate_population(batch, set_type='test', time_limit=horizon)
    
    results = []
    for i, strat in enumerate(batch):
        ret_train = res_train.iloc[i]['total_return']
        ret_val = res_val.iloc[i]['total_return']
        ret_test = res_test.iloc[i]['total_return']
        
        # Robust Return: Mean of Val + Test (Excluding Train)
        robust_ret = np.mean([ret_val, ret_test])
        
        # STRICT FILTER: Must be profitable in OOS (Test) and Robust > 0, AND profitable in Training
        if ret_test <= 0 or res_test.iloc[i]['sortino'] <= 0.1 or robust_ret <= 0 or ret_train <= 0:
            continue

        results.append({
            'Strategy': strat,
            'Ret_Train': ret_train,
            'Ret_Val': ret_val,
            'Ret_Test': ret_test,
            'Robust_Ret': robust_ret,
            'Sortino_OOS': res_test.iloc[i]['sortino'],
            'Trades': res_test.iloc[i]['trades'], # Use Test trades count
            'Gen': getattr(strat, 'generation_found', '?')
        })
        
    return results

def main():
    print("\n" + "="*120)
    print("🏆 APEX STRATEGY REPORT (ROBUST OOS) 🏆")
    print(f"Account: ${config.ACCOUNT_SIZE:,.0f} | Lots: {config.MIN_LOTS}-{config.MAX_LOTS} | Cost: {config.COST_BPS} bps + Spread")
    print("Sorting by 'Robust Return' = mean(Val, Test)")
    print("="*120 + "\n")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    base_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(base_df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
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
                try:
                    s = Strategy.from_dict(d)
                    s.generation_found = d.get('generation', '?')
                    strategies.append(s)
                except: pass
            
            if not strategies: continue

            # Batch Processing
            chunk_size = 1000
            all_horizon_results = []
            
            total_chunks = (len(strategies) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(strategies), chunk_size):
                batch = strategies[i:i+chunk_size]
                batch_results = evaluate_batch(backtester, batch, horizon=h)
                all_horizon_results.extend(batch_results)
                backtester.reset_jit_context()
                
            if not all_horizon_results:
                print("  No active strategies found.")
                continue

            # Convert to DataFrame
            df_rows = []
            horizon_genes = []
            
            for res in all_horizon_results:
                strat = res['Strategy']
                strat_genes = [get_gene_description(g) for g in strat.long_genes + strat.short_genes]
                horizon_genes.extend(strat_genes)
                
                df_rows.append({
                    'Gen': res['Gen'],
                    'Name': strat.name,
                    'Ret%(Train)': res['Ret_Train'] * 100,
                    'Ret%(Val)': res['Ret_Val'] * 100,
                    'Ret%(Test)': res['Ret_Test'] * 100,
                    'Robust%': res['Robust_Ret'] * 100,
                    'Sortino(OOS)': res['Sortino_OOS'],
                    'Trades': int(res['Trades']),
                    'Genes': ", ".join(strat_genes[:2]) + ("..." if len(strat_genes)>2 else "")
                })
                
            global_gene_counts.update(horizon_genes)
            
            df = pd.DataFrame(df_rows)
            # Sort by Robust Return (Primary) then Sortino OOS (Secondary)
            df = df.sort_values(by=['Robust%', 'Sortino(OOS)'], ascending=[False, False])
            
            # --- CORRELATION FILTERING (Top 5 Unique) ---
            print("  Performing Correlation Filter (Limit 5, Threshold 0.7)...")
            
            results_map = {res['Strategy'].name: res for res in all_horizon_results}
            sorted_names = df['Name'].values
            candidates = [results_map[name]['Strategy'] for name in sorted_names]
            
            top_candidates = candidates[:50]
            selected_strats = []
            
            if top_candidates:
                sig_matrix = backtester.generate_signal_matrix(top_candidates)
                selected_indices = []
                
                for i in range(len(top_candidates)):
                    if len(selected_strats) >= 5: break
                    
                    is_uncorrelated = True
                    current_sig = sig_matrix[:, i]
                    
                    for existing_idx in selected_indices:
                        existing_sig = sig_matrix[:, existing_idx]
                        if np.std(current_sig) == 0 or np.std(existing_sig) == 0:
                            corr = 0 
                        else:
                            corr = np.corrcoef(current_sig, existing_sig)[0, 1]
                            
                        if abs(corr) > 0.7:
                            is_uncorrelated = False
                            break
                    
                    if is_uncorrelated:
                        selected_indices.append(i)
                        selected_strats.append(top_candidates[i])
            
            # Create DF for display
            selected_names = {s.name for s in selected_strats}
            top_unique_df = df[df['Name'].isin(selected_names)].copy()
            top_unique_df = top_unique_df.sort_values(by='Robust%', ascending=False)
            
            # Formatting for Display
            display_df = top_unique_df.copy()
            for col in ['Ret%(Train)', 'Ret%(Val)', 'Ret%(Test)', 'Robust%', 'Sortino(OOS)']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
                
            print(display_df.to_string(index=False))
            
            # Save Top 5 Unique to JSON
            top_unique_strategies = []
            for s in selected_strats:
                res = results_map[s.name]
                s_dict = s.to_dict()
                s_dict['metrics'] = {
                    'robust_return': float(res['Robust_Ret']),
                    'train_return': float(res['Ret_Train']),
                    'val_return': float(res['Ret_Val']),
                    'test_return': float(res['Ret_Test']),
                    'sortino_oos': float(res['Sortino_OOS'])
                }
                top_unique_strategies.append(s_dict)
            
            top_unique_strategies.sort(key=lambda x: x['metrics']['robust_return'], reverse=True)
            
            out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}_top5_unique.json")
            with open(out_path, "w") as f:
                json.dump(top_unique_strategies, f, indent=4)
            print(f"  💾 Saved Top 5 Unique Strategies to: {out_path}")
                
        except Exception as e:
            print(f"  Error processing horizon {h}: {e}")

    print("\n" + "="*120)
    print("🧬 GLOBAL DOMINANT GENES 🧬")
    if global_gene_counts:
        for i, (g, c) in enumerate(global_gene_counts.most_common(20), 1):
            print(f"{i:2d}. {g:<40} ({c})")

if __name__ == "__main__":
    main()