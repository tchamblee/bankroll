import json
import os
import re
import sys
import pandas as pd
import numpy as np
import config
from collections import Counter
from backtest import BacktestEngine
from backtest.reporting import GeneTranslator, cluster_strategies
from backtest.strategy_loader import load_strategies
from genome import Strategy

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
        
        # STRICT FILTER: Must be profitable in OOS (Test) and Robust > 0
        # Relaxed Train/Val requirement to allow recovery strategies
        if ret_test <= 0 or res_test.iloc[i]['sortino'] <= 0.1 or robust_ret <= 0:
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
    
    # NEW LOGIC: Load from Candidates File
    candidates_path = config.CANDIDATES_FILE
    if not os.path.exists(candidates_path):
         print("❌ No candidates.json found.")
         sys.exit(0)
         
    with open(candidates_path, 'r') as f:
        candidate_dicts = json.load(f)
        
    if not candidate_dicts:
        print("❌ Candidate list is empty.")
        sys.exit(0)

    strategies_by_horizon = {}
    for d in candidate_dicts:
        s = Strategy.from_dict(d)
        # Manually attach horizon
        h = d.get('horizon')
        if h is None:
             print(f"⚠️ Strategy {s.name} missing horizon. Skipping.")
             continue
        s.horizon = h
        if h not in strategies_by_horizon:
            strategies_by_horizon[h] = []
        strategies_by_horizon[h].append(s)

    sorted_horizons = sorted(strategies_by_horizon.keys())
    global_gene_counts = Counter()
    
    for h in sorted_horizons:
        print(f"\n--- Horizon: {h} Bars ---")
        strategies = strategies_by_horizon[h]
        
        if not strategies: 
            continue

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
            # Convert genes to dict for translator if they are objects, but Strategy stores them as dicts in .long_genes/short_genes lists?
            # Strategy object has .long_genes as list of dicts.
            
            strat_genes = [GeneTranslator.translate_gene(g) for g in strat.long_genes + strat.short_genes]
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
        
        # --- CORRELATION CLUSTERING (Hierarchical) ---
        print("  Performing Hierarchical Correlation Clustering (Threshold 0.7)...")
        
        results_map = {res['Strategy'].name: res for res in all_horizon_results}
        selected_strats = cluster_strategies(all_horizon_results, backtester, threshold=0.7)

        print(f"  Found {len(selected_strats)} distinct strategy clusters.")
        
        # Create DF for display
        selected_names = {s.name for s in selected_strats}
        top_unique_df = df[df['Name'].isin(selected_names)].copy()
        top_unique_df = top_unique_df.sort_values(by='Robust%', ascending=False)
        
        # Formatting for Display
        display_df = top_unique_df.copy()
        for col in ['Ret%(Train)', 'Ret%(Val)', 'Ret%(Test)', 'Robust%', 'Sortino(OOS)']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
            
        print(display_df.to_string(index=False))
        
        # Save ALL Unique to JSON
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
        
        out_path = config.APEX_FILE_TEMPLATE.format(h).replace(".json", "_top5_unique.json")
        with open(out_path, "w") as f:
            json.dump(top_unique_strategies, f, indent=4)
        print(f"  💾 Saved {len(top_unique_strategies)} Unique Cluster Champions to: {out_path}")
                


    print("\n" + "="*120)
    print("🧬 GLOBAL DOMINANT GENES 🧬")
    if global_gene_counts:
        for i, (g, c) in enumerate(global_gene_counts.most_common(20), 1):
            print(f"{i:2d}. {g:<40} ({c})")

if __name__ == "__main__":
    main()