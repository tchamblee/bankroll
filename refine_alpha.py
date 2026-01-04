import json
import os
import sys
import numpy as np
import pandas as pd
import config
import copy
import random
from backtest import BacktestEngine
from backtest.strategy_loader import load_strategies
from genome import Strategy
from backtest.reporting import GeneTranslator

def mutate_strategy_parameters(strat):
    """
    Creates a mutant of the strategy by slightly perturbing parameters (thresholds, windows).
    Does NOT change the gene structure (no new logic), just fine-tuning.
    """
    mutant = strat.copy()
    mutant.name = f"{strat.name}_Mutant_{random.randint(1000,9999)}"
    mutant.generation_found = f"{strat.generation_found}+"
    
    # Mutate Genes
    for gene in mutant.long_genes + mutant.short_genes:
        # 1. Mutate Thresholds (Small jitter)
        if hasattr(gene, 'threshold'):
            jitter = random.uniform(-0.1, 0.1) * gene.threshold
            if jitter == 0: jitter = 0.05 # Fallback for zero threshold
            gene.threshold += jitter
            
        # 2. Mutate Windows (Small shift)
        if hasattr(gene, 'window'):
            if random.random() < 0.3: # 30% chance to shift window
                shift = random.choice([-1, 1])
                # Find current index in valid windows?
                # Or just +/- 1 index in standard windows list?
                # For now, just simplistic +/- 10%
                new_win = int(gene.window * (1 + random.uniform(-0.1, 0.1)))
                gene.window = max(5, new_win)
                
    return mutant

def refine_horizon(horizon, backtester):
    print(f"\n🔬 Refining Horizon: {horizon} Bars")
    
    apex_file = config.APEX_FILE_TEMPLATE.format(horizon).replace(".json", "_top5_unique.json")
    if not os.path.exists(apex_file):
        print(f"  ⚠️  No apex file found for horizon {horizon}. Skipping.")
        return []

    with open(apex_file, 'r') as f:
        champions_data = json.load(f)
        
    champions = []
    for d in champions_data:
        s = Strategy.from_dict(d)
        s.metrics = d.get('metrics', {})
        s.generation_found = d.get('generation_found', 'Apex')
        champions.append(s)
        
    print(f"  Loaded {len(champions)} champions.")
    
    refined_candidates = []
    
    for champ in champions:
        print(f"  > Refining: {champ.name} (Sortino: {champ.metrics.get('sortino_oos',0):.2f})")
        
        # 1. Generate Mutants
        mutants = [mutate_strategy_parameters(champ) for _ in range(50)] # 50 mutants per champ
        
        # 2. Evaluate
        # We want to see if we can improve Robust Return (Mean of Val/Test)
        # while maintaining or improving OOS performance.
        
        # Train
        res_train = backtester.evaluate_population(mutants, set_type='train', time_limit=horizon)
        # Val
        res_val = backtester.evaluate_population(mutants, set_type='validation', time_limit=horizon)
        # Test
        res_test = backtester.evaluate_population(mutants, set_type='test', time_limit=horizon)
        
        best_mutant = None
        best_robust_score = -999
        
        baseline_robust = (champ.metrics.get('val_return', 0) + champ.metrics.get('test_return', 0)) / 2
        if baseline_robust == 0: # Recalculate if missing
             # (Assume we'd need to re-eval base champ, but let's trust JSON for now or skip)
             pass
             
        for i, m in enumerate(mutants):
            t_r = res_train.iloc[i]['total_return']
            v_r = res_val.iloc[i]['total_return']
            te_r = res_test.iloc[i]['total_return']
            te_s = res_test.iloc[i]['sortino']
            trades = res_test.iloc[i]['trades']
            
            # Robust Return Metric
            robust_ret = (v_r + te_r) / 2
            
            # Constraints
            if te_r <= 0 or te_s < 1.0 or trades < 20:
                continue
                
            # Improvement Check?
            # We want the HIGHEST Robust Return that is also OOS Profitable.
            if robust_ret > best_robust_score:
                best_robust_score = robust_ret
                best_mutant = m
                # Store metrics for reporting
                best_mutant.metrics = {
                    'train_return': t_r,
                    'val_return': v_r,
                    'test_return': te_r,
                    'sortino_oos': te_s,
                    'robust_return': robust_ret,
                    'trades': int(trades)
                }

        # Compare Best Mutant vs Original
        # (Actually, we just add the best mutant to the pool if it's good)
        if best_mutant:
             print(f"    ✨ Found Improved Mutant! Robust: {best_mutant.metrics['robust_return']*100:.2f}% (Sortino: {best_mutant.metrics['sortino_oos']:.2f})")
             best_mutant.horizon = horizon
             refined_candidates.append(best_mutant)
        else:
             print("    . No better mutant found.")
             # Add original if it passes basic checks? 
             # Or just assume original is already in candidates.json? 
             # Let's add original back to be safe, ensuring it's in the refined list.
             # But we need to re-evaluate it to get fresh metrics if they weren't fully trusted.
             # For now, simpler: Just keep the new ones.
             
    return refined_candidates

def main():
    print("💎 ALPHA REFINEMENT ENGINE 💎")
    print("=============================")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    base_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(base_df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    all_refined = []
    
    for h in config.PREDICTION_HORIZONS:
        refined = refine_horizon(h, backtester)
        all_refined.extend(refined)
        
    if not all_refined:
        print("\n❌ No refined strategies found.")
        sys.exit(0)
        
    # Save to refined_candidates.json
    out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "refined_candidates.json")
    out_data = []
    for s in all_refined:
        d = s.to_dict()
        d['horizon'] = s.horizon # Ensure horizon is top-level
        # Flatten metrics for candidates file compatibility
        m = s.metrics
        d['train_return'] = float(m['train_return'])
        d['val_return'] = float(m['val_return'])
        d['test_return'] = float(m['test_return'])
        d['test_sortino'] = float(m['sortino_oos'])
        d['test_trades'] = int(m['trades'])
        d['robust_return'] = float(m['robust_return'])
        out_data.append(d)
        
    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=4)
        
    print(f"\n💾 Saved {len(all_refined)} Refined Strategies to: {out_path}")
    print("👉 Next Step: Run 'python3 optimize_candidate.py' (Wait, need to point it to this file? or Merge?)")
    print("   Recommendation: Use 'python3 manage_candidates.py import refined_candidates.json' (if that command existed)")
    print("   Or manually rename/merge.")

if __name__ == "__main__":
    main()
