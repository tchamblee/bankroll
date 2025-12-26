import pandas as pd
import numpy as np
import os
import json
import argparse
import config
from genome import Strategy
from backtest import BacktestEngine

def load_strategies(path):
    print(f"Loading strategies from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
        
    strategies = []
    for d in data:
        s = Strategy.from_dict(d)
        s.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)
        # Hydrate legacy params
        if not hasattr(s, 'stop_loss_pct'):
            s.stop_loss_pct = d.get('stop_loss_pct', config.DEFAULT_STOP_LOSS)
        if not hasattr(s, 'take_profit_pct'):
            s.take_profit_pct = d.get('take_profit_pct', config.DEFAULT_TAKE_PROFIT)
        strategies.append(s)
    
    print(f"Loaded {len(strategies)} strategies.")
    return strategies

def run_stress_test(strategy_file):
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        return

    # 1. Load Data
    print("Loading Market Data...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # 2. Load Strategies
    strategies = load_strategies(strategy_file)
    if not strategies: return
    
    # 3. Run CPCV
    # N=6, k=2 => 15 Combinations (Reasonable for a quick stress test)
    # N=10, k=2 => 45 Combinations (More rigorous)
    print("\nRunning Combinatorial Purged Cross-Validation (CPCV)...")
    print("Generating 15 alternative market histories (N=6, k=2)...")
    
    backtester.ensure_context(strategies)
    
    # We use the horizon of the first strategy for safety/filtering, 
    # but run_simulation_batch handles individual horizons if set on objects.
    # Pass max horizon to ensure context is generated
    max_h = max([s.horizon for s in strategies])
    
    results = backtester.evaluate_combinatorial_purged_cv(
        strategies, 
        n_folds=6, 
        n_test_folds=2, 
        time_limit=max_h
    )
    
    # 4. Report
    print("\n" + "="*80)
    print("🛡️  STRESS TEST REPORT (CPCV)")
    print("="*80)
    print(f"{ 'Name':<20} | {'P5 Sortino':<12} | {'Median':<10} | {'Std Dev':<10} | {'Min':<10} | {'Verdict'}")
    print("-" * 80)
    
    passed_strategies = []
    
    for idx, r in results.iterrows():
        name = r['id']
        p5 = r['cpcv_p5_sortino']
        med = r['cpcv_median']
        std = r['cpcv_std']
        min_s = r['cpcv_min']
        
        # Rating Logic
        # P5 (5th Percentile) Sortino > 1.5 means in 95% of market splits, Sortino was > 1.5
        if p5 > 1.5 and min_s > -1.0:
            verdict = "✅ PASSED"
            passed_strategies.append(name)
        elif p5 > 0:
            verdict = "⚠️  RISKY"
        else:
            verdict = "❌ FAILED"
            
        print(f"{name:<20} | {p5:<12.2f} | {med:<10.2f} | {std:<10.2f} | {min_s:<10.2f} | {verdict}")
        
    print("="*80)
    
    if passed_strategies:
        print(f"\n🎉 {len(passed_strategies)}/{len(strategies)} strategies passed the Stress Test.")
    else:
        print("\n💀 No strategies passed the Stress Test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to strategy JSON file")
    args = parser.parse_args()
    
    run_stress_test(args.file)
