import argparse
import json
import os
import copy
import random
import numpy as np
import pandas as pd
import config
from backtest import BacktestEngine
from backtest.statistics import calculate_sortino_ratio
from backtest.utils import find_strategy_in_files
from genome import Strategy, GenomeFactory

class StrategyOptimizer:
    def __init__(self, target_name, source_file=None, horizon=180, strategy_dict=None):
        self.target_name = target_name
        self.source_file = source_file
        self.horizon = horizon
        self.strategy_dict = strategy_dict
        self.parent_strategy = None
        self.variants = []
        
        # Load Data
        print("Loading Feature Matrix...")
        self.data = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
        
        # Initialize Factory (for mutations) and Backtester
        self.factory = GenomeFactory()
        self.factory.set_stats(self.data)
        
        self.backtester = BacktestEngine(
            self.data, 
            cost_bps=config.COST_BPS, 
            annualization_factor=config.ANNUALIZATION_FACTOR
        )

    def load_parent(self):
        if self.strategy_dict:
            print(f"Using provided strategy data for {self.target_name}...")
            self.parent_strategy = Strategy.from_dict(self.strategy_dict)
        else:
            print(f"Searching for {self.target_name} in {self.source_file}...")
            with open(self.source_file, 'r') as f:
                strategies = json.load(f)
                
            target_data = next((s for s in strategies if s['name'] == self.target_name), None)
            if not target_data:
                raise ValueError(f"Strategy {self.target_name} not found in {self.source_file}")
                
            self.parent_strategy = Strategy.from_dict(target_data)
            
        print(f"✅ Loaded Parent: {self.parent_strategy.name}")
        print(f"   Genes: {len(self.parent_strategy.long_genes)} Long, {len(self.parent_strategy.short_genes)} Short")
        print(f"   Concordance: {self.parent_strategy.min_concordance}")

    def generate_variants(self, n_jitter=20, n_mutants=20):
        print("\n🧬 Generating Variants...")
        
        # 1. Simplification (Ablation) - Remove 1 gene at a time
        # This checks if "Less is More"
        all_genes = self.parent_strategy.long_genes + self.parent_strategy.short_genes
        if len(all_genes) > 1:
            for i in range(len(self.parent_strategy.long_genes)):
                variant = copy.deepcopy(self.parent_strategy)
                variant.name = f"{self.target_name}_Simple_L{i}"
                variant.long_genes.pop(i)
                variant.recalculate_concordance()
                self.variants.append(variant)

            for i in range(len(self.parent_strategy.short_genes)):
                variant = copy.deepcopy(self.parent_strategy)
                variant.name = f"{self.target_name}_Simple_S{i}"
                variant.short_genes.pop(i)
                variant.recalculate_concordance()
                self.variants.append(variant)

        # 2. Relaxation - Lower Concordance
        if self.parent_strategy.min_concordance > 1:
            variant = copy.deepcopy(self.parent_strategy)
            variant.name = f"{self.target_name}_Relaxed"
            variant.min_concordance -= 1
            self.variants.append(variant)

        # 3. Jitter (Parameter Noise) - Test Robustness / Fine Tuning
        for i in range(n_jitter):
            variant = copy.deepcopy(self.parent_strategy)
            variant.name = f"{self.target_name}_Jitter_{i}"
            
            # Jitter every gene slightly
            for g in variant.long_genes + variant.short_genes:
                # Jitter Threshold
                if hasattr(g, 'threshold'):
                    # ± 5% change
                    g.threshold *= random.uniform(0.95, 1.05)
                
                # Jitter Window
                if hasattr(g, 'window') and isinstance(g.window, int):
                    # ± 5% change, min 2
                    g.window = max(2, int(g.window * random.uniform(0.95, 1.05)))
                    
            self.variants.append(variant)

        # 4. Mutation - Swap logic
        for i in range(n_mutants):
            variant = copy.deepcopy(self.parent_strategy)
            variant.name = f"{self.target_name}_Mutant_{i}"
            
            # Mutate 1 random gene
            genes = variant.long_genes + variant.short_genes
            if genes:
                target_gene = random.choice(genes)
                target_gene.mutate(self.factory.features)
                
            variant.recalculate_concordance()
            self.variants.append(variant)

        print(f"   Generated {len(self.variants)} variants.")

    def evaluate_and_report(self):
        print("\n🔬 Evaluating Variants on All Sets (Train / Val / Test)...")
        
        population = [self.parent_strategy] + self.variants
        
        # Hydrate Horizon on Strategy Objects (Critical for Signal Generation Checks)
        for s in population:
            s.horizon = self.horizon
        
        # 1. Generate Full Signal Matrix
        print("   Generating signals...")
        # Pass horizon explicitly to enforce Safe Entry logic (Time Filter)
        full_signals = self.backtester.generate_signal_matrix(population, horizon=self.horizon)
        
        # --- FIX: LOOKAHEAD BIAS (Next Open Execution) ---
        # Signals generated at Close[t] must be executed at Open[t+1].
        # We shift signals forward by 1.
        full_signals = np.vstack([np.zeros((1, full_signals.shape[1]), dtype=full_signals.dtype), full_signals[:-1]])
        
        # 2. Define Split Indices
        train_end = self.backtester.train_idx
        val_end = self.backtester.val_idx
        
        # Print Date Ranges
        t_vec = self.backtester.times_vec
        def fmt_date(idx):
            if hasattr(t_vec, 'iloc'): return str(t_vec.iloc[idx])
            return str(t_vec[idx])
            
        print(f"   📅 Data Splits:")
        print(f"      Train: {fmt_date(0)} -> {fmt_date(train_end-1)}")
        print(f"      Val:   {fmt_date(train_end)} -> {fmt_date(val_end-1)}")
        print(f"      Test:  {fmt_date(val_end)} -> {fmt_date(len(t_vec)-1)}")
        
        # 3. Batch Simulation for Each Split
        print("   Simulating Train...")
        train_rets, train_trades = self.backtester.run_simulation_batch(
            full_signals[:train_end], population, 
            self.backtester.open_vec[:train_end], 
            self.backtester.times_vec.iloc[:train_end] if hasattr(self.backtester.times_vec, 'iloc') else self.backtester.times_vec[:train_end],
            time_limit=self.horizon, highs=self.backtester.high_vec[:train_end], lows=self.backtester.low_vec[:train_end], atr=self.backtester.atr_vec[:train_end]
        )
        
        print("   Simulating Validation...")
        val_rets, val_trades = self.backtester.run_simulation_batch(
            full_signals[train_end:val_end], population, 
            self.backtester.open_vec[train_end:val_end], 
            self.backtester.times_vec.iloc[train_end:val_end] if hasattr(self.backtester.times_vec, 'iloc') else self.backtester.times_vec[train_end:val_end],
            time_limit=self.horizon, highs=self.backtester.high_vec[train_end:val_end], lows=self.backtester.low_vec[train_end:val_end], atr=self.backtester.atr_vec[train_end:val_end]
        )
        
        print("   Simulating Test...")
        test_rets, test_trades = self.backtester.run_simulation_batch(
            full_signals[val_end:], population, 
            self.backtester.open_vec[val_end:], 
            self.backtester.times_vec.iloc[val_end:] if hasattr(self.backtester.times_vec, 'iloc') else self.backtester.times_vec[val_end:],
            time_limit=self.horizon, highs=self.backtester.high_vec[val_end:], lows=self.backtester.low_vec[val_end:], atr=self.backtester.atr_vec[val_end:]
        )
        
        # Calculate Hamming Distance (Signal Diff) vs Parent (Index 0)
        parent_sig = full_signals[:, 0]
        
        # Process Results
        results_data = []
        for i, strat in enumerate(population):
            # Calculate metrics
            t_r = np.sum(train_rets[:, i])
            v_r = np.sum(val_rets[:, i])
            te_r = np.sum(test_rets[:, i])
            
            t_s = calculate_sortino_ratio(train_rets[:, i], config.ANNUALIZATION_FACTOR)
            v_s = calculate_sortino_ratio(val_rets[:, i], config.ANNUALIZATION_FACTOR)
            te_s = calculate_sortino_ratio(test_rets[:, i], config.ANNUALIZATION_FACTOR)
            
            # Hamming Distance
            diff_bits = np.count_nonzero(full_signals[:, i] != parent_sig)
            
            results_data.append({
                'name': strat.name,
                'strat': strat,
                'diff': diff_bits,
                'train': {'ret': t_r, 'sortino': t_s, 'trades': train_trades[i]},
                'val': {'ret': v_r, 'sortino': v_s, 'trades': val_trades[i]},
                'test': {'ret': te_r, 'sortino': te_s, 'trades': test_trades[i]}
            })

        parent = results_data[0]
        print(f"\n🏛️  PARENT ({parent['name']}):")
        print(f"   TRAIN: Ret {parent['train']['ret']*100:6.2f}% | Sort {parent['train']['sortino']:5.2f} | Tr {parent['train']['trades']}")
        print(f"   VAL  : Ret {parent['val']['ret']*100:6.2f}% | Sort {parent['val']['sortino']:5.2f} | Tr {parent['val']['trades']}")
        print(f"   TEST : Ret {parent['test']['ret']*100:6.2f}% | Sort {parent['test']['sortino']:5.2f} | Tr {parent['test']['trades']}")
        print("-" * 80)

        better_variants = []
        for res in results_data[1:]:
            # Criteria: 
            # 1. Must be >= Parent in ALL sets (Train, Val, Test) - "Do No Harm"
            # 2. Must be Profitable (Non-Negative) in ALL sets - "No Losers"
            
            # Non-Inferiority
            train_ok = res['train']['sortino'] >= parent['train']['sortino']
            val_ok = res['val']['sortino'] >= parent['val']['sortino']
            test_ok = res['test']['sortino'] >= parent['test']['sortino']
            
            # Profitability
            train_pos = res['train']['sortino'] >= 0 and res['train']['ret'] >= 0
            val_pos = res['val']['sortino'] >= 0 and res['val']['ret'] >= 0
            test_pos = res['test']['sortino'] >= 0 and res['test']['ret'] >= 0
            
            if train_ok and val_ok and test_ok and train_pos and val_pos and test_pos:
                res['note'] = "💎 Universal"
                better_variants.append(res)
        
        better_variants.sort(key=lambda x: x['test']['sortino'], reverse=True)
        
        print(f"\n🏆 Top {min(10, len(better_variants))} Variants (Sorted by Test Sortino):")
        print(f"{'Name':<30} | {'Diff':<6} | {'Train Sort':<10} | {'Val Sort':<9} | {'Test Sort':<9} | {'Test Ret':<9} | {'Note'}")
        print("-" * 105)
        for v in better_variants[:10]:
            print(f"{v['name']:<30} | {v['diff']:<6} | {v['train']['sortino']:10.2f} | {v['val']['sortino']:9.2f} | {v['test']['sortino']:9.2f} | {v['test']['ret']*100:8.2f}% | {v['note']}")

        # Save
        if better_variants:
            out_file = f"output/strategies/optimized_{self.target_name}.json"
            
            def convert_stats(stats_dict):
                return {
                    'ret': float(stats_dict['ret']),
                    'sortino': float(stats_dict['sortino']),
                    'trades': int(stats_dict['trades'])
                }

            to_save = []
            for v in better_variants:
                d = v['strat'].to_dict()
                d.update({
                    'horizon': self.horizon,
                    'train_stats': convert_stats(v['train']),
                    'val_stats': convert_stats(v['val']),
                    'test_stats': convert_stats(v['test'])
                })
                to_save.append(d)
                
            with open(out_file, 'w') as f:
                json.dump(to_save, f, indent=4)
            print(f"\n💾 Saved {len(to_save)} variants to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize a specific strategy variant")
    parser.add_argument("name", type=str, help="Name of the strategy (e.g., Child_3604)")
    parser.add_argument("--file", type=str, default=None, help="Path to strategy file (Optional, auto-detected if omitted)")
    parser.add_argument("--horizon", type=int, default=None, help="Time horizon (Optional, auto-detected if omitted)")
    
    args = parser.parse_args()
    
    target_dict = None
    file_path = args.file
    horizon = args.horizon
    
    # Auto-discovery if file/horizon missing
    if not file_path:
        print(f"🔍 Searching for strategy '{args.name}'...")
        # find_strategy_in_files returns the dict, but we can't get the filename easily from it currently.
        # But StrategyOptimizer needs the logic. We can instantiate Strategy directly.
        found_dict = find_strategy_in_files(args.name)
        if found_dict:
            target_dict = found_dict
            if not horizon:
                horizon = found_dict.get('horizon', 60)
                print(f"   ✅ Inferred Horizon: {horizon}")
            print(f"   ✅ Found strategy data in index.")
        else:
            print(f"❌ Error: Could not locate strategy '{args.name}' in standard files.")
            exit(1)
    
    if not horizon:
        print(f"❌ Error: Could not infer horizon for '{args.name}'. Please provide --horizon.")
        exit(1)
    
    optimizer = StrategyOptimizer(args.name, source_file=file_path, horizon=horizon, strategy_dict=target_dict)
    optimizer.load_parent()
    optimizer.generate_variants(n_jitter=20, n_mutants=20)
    optimizer.evaluate_and_report()

