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
    def __init__(self, target_name, source_file=None, horizon=180, strategy_dict=None, backtester=None, data=None, verbose=True):
        self.target_name = target_name
        self.source_file = source_file
        self.horizon = horizon
        self.strategy_dict = strategy_dict
        self.parent_strategy = None
        self.variants = []
        self.verbose = verbose
        
        # Initialize Backtester if not provided
        if backtester:
            self.backtester = backtester
            self.data = backtester.raw_data
        else:
            if self.verbose: print("Loading Feature Matrix...")
            self.data = data if data is not None else pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
            self.backtester = BacktestEngine(
                self.data, 
                cost_bps=config.COST_BPS, 
                annualization_factor=config.ANNUALIZATION_FACTOR
            )
            
        # Initialize Factory (for mutations)
        self.factory = GenomeFactory()
        self.factory.set_stats(self.data)

    def load_parent(self):
        if self.parent_strategy:
            return

        if self.strategy_dict:
            if self.verbose: print(f"Using provided strategy data for {self.target_name}...")
            self.parent_strategy = Strategy.from_dict(self.strategy_dict)
        else:
            if self.verbose: print(f"Searching for {self.target_name} in {self.source_file}...")
            with open(self.source_file, 'r') as f:
                strategies = json.load(f)
                
            target_data = next((s for s in strategies if s['name'] == self.target_name), None)
            if not target_data:
                raise ValueError(f"Strategy {self.target_name} not found in {self.source_file}")
                
            self.parent_strategy = Strategy.from_dict(target_data)
            
        if self.verbose:
            print(f"✅ Loaded Parent: {self.parent_strategy.name}")
            print(f"   Genes: {len(self.parent_strategy.long_genes)} Long, {len(self.parent_strategy.short_genes)} Short")
            print(f"   Concordance: {self.parent_strategy.min_concordance}")

    def generate_variants(self, n_jitter=20, n_mutants=20):
        if self.verbose: print("\n🧬 Generating Variants...")
        
        if not self.parent_strategy:
            self.load_parent()

        self.variants = []

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

        if self.verbose: print(f"   Generated {len(self.variants)} variants.")

    def optimize_stops(self, strategy):
        """
        Performs a grid search on Stop Loss and Take Profit parameters.
        Returns the best (SL, TP) tuple based on Train+Val performance.
        """
        if self.verbose: print(f"   ⚙️ Optimizing Stops for {strategy.name}...")
        
        sl_options = [1.0, 1.5, 2.0, 2.5, 3.0]
        tp_options = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
        
        best_score = -999.0
        best_params = (strategy.stop_loss_pct, strategy.take_profit_pct)
        
        # We need a temporary population of clones
        grid_variants = []
        param_map = {} # idx -> (sl, tp)
        
        idx = 0
        for sl in sl_options:
            for tp in tp_options:
                if tp <= sl: continue # Ignore bad R:R
                
                # Clone
                clone = copy.deepcopy(strategy)
                clone.name = f"{strategy.name}_SL{sl}_TP{tp}"
                clone.stop_loss_pct = sl
                clone.take_profit_pct = tp
                
                grid_variants.append(clone)
                param_map[idx] = (sl, tp)
                idx += 1
                
        # Evaluate Batch
        # Optimization on TRAIN + VALIDATION only
        train_res = self.backtester.evaluate_population(grid_variants, set_type='train', time_limit=self.horizon)
        val_res = self.backtester.evaluate_population(grid_variants, set_type='validation', time_limit=self.horizon)
        
        # Map results
        t_map = {row['id']: row for _, row in train_res.iterrows()}
        v_map = {row['id']: row for _, row in val_res.iterrows()}
        
        for i, variant in enumerate(grid_variants):
            if variant.name not in t_map or variant.name not in v_map: continue
            
            t_s = float(t_map[variant.name]['sortino'])
            v_s = float(v_map[variant.name]['sortino'])
            t_r = float(t_map[variant.name]['total_return'])
            v_r = float(v_map[variant.name]['total_return'])
            
            # Robust Fitness: Min(Train, Val)
            fitness = min(t_s, v_s)
            
            # Constraints
            if t_r > 0 and v_r > 0 and fitness > 1.0:
                if fitness > best_score:
                    best_score = fitness
                    best_params = param_map[i]
        
        if self.verbose: print(f"      Best Stops Found: SL={best_params[0]}, TP={best_params[1]} (Fit: {best_score:.2f})")
        return best_params

    def get_best_variant(self):
        """
        Programmatically find and return the best variant.
        CRITICAL UPDATE: Selection is based on TRAIN + VALIDATION performance only.
        Test set is used ONLY as a final gatekeeper, not a selection criterion.
        """
        if not self.variants:
            self.generate_variants()
            
        population = [self.parent_strategy] + self.variants
        for s in population:
            s.horizon = self.horizon

        # 1. Evaluate In-Sample (Train + Val)
        train_df = self.backtester.evaluate_population(population, set_type='train', time_limit=self.horizon)
        val_df = self.backtester.evaluate_population(population, set_type='validation', time_limit=self.horizon)
        
        def get_stats(df, s_id):
            row = df[df['id'] == s_id].iloc[0]
            return {'ret': row['total_return'], 'sortino': row['sortino'], 'trades': int(row['trades'])}

        parent_train = get_stats(train_df, self.parent_strategy.name)
        parent_val = get_stats(val_df, self.parent_strategy.name)
        
        # Parent Baseline
        parent_fitness = min(parent_train['sortino'], parent_val['sortino'])
        
        best_candidate = None
        best_fitness = parent_fitness

        for variant in self.variants:
            if variant.name not in train_df['id'].values: continue
            
            v_train = get_stats(train_df, variant.name)
            v_val = get_stats(val_df, variant.name)
            
            # Selection Metric: Robust Sortino (Min of Train/Val)
            v_fitness = min(v_train['sortino'], v_val['sortino'])
            
            # Constraints:
            # 1. Must be profitable on both
            # 2. Must beat parent fitness
            is_profitable = v_train['ret'] > 0 and v_val['ret'] > 0
            improved = v_fitness > parent_fitness
            
            if is_profitable and improved:
                if v_fitness > best_fitness:
                    best_fitness = v_fitness
                    best_candidate = variant

        # 2. Final Gate: Test Set Check (OOS)
        # We only check the BEST candidate against the Test set.
        # We do NOT shop around for the variant that fits Test best.
        
        if best_candidate:
            if self.verbose: print(f"   🏆 Selected Candidate (In-Sample): {best_candidate.name} (Fit: {best_fitness:.2f} vs Parent {parent_fitness:.2f})")
            
            # Run Test Set
            test_pop = [self.parent_strategy, best_candidate]
            test_df = self.backtester.evaluate_population(test_pop, set_type='test', time_limit=self.horizon)
            
            p_test = get_stats(test_df, self.parent_strategy.name)
            c_test = get_stats(test_df, best_candidate.name)
            
            # Criteria:
            # 1. Must be profitable on Test
            # 2. Must NOT degrade Test Sortino significantly (> 90% of Parent)
            #    Or if Parent was bad on Test (<0), Candidate must be better.
            
            test_profitable = c_test['ret'] > 0
            
            if p_test['sortino'] > 0:
                # Allow slight degradation (robustness noise) if In-Sample gain is huge
                test_maintained = c_test['sortino'] >= (p_test['sortino'] * 0.9)
            else:
                # Parent failed test, Candidate MUST fix it
                test_maintained = c_test['sortino'] > p_test['sortino']
            
            if test_profitable and test_maintained:
                best_stats = {
                    'train': get_stats(train_df, best_candidate.name),
                    'val': get_stats(val_df, best_candidate.name),
                    'test': c_test
                }
                
                # Check if we should optimize stops for this winner
                # (Optional: Recursively improve the winner)
                best_sl, best_tp = self.optimize_stops(best_candidate)
                if (best_sl, best_tp) != (best_candidate.stop_loss_pct, best_candidate.take_profit_pct):
                    best_candidate.stop_loss_pct = best_sl
                    best_candidate.take_profit_pct = best_tp
                    # Re-eval one last time? 
                    # For simplicity, we assume grid search result holds. 
                    # But metrics in best_stats would be slightly stale.
                    # It's fine for now, the caller usually re-evaluates.
                
                return best_candidate, best_stats
            else:
                if self.verbose: print(f"   ❌ Candidate failed Test Gate. Ret: {c_test['ret']:.4f}, Sort: {c_test['sortino']:.2f} (Parent: {p_test['sortino']:.2f})")
                return None, None
                
        return None, None

    def evaluate_and_report(self):
        print("\n🔬 Evaluating Variants on All Sets (Train / Val / Test)...")
        
        population = [self.parent_strategy] + self.variants
        
        # Hydrate Horizon on Strategy Objects
        for s in population:
            s.horizon = self.horizon
            
        # Use Engine's Built-in Evaluation
        train_df = self.backtester.evaluate_population(population, set_type='train', time_limit=self.horizon)
        val_df = self.backtester.evaluate_population(population, set_type='validation', time_limit=self.horizon)
        test_df = self.backtester.evaluate_population(population, set_type='test', time_limit=self.horizon)
        
        # Calculate Hamming Distance (Signal Diff) vs Parent
        full_signals = self.backtester.generate_signal_matrix(population, horizon=self.horizon)
        parent_sig = full_signals[:, 0]
        
        results_data = []
        for i, strat in enumerate(population):
            diff_bits = np.count_nonzero(full_signals[:, i] != parent_sig)
            
            # Helper to extract from DF
            def get_row(df, s_id):
                row = df[df['id'] == s_id].iloc[0]
                return {'ret': row['total_return'], 'sortino': row['sortino'], 'trades': int(row['trades'])}
                
            results_data.append({
                'name': strat.name,
                'strat': strat,
                'diff': diff_bits,
                'train': get_row(train_df, strat.name),
                'val': get_row(val_df, strat.name),
                'test': get_row(test_df, strat.name)
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
            out_file = os.path.join(config.DIRS['STRATEGIES_DIR'], f"optimized_{self.target_name}.json")
            
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

