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
    def __init__(self, target_name, source_file=None, horizon=180, strategy_dict=None, backtester=None, data=None, verbose=True, seed=None):
        self.target_name = target_name
        self.source_file = source_file
        self.horizon = horizon
        self.strategy_dict = strategy_dict
        self.parent_strategy = None
        self.variants = []
        self.verbose = verbose

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        else:
            # Default: deterministic seed based on strategy name
            default_seed = hash(target_name) % (2**32)
            random.seed(default_seed)
            np.random.seed(default_seed)
            if verbose:
                print(f"Using default seed {default_seed} (based on strategy name)")

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

        # 2a. Relaxation - Lower Concordance (more permissive signals)
        if self.parent_strategy.min_concordance > 1:
            variant = copy.deepcopy(self.parent_strategy)
            variant.name = f"{self.target_name}_Relaxed"
            variant.min_concordance -= 1
            self.variants.append(variant)

        # 2b. Strictness - Higher Concordance (more selective signals)
        max_concordance = max(len(self.parent_strategy.long_genes), len(self.parent_strategy.short_genes))
        if self.parent_strategy.min_concordance < max_concordance:
            variant = copy.deepcopy(self.parent_strategy)
            variant.name = f"{self.target_name}_Strict"
            variant.min_concordance += 1
            self.variants.append(variant)

        # 3. Jitter (Parameter Noise) - Test Robustness / Fine Tuning
        jitter_pct = getattr(config, 'OPTIMIZE_JITTER_PCT', 0.05)
        for i in range(n_jitter):
            variant = copy.deepcopy(self.parent_strategy)
            variant.name = f"{self.target_name}_Jitter_{i}"

            # Jitter every gene slightly
            for g in variant.long_genes + variant.short_genes:
                # Jitter Threshold
                if hasattr(g, 'threshold'):
                    g.threshold *= random.uniform(1 - jitter_pct, 1 + jitter_pct)

                # Jitter Window
                if hasattr(g, 'window') and isinstance(g.window, int):
                    g.window = max(2, int(g.window * random.uniform(1 - jitter_pct, 1 + jitter_pct)))

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

        # 5. Expansion - Add 1 random gene (test if more complexity helps)
        # Only expand if we're below the max gene count
        current_genes = len(self.parent_strategy.long_genes) + len(self.parent_strategy.short_genes)
        if current_genes < config.GENE_COUNT_MAX * 2:  # *2 because long+short
            for i in range(5):
                variant = copy.deepcopy(self.parent_strategy)
                variant.name = f"{self.target_name}_Expand_{i}"
                new_gene = self.factory.create_random_gene()
                if random.random() < 0.5:
                    variant.long_genes.append(new_gene)
                else:
                    variant.short_genes.append(new_gene)
                variant.recalculate_concordance()
                self.variants.append(variant)

        if self.verbose: print(f"   Generated {len(self.variants)} variants.")

    def optimize_stops(self, strategy):
        """
        Performs a grid search on Stop Loss and Take Profit parameters.
        Returns the best (SL, TP) tuple based on Train+Val performance.
        """
        if self.verbose: print(f"   ⚙️ Optimizing Stops for {strategy.name}...")

        sl_options = getattr(config, 'OPTIMIZE_SL_OPTIONS', [1.0, 1.5, 2.0, 2.5, 3.0])
        tp_options = getattr(config, 'OPTIMIZE_TP_OPTIONS', [2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
        
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

    def get_best_variant(self, use_walk_forward=None):
        """
        Programmatically find and return the best variant.
        CRITICAL: Selection is based on CPCV (Combinatorial Purged Cross-Validation) metrics.
        This ensures robustness across multiple time period combinations.

        Args:
            use_walk_forward: Deprecated, kept for compatibility. CPCV is always used.
        """
        if not self.variants:
            self.generate_variants()

        population = [self.parent_strategy] + self.variants
        for s in population:
            s.horizon = self.horizon

        def get_stats(df, s_id):
            row = df[df['id'] == s_id].iloc[0]
            return {'ret': row['total_return'], 'sortino': row['sortino'], 'trades': int(row['trades'])}

        def get_cpcv_stats(df, s_id):
            row = df[df['id'] == s_id].iloc[0]
            return {
                'cpcv_min': row['cpcv_min'],
                'cpcv_p5': row['cpcv_p5_sortino'],
                'cpcv_median': row['cpcv_median']
            }

        # Complexity penalty: prefer simpler strategies (fewer genes)
        def calc_fitness_cpcv(cpcv_min, n_genes):
            # Use CPCV min as the primary fitness metric
            penalty = n_genes * config.COMPLEXITY_PENALTY_PER_GENE
            return cpcv_min - penalty

        # Run CPCV evaluation
        if self.verbose:
            print("   Running CPCV evaluation for robust selection...")
        cpcv_df = self.backtester.evaluate_combinatorial_purged_cv(population, time_limit=self.horizon)

        # Also get train/val for profitability checks
        train_df = self.backtester.evaluate_population(population, set_type='train', time_limit=self.horizon)
        val_df = self.backtester.evaluate_population(population, set_type='validation', time_limit=self.horizon)

        # Parent baseline
        parent_cpcv = get_cpcv_stats(cpcv_df, self.parent_strategy.name)
        parent_n_genes = len(self.parent_strategy.long_genes) + len(self.parent_strategy.short_genes)
        parent_fitness = calc_fitness_cpcv(parent_cpcv['cpcv_min'], parent_n_genes)

        if self.verbose:
            print(f"   Parent CPCV: Min={parent_cpcv['cpcv_min']:.2f}, P5={parent_cpcv['cpcv_p5']:.2f}, Fitness={parent_fitness:.2f}")

        best_candidate = None
        best_fitness = parent_fitness

        for variant in self.variants:
            v_n_genes = len(variant.long_genes) + len(variant.short_genes)

            if variant.name not in cpcv_df['id'].values:
                continue
            if variant.name not in train_df['id'].values:
                continue

            v_cpcv = get_cpcv_stats(cpcv_df, variant.name)
            v_train = get_stats(train_df, variant.name)
            v_val = get_stats(val_df, variant.name)

            # CRITICAL: Reject any variant below CPCV threshold
            if v_cpcv['cpcv_min'] < config.MIN_CPCV_THRESHOLD:
                continue

            v_fitness = calc_fitness_cpcv(v_cpcv['cpcv_min'], v_n_genes)

            # Constraints:
            # 1. Must be profitable on both train and val
            # 2. Must meet CPCV threshold (checked above)
            # 3. Must meet sortino threshold on both train and val
            # 4. Must beat parent fitness by minimum threshold (avoid noise)
            is_profitable = v_train['ret'] > 0 and v_val['ret'] > 0
            min_val_sortino = getattr(config, 'MIN_VAL_SORTINO', config.MIN_SORTINO_THRESHOLD)
            meets_sortino = v_train['sortino'] >= config.MIN_SORTINO_THRESHOLD and v_val['sortino'] >= min_val_sortino

            min_improvement = getattr(config, 'OPTIMIZE_MIN_IMPROVEMENT', 0.05)
            improvement_threshold = abs(parent_fitness) * min_improvement if parent_fitness != 0 else 0.1
            improved = v_fitness > (parent_fitness + improvement_threshold)

            if is_profitable and meets_sortino and improved:
                if v_fitness > best_fitness:
                    best_fitness = v_fitness
                    best_candidate = variant

        # 2. Final Gate: Test Set Check (OOS)
        # We only check the BEST candidate against the Test set.
        # We do NOT shop around for the variant that fits Test best.

        if best_candidate:
            best_cpcv = get_cpcv_stats(cpcv_df, best_candidate.name)
            if self.verbose:
                print(f"   🏆 Selected Candidate: {best_candidate.name}")
                print(f"      CPCV: Min={best_cpcv['cpcv_min']:.2f}, P5={best_cpcv['cpcv_p5']:.2f}, Fitness={best_fitness:.2f}")

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
                # Check if we should optimize stops for this winner
                best_sl, best_tp = self.optimize_stops(best_candidate)
                if (best_sl, best_tp) != (best_candidate.stop_loss_pct, best_candidate.take_profit_pct):
                    best_candidate.stop_loss_pct = best_sl
                    best_candidate.take_profit_pct = best_tp

                # Re-evaluate with final stops to get accurate stats including CPCV
                final_cpcv = self.backtester.evaluate_combinatorial_purged_cv(
                    [best_candidate], time_limit=self.horizon
                )
                final_eval = self.backtester.evaluate_population(
                    [best_candidate], set_type='train', time_limit=self.horizon
                )
                final_val = self.backtester.evaluate_population(
                    [best_candidate], set_type='validation', time_limit=self.horizon
                )
                final_test = self.backtester.evaluate_population(
                    [best_candidate], set_type='test', time_limit=self.horizon
                )

                final_cpcv_stats = get_cpcv_stats(final_cpcv, best_candidate.name)
                best_stats = {
                    'train': get_stats(final_eval, best_candidate.name),
                    'val': get_stats(final_val, best_candidate.name),
                    'test': get_stats(final_test, best_candidate.name),
                    'cpcv_min': final_cpcv_stats['cpcv_min'],
                    'cpcv_p5': final_cpcv_stats['cpcv_p5']
                }

                if self.verbose:
                    print(f"   ✅ Final CPCV after stops: Min={final_cpcv_stats['cpcv_min']:.2f}, P5={final_cpcv_stats['cpcv_p5']:.2f}")

                return best_candidate, best_stats
            else:
                if self.verbose: print(f"   ❌ Candidate failed Test Gate. Ret: {c_test['ret']:.4f}, Sort: {c_test['sortino']:.2f} (Parent: {p_test['sortino']:.2f})")
                return None, None

        return None, None

    def evaluate_and_report(self):
        print("\n🔬 Evaluating Variants using CPCV...")

        population = [self.parent_strategy] + self.variants

        # Hydrate Horizon on Strategy Objects
        for s in population:
            s.horizon = self.horizon

        def get_stats(df, s_id):
            row = df[df['id'] == s_id].iloc[0]
            return {'ret': row['total_return'], 'sortino': row['sortino'], 'trades': int(row['trades'])}

        def get_cpcv_stats(df, s_id):
            row = df[df['id'] == s_id].iloc[0]
            return {
                'cpcv_min': row['cpcv_min'],
                'cpcv_p5': row['cpcv_p5_sortino'],
                'cpcv_median': row['cpcv_median']
            }

        # Run CPCV evaluation for robust selection
        cpcv_df = self.backtester.evaluate_combinatorial_purged_cv(population, time_limit=self.horizon)

        # Also get train/val/test for profitability checks and display
        train_df = self.backtester.evaluate_population(population, set_type='train', time_limit=self.horizon)
        val_df = self.backtester.evaluate_population(population, set_type='validation', time_limit=self.horizon)
        test_df = self.backtester.evaluate_population(population, set_type='test', time_limit=self.horizon)

        # Calculate Hamming Distance (Signal Diff) vs Parent
        full_signals = self.backtester.generate_signal_matrix(population, horizon=self.horizon)
        parent_sig = full_signals[:, 0]

        results_data = []
        for i, strat in enumerate(population):
            diff_bits = np.count_nonzero(full_signals[:, i] != parent_sig)

            results_data.append({
                'name': strat.name,
                'strat': strat,
                'diff': diff_bits,
                'train': get_stats(train_df, strat.name),
                'val': get_stats(val_df, strat.name),
                'test': get_stats(test_df, strat.name),
                'cpcv': get_cpcv_stats(cpcv_df, strat.name)
            })

        parent = results_data[0]
        parent_cpcv_min = parent['cpcv']['cpcv_min']

        print(f"\n🏛️  PARENT ({parent['name']}):")
        print(f"   CPCV : Min {parent_cpcv_min:5.2f} | P5 {parent['cpcv']['cpcv_p5']:5.2f}")
        print(f"   TRAIN: Ret {parent['train']['ret']*100:6.2f}% | Sort {parent['train']['sortino']:5.2f} | Tr {parent['train']['trades']}")
        print(f"   VAL  : Ret {parent['val']['ret']*100:6.2f}% | Sort {parent['val']['sortino']:5.2f} | Tr {parent['val']['trades']}")
        print(f"   TEST : Ret {parent['test']['ret']*100:6.2f}% | Sort {parent['test']['sortino']:5.2f} | Tr {parent['test']['trades']}")
        print("-" * 130)

        better_variants = []
        min_improvement = getattr(config, 'OPTIMIZE_MIN_IMPROVEMENT', 0.05)
        improvement_threshold = abs(parent_cpcv_min) * min_improvement if parent_cpcv_min != 0 else 0.1

        for res in results_data[1:]:
            # CRITICAL: Reject any variant below CPCV threshold
            if res['cpcv']['cpcv_min'] < config.MIN_CPCV_THRESHOLD:
                continue

            # Profitability check (train/val)
            train_pos = res['train']['ret'] > 0
            val_pos = res['val']['ret'] > 0

            if not (train_pos and val_pos):
                continue

            # Must beat parent CPCV min by threshold
            v_cpcv_min = res['cpcv']['cpcv_min']
            if v_cpcv_min <= (parent_cpcv_min + improvement_threshold):
                continue

            # Note test performance for information (not selection)
            test_ok = res['test']['sortino'] > 0 and res['test']['ret'] > 0
            res['note'] = "✅ OOS+" if test_ok else "⚠️ OOS-"

            # Apply complexity penalty for sorting (prefer simpler strategies)
            n_genes = len(res['strat'].long_genes) + len(res['strat'].short_genes)
            penalty = n_genes * config.COMPLEXITY_PENALTY_PER_GENE
            res['fitness'] = v_cpcv_min - penalty
            res['n_genes'] = n_genes
            better_variants.append(res)

        better_variants.sort(key=lambda x: x['fitness'], reverse=True)

        print(f"\n🏆 Top {min(10, len(better_variants))} Variants (Sorted by CPCV Min - Complexity Penalty):")
        print(f"{'Name':<30} | {'Genes':<5} | {'CMin':<6} | {'CP5':<6} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Test Ret':<9} | {'Note'}")
        print("-" * 120)
        for v in better_variants[:10]:
            print(f"{v['name']:<30} | {v['n_genes']:<5} | {v['cpcv']['cpcv_min']:6.2f} | {v['cpcv']['cpcv_p5']:6.2f} | {v['train']['sortino']:6.2f} | {v['val']['sortino']:6.2f} | {v['test']['sortino']:6.2f} | {v['test']['ret']*100:8.2f}% | {v['note']}")

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
                    'test_stats': convert_stats(v['test']),
                    'cpcv_min': float(v['cpcv']['cpcv_min']),
                    'cpcv_p5': float(v['cpcv']['cpcv_p5'])
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
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (default: hash of strategy name)")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward validation for more robust selection")

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
    
    optimizer = StrategyOptimizer(args.name, source_file=file_path, horizon=horizon, strategy_dict=target_dict, seed=args.seed)
    optimizer.load_parent()
    optimizer.generate_variants(n_jitter=20, n_mutants=20)

    if args.walk_forward:
        print("\n🔄 Using Walk-Forward Validation for selection...")
        best, stats = optimizer.get_best_variant(use_walk_forward=True)
        if best:
            print(f"\n✅ Best Variant: {best.name}")
            print(f"   Train: {stats['train']['sortino']:.2f} | Val: {stats['val']['sortino']:.2f} | Test: {stats['test']['sortino']:.2f}")

            # Save to file for promote_strategy.py compatibility
            out_file = os.path.join(config.DIRS['STRATEGIES_DIR'], f"optimized_{args.name}.json")
            d = best.to_dict()

            # Convert numpy types to Python native types for JSON serialization
            def convert_stats(s):
                result = {}
                for k, v in s.items():
                    if isinstance(v, (np.floating, np.float32, np.float64)):
                        result[k] = float(v)
                    elif isinstance(v, (np.integer, np.int32, np.int64)):
                        result[k] = int(v)
                    else:
                        result[k] = v
                return result

            d.update({
                'horizon': horizon,
                'train_stats': convert_stats(stats['train']),
                'val_stats': convert_stats(stats['val']),
                'test_stats': convert_stats(stats['test']),
                'cpcv_min': float(stats['cpcv_min']),
                'cpcv_p5': float(stats['cpcv_p5'])
            })
            with open(out_file, 'w') as f:
                json.dump([d], f, indent=4)
            print(f"💾 Saved to {out_file}")
        else:
            print("\n❌ No variant passed all gates.")
    else:
        optimizer.evaluate_and_report()

