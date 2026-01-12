import argparse
import json
import os
import copy
import random
import numpy as np
import pandas as pd
import config
from backtest import BacktestEngine
from backtest.utils import find_strategy_in_files
from genome import Strategy

class StopLossOptimizer:
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
            
        # Ensure parent has the correct horizon set
        self.parent_strategy.horizon = self.horizon

        if self.verbose:
            print(f"✅ Loaded Parent: {self.parent_strategy.name}")
            print(f"   Current SL: {self.parent_strategy.stop_loss_pct} | TP: {self.parent_strategy.take_profit_pct}")

    def generate_grid(self, sl_range, tp_range, min_rr_ratio=None, directional=False):
        """
        Generates variants for a grid of SL and TP values.
        sl_range: (start, stop, step)
        tp_range: (start, stop, step)
        min_rr_ratio: Minimum TP/SL ratio (e.g., 1.0 means TP >= SL). None to allow all.
        directional: If True, set directional stops (sl_long, sl_short, tp_long, tp_short)
                     instead of symmetric stops.
        """
        if self.verbose:
            mode = "Directional" if directional else "Symmetric"
            print(f"\n🕸️  Generating {mode} Grid Variants...")

        if not self.parent_strategy:
            self.load_parent()

        self.variants = []

        sl_values = np.arange(sl_range[0], sl_range[1] + sl_range[2], sl_range[2])
        tp_values = np.arange(tp_range[0], tp_range[1] + tp_range[2], tp_range[2])

        total_possible = len(sl_values) * len(tp_values)
        if self.verbose and total_possible > 500:
            print(f"   ⚠️  Large grid: {total_possible} combinations. This may take a while...")

        count = 0
        skipped = 0
        for sl in sl_values:
            for tp in tp_values:
                # Filter by R:R ratio if specified
                if min_rr_ratio is not None and sl > 0:
                    rr_ratio = tp / sl
                    if rr_ratio < min_rr_ratio:
                        skipped += 1
                        continue

                # Create variant
                variant = copy.deepcopy(self.parent_strategy)
                # Round to 2 decimal places to avoid float ugliness
                sl_r = round(sl, 2)
                tp_r = round(tp, 2)

                variant.name = f"{self.target_name}_SL{sl_r}_TP{tp_r}"
                variant.horizon = self.horizon

                if directional:
                    # Set directional stops (allows asymmetric long/short later)
                    variant.sl_long = sl_r
                    variant.sl_short = sl_r
                    variant.tp_long = tp_r
                    variant.tp_short = tp_r
                    # Keep symmetric as fallback
                    variant.stop_loss_pct = sl_r
                    variant.take_profit_pct = tp_r
                else:
                    # Standard symmetric stops
                    variant.stop_loss_pct = sl_r
                    variant.take_profit_pct = tp_r

                self.variants.append(variant)
                count += 1

        if self.verbose:
            msg = f"   Generated {count} variants."
            if skipped > 0:
                msg += f" (Skipped {skipped} with R:R < {min_rr_ratio})"
            print(msg)

    def get_best_variant(self):
        """
        Programmatically find and return the best stop variant.
        CRITICAL: Selection is based on CPCV (Combinatorial Purged Cross-Validation) metrics.
        Returns (best_strategy, stats_dict) or (None, None) if no improvement found.
        """
        if not self.variants:
            if self.verbose:
                print("No variants to evaluate.")
            return None, None

        if not self.parent_strategy:
            self.load_parent()

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

        # Run CPCV evaluation for robust selection
        if self.verbose:
            print("   Running CPCV evaluation for robust stop optimization...")
        cpcv_df = self.backtester.evaluate_combinatorial_purged_cv(population, time_limit=self.horizon)

        # Also get train/val/test for profitability checks
        train_df = self.backtester.evaluate_population(population, set_type='train', time_limit=self.horizon)
        val_df = self.backtester.evaluate_population(population, set_type='validation', time_limit=self.horizon)
        test_df = self.backtester.evaluate_population(population, set_type='test', time_limit=self.horizon)

        # Parent baseline using CPCV min
        parent_cpcv = get_cpcv_stats(cpcv_df, self.parent_strategy.name)
        parent_fitness = parent_cpcv['cpcv_min']

        if self.verbose:
            print(f"   Parent CPCV: Min={parent_cpcv['cpcv_min']:.2f}, P5={parent_cpcv['cpcv_p5']:.2f}")

        # Get configurable thresholds
        min_improvement = getattr(config, 'OPTIMIZE_MIN_IMPROVEMENT', 0.05)
        improvement_threshold = abs(parent_fitness) * min_improvement if parent_fitness != 0 else 0.1

        best_candidate = None
        best_fitness = parent_fitness

        for variant in self.variants:
            if variant.name not in cpcv_df['id'].values:
                continue
            if variant.name not in train_df['id'].values:
                continue

            v_cpcv = get_cpcv_stats(cpcv_df, variant.name)
            v_train = get_stats(train_df, variant.name)
            v_val = get_stats(val_df, variant.name)

            # CRITICAL: Reject any variant with negative CPCV min
            if v_cpcv['cpcv_min'] < 0:
                continue

            # Must be profitable on train/val
            if v_train['ret'] <= 0 or v_val['ret'] <= 0:
                continue

            v_fitness = v_cpcv['cpcv_min']

            # Must beat parent by threshold
            if v_fitness > (parent_fitness + improvement_threshold) and v_fitness > best_fitness:
                best_fitness = v_fitness
                best_candidate = variant

        if best_candidate:
            best_cpcv = get_cpcv_stats(cpcv_df, best_candidate.name)
            # Get final stats including test
            best_stats = {
                'train': get_stats(train_df, best_candidate.name),
                'val': get_stats(val_df, best_candidate.name),
                'test': get_stats(test_df, best_candidate.name),
                'cpcv_min': best_cpcv['cpcv_min'],
                'cpcv_p5': best_cpcv['cpcv_p5']
            }
            if self.verbose:
                print(f"   🏆 Best: {best_candidate.name}")
                print(f"      CPCV: Min={best_cpcv['cpcv_min']:.2f}, P5={best_cpcv['cpcv_p5']:.2f} (Parent Min: {parent_fitness:.2f})")
            return best_candidate, best_stats

        if self.verbose:
            print("   No variant beat parent by required threshold.")
        return None, None

    def evaluate_and_report(self):
        if not self.variants:
            print("No variants to evaluate.")
            return

        print(f"\n🔬 Evaluating {len(self.variants)} Variants using CPCV...")

        population = [self.parent_strategy] + self.variants

        # Hydrate Horizon on Strategy Objects (redundant safety)
        for s in population:
            s.horizon = self.horizon

        def get_stats(df, s_id):
            row = df[df['id'] == s_id].iloc[0]
            return {
                'ret': row['total_return'],
                'sortino': row['sortino'],
                'trades': int(row['trades'])
            }

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

        parent_stats = {
            'train': get_stats(train_df, self.parent_strategy.name),
            'val': get_stats(val_df, self.parent_strategy.name),
            'test': get_stats(test_df, self.parent_strategy.name)
        }
        parent_cpcv = get_cpcv_stats(cpcv_df, self.parent_strategy.name)

        # Parent baseline uses CPCV min (robust metric)
        parent_cpcv_min = parent_cpcv['cpcv_min']

        print(f"\n🏛️  PARENT ({self.parent_strategy.name}) [SL:{self.parent_strategy.stop_loss_pct} TP:{self.parent_strategy.take_profit_pct}]:")
        print(f"   CPCV : Min {parent_cpcv_min:5.2f} | P5 {parent_cpcv['cpcv_p5']:5.2f}")
        print(f"   TRAIN: Ret {parent_stats['train']['ret']*100:6.2f}% | Sort {parent_stats['train']['sortino']:5.2f} | Tr {parent_stats['train']['trades']}")
        print(f"   VAL  : Ret {parent_stats['val']['ret']*100:6.2f}% | Sort {parent_stats['val']['sortino']:5.2f} | Tr {parent_stats['val']['trades']}")
        print(f"   TEST : Ret {parent_stats['test']['ret']*100:6.2f}% | Sort {parent_stats['test']['sortino']:5.2f} | Tr {parent_stats['test']['trades']}")
        print("-" * 130)

        better_variants = []

        # Get configurable thresholds
        min_improvement = getattr(config, 'OPTIMIZE_MIN_IMPROVEMENT', 0.05)
        improvement_threshold = abs(parent_cpcv_min) * min_improvement if parent_cpcv_min != 0 else 0.1

        for i, variant in enumerate(self.variants):
            if variant.name not in cpcv_df['id'].values:
                continue

            v_cpcv = get_cpcv_stats(cpcv_df, variant.name)
            v_train = get_stats(train_df, variant.name)
            v_val = get_stats(val_df, variant.name)
            v_test = get_stats(test_df, variant.name)

            # CRITICAL: Reject any variant with negative CPCV min
            if v_cpcv['cpcv_min'] < 0:
                continue

            # Must be profitable in Train/Val
            if v_train['ret'] <= 0 or v_val['ret'] <= 0:
                continue

            # Selection metric: CPCV min (robust across all folds)
            v_cpcv_min = v_cpcv['cpcv_min']

            # Must beat parent by minimum improvement threshold (avoid noise)
            if v_cpcv_min <= (parent_cpcv_min + improvement_threshold):
                continue

            # Note test performance for informational display
            test_ok = v_test['sortino'] > 0 and v_test['ret'] > 0
            note = "✅ OOS+" if test_ok else "⚠️ OOS-"

            res = {
                'name': variant.name,
                'sl': variant.stop_loss_pct,
                'tp': variant.take_profit_pct,
                'strat': variant,
                'train': v_train,
                'val': v_val,
                'test': v_test,
                'cpcv_min': v_cpcv_min,
                'cpcv_p5': v_cpcv['cpcv_p5'],
                'note': note
            }

            better_variants.append(res)

        # Sort by CPCV Min (robust metric)
        better_variants.sort(key=lambda x: x['cpcv_min'], reverse=True)

        print(f"\n🏆 Top {min(20, len(better_variants))} Variants (Sorted by CPCV Min):")
        print(f"{'SL':<5} | {'TP':<5} | {'CMin':<6} | {'CP5':<6} | {'Train':<6} | {'Val':<6} | {'Test':<6} | {'Test Ret':<9} | {'OOS':<6} | {'Name'}")
        print("-" * 110)

        for v in better_variants[:20]:
            is_best = (v['cpcv_min'] > parent_cpcv_min)
            marker = "⭐" if is_best else "  "
            print(f"{v['sl']:<5.2f} | {v['tp']:<5.2f} | {v['cpcv_min']:6.2f} | {v['cpcv_p5']:6.2f} | {v['train']['sortino']:6.2f} | {v['val']['sortino']:6.2f} | {v['test']['sortino']:6.2f} | {v['test']['ret']*100:8.2f}% | {v['note']:<6} {marker}| {v['name']}")

        # Save Best
        if better_variants:
            best = better_variants[0]
            out_file = os.path.join(config.DIRS['STRATEGIES_DIR'], f"optimized_stops_{self.target_name}.json")

            def convert_stats(stats_dict):
                return {
                    'ret': float(stats_dict['ret']),
                    'sortino': float(stats_dict['sortino']),
                    'trades': int(stats_dict['trades'])
                }

            # We can save a few top ones
            to_save = []
            for v in better_variants[:5]:
                d = v['strat'].to_dict()
                d.update({
                    'horizon': self.horizon,
                    'train_stats': convert_stats(v['train']),
                    'val_stats': convert_stats(v['val']),
                    'test_stats': convert_stats(v['test']),
                    'cpcv_min': float(v['cpcv_min']),
                    'cpcv_p5': float(v['cpcv_p5'])
                })
                to_save.append(d)

            with open(out_file, 'w') as f:
                json.dump(to_save, f, indent=4)
            print(f"\n💾 Saved top 5 variants to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Stop Loss and Take Profit for a strategy")
    parser.add_argument("name", type=str, help="Name of the strategy")
    parser.add_argument("--file", type=str, default=None, help="Path to strategy file (Optional)")
    parser.add_argument("--horizon", type=int, default=None, help="Time horizon (Optional)")
    
    # Grid definition
    parser.add_argument("--sl-start", type=float, default=0.5, help="Start of SL range (ATR multiple)")
    parser.add_argument("--sl-end", type=float, default=5.0, help="End of SL range")
    parser.add_argument("--sl-step", type=float, default=0.25, help="Step size for SL")
    
    parser.add_argument("--tp-start", type=float, default=1.0, help="Start of TP range")
    parser.add_argument("--tp-end", type=float, default=10.0, help="End of TP range")
    parser.add_argument("--tp-step", type=float, default=0.5, help="Step size for TP")

    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--min-rr", type=float, default=None, help="Minimum R:R ratio (TP/SL). E.g., 1.5 means TP must be >= 1.5x SL")
    parser.add_argument("--directional", action="store_true", help="Set directional stops (sl_long, sl_short, tp_long, tp_short)")

    args = parser.parse_args()
    
    target_dict = None
    file_path = args.file
    horizon = args.horizon
    
    # Auto-discovery
    if not file_path:
        print(f"🔍 Searching for strategy '{args.name}'...")
        found_dict = find_strategy_in_files(args.name)
        if found_dict:
            target_dict = found_dict
            if not horizon:
                horizon = found_dict.get('horizon', 60)
                print(f"   ✅ Inferred Horizon: {horizon}")
        else:
            print(f"❌ Error: Could not locate strategy '{args.name}' in standard files.")
            exit(1)
            
    if not horizon:
        print("❌ Error: Horizon required.")
        exit(1)
        
    optimizer = StopLossOptimizer(
        args.name,
        source_file=file_path,
        horizon=horizon,
        strategy_dict=target_dict,
        seed=args.seed
    )
    
    optimizer.load_parent()
    optimizer.generate_grid(
        sl_range=(args.sl_start, args.sl_end, args.sl_step),
        tp_range=(args.tp_start, args.tp_end, args.tp_step),
        min_rr_ratio=args.min_rr,
        directional=args.directional
    )
    optimizer.evaluate_and_report()
