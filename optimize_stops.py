import argparse
import json
import os
import copy
import numpy as np
import pandas as pd
import config
from backtest import BacktestEngine
from backtest.utils import find_strategy_in_files
from genome import Strategy

class StopLossOptimizer:
    def __init__(self, target_name, source_file=None, horizon=180, strategy_dict=None, backtester=None, data=None, verbose=True):
        self.target_name = target_name
        self.source_file = source_file
        self.horizon = horizon
        self.strategy_dict = strategy_dict
        self.parent_strategy = None
        self.variants = []
        self.verbose = verbose
        
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

    def generate_grid(self, sl_range, tp_range):
        """
        Generates variants for a grid of SL and TP values.
        sl_range: (start, stop, step)
        tp_range: (start, stop, step)
        """
        if self.verbose: print("\n🕸️  Generating Grid Variants...")
        
        if not self.parent_strategy:
            self.load_parent()

        self.variants = []
        
        sl_values = np.arange(sl_range[0], sl_range[1] + sl_range[2], sl_range[2])
        tp_values = np.arange(tp_range[0], tp_range[1] + tp_range[2], tp_range[2])
        
        count = 0
        for sl in sl_values:
            for tp in tp_values:
                # Basic sanity: TP should generally be > SL (or at least close)
                # But we'll allow all combinations to see what happens, maybe scalping works with TP < SL (though risky)
                
                # Create variant
                variant = copy.deepcopy(self.parent_strategy)
                # Round to 2 decimal places to avoid float ugliness
                sl_r = round(sl, 2)
                tp_r = round(tp, 2)
                
                variant.name = f"{self.target_name}_SL{sl_r}_TP{tp_r}"
                variant.stop_loss_pct = sl_r
                variant.take_profit_pct = tp_r
                variant.horizon = self.horizon # Ensure horizon is set
                
                self.variants.append(variant)
                count += 1
                
        if self.verbose: print(f"   Generated {count} variants.")

    def evaluate_and_report(self):
        if not self.variants:
            print("No variants to evaluate.")
            return

        print(f"\n🔬 Evaluating {len(self.variants)} Variants on All Sets (Train / Val / Test)...")
        
        population = [self.parent_strategy] + self.variants
        
        # Hydrate Horizon on Strategy Objects (redundant safety)
        for s in population:
            s.horizon = self.horizon
            
        # Use Engine's Built-in Evaluation
        # We can do this in one go.
        train_df = self.backtester.evaluate_population(population, set_type='train', time_limit=self.horizon)
        val_df = self.backtester.evaluate_population(population, set_type='validation', time_limit=self.horizon)
        test_df = self.backtester.evaluate_population(population, set_type='test', time_limit=self.horizon)
        
        # Consolidate Results
        results_data = []
        
        def get_stats(df, s_id):
            row = df[df['id'] == s_id].iloc[0]
            return {
                'ret': row['total_return'], 
                'sortino': row['sortino'], 
                'trades': int(row['trades'])
            }

        parent_stats = {
            'train': get_stats(train_df, self.parent_strategy.name),
            'val': get_stats(val_df, self.parent_strategy.name),
            'test': get_stats(test_df, self.parent_strategy.name)
        }
        
        parent_avg_sort = (parent_stats['train']['sortino'] + parent_stats['val']['sortino'] + parent_stats['test']['sortino']) / 3.0
        
        print(f"\n🏛️  PARENT ({self.parent_strategy.name}) [SL:{self.parent_strategy.stop_loss_pct} TP:{self.parent_strategy.take_profit_pct}]:")
        print(f"   AVG  : Sort {parent_avg_sort:5.2f}")
        print(f"   TRAIN: Ret {parent_stats['train']['ret']*100:6.2f}% | Sort {parent_stats['train']['sortino']:5.2f} | Tr {parent_stats['train']['trades']}")
        print(f"   VAL  : Ret {parent_stats['val']['ret']*100:6.2f}% | Sort {parent_stats['val']['sortino']:5.2f} | Tr {parent_stats['val']['trades']}")
        print(f"   TEST : Ret {parent_stats['test']['ret']*100:6.2f}% | Sort {parent_stats['test']['sortino']:5.2f} | Tr {parent_stats['test']['trades']}")
        print("-" * 120)

        better_variants = []
        
        for i, variant in enumerate(self.variants):
            v_train = get_stats(train_df, variant.name)
            v_val = get_stats(val_df, variant.name)
            v_test = get_stats(test_df, variant.name)
            
            # Filtering Logic
            # 1. Must be profitable in Train/Val (Sanity)
            if v_train['ret'] <= 0 or v_val['ret'] <= 0:
                continue
                
            # 2. Comparison to Parent
            # We want to see significant improvements or at least similar performance with better risk profile
            # Let's be permissive and rank them later.
            
            res = {
                'name': variant.name,
                'sl': variant.stop_loss_pct,
                'tp': variant.take_profit_pct,
                'strat': variant,
                'train': v_train,
                'val': v_val,
                'test': v_test,
                'avg_sort': (v_train['sortino'] + v_val['sortino'] + v_test['sortino']) / 3.0,
                'note': ""
            }
            
            # Check for robustness
            is_robust = (v_train['sortino'] > 1.0 and v_val['sortino'] > 1.0 and v_test['sortino'] > 0)
            
            if is_robust:
                better_variants.append(res)

        # Sort by Average Sortino
        better_variants.sort(key=lambda x: x['avg_sort'], reverse=True)
        
        print(f"\n🏆 Top {min(20, len(better_variants))} Variants (Sorted by Avg Sortino):")
        print(f"{ 'SL':<5} | { 'TP':<5} | { 'Avg Sort':<10} | { 'Trn Sort':<9} | { 'Val Sort':<9} | { 'Test Sort':<9} | { 'Test Ret':<9} | {'Trds':<4} | {'Name'}")
        print("-" * 130)
        
        for v in better_variants[:20]:
            is_best = (v['avg_sort'] > parent_avg_sort)
            marker = "⭐" if is_best else ""
            print(f"{v['sl']:<5.2f} | {v['tp']:<5.2f} | {v['avg_sort']:10.2f} | {v['train']['sortino']:9.2f} | {v['val']['sortino']:9.2f} | {v['test']['sortino']:9.2f} | {v['test']['ret']*100:8.2f}% | {v['test']['trades']:<4} {marker} | {v['name']}")

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
                    'test_stats': convert_stats(v['test'])
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
        strategy_dict=target_dict
    )
    
    optimizer.load_parent()
    optimizer.generate_grid(
        sl_range=(args.sl_start, args.sl_end, args.sl_step),
        tp_range=(args.tp_start, args.tp_end, args.tp_step)
    )
    optimizer.evaluate_and_report()
