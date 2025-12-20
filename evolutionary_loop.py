import pandas as pd
import numpy as np
import os
import json
import random
import time
import config
from feature_engine import FeatureEngine
from strategy_genome import GenomeFactory, Strategy
from backtest_engine import BacktestEngine
from validate_features import triple_barrier_labels

class EvolutionaryAlphaFactory:
    def __init__(self, data, survivors_file, population_size=20000, generations=100, decay_rate=0.99, target_col='log_ret', prediction_mode=False):
        self.data = data
        self.pop_size = population_size
        self.generations = generations
        self.decay_rate = decay_rate
        self.survivors_file = survivors_file
        self.prediction_mode = prediction_mode
        
        self.factory = GenomeFactory(survivors_file)
        self.factory.set_stats(data)
        
        # Aligned with refactored BacktestEngine
        self.backtester = BacktestEngine(
            data, 
            cost_bps=0.5, 
            target_col=target_col,
            annualization_factor=181440,
            account_size=config.ACCOUNT_SIZE
        )
        
        self.population = []
        self.hall_of_fame = []

    def initialize_population(self):
        print(f"🧬 Spawning {self.pop_size} lot-based strategies (1-3 lots)...")
        # strategies will now have 3-5 genes to allow up/downsizing
        self.population = [self.factory.create_strategy(num_genes_range=(3, 5)) for _ in range(self.pop_size)]

    def crossover(self, p1, p2):
        child = Strategy(name=f"Child_{random.randint(1000,9999)}")
        
        # Pick random number of genes from parents
        n_long = random.randint(3, 5)
        n_short = random.randint(3, 5)
        
        combined_long = p1.long_genes + p2.long_genes
        combined_short = p1.short_genes + p2.short_genes
        
        child.long_genes = [g.copy() for g in random.sample(combined_long, min(len(combined_long), n_long))]
        child.short_genes = [g.copy() for g in random.sample(combined_short, min(len(combined_short), n_short))]
        
        return child

    def evolve(self, horizon=60):
        # Reproducibility
        random.seed(42)
        np.random.seed(42)
        
        self.initialize_population()
        
        generations_without_improvement = 0
        global_best_fitness = -999.0
        
        for gen in range(self.generations):
            start_time = time.time()
            
            # 1. Evaluate on TRAIN (Using TRADING mode)
            results = self.backtester.evaluate_population(self.population, set_type='train', prediction_mode=False)
            
            # 2. Filtering & Scoring
            # Need at least 10 lot changes to be significant
            results.loc[results['trades'] < 10, 'sortino'] = -1.0
            
            raw_best = results['sortino'].max()
            decay_factor = self.decay_rate ** gen
            results['sortino'] *= decay_factor
            
            best_fitness = results['sortino'].max()
            print(f"\n--- Generation {gen} | Best [Train] Sortino: {best_fitness:.4f} (Raw: {raw_best:.4f}) ---")
            
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            if generations_without_improvement >= 7:
                print(f"🛑 Early Stopping Triggered.")
                break
            
            # 3. Selection
            num_elite = int(self.pop_size * 0.2)
            elites_ids = results.sort_values('sortino', ascending=False).head(num_elite).index.tolist()
            elites = [self.population[idx] for idx in elites_ids]
            
            # 4. Cross-Validation
            val_results = self.backtester.evaluate_population(elites, set_type='validation', prediction_mode=False)
            for i, elite in enumerate(elites):
                val_score = val_results.iloc[i]['sortino']
                if val_score > 0.0:
                    self.hall_of_fame.append((elite, val_score))
            
            # 5. Next Gen
            new_population = elites[:50]
            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
                
                # Mutation
                mut_rate = 0.15
                for g in child.long_genes + child.short_genes:
                    if random.random() < mut_rate:
                        g.mutate(self.factory.features)
                
                new_population.append(child)
                
            self.population = new_population
            print(f"  Gen completed in {time.time()-start_time:.2f}s")

        # 6. OOS Test
        print("\n--- 🏆 APEX TRADERS (OOS TEST) ---")
        unique_hof = []
        seen = set()
        for s, v in sorted(self.hall_of_fame, key=lambda x: x[1], reverse=True):
            if str(s) not in seen:
                unique_hof.append(s)
                seen.add(str(s))
            if len(unique_hof) >= 50: break
            
        if unique_hof:
            test_res, net_returns = self.backtester.evaluate_population(unique_hof, set_type='test', return_series=True, prediction_mode=False)
            
            sorted_indices = test_res.sort_values('sortino', ascending=False).index.tolist()
            returns_df = pd.DataFrame(net_returns)
            corr_matrix = returns_df.corr().abs()
            
            selected_indices = []
            for idx in sorted_indices:
                if len(selected_indices) >= 5: break
                # current_score = test_res.loc[idx, 'sortino'] 
                # We allow negative scores now to avoid Empty DataFrame, 
                # so users can see the "best of the worst" if all fail.
                
                if not selected_indices:
                    selected_indices.append(idx)
                else:
                    rho = corr_matrix.loc[idx, selected_indices].max()
                    if rho < 0.7: selected_indices.append(idx)
            
            final_res = test_res.iloc[selected_indices]
            
            if not final_res.empty and final_res['sortino'].max() <= 0:
                print("\n⚠️  WARNING: All selected strategies have negative OOS performance.")

            print("\nFinal Account Performance Stats (OOS):")
            print(final_res[['id', 'sortino', 'total_return', 'trades']])
            
            output = []
            for idx in selected_indices:
                s = unique_hof[idx]
                # Serialize Strategy to dict
                strat_data = s.to_dict()
                # Append performance metrics
                strat_data['test_sortino'] = test_res.loc[idx, 'sortino']
                output.append(strat_data)
            
            out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")
            with open(out_path, "w") as f: json.dump(output, f, indent=4)
            print(f"\n💾 Saved {len(output)} Apex Strategies to {out_path}")
        else:
            print("No strategies survived validation.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--pop_size", type=int, default=2000)
    parser.add_argument("--gens", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        exit(1)
        
    bars_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    print(f"\n🚀 Starting Real-Money Evolution ($30k Account, 1-3 Lots)")
    
    factory = EvolutionaryAlphaFactory(
        bars_df, 
        args.survivors, 
        population_size=args.pop_size, 
        generations=args.gens,
        target_col='log_ret',
        prediction_mode=False
    )
    factory.evolve(horizon=args.horizon)