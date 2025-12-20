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
    def __init__(self, data, survivors_file, population_size=2000, generations=100, decay_rate=0.99, target_col='log_ret', prediction_mode=False):
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
            
            # 1. Evaluate on TRAIN and VALIDATION simultaneously (Robustness Check)
            train_results = self.backtester.evaluate_population(self.population, set_type='train', prediction_mode=False)
            val_results = self.backtester.evaluate_population(self.population, set_type='validation', prediction_mode=False)
            
            # 2. Filtering & Scoring
            # Calculate penalties on Train (primary driver of complexity)
            train_sortino = train_results['sortino'].values
            val_sortino = val_results['sortino'].values
            trade_counts = train_results['trades'].values
            
            # Penalties
            for i, strat in enumerate(self.population):
                n_genes = len(strat.long_genes) + len(strat.short_genes)
                complexity_penalty = n_genes * 0.02
                
                # Apply penalties to raw scores
                train_sortino[i] -= complexity_penalty
                val_sortino[i] -= complexity_penalty
                
                # Minimum Trade Activity Filter (on Train)
                if trade_counts[i] < 10:
                    train_sortino[i] = -1.0
                    val_sortino[i] = -1.0

            # 3. Robust Fitness Calculation
            # Fitness = min(Train, Val) -> Strategy must work in BOTH regimes to survive
            # This aggressively kills overfitting
            robust_fitness = np.minimum(train_sortino, val_sortino)
            
            # Update Strategy objects for selection
            for i, strat in enumerate(self.population):
                strat.fitness = robust_fitness[i]
                # Store split metrics for debugging/HOF
                strat.train_score = train_sortino[i]
                strat.val_score = val_sortino[i]

            # Statistics
            best_idx = np.argmax(robust_fitness)
            best_fitness = robust_fitness[best_idx]
            best_strat = self.population[best_idx]
            
            avg_train = np.mean(train_sortino)
            avg_val = np.mean(val_sortino)
            gen_gap = avg_train - avg_val
            
            print(f"\n--- Gen {gen} | Best Robust Fitness: {best_fitness:.4f} (Tr:{best_strat.train_score:.2f}/Val:{best_strat.val_score:.2f}) ---")
            print(f"    Avg Gap: {gen_gap:.4f} | Avg Train: {avg_train:.4f} | Avg Val: {avg_val:.4f}")
            
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            if generations_without_improvement >= 10: # Increased patience for robust convergence
                print(f"🛑 Early Stopping Triggered (No Robust Improvement).")
                break
            
            # 4. Selection (Elitism on Robust Fitness)
            num_elite = int(self.pop_size * 0.2)
            # Sort by ROBUST fitness
            sorted_indices = np.argsort(robust_fitness)[::-1]
            elites = [self.population[i] for i in sorted_indices[:num_elite]]
            
            # Update Hall of Fame (Survivors of the Robust Filter)
            # We keep unique strategies that have > 0.1 Robust Score
            for elite in elites:
                if elite.fitness > 0.1:
                    # Check for duplicates in HOF (simple name check or gene hash ideally)
                    self.hall_of_fame.append((elite, elite.fitness))
            
            # 5. Next Gen
            new_population = elites[:50] # Keep top 50 unchanged
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

        # 6. Final Selection
        print("\n--- 🛡️ FINAL SELECTION (TEST PHASE) ---")
        unique_hof = []
        seen = set()
        # sort HOF by Robust Score
        for s, v in sorted(self.hall_of_fame, key=lambda x: x[1], reverse=True):
            s_str = str(s)
            if s_str not in seen:
                unique_hof.append(s)
                seen.add(s_str)
            if len(unique_hof) >= 50: break

        selected_indices = []
        final_strategies = []
        
        if unique_hof:
            # We already know they are robust (min(train, val) > 0.1).
            # Now we check correlation on Validation returns to diversify.
            val_res, val_returns = self.backtester.evaluate_population(unique_hof, set_type='validation', return_series=True, prediction_mode=False)
            
            returns_df = pd.DataFrame(val_returns)
            corr_matrix = returns_df.corr().abs()
            
            # Select top 5 uncorrelated
            # They are already sorted by robustness from HOF extraction
            for i in range(len(unique_hof)):
                if len(selected_indices) >= 5: break
                
                if not selected_indices:
                    selected_indices.append(i)
                else:
                    rho = corr_matrix.loc[i, selected_indices].max()
                    if rho < 0.7: selected_indices.append(i)
            
            final_strategies = [unique_hof[i] for i in selected_indices]
            
            # 7. OOS Reporting (Test Phase)
            print("\n--- 🏆 APEX TRADERS (OOS PERFORMANCE REPORT) ---")
            test_res, _ = self.backtester.evaluate_population(final_strategies, set_type='test', return_series=True, prediction_mode=False)
            
            print("\nFinal Account Performance Stats (OOS):")
            if not test_res.empty:
                print(test_res[['id', 'sortino', 'total_return', 'trades']])
            
            output = []
            for i, idx in enumerate(selected_indices):
                s = final_strategies[i]
                strat_data = s.to_dict()
                strat_data['test_sortino'] = test_res.iloc[i]['sortino'] if not test_res.empty else 0.0
                strat_data['robust_score'] = s.fitness
                output.append(strat_data)
            
            out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")
            with open(out_path, "w") as f: json.dump(output, f, indent=4)
            print(f"\n💾 Saved {len(output)} Apex Strategies to {out_path}")
        else:
            print("No strategies survived robustness filter.")

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