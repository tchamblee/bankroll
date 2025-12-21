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
        # print(f"🧬 Spawning {self.pop_size} lot-based strategies (1-3 lots)...")
        # strategies will now have 1-2 genes to allow up/downsizing
        self.population = [self.factory.create_strategy(num_genes_range=(1, 2)) for _ in range(self.pop_size)]

    def crossover(self, p1, p2):
        child = Strategy(name=f"Child_{random.randint(1000,9999)}")
        
        # Pick random number of genes from parents
        n_long = random.randint(1, 2)
        n_short = random.randint(1, 2)
        
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
            
            # 1. Evaluate using Rolling Walk-Forward Validation (4 Folds)
            wfv_results = self.backtester.evaluate_walk_forward(self.population, folds=4)
            
            # 2. Filtering & Scoring
            wfv_scores = wfv_results['sortino'].values # Already penalized for fold variance
            
            # Count feature usage
            feature_counts = {}
            for strat in self.population:
                used_features = set()
                for gene in strat.long_genes + strat.short_genes:
                    if hasattr(gene, 'feature'):
                        used_features.add(gene.feature)
                    elif hasattr(gene, 'feature_left'):
                        used_features.add(gene.feature_left)
                        used_features.add(gene.feature_right)
                    elif hasattr(gene, 'mode'): # TimeGene
                        used_features.add(f"time_{gene.mode}")
                    elif hasattr(gene, 'direction'): # ConsecutiveGene
                        used_features.add(f"consecutive_{gene.direction}")
                
                for f in used_features:
                    feature_counts[f] = feature_counts.get(f, 0) + 1
            
            # Identify dominant features (>15% of population)
            threshold = self.pop_size * 0.15
            dominant_features = {f for f, count in feature_counts.items() if count > threshold}
            
            # Log dominant features
            if dominant_features:
                sorted_dom = sorted([(f, feature_counts[f]) for f in dominant_features], key=lambda x: x[1], reverse=True)[:3]
                # print(f"  ⚠️  Dominant Features: {', '.join([f'{f}({c})' for f, c in sorted_dom])}")
            
            # Penalties
            for i, strat in enumerate(self.population):
                n_genes = len(strat.long_genes) + len(strat.short_genes)
                
                # Complexity Penalty (Aggressive: 0.5 per gene)
                # This strongly favors 1-gene strategies over 2-gene strategies.
                complexity_penalty = n_genes * 0.5
                
                # Apply Dynamic Dominance Penalty
                dom_penalty = 0.0
                strat_features = set()
                
                for gene in strat.long_genes + strat.short_genes:
                    if hasattr(gene, 'feature'):
                        strat_features.add(gene.feature)
                    elif hasattr(gene, 'feature_left'):
                        strat_features.add(gene.feature_left)
                        strat_features.add(gene.feature_right)
                    elif hasattr(gene, 'mode'):
                        strat_features.add(f"time_{gene.mode}")
                    elif hasattr(gene, 'direction'):
                        strat_features.add(f"consecutive_{gene.direction}")

                for f in strat_features:
                    # Penalty scales with popularity: Aggressive Diversification
                    # >10% usage triggers massive penalty (usage_ratio * 10.0)
                    # e.g. 20% usage -> 2.0 penalty (kills Sortino)
                    usage_ratio = feature_counts.get(f, 0) / self.pop_size
                    if usage_ratio > 0.10:
                         dom_penalty += usage_ratio * 10.0
                
                total_penalty = complexity_penalty + dom_penalty
                wfv_scores[i] -= total_penalty
                
                strat.fitness = wfv_scores[i]

            # Statistics
            best_idx = np.argmax(wfv_scores)
            best_fitness = wfv_scores[best_idx]
            best_strat = self.population[best_idx]
            
            # Extract detailed WFV stats for best strat
            best_stats = wfv_results.iloc[best_idx]
            
            if gen % 5 == 0:
                print(f"Gen {gen} | Best WFV Score: {best_fitness:.4f} (Avg:{best_stats['avg_sortino']:.2f})")
            
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
            sorted_indices = np.argsort(wfv_scores)[::-1]
            elites = [self.population[i] for i in sorted_indices[:num_elite]]
            
            # Update Hall of Fame (Survivors of the Robust Filter)
            # We keep unique strategies that have > 0.1 Robust Score
            for elite in elites:
                if elite.fitness > 0.1:
                    # Tag with generation found
                    if not hasattr(elite, 'generation_found'):
                        elite.generation_found = gen
                        
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
            # print(f"  Gen completed in {time.time()-start_time:.2f}s")

        # 6. Final Selection
        # print(f"\n--- 🛡️ FINAL SELECTION (TEST PHASE: All Generations) ---")
        unique_hof = []
        seen = set()
        
        # Sort HOF by Robust Score (Validation) initially
        for s, v in sorted(self.hall_of_fame, key=lambda x: x[1], reverse=True):
            s_str = str(s)
            if s_str not in seen:
                unique_hof.append(s)
                seen.add(s_str)
        
        # print(f"Testing {len(unique_hof)} unique robust strategies collected across all generations...")

        if unique_hof:
            # 7. OOS Reporting (Test Phase) on EVERYTHING
            # print("\n--- 🏆 APEX TRADERS (OOS PERFORMANCE REPORT) ---")
            
            # Evaluate entire history on Test Set
            test_res, _ = self.backtester.evaluate_population(unique_hof, set_type='test', return_series=True, prediction_mode=False)
            
            # Filter for OOS Profitability
            profitable_indices = test_res[test_res['sortino'] > 0.0].index.tolist()
            
            output = []
            
            # If no profitable strategies, save top 5 by Sortino anyway for debugging
            if not profitable_indices:
                print("No profitable OOS strategies found. Saving top 10 losers for analysis.")
                profitable_indices = test_res.sort_values('sortino', ascending=False).head(10).index.tolist()
            
            # Collect results
            for i in profitable_indices:
                s = unique_hof[i]
                strat_data = s.to_dict()
                
                # Add Metadata
                strat_data['generation'] = getattr(s, 'generation_found', -1)
                strat_data['test_sortino'] = float(test_res.iloc[i]['sortino'])
                strat_data['test_return'] = float(test_res.iloc[i]['total_return'])
                strat_data['test_trades'] = int(test_res.iloc[i]['trades'])
                strat_data['robust_score'] = float(s.fitness)
                
                output.append(strat_data)
            
            # Sort by TEST Sortino (Real Performance)
            output.sort(key=lambda x: x['test_sortino'], reverse=True)
            
            # Limit to Top 100 as requested
            output = output[:100]
            
            # Print Top 5
            print("\nTop 5 OOS Performers:")
            for s in output[:5]:
                print(f"  Gen {s['generation']:<2} | Sortino: {s['test_sortino']:.2f} | Ret: {s['test_return']*100:.2f}% | ID: {s['name']}")
            
            out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")
            with open(out_path, "w") as f: json.dump(output, f, indent=4)
            print(f"\n💾 Saved Top {len(output)} Profitable Strategies from All Generations to {out_path}")
        else:
            print("No strategies survived robustness filter.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--pop_size", type=int, default=3000)
    parser.add_argument("--gens", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        exit(1)
        
    bars_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # print(f"\n🚀 Starting Real-Money Evolution ($30k Account, 1-3 Lots)")
    
    factory = EvolutionaryAlphaFactory(
        bars_df, 
        args.survivors, 
        population_size=args.pop_size, 
        generations=args.gens,
        target_col='log_ret',
        prediction_mode=False
    )
    factory.evolve(horizon=args.horizon)