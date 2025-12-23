import pandas as pd
import numpy as np
import os
import shutil
import json
import random
import time
import config
from feature_engine import FeatureEngine
from genome import GenomeFactory, Strategy
from backtest import BacktestEngine
from validate_features import triple_barrier_labels

import uuid

class EvolutionaryAlphaFactory:
    def __init__(self, data, survivors_file, population_size=None, generations=100, decay_rate=0.99, target_col='log_ret', prediction_mode=False):
        self.training_id = uuid.uuid4().hex[:8]  # Unique ID for this run
        print(f"🆔 Training Run ID: {self.training_id}")
        
        self.data = data
        self.pop_size = population_size if population_size else config.EVO_BATCH_SIZE
        self.generations = generations
        self.decay_rate = decay_rate
        self.survivors_file = survivors_file
        self.prediction_mode = prediction_mode
        
        # Aligned with refactored BacktestEngine
        self.backtester = BacktestEngine(
            data, 
            cost_bps=config.COST_BPS, 
            target_col=target_col,
            annualization_factor=config.ANNUALIZATION_FACTOR,
            account_size=config.ACCOUNT_SIZE
        )
        
        self.factory = GenomeFactory(survivors_file)
        # FIX: Use only Training Data for Genome Stats (avoid lookahead in mutation)
        train_data = data.iloc[:self.backtester.train_idx]
        self.factory.set_stats(train_data)
        
        self.population = []
        self.hall_of_fame = []

    def initialize_population(self):
        # print(f"🧬 Spawning {self.pop_size} lot-based strategies (1-3 lots)...")
        # strategies will now have 3-4 genes to allow more complex logic
        self.population = [self.factory.create_strategy((2, 4)) for _ in range(self.pop_size)]

    def crossover(self, p1, p2):
        child = Strategy(name=f"Child_{random.randint(1000,9999)}")
        
        # Pick random number of genes from parents
        n_long = random.randint(1, 4)
        n_short = random.randint(1, 4)
        
        combined_long = p1.long_genes + p2.long_genes
        combined_short = p1.short_genes + p2.short_genes
        combined_regime = p1.regime_genes + p2.regime_genes
        
        child.long_genes = [g.copy() for g in random.sample(combined_long, min(len(combined_long), n_long))]
        child.short_genes = [g.copy() for g in random.sample(combined_short, min(len(combined_short), n_short))]
        
        # Inherit Regime Genes (0-2 max)
        if combined_regime:
            n_regime = random.randint(0, min(len(combined_regime), 2))
            child.regime_genes = [g.copy() for g in random.sample(combined_regime, n_regime)]
        
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
            wfv_results = self.backtester.evaluate_walk_forward(self.population, folds=4, time_limit=horizon)
            
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
            # dominant_features = {f for f, count in feature_counts.items() if count > threshold}
            
            # # Log dominant features
            # if dominant_features:
            #     sorted_dom = sorted([(f, feature_counts[f]) for f in dominant_features], key=lambda x: x[1], reverse=True)[:3]
            #     print(f"Dominant Features: {', '.join([f'{f}({c})' for f, c in sorted_dom])}")
            
            # Penalties
            for i, strat in enumerate(self.population):
                n_genes = len(strat.long_genes) + len(strat.short_genes)
                
                # Complexity Penalty
                # This allows for slightly more complex strategies while still preventing bloat.
                complexity_penalty = n_genes * config.COMPLEXITY_PENALTY_PER_GENE
                
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

                # for f in strat_features:
                #     # Penalty scales with popularity: Ruthless Diversification
                #     # >20% usage triggers penalty (usage_ratio * 10.0)
                #     usage_ratio = feature_counts.get(f, 0) / self.pop_size
                #     if usage_ratio > config.DOMINANCE_PENALTY_THRESHOLD:
                #          dom_penalty += usage_ratio * config.DOMINANCE_PENALTY_MULTIPLIER
                
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
                print(f"Gen {gen} | Best WFV Score: {best_fitness:.4f} (Avg:{best_stats['avg_sortino']:.2f} | Min:{best_stats['min_sortino']:.2f})")
            
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            if generations_without_improvement >= 10: # Increased patience for robust convergence
                print(f"🛑 Early Stopping Triggered (No Robust Improvement).")
                break
            
            # 4. Selection (Elitism on Robust Fitness)
            num_elite = int(self.pop_size * config.ELITE_PERCENTAGE)

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
            
            # Cap Hall of Fame Size to prevent memory explosion
            if len(self.hall_of_fame) > 20000:
                self.hall_of_fame.sort(key=lambda x: x[1], reverse=True)
                self.hall_of_fame = self.hall_of_fame[:20000]

            # 5. Next Gen
            new_population = elites[:50] # Keep top 50 unchanged (Elitism)
            
            # --- ELITE MUTATION (Refinement) ---
            # Take top 20% of elites and create slightly tweaked clones
            # This allows fine-tuning without the destructive nature of crossover
            n_mutants = int(self.pop_size * 0.20)
            mutant_source = elites[:max(1, int(len(elites)*0.2))] # Top 20% of elites
            
            for _ in range(n_mutants):
                if not mutant_source: break
                parent = random.choice(mutant_source)
                
                # Clone
                child = Strategy(name=f"Mutant_{random.randint(1000,9999)}")
                child.long_genes = [g.copy() for g in parent.long_genes]
                child.short_genes = [g.copy() for g in parent.short_genes]
                child.regime_genes = [g.copy() for g in parent.regime_genes]
                child.min_concordance = parent.min_concordance
                
                # Apply Mutation (Aggressive Fine-Tuning)
                # Mutate 1-2 genes guaranteed
                genes_to_mutate = child.long_genes + child.short_genes + child.regime_genes
                if genes_to_mutate:
                    target_genes = random.sample(genes_to_mutate, min(len(genes_to_mutate), 2))
                    for g in target_genes:
                        g.mutate(self.factory.features)
                        
                child.cleanup()
                new_population.append(child)
            
            # --- IMMIGRATION (Fresh Blood Injection) ---
            # Prune the bottom and replace with 10% completely new random strategies
            n_immigrants = int(self.pop_size * config.IMMIGRATION_PERCENTAGE)
            for _ in range(n_immigrants):
                new_population.append(self.factory.create_strategy((2, 4)))
            
            # Fill the rest with Children of Elites (Crossover)
            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
                
                # Mutation
                mut_rate = config.MUTATION_RATE

                for g in child.long_genes + child.short_genes:
                    if random.random() < mut_rate:
                        g.mutate(self.factory.features)
                
                child.cleanup()
                new_population.append(child)
                
            self.population = new_population
            
            # CLEANUP: Free memory from temporary features (deltas, zscores) created this generation
            # self.backtester.reset_jit_context()
            
            print(f"  Gen completed in {time.time()-start_time:.2f}s")

        # 6. Final Selection
        # print(f"\n--- 🛡️ FINAL SELECTION (VALIDATION PHASE) ---")
        unique_hof = []
        seen = set()
        
        # Sort HOF by Robust Score (Validation) - STRICTLY BLIND TO TEST SET
        for s, v in sorted(self.hall_of_fame, key=lambda x: x[1], reverse=True):
            # Create a Gene-Only Signature (Ignore Name) to remove clones
            l_sig = "|".join(sorted([str(g) for g in s.long_genes]))
            s_sig = "|".join(sorted([str(g) for g in s.short_genes]))
            r_sig = "|".join(sorted([str(g) for g in s.regime_genes]))
            gene_signature = f"L:{l_sig}_S:{s_sig}_R:{r_sig}"
            
            if gene_signature not in seen:
                unique_hof.append(s)
                seen.add(gene_signature)
        
        # Select Top 100 Candidates based on VALIDATION ROBUSTNESS
        top_candidates = unique_hof[:100]
        
        if top_candidates:
            # 7. OOS Reporting (Test Phase) on SELECTED CANDIDATES ONLY
            # print("\n--- 🏆 APEX TRADERS (OOS PERFORMANCE REPORT) ---")
            
            # PRESERVE Validation Fitness (Backtester overwrites s.fitness)
            val_fitness_map = {s.name: s.fitness for s in top_candidates}
            
            # Evaluate on Test Set (One-Shot)
            test_res = self.backtester.evaluate_population(top_candidates, set_type='test', return_series=False, prediction_mode=False, time_limit=horizon)
            
            output = []
            
            # Collect results (Preserve Order of Robustness)
            for i, s in enumerate(top_candidates):
                strat_data = s.to_dict()
                
                # Test Metrics (For Reporting Only)
                t_sortino = float(test_res.iloc[i]['sortino'])
                t_ret = float(test_res.iloc[i]['total_return'])
                t_trades = int(test_res.iloc[i]['trades'])
                
                # Add Metadata
                strat_data['generation'] = getattr(s, 'generation_found', -1)
                strat_data['test_sortino'] = t_sortino
                strat_data['test_return'] = t_ret
                strat_data['test_trades'] = t_trades
                # RETRIEVE preserved Validation Score
                strat_data['robust_score'] = float(val_fitness_map.get(s.name, -999.0)) 
                strat_data['training_id'] = self.training_id
                
                output.append(strat_data)
            
            # Save to Disk (Sorted by Robustness)
            os.makedirs(config.DIRS['STRATEGIES_DIR'], exist_ok=True)
            out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")

            # --- PERSISTENCE: APPEND ONLY ---
            # We append new robust strategies. We do not re-rank by Test Score globally.
            existing_data = []
            if os.path.exists(out_path):
                try:
                    with open(out_path, "r") as f:
                        existing_data = json.load(f)
                except Exception as e:
                    print(f"⚠️ Failed to load existing strategies: {e}")

            # Merge
            combined = existing_data + output
            
            # Deduplicate by Name (Strategy Structure)
            seen_names = set()
            unique_combined = []
            # Prefer newer runs if duplicate? No, prefer higher robustness.
            # Sort combined by Robust Score
            combined.sort(key=lambda x: x.get('robust_score', -999), reverse=True)
            
            for s in combined:
                if s['name'] not in seen_names:
                    unique_combined.append(s)
                    seen_names.add(s['name'])
            
            # Limit global file to top 1000 Robust Strategies
            final_output = unique_combined[:1000]
            
            # Print Top 5 (Current Run)
            print("\nTop 5 Robust Strategies (Current Run) & Their OOS Results:")
            print(f"{'Rank':<4} | {'Robust (Val)':<12} | {'Sortino (Test)':<14} | {'Ret (Test)':<10} | {'Trades':<6} | {'ID'}")
            print("-" * 80)
            for i, s in enumerate(output[:5]):
                print(f"{i+1:<4} | {s['robust_score']:<12.4f} | {s['test_sortino']:<14.4f} | {s['test_return']*100:<9.2f}% | {s['test_trades']:<6} | {s['name']}")
            
            with open(out_path, "w") as f: json.dump(final_output, f, indent=4)
            print(f"\n💾 Saved {len(output)} Candidates to {out_path} (Merged Total: {len(final_output)})")
        else:
            print("No strategies survived robustness filter.")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--pop_size", type=int, default=5000)
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
    
    # Cleanup Temp Dir
    if hasattr(factory.backtester, 'temp_dir') and os.path.exists(factory.backtester.temp_dir):
        shutil.rmtree(factory.backtester.temp_dir)