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
        self.decay_rate = decay_rate # Penalty for later generations to prevent overfitting
        self.survivors_file = survivors_file
        self.prediction_mode = prediction_mode
        
        self.factory = GenomeFactory(survivors_file)
        self.factory.set_stats(data)
        self.backtester = BacktestEngine(data, cost_bps=0.2, target_col=target_col) # Lower costs for easier discovery
        
        self.population = []
        self.hall_of_fame = []

    def initialize_population(self):
        print(f"🧬 Spawning {self.pop_size} bidirectional strategies...")
        self.population = [self.factory.create_strategy() for _ in range(self.pop_size)]

    def crossover(self, p1, p2):
        # Strict Crossover: Maintain Regime (0) + Trigger (1) structure
        child = Strategy(name=f"Child_{random.randint(1000,9999)}")
        
        # Long Leg: Pick Regime from one parent, Trigger from the other (Deep Copy!)
        l_regime = random.choice([p1.long_genes[0], p2.long_genes[0]]).copy()
        l_trigger = random.choice([p1.long_genes[1], p2.long_genes[1]]).copy()
        child.long_genes = [l_regime, l_trigger]
        
        # Short Leg (Deep Copy!)
        s_regime = random.choice([p1.short_genes[0], p2.short_genes[0]]).copy()
        s_trigger = random.choice([p1.short_genes[1], p2.short_genes[1]]).copy()
        child.short_genes = [s_regime, s_trigger]
        
        return child

    def evolve(self, horizon=60):
        # Reproducibility
        random.seed(42)
        np.random.seed(42)
        
        self.initialize_population()
        
        generations_without_improvement = 0
        global_best_sharpe = -999.0
        
        for gen in range(self.generations):
            start_time = time.time()
            
            # 1. Evaluate on TRAIN
            results = self.backtester.evaluate_population(self.population, set_type='train', prediction_mode=self.prediction_mode)
            
            # 2. Apply Trade Penalty
            results.loc[results['trades'] < 5, 'sharpe'] = -1.0
            
            # Capture Raw Metrics
            raw_best_sharpe = results['sharpe'].max()
            
            # 2.5 Apply Generational Decay (Alpha Decay Simulation)
            # Penalize later generations to force significant improvements
            decay_factor = self.decay_rate ** gen
            results['sharpe'] = results['sharpe'] * decay_factor
            
            best_sharpe = results['sharpe'].max()
            print(f"\n--- Generation {gen} | Best [Train] Sharpe: {best_sharpe:.4f} (Raw: {raw_best_sharpe:.4f} | Decay: {decay_factor:.4f}) ---")
            
            # Early Stopping Check (using Decayed Sharpe to force robustness)
            if best_sharpe > global_best_sharpe:
                global_best_sharpe = best_sharpe
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            if generations_without_improvement >= 10: # Relaxed slightly for decay
                print(f"🛑 Early Stopping Triggered: No improvement for 10 generations (Best: {global_best_sharpe:.4f})")
                break
            
            # 3. Selection
            num_elite = int(self.pop_size * 0.2)
            elites_ids = results.sort_values('sharpe', ascending=False).head(num_elite).index.tolist()
            elites = [self.population[idx] for idx in elites_ids]
            
            # 4. Cross-Validation Gating
            val_results = self.backtester.evaluate_population(elites, set_type='validation', prediction_mode=self.prediction_mode)
            val_results.loc[val_results['trades'] < 3, 'sharpe'] = -1.0
            
            for i, elite in enumerate(elites):
                val_sharpe = val_results.iloc[i]['sharpe']
                if val_sharpe > 0.0:
                    self.hall_of_fame.append((elite, val_sharpe))
            
            # 5. Create Next Gen
            new_population = elites[:20]
            
            # Migration (Higher diversity to avoid monoculture)
            if gen % 5 == 0 and gen > 0:
                for _ in range(int(self.pop_size * 0.4)):
                    new_population.append(self.factory.create_strategy())
            
            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
                
                # Structural Mutation
                # Mutate aggressively if decaying best_sharpe is low
                mut_rate = 0.25 if best_sharpe < 0.01 else 0.1
                
                # Long Genes
                if random.random() < mut_rate: # Mutate Regime
                    child.long_genes[0].mutate(self.factory.regime_pool)
                if random.random() < mut_rate: # Mutate Trigger
                    child.long_genes[1].mutate(self.factory.trigger_pool)
                    
                # Short Genes
                if random.random() < mut_rate:
                    child.short_genes[0].mutate(self.factory.regime_pool)
                if random.random() < mut_rate:
                    child.short_genes[1].mutate(self.factory.trigger_pool)
                
                new_population.append(child)
                
            self.population = new_population
            print(f"  Gen completed in {time.time()-start_time:.2f}s")

        # 6. Final OOS Test
        print("\n--- 🏆 APEX PREDATORS (OOS TEST) ---")
        unique_hof = []
        seen = set()
        for s, v in sorted(self.hall_of_fame, key=lambda x: x[1], reverse=True):
            if str(s) not in seen:
                unique_hof.append(s)
                seen.add(str(s))
            # Keep more candidates for orthogonality check (Top 50 instead of 5)
            if len(unique_hof) >= 50: break
            
        if unique_hof:
            # Get Metrics AND Returns Series
            test_res, net_returns = self.backtester.evaluate_population(unique_hof, set_type='test', return_series=True, prediction_mode=self.prediction_mode)
            
            # --- FIX: ORTHOGONALITY CHECK (Monoculture Prevention) ---
            # Greedy Selection: Pick Best, then pick next Best that is uncorrelated (< 0.7)
            
            # 1. Sort by Sharpe
            sorted_indices = test_res.sort_values('sharpe', ascending=False).index.tolist()
            
            # 2. Calculate Correlation Matrix of Strategies
            # (Strategies are columns in net_returns)
            returns_df = pd.DataFrame(net_returns)
            corr_matrix = returns_df.corr().abs() # Absolute correlation
            
            selected_indices = []
            
            print(f"Selecting uncorrelated strategies from {len(unique_hof)} candidates...")
            
            for idx in sorted_indices:
                if len(selected_indices) >= 5: break
                
                # Check for positive performance
                current_sharpe = test_res.loc[idx, 'sharpe']
                if current_sharpe <= 0:
                    continue
                
                if not selected_indices:
                    # Always pick the absolute best first
                    selected_indices.append(idx)
                    print(f"  1. [Best] {test_res.loc[idx, 'id']} (Sharpe: {current_sharpe:.4f})")
                else:
                    # Check correlation with ALREADY SELECTED
                    is_uncorrelated = True
                    for selected_idx in selected_indices:
                        rho = corr_matrix.loc[idx, selected_idx]
                        if rho > 0.7:
                            is_uncorrelated = False
                            # print(f"     Skipping {test_res.loc[idx, 'id']} (Corr: {rho:.2f} with {test_res.loc[selected_idx, 'id']})")
                            break
                    
                    if is_uncorrelated:
                        selected_indices.append(idx)
                        print(f"  {len(selected_indices)}. [Add ] {test_res.loc[idx, 'id']} (Sharpe: {current_sharpe:.4f})")
            
            # Subset results to selected
            final_apex = [unique_hof[i] for i in selected_indices]
            final_res = test_res.iloc[selected_indices]
            
            print("\nFinal Portfolio Stats:")
            print(final_res[['id', 'sharpe', 'total_return', 'trades']])
            
            output = []
            for idx in selected_indices:
                s = unique_hof[idx]
                output.append({'name': s.name, 'logic': str(s), 'test_sharpe': test_res.loc[idx, 'sharpe']})
            
            out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")
            with open(out_path, "w") as f: json.dump(output, f, indent=4)
            print(f"\n💾 Saved {len(output)} Apex Strategies to {out_path}")
        else:
            print("No strategies survived validation.")
            
        # 7. Save Final Population Snapshot (Top 100) for DNA Analysis
        # We need to re-evaluate current population to get latest scores if needed, 
        # or just use the last results.
        print("\nSaving Population Snapshot for DNA Analysis...")
        pop_results = self.backtester.evaluate_population(self.population, set_type='train', prediction_mode=self.prediction_mode)
        top_100_idx = pop_results.sort_values('sharpe', ascending=False).head(100).index
        
        dna_dump = []
        for idx in top_100_idx:
            strat = self.population[idx]
            # Extract raw genes
            genes = []
            
            def extract_gene_data(g, type_lbl):
                if hasattr(g, 'type'):
                    if g.type == 'delta':
                        return {'feature': g.feature, 'op': g.operator, 'threshold': g.threshold, 'lookback': g.lookback, 'type': type_lbl, 'mode': 'delta'}
                    elif g.type == 'zscore':
                        return {'feature': g.feature, 'op': g.operator, 'threshold': g.threshold, 'window': g.window, 'type': type_lbl, 'mode': 'zscore'}
                    elif g.type == 'relational':
                        return {'feature': g.feature_left, 'op': g.operator, 'threshold': g.feature_right, 'type': type_lbl, 'mode': 'relational'}
                    
                # Fallback for Static or unknown
                if hasattr(g, 'threshold'): 
                    return {'feature': g.feature, 'op': g.operator, 'threshold': g.threshold, 'type': type_lbl, 'mode': 'static'}
                else: 
                    return {'feature': g.feature_left, 'op': g.operator, 'threshold': g.feature_right, 'type': type_lbl, 'mode': 'relational'}

            for g in strat.long_genes: genes.append(extract_gene_data(g, 'long'))
            for g in strat.short_genes: genes.append(extract_gene_data(g, 'short'))
            
            dna_dump.append({
                'name': strat.name,
                'sharpe': pop_results.loc[idx, 'sharpe'],
                'genes': genes
            })
            
        with open(os.path.join(config.DIRS['OUTPUT_DIR'], "final_population.json"), "w") as f:
            json.dump(dna_dump, f, indent=4)
        print(f"💾 Saved Top 100 Population Genomes to {os.path.join(config.DIRS['OUTPUT_DIR'], 'final_population.json')}")

import argparse

if __name__ == "__main__":
    from feature_engine import create_full_feature_engine
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", type=str, required=True, help="Path to survivors JSON file")
    parser.add_argument("--horizon", type=int, default=60, help="Target prediction horizon")
    parser.add_argument("--pop_size", type=int, default=20000, help="Population Size")
    parser.add_argument("--gens", type=int, default=100, help="Number of Generations")
    parser.add_argument("--decay", type=float, default=0.99, help="Generational Sharpe Decay Rate")
    args = parser.parse_args()

    print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found. Run generate_features.py first.")
        exit(1)
        
    bars_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    survivors_file = args.survivors
    
    print(f"\n🚀 Starting Evolution for Horizon: {args.horizon}")
    print(f"📂 Using Survivors: {survivors_file}")
    
    # --- FIX: ALIGNMENT WITH PURGE PROCESS ---
    # We must train strategies to predict the SAME target used to select the features.
    # Calculating Triple Barrier Labels...
    print(f"🎯 calculating Triple Barrier Labels (Horizon: {args.horizon})...")
    bars_df['target_return'] = triple_barrier_labels(bars_df, lookahead=args.horizon, pt_sl_multiple=2.0)
    
    # Fill NaNs in target (usually at the end of the series) with 0 to prevent crashes
    bars_df['target_return'] = bars_df['target_return'].fillna(0.0)
    # -----------------------------------------

    print(f"👥 Population: {args.pop_size} | 🧬 Generations: {args.gens} | 📉 Decay: {args.decay}")
    
    factory = EvolutionaryAlphaFactory(
        bars_df, 
        survivors_file, 
        population_size=args.pop_size, 
        generations=args.gens,
        decay_rate=args.decay,
        target_col='target_return',
        prediction_mode=True
    )
    factory.evolve(horizon=args.horizon)