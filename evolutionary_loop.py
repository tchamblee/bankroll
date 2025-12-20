import pandas as pd
import numpy as np
import os
import json
import random
import time
import config
from feature_engine import FeatureEngine
from strategy_genome import GenomeFactory, Strategy, Gene
from backtest_engine import BacktestEngine

class EvolutionaryAlphaFactory:
    def __init__(self, data, survivors_file, population_size=10000, generations=100):
        self.data = data
        self.pop_size = population_size
        self.generations = generations
        self.survivors_file = survivors_file
        
        self.factory = GenomeFactory(survivors_file)
        self.factory.set_stats(data)
        self.backtester = BacktestEngine(data, cost_bps=0.2) # Lower costs for easier discovery
        
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
            results = self.backtester.evaluate_population(self.population, set_type='train')
            
            # 2. Apply Trade Penalty
            results.loc[results['trades'] < 5, 'sharpe'] = -1.0
            
            best_sharpe = results['sharpe'].max()
            print(f"\n--- Generation {gen} | Best [Train] Sharpe: {best_sharpe:.4f} ---")
            
            # Early Stopping Check
            if best_sharpe > global_best_sharpe:
                global_best_sharpe = best_sharpe
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            if generations_without_improvement >= 5:
                print(f"🛑 Early Stopping Triggered: No improvement for 5 generations (Best: {global_best_sharpe:.4f})")
                break
            
            # 3. Selection
            num_elite = int(self.pop_size * 0.2)
            elites_ids = results.sort_values('sharpe', ascending=False).head(num_elite).index.tolist()
            elites = [self.population[idx] for idx in elites_ids]
            
            # 4. Cross-Validation Gating
            val_results = self.backtester.evaluate_population(elites, set_type='validation')
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
            if len(unique_hof) >= 5: break
            
        if unique_hof:
            test_res = self.backtester.evaluate_population(unique_hof, set_type='test')
            print(test_res)
            
            output = []
            for i, s in enumerate(unique_hof):
                output.append({'name': s.name, 'logic': str(s), 'test_sharpe': test_res.iloc[i]['sharpe']})
            
            out_path = f"data/apex_strategies_{horizon}.json"
            with open(out_path, "w") as f: json.dump(output, f, indent=4)
            print(f"\n💾 Saved {len(output)} Apex Strategies to {out_path}")
        else:
            print("No strategies survived validation.")
            
        # 7. Save Final Population Snapshot (Top 100) for DNA Analysis
        # We need to re-evaluate current population to get latest scores if needed, 
        # or just use the last results.
        print("\nSaving Population Snapshot for DNA Analysis...")
        pop_results = self.backtester.evaluate_population(self.population, set_type='train')
        top_100_idx = pop_results.sort_values('sharpe', ascending=False).head(100).index
        
        dna_dump = []
        for idx in top_100_idx:
            strat = self.population[idx]
            # Extract raw genes
            genes = []
            for g in strat.long_genes: genes.append({'feature': g.feature, 'op': g.operator, 'threshold': g.threshold, 'type': 'long'})
            for g in strat.short_genes: genes.append({'feature': g.feature, 'op': g.operator, 'threshold': g.threshold, 'type': 'short'})
            
            dna_dump.append({
                'name': strat.name,
                'sharpe': pop_results.loc[idx, 'sharpe'],
                'genes': genes
            })
            
        with open("data/final_population.json", "w") as f:
            json.dump(dna_dump, f, indent=4)
        print(f"💾 Saved Top 100 Population Genomes to data/final_population.json")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", type=str, required=True, help="Path to survivors JSON file")
    parser.add_argument("--horizon", type=int, default=60, help="Target prediction horizon")
    args = parser.parse_args()

    engine = FeatureEngine(config.DIRS['DATA_RAW_TICKS'])
    survivors_file = args.survivors
    
    print(f"\n🚀 Starting Evolution for Horizon: {args.horizon}")
    print(f"📂 Using Survivors: {survivors_file}")

    df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    engine.create_volume_bars(df, volume_threshold=250)
    engine.add_features_to_bars(windows=[50, 100, 200, 400])
    
    # --- CRYPTO & GDELT INTEGRATION ---
    engine.add_crypto_features("CLEAN_IBIT.parquet")
    
    gdelt_df = engine.load_gdelt_data()
    if gdelt_df is not None:
        engine.add_gdelt_features(gdelt_df)

    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_advanced_physics_features()
    engine.add_delta_features(lookback=10)
    engine.add_delta_features(lookback=50)
    
    factory = EvolutionaryAlphaFactory(engine.bars, survivors_file)
    factory.evolve(horizon=args.horizon)