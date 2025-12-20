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
    def __init__(self, data, survivors_file, population_size=2000, generations=30):
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
        
        # Long Leg: Pick Regime from one parent, Trigger from the other
        l_regime = random.choice([p1.long_genes[0], p2.long_genes[0]])
        l_trigger = random.choice([p1.long_genes[1], p2.long_genes[1]])
        child.long_genes = [l_regime, l_trigger]
        
        # Short Leg
        s_regime = random.choice([p1.short_genes[0], p2.short_genes[0]])
        s_trigger = random.choice([p1.short_genes[1], p2.short_genes[1]])
        child.short_genes = [s_regime, s_trigger]
        
        return child

    def evolve(self):
        self.initialize_population()
        
        for gen in range(self.generations):
            start_time = time.time()
            
            # 1. Evaluate on TRAIN
            results = self.backtester.evaluate_population(self.population, set_type='train')
            
            # 2. Apply Trade Penalty
            results.loc[results['trades'] < 5, 'sharpe'] = -1.0
            
            best_sharpe = results['sharpe'].max()
            print(f"\n--- Generation {gen} | Best [Train] Sharpe: {best_sharpe:.4f} ---")
            
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
            
            # Migration
            if gen % 5 == 0 and gen > 0:
                for _ in range(int(self.pop_size * 0.3)):
                    new_population.append(self.factory.create_strategy())
            
            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = self.crossover(p1, p2)
                
                # Structural Mutation
                mut_rate = 0.2 if best_sharpe < 0.01 else 0.05
                
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
            with open("data/apex_strategies.json", "w") as f: json.dump(output, f, indent=4)
        else:
            print("No strategies survived validation.")

if __name__ == "__main__":
    engine = FeatureEngine(config.DIRS['DATA_RAW_TICKS'])
    survivors_file = os.path.join(config.DIRS['DATA_DIR'], "survivors_60.json")
    df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    engine.create_volume_bars(df, volume_threshold=250)
    engine.add_features_to_bars(windows=[50, 100, 200, 400])
    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_delta_features(lookback=10)
    engine.add_delta_features(lookback=50)
    
    factory = EvolutionaryAlphaFactory(engine.bars, survivors_file)
    factory.evolve()