import numpy as np
import random
import time
import config
import uuid
import warnings
from genome import GenomeFactory, Strategy
from backtest import BacktestEngine

# Local helper imports
from .reproduction import crossover_strategies, mutate_strategy
from .selection import update_hall_of_fame
from .reporting import save_campaign_results

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")

class EvolutionaryAlphaFactory:
    def __init__(self, data, survivors_file, population_size=None, generations=100, decay_rate=0.99, target_col='log_ret', prediction_mode=False):
        self.training_id = uuid.uuid4().hex[:8]
        print(f"🆔 Training Run ID: {self.training_id}")
        
        self.data = data
        self.pop_size = population_size if population_size else config.EVO_BATCH_SIZE
        self.generations = generations
        self.decay_rate = decay_rate
        self.survivors_file = survivors_file
        self.prediction_mode = prediction_mode
        
        self.backtester = BacktestEngine(
            data, 
            cost_bps=config.COST_BPS, 
            target_col=target_col,
            annualization_factor=config.ANNUALIZATION_FACTOR,
            account_size=config.ACCOUNT_SIZE
        )
        
        self.factory = GenomeFactory(survivors_file)
        train_data = data.iloc[:self.backtester.train_idx]
        self.factory.set_stats(train_data)
        
        self.population = []
        self.hall_of_fame = [] 
        self.total_strategies_evaluated = 0

    def cleanup(self):
        if hasattr(self, 'backtester'):
            self.backtester.shutdown()

    def initialize_population(self, horizon=None):
        self.population = []
        print(f"  🎲 Generating {self.pop_size} Random Strategies to initialize population.")
        self.population = [self.factory.create_strategy((config.GENE_COUNT_MIN, config.GENE_COUNT_MAX)) for _ in range(self.pop_size)]

    def evolve(self, horizon=60):
        self.initialize_population(horizon=horizon)
        
        generations_without_improvement = 0
        global_best_fitness = -999.0
        
        for gen in range(self.generations):
            start_time = time.time()
            
            # Dynamic Trade Filter (Horizon-Aware Scaling)
            target_final = max(config.MIN_TRADES_FOR_METRICS, int(config.MIN_TRADES_COEFFICIENT / horizon + 5))
            scaling_factor = target_final / 10.0
            
            current_min_trades = int(3 * scaling_factor)
            if gen >= 5: current_min_trades = int(6 * scaling_factor)
            if gen >= 15: current_min_trades = target_final
            
            # Increment trial counter for DSR
            self.total_strategies_evaluated += len(self.population)

            # 1. Evaluate using Rolling Walk-Forward Validation
            wfv_results = self.backtester.evaluate_walk_forward(self.population, folds=4, time_limit=horizon, min_trades=current_min_trades)
            wfv_scores = wfv_results['sortino'].values
            
            avg_wfv = np.mean(wfv_scores)
            avg_trades_debug = wfv_results.get('avg_trades', wfv_results.get('trades', np.zeros(len(wfv_scores)))).mean() # Fallback safely
            print(f"  DEBUG Gen {gen}: Avg Raw WFV: {avg_wfv:.2f} | Avg Trades: {avg_trades_debug:.1f}")

            # 2. Penalties (Dominance & Complexity)
            feature_counts = {}
            for strat in self.population:
                used_features = set()
                for gene in strat.long_genes + strat.short_genes:
                    if hasattr(gene, 'feature'): used_features.add(gene.feature)
                    elif hasattr(gene, 'feature_left'): 
                        used_features.add(gene.feature_left)
                        used_features.add(gene.feature_right)
                for f in used_features:
                    feature_counts[f] = feature_counts.get(f, 0) + 1
            
            for i, strat in enumerate(self.population):
                n_genes = len(strat.long_genes) + len(strat.short_genes)
                complexity_penalty = 0.5 * (n_genes ** 2) * config.COMPLEXITY_PENALTY_PER_GENE
                
                dom_penalty = 0.0
                strat_features = set()
                for gene in strat.long_genes + strat.short_genes:
                    if hasattr(gene, 'feature'): strat_features.add(gene.feature)
                    elif hasattr(gene, 'feature_left'): 
                        strat_features.add(gene.feature_left)
                        strat_features.add(gene.feature_right)
                
                for f in strat_features:
                    count = feature_counts.get(f, 0)
                    ratio = count / self.pop_size
                    if ratio > config.DOMINANCE_PENALTY_THRESHOLD:
                         dom_penalty += (ratio - config.DOMINANCE_PENALTY_THRESHOLD) * config.DOMINANCE_PENALTY_MULTIPLIER
                
                wfv_scores[i] -= (complexity_penalty + dom_penalty)
                strat.fitness = wfv_scores[i]

            # 3. Stats & Early Stopping
            best_idx = np.argmax(wfv_scores)
            best_fitness = wfv_scores[best_idx]
            
            if gen % 5 == 0:
                print(f"Gen {gen} | Best WFV Score: {best_fitness:.4f}")
            
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
                
            if generations_without_improvement >= 15:
                print(f"🛑 Early Stopping Triggered.")
                break
            
            # 4. Update Diverse Hall of Fame
            sorted_indices = np.argsort(wfv_scores)[::-1]
            top_50_indices = sorted_indices[:50]
            top_50_strats = [self.population[i] for i in top_50_indices]
            
            for s in top_50_strats:
                if not hasattr(s, 'generation_found'): s.generation_found = gen
            
            update_hall_of_fame(self.hall_of_fame, self.backtester, top_50_strats, gen, max_gens=self.generations, horizon=horizon)

            # 5. Selection for Next Gen
            num_elite = int(self.pop_size * config.ELITE_PERCENTAGE)
            elites = [self.population[i] for i in sorted_indices[:num_elite]]
            
            new_population = elites[:50] # Keep top 50 strictly
            
            # Elite Mutation
            n_mutants = int(self.pop_size * 0.20)
            mutant_source = elites[:max(1, int(len(elites)*0.2))]
            for _ in range(n_mutants):
                if not mutant_source: break
                parent = random.choice(mutant_source)
                child = Strategy(name=f"Mutant_{random.randint(1000,9999)}")
                child.long_genes = [g.copy() for g in parent.long_genes]
                child.short_genes = [g.copy() for g in parent.short_genes]
                child.stop_loss_pct = parent.stop_loss_pct
                child.take_profit_pct = parent.take_profit_pct
                
                mutate_strategy(child, self.factory.features)
                new_population.append(child)
            
            # Immigration
            n_immigrants = int(self.pop_size * config.IMMIGRATION_PERCENTAGE)
            for _ in range(n_immigrants):
                new_population.append(self.factory.create_strategy((config.GENE_COUNT_MIN, config.GENE_COUNT_MAX)))
            
            # Crossover
            while len(new_population) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = crossover_strategies(p1, p2)
                
                # Chance of mutation after crossover
                mut_rate = config.MUTATION_RATE
                if random.random() < mut_rate: # Apply mutation to the child as a whole (probabilistic in mutate_strategy?)
                    # No, logic was per gene in original:
                    # for g in child.long_genes + child.short_genes:
                    #   if random.random() < mut_rate: g.mutate(self.factory.features)
                    # The mutate_strategy function mutates 2 genes randomly.
                    # I'll stick to mutate_strategy logic for simplicity as it's cleaner.
                    mutate_strategy(child, self.factory.features)
                
                new_population.append(child)
                
            self.population = new_population
            print(f"  Gen {gen} completed in {time.time()-start_time:.2f}s | HOF Size: {len(self.hall_of_fame)}")

        # --- FINAL SELECTION ---
        save_campaign_results(self.hall_of_fame, self.backtester, horizon, self.training_id, self.total_strategies_evaluated)
