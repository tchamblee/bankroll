import numpy as np
import pandas as pd
import random
import time
import config
import uuid
import warnings
import json
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
        
        # --- DATA FILTERING ---
        if hasattr(config, 'TRAIN_START_DATE') and config.TRAIN_START_DATE:
            if 'time_start' in data.columns:
                 # Ensure datetime compatibility
                 if not pd.api.types.is_datetime64_any_dtype(data['time_start']):
                     data['time_start'] = pd.to_datetime(data['time_start'], utc=True)
                 
                 # Handle Timezone
                 ts_col = data['time_start']
                 if ts_col.dt.tz is None:
                     ts_col = ts_col.dt.tz_localize('UTC')
                 else:
                     ts_col = ts_col.dt.tz_convert('UTC')
                     
                 start_ts = pd.Timestamp(config.TRAIN_START_DATE).tz_localize('UTC')
                 
                 if ts_col.min() < start_ts:
                     original_len = len(data)
                     data = data[ts_col >= start_ts].reset_index(drop=True)
                     print(f"  📅 Time Filter: Applied {config.TRAIN_START_DATE}. Dropped {original_len - len(data)} rows.")

        self.data = data # Store original just in case, but usually not needed
        self.pop_size = population_size if population_size else config.EVO_BATCH_SIZE
        self.generations = generations
        self.decay_rate = decay_rate
        self.survivors_file = survivors_file
        self.prediction_mode = prediction_mode
        
        # --- OPTIMIZATION: Filter Data to Survivors + Essentials ---
        # This prevents the BacktestEngine from precomputing context for hundreds of useless features.
        try:
            with open(survivors_file, 'r') as f:
                survivor_features = set(json.load(f))
            
            essentials = {'time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'log_ret'}
            if target_col: essentials.add(target_col)
            
            # Keep columns that are in survivors OR essentials
            keep_cols = [c for c in data.columns if c in survivor_features or c in essentials]
            data_subset = data[keep_cols]
            # print(f"  📉 Optimization: Reduced Feature Matrix from {len(data.columns)} to {len(keep_cols)} columns (Survivors + Essentials).")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not filter feature matrix: {e}. Using full matrix.")
            data_subset = data

        self.backtester = BacktestEngine(
            data_subset, 
            cost_bps=config.COST_BPS, 
            target_col=target_col,
            annualization_factor=config.ANNUALIZATION_FACTOR,
            account_size=config.ACCOUNT_SIZE
        )
        
        self.factory = GenomeFactory(survivors_file)
        train_data = data_subset.iloc[:self.backtester.train_idx]
        self.factory.set_stats(train_data)
        
        self.population = []
        self.hall_of_fame = [] 
        self.total_strategies_evaluated = 0

    def cleanup(self):
        if hasattr(self, 'backtester'):
            self.backtester.shutdown()

    def generate_viable_strategies(self, target_count, horizon, batch_size=1000, verbose=True):
        """
        Generate random strategies that pass a minimum viability bar.

        Instead of generating pure random strategies (many garbage), this:
        1. Generates batches of random strategies
        2. Quick-evaluates them on train set only
        3. Keeps only those with positive return and minimum trades
        4. Repeats until target count is reached (no giving up)

        This ensures all strategies entering evolution have baseline quality.
        """
        viable = []
        total_generated = 0
        min_trades_filter = max(3, int(config.MIN_TRADES_COEFFICIENT / horizon / 3))
        start_time = time.time()

        while len(viable) < target_count:
            # Generate batch
            batch = [
                self.factory.create_strategy((config.GENE_COUNT_MIN, config.GENE_COUNT_MAX))
                for _ in range(batch_size)
            ]
            total_generated += batch_size

            # Quick eval on train set only (fast)
            results = self.backtester.evaluate_population(
                batch, set_type='train', time_limit=horizon, min_trades=1
            )

            # Filter for viability: positive return + minimum trades
            for i, row in results.iterrows():
                if len(viable) >= target_count:
                    break

                strat_name = row['id']
                total_return = row.get('total_return', 0)
                trades = row.get('trades', 0)

                # Minimum bar: positive return and enough trades
                if total_return > 0 and trades >= min_trades_filter:
                    # Find the strategy object by name
                    strat = next((s for s in batch if s.name == strat_name), None)
                    if strat:
                        viable.append(strat)

            if verbose:
                elapsed = time.time() - start_time
                pass_rate = len(viable) / max(1, total_generated) * 100
                # Estimate time remaining
                if len(viable) > 0:
                    rate_per_sec = len(viable) / max(1, elapsed)
                    remaining = (target_count - len(viable)) / max(0.001, rate_per_sec)
                    eta_str = f", ~{remaining:.0f}s remaining" if remaining < 3600 else f", ~{remaining/60:.0f}m remaining"
                else:
                    eta_str = ""
                print(f"    ... Viability filter: {len(viable)}/{target_count} ({pass_rate:.1f}% of {total_generated} pass{eta_str})")

        return viable

    def initialize_population(self, horizon=None):
        self.population = []
        print(f"  🎲 Generating {self.pop_size} Random Strategies to initialize population.")

        # Use viable strategy generation for better starting quality
        if horizon:
            self.population = self.generate_viable_strategies(self.pop_size, horizon)
            viable_count = len([s for s in self.population])
            print(f"    ... Generated {viable_count}/{self.pop_size} strategies (pre-filtered for viability). Done.")
        else:
            # Fallback to pure random if no horizon specified
            for i in range(self.pop_size):
                self.population.append(self.factory.create_strategy((config.GENE_COUNT_MIN, config.GENE_COUNT_MAX)))
                if (i + 1) % 500 == 0:
                    print(f"    ... Generated {i + 1}/{self.pop_size} strategies")
            print(f"    ... Generated {self.pop_size}/{self.pop_size} strategies. Done.")

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
            wfv_results = self.backtester.evaluate_walk_forward(self.population, folds=config.WFV_FOLDS, time_limit=horizon, min_trades=current_min_trades)
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
                # Inherit all barrier parameters (symmetric + directional)
                child.stop_loss_pct = parent.stop_loss_pct
                child.take_profit_pct = parent.take_profit_pct
                child.limit_dist_atr = parent.limit_dist_atr
                child.sl_long = parent.sl_long
                child.sl_short = parent.sl_short
                child.tp_long = parent.tp_long
                child.tp_short = parent.tp_short

                mutate_strategy(child, self.factory.features)
                new_population.append(child)
            
            # Immigration (use viable strategies for better quality immigrants)
            n_immigrants = int(self.pop_size * config.IMMIGRATION_PERCENTAGE)
            if n_immigrants > 0:
                immigrants = self.generate_viable_strategies(n_immigrants, horizon, batch_size=max(100, n_immigrants * 5), verbose=False)
                new_population.extend(immigrants)
            
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
