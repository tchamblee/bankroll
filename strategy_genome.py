import numpy as np
import pandas as pd
import random
import json
import os

VALID_DELTA_LOOKBACKS = [1, 3, 5, 10, 20, 50]
VALID_ZSCORE_WINDOWS = [10, 20, 50, 100, 200]

class StaticGene:
    """
    Classic 'Magic Number' Gene.
    Format: Feature <Operator> Threshold
    Example: frac_diff_02 > 0.45
    """
    def __init__(self, feature: str, operator: str, threshold: float):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.type = 'static'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        # Cache Key
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold)
            if key in cache: return cache[key]

        # Context is a dict of numpy arrays
        if self.feature not in context:
            # Fallback for safety, though precalc should handle this
            res = np.zeros(context['__len__'], dtype=bool) if '__len__' in context else np.array([])
        else:
            data = context[self.feature]
            if self.operator == '>': res = data > self.threshold
            elif self.operator == '<': res = data < self.threshold
            else: res = np.zeros(len(data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Threshold
        if random.random() < 0.5:
            change = self.threshold * 0.1 * (1 if random.random() > 0.5 else -1)
            # Handle near-zero thresholds
            if abs(self.threshold) < 0.001: 
                change = 0.001 * (1 if random.random() > 0.5 else -1)
            self.threshold += change
            
        # 2. Mutate Operator
        if random.random() < 0.2: 
            self.operator = '>' if self.operator == '<' else '<'
            
        # 3. Mutate Feature
        if random.random() < 0.1: 
            self.feature = random.choice(features_pool)

    def copy(self):
        return StaticGene(self.feature, self.operator, self.threshold)

    def __repr__(self):
        return f"{self.feature} {self.operator} {self.threshold:.4f}"

class RelationalGene:
    """
    Sophisticated 'Context' Gene.
    Format: Feature_A <Operator> Feature_B
    Example: volatility_50 > volatility_200 (Volatility Expansion)
    """
    def __init__(self, feature_left: str, operator: str, feature_right: str):
        self.feature_left = feature_left
        self.operator = operator
        self.feature_right = feature_right
        self.type = 'relational'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature_left, self.operator, self.feature_right)
            if key in cache: return cache[key]

        if self.feature_left not in context or self.feature_right not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            left_data = context[self.feature_left]
            right_data = context[self.feature_right]
            
            if self.operator == '>': res = left_data > right_data
            elif self.operator == '<': res = left_data < right_data
            else: res = np.zeros(len(left_data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Operator
        if random.random() < 0.3: 
            self.operator = '>' if self.operator == '<' else '<'
            
        # 2. Mutate Left Feature
        if random.random() < 0.3: 
            self.feature_left = random.choice(features_pool)
            
        # 3. Mutate Right Feature
        if random.random() < 0.3: 
            self.feature_right = random.choice(features_pool)

    def copy(self):
        return RelationalGene(self.feature_left, self.operator, self.feature_right)

    def __repr__(self):
        return f"{self.feature_left} {self.operator} {self.feature_right}"

class DeltaGene:
    """
    'Momentum' Gene.
    Checks the change in a feature over time.
    Format: Delta(Feature, Lookback) <Operator> Threshold
    Example: delta(volatility, 5) > 0.05 (Volatility Spiking)
    """
    def __init__(self, feature: str, operator: str, threshold: float, lookback: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        # Enforce valid lookback
        if lookback not in VALID_DELTA_LOOKBACKS:
             # Snap to nearest
             self.lookback = min(VALID_DELTA_LOOKBACKS, key=lambda x:abs(x-lookback))
        else:
            self.lookback = lookback
        self.type = 'delta'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.lookback)
            if key in cache: return cache[key]

        ctx_key = f"delta_{self.feature}_{self.lookback}"
        if ctx_key not in context:
            # Fallback if precalc missed it (shouldn't happen if engine is sync'd)
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[ctx_key]
            if self.operator == '>': res = data > self.threshold
            elif self.operator == '<': res = data < self.threshold
            else: res = np.zeros(len(data), dtype=bool)
        
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Lookback
        if random.random() < 0.3:
            # Pick a new random lookback from the valid list
            self.lookback = random.choice(VALID_DELTA_LOOKBACKS)
        
        # 2. Mutate Threshold
        if random.random() < 0.3:
            change = self.threshold * 0.1 * (1 if random.random() > 0.5 else -1)
            if abs(self.threshold) < 0.001: 
                change = 0.001 * (1 if random.random() > 0.5 else -1)
            self.threshold += change
            
        # 3. Mutate Feature
        if random.random() < 0.1: 
            self.feature = random.choice(features_pool)
            
        # 4. Mutate Operator
        if random.random() < 0.2: 
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return DeltaGene(self.feature, self.operator, self.threshold, self.lookback)

    def __repr__(self):
        return f"Delta({self.feature}, {self.lookback}) {self.operator} {self.threshold:.4f}"

class ZScoreGene:
    """
    'Statistical' Gene (Prop Desk Favorite).
    Checks if a feature is an outlier relative to its recent history (Bollinger-style logic).
    Format: ZScore(Feature, Window) <Operator> Sigma
    Example: zscore(close, 20) < -2.0 (Price 2 sigmas below 20d mean)
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold # Sigma value
        
        if window not in VALID_ZSCORE_WINDOWS:
            self.window = min(VALID_ZSCORE_WINDOWS, key=lambda x:abs(x-window))
        else:
            self.window = window
            
        self.type = 'zscore'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window)
            if key in cache: return cache[key]

        ctx_key = f"zscore_{self.feature}_{self.window}"
        if ctx_key not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            z_score = context[ctx_key]
            if self.operator == '>': res = z_score > self.threshold
            elif self.operator == '<': res = z_score < self.threshold
            else: res = np.zeros(len(z_score), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Window
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)
            
        # 2. Mutate Threshold (Sigma)
        if random.random() < 0.3:
            self.threshold += random.uniform(-0.5, 0.5)
            
        # 3. Mutate Feature
        if random.random() < 0.1: 
            self.feature = random.choice(features_pool)
            
        # 4. Mutate Operator
        if random.random() < 0.2: 
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return ZScoreGene(self.feature, self.operator, self.threshold, self.window)

    def __repr__(self):
        return f"Z({self.feature}, {self.window}) {self.operator} {self.threshold:.2f}σ"

class Strategy:
    """
    Represents a Bidirectional Trading Strategy.
    """
    def __init__(self, name="Strategy", long_genes=None, short_genes=None, min_concordance=None):
        self.name = name
        self.long_genes = long_genes if long_genes else []
        self.short_genes = short_genes if short_genes else []
        self.min_concordance = min_concordance # None = ALL (AND), 1 = OR, etc.
        self.fitness = 0.0
        
    def generate_signal(self, context: dict, cache: dict = None) -> np.array:
        # Fix: len(context) returns key count for dict, but we want data rows.
        # BacktestEngine inserts '__len__' into the context.
        n_rows = context.get('__len__', 0)
        
        # Fallback for safety (e.g. if context is raw dict without metadata)
        if n_rows == 0 and len(context) > 0:
            # Try to get length from first array value
            for val in context.values():
                 if hasattr(val, 'shape'):
                     n_rows = val.shape[0]
                     break
        
        # Helper for Voting/AND Logic
        def evaluate_leg(genes, threshold):
            if not genes: return np.zeros(n_rows, dtype=bool)
            
            # Optimization: If strict AND (threshold is None or len(genes)), use fast bitwise
            eff_threshold = threshold if threshold is not None else len(genes)
            
            if eff_threshold == len(genes):
                # Strict AND
                mask = np.ones(n_rows, dtype=bool)
                for gene in genes:
                    mask &= gene.evaluate(context, cache)
                return mask
            else:
                # Voting Logic
                votes = np.zeros(n_rows, dtype=int)
                for gene in genes:
                    votes += gene.evaluate(context, cache).astype(int)
                return votes >= eff_threshold

        l_mask = evaluate_leg(self.long_genes, self.min_concordance)
        s_mask = evaluate_leg(self.short_genes, self.min_concordance)
            
        # Final: +1, -1, or 0 (Short priority or cancellation if both True)
        return l_mask.astype(int) - s_mask.astype(int)

    def __repr__(self):
        logic_str = "AND" if self.min_concordance is None else f"VOTE({self.min_concordance})"
        l_str = f" {logic_str} ".join([str(g) for g in self.long_genes]) if self.long_genes else "None"
        s_str = f" {logic_str} ".join([str(g) for g in self.short_genes]) if self.short_genes else "None"
        return f"[{self.name}][{logic_str}] LONG:({l_str}) | SHORT:({s_str})"

class GenomeFactory:
    def __init__(self, survivors_file):
        with open(survivors_file, 'r') as f:
            self.features = json.load(f)
        self.feature_stats = {} 
        
        # Categorize Features for Gated Logic
        self.regime_keywords = ['hurst', 'volatility', 'efficiency', 'entropy', 'skew', 'trend_strength', 
                               'yang_zhang', 'lambda', 'force', 'fdi',
                               'Vol_Ratio', 'news', 'panic', 'crisis', 'epu', 'total_vol']
        self.regime_pool = [f for f in self.features if any(k in f for k in self.regime_keywords)]
        self.trigger_pool = [f for f in self.features if f not in self.regime_pool]
        
        print(f"Factory Loaded: {len(self.regime_pool)} Regime Features | {len(self.trigger_pool)} Trigger Features")

    def set_stats(self, df):
        for f in self.features:
            if f in df.columns:
                self.feature_stats[f] = {'mean': df[f].mean(), 'std': df[f].std()}

    def create_gene_from_pool(self, pool):
        if not pool: return self.create_random_gene() # Fallback
        
        rand_val = random.random()
        
        # 20% Chance of Relational Gene (Context)
        if rand_val < 0.20:
            feature_left = random.choice(pool)
            feature_right = random.choice(pool) 
            operator = random.choice(['>', '<'])
            return RelationalGene(feature_left, operator, feature_right)
            
        # 20% Chance of Delta Gene (Momentum)
        elif rand_val < 0.40:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            # Delta threshold usually smaller than absolute values. 
            # We'll start with small random fraction of std dev.
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = random.uniform(-0.5, 0.5) * stats['std']
            # Use valid lookback
            lookback = random.choice(VALID_DELTA_LOOKBACKS)
            return DeltaGene(feature, operator, threshold, lookback)
            
        # 20% Chance of ZScore Gene (Statistical Extreme)
        elif rand_val < 0.60:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            # Thresholds usually -2, -1, 1, 2 sigmas
            threshold = random.choice([-3.0, -2.0, -1.5, 1.5, 2.0, 3.0])
            # Use valid window
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return ZScoreGene(feature, operator, threshold, window)
        
        # 40% Chance of Static Gene (Classic)
        else:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = stats['mean'] + random.uniform(-1.5, 1.5) * stats['std']
            return StaticGene(feature, operator, threshold)

    def create_random_gene(self):
        # Legacy fallback
        feature = random.choice(self.features)
        operator = random.choice(['>', '<'])
        stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
        threshold = stats['mean'] + random.uniform(-2, 2) * stats['std']
        return StaticGene(feature, operator, threshold)

    def create_strategy(self, num_genes_range=(2, 2)):
        # Force Structure: 1 Regime Gene + 1 Trigger Gene
        # Only supports 2 genes for now to enforce the Regime+Trigger pair
        
        long_genes = [self.create_gene_from_pool(self.regime_pool), self.create_gene_from_pool(self.trigger_pool)]
        short_genes = [self.create_gene_from_pool(self.regime_pool), self.create_gene_from_pool(self.trigger_pool)]
        
        return Strategy(
            name=f"Strat_{random.randint(1000,9999)}",
            long_genes=long_genes,
            short_genes=short_genes
        )