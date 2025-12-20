import numpy as np
import pandas as pd
import random
import json
import os

class Gene:
    """
    Represents a single trading rule (The Atom of the Strategy).
    Format: Feature <Operator> Threshold
    Example: frac_diff_02 > 0.45
    """
    def __init__(self, feature: str, operator: str, threshold: float):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold

    def evaluate(self, df: pd.DataFrame) -> np.array:
        if self.feature not in df.columns:
            return np.zeros(len(df), dtype=bool)
            
        data = df[self.feature].values
        
        if self.operator == '>': return data > self.threshold
        elif self.operator == '<': return data < self.threshold
        elif self.operator == '>=': return data >= self.threshold
        elif self.operator == '<=': return data <= self.threshold
        else: return np.zeros(len(df), dtype=bool)

    def mutate(self, features_pool):
        if random.random() < 0.5:
            change = self.threshold * 0.1 * (1 if random.random() > 0.5 else -1)
            if abs(self.threshold) < 0.001: change = 0.001 * (1 if random.random() > 0.5 else -1)
            self.threshold += change
        if random.random() < 0.2: self.operator = '>' if self.operator == '<' else '<'
        if random.random() < 0.1: self.feature = random.choice(features_pool)

    def copy(self):
        """Returns a deep copy of the Gene."""
        return Gene(self.feature, self.operator, self.threshold)

    def __repr__(self):
        return f"{self.feature} {self.operator} {self.threshold:.4f}"

class Strategy:
    """
    Represents a Bidirectional Trading Strategy.
    """
    def __init__(self, name="Strategy", long_genes=None, short_genes=None):
        self.name = name
        self.long_genes = long_genes if long_genes else []
        self.short_genes = short_genes if short_genes else []
        self.fitness = 0.0
        
    def generate_signal(self, df: pd.DataFrame) -> np.array:
        # Long Signal (AND logic)
        l_mask = np.ones(len(df), dtype=bool) if self.long_genes else np.zeros(len(df), dtype=bool)
        for gene in self.long_genes:
            l_mask &= gene.evaluate(df)
            
        # Short Signal (AND logic)
        s_mask = np.ones(len(df), dtype=bool) if self.short_genes else np.zeros(len(df), dtype=bool)
        for gene in self.short_genes:
            s_mask &= gene.evaluate(df)
            
        # Final: +1, -1, or 0 (Short priority or cancellation if both True)
        return l_mask.astype(int) - s_mask.astype(int)

    def __repr__(self):
        l_str = " AND ".join([str(g) for g in self.long_genes]) if self.long_genes else "None"
        s_str = " AND ".join([str(g) for g in self.short_genes]) if self.short_genes else "None"
        return f"[{self.name}] LONG:({l_str}) | SHORT:({s_str})"

class GenomeFactory:
    def __init__(self, survivors_file):
        with open(survivors_file, 'r') as f:
            self.features = json.load(f)
        self.feature_stats = {} 
        
        # Categorize Features for Gated Logic
        self.regime_keywords = ['hurst', 'volatility', 'efficiency', 'entropy', 'skew', 'trend_strength', 
                               'yang_zhang', 'lambda', 'force', 'fdi']
        self.regime_pool = [f for f in self.features if any(k in f for k in self.regime_keywords)]
        self.trigger_pool = [f for f in self.features if f not in self.regime_pool]
        
        print(f"Factory Loaded: {len(self.regime_pool)} Regime Features | {len(self.trigger_pool)} Trigger Features")

    def set_stats(self, df):
        for f in self.features:
            if f in df.columns:
                self.feature_stats[f] = {'mean': df[f].mean(), 'std': df[f].std()}

    def create_gene_from_pool(self, pool):
        if not pool: return self.create_random_gene() # Fallback
        feature = random.choice(pool)
        operator = random.choice(['>', '<'])
        stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
        threshold = stats['mean'] + random.uniform(-1.5, 1.5) * stats['std']
        return Gene(feature, operator, threshold)

    def create_random_gene(self):
        # Legacy fallback
        feature = random.choice(self.features)
        operator = random.choice(['>', '<'])
        stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
        threshold = stats['mean'] + random.uniform(-2, 2) * stats['std']
        return Gene(feature, operator, threshold)

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