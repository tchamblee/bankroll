import random
import json
import math
from .constants import (
    VALID_DELTA_LOOKBACKS,
    VALID_ZSCORE_WINDOWS,
    VALID_CORR_WINDOWS,
    VALID_FLUX_LAGS,
    VALID_EFF_WINDOWS,
    VALID_SLOPE_WINDOWS
)
from .genes import (
    ZScoreGene, RelationalGene, SqueezeGene, CorrelationGene, FluxGene,
    EfficiencyGene, DivergenceGene, EventGene, CrossGene, PersistenceGene,
    ExtremaGene, ConsecutiveGene, DeltaGene, gene_from_dict
)
from .strategy import Strategy

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
        
        # print(f"Factory Loaded: {len(self.regime_pool)} Regime Features | {len(self.trigger_pool)} Trigger Features")

    def set_stats(self, df):
        for f in self.features:
            if f in df.columns:
                self.feature_stats[f] = {'mean': df[f].mean(), 'std': df[f].std()}

    def create_gene_from_pool(self, pool):
        if not pool: return self.create_random_gene() # Fallback
        
        rand_val = random.random()
        
        # 30% ZScore Gene (The "Super Gene" - Adaptive, Robust, Statistical)
        if rand_val < 0.30:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-3.0, -2.5, -2.0, -1.5, 1.5, 2.0, 2.5, 3.0])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return ZScoreGene(feature, operator, threshold, window)

        # 15% Relational Gene (Context - "Is A > B?")
        elif rand_val < 0.45:
            feature_left = random.choice(pool)
            feature_right = random.choice(pool)
            while feature_right == feature_left and len(pool) > 1:
                feature_right = random.choice(pool)
            operator = random.choice(['>', '<'])
            return RelationalGene(feature_left, operator, feature_right)

        # 5% Squeeze Gene (Regime Detector - Volatility Compression)
        elif rand_val < 0.50:
            feature_short = random.choice(pool)
            feature_long = random.choice(pool)
            while feature_long == feature_short and len(pool) > 1:
                feature_long = random.choice(pool)
            multiplier = random.uniform(0.5, 0.95)
            return SqueezeGene(feature_short, feature_long, multiplier)

        # 10% Correlation Gene (Synergy - "Are A and B moving together?")
        elif rand_val < 0.60:
            feature_left = random.choice(pool)
            feature_right = random.choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-0.8, -0.5, 0.5, 0.8])
            window = random.choice(VALID_CORR_WINDOWS)
            return CorrelationGene(feature_left, feature_right, operator, threshold, window)

        # 5% Flux Gene (Acceleration)
        elif rand_val < 0.65:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = random.uniform(-0.1, 0.1) * stats['std']
            lag = random.choice(VALID_FLUX_LAGS)
            return FluxGene(feature, operator, threshold, lag)

        # 5% Efficiency Gene (Path)
        elif rand_val < 0.70:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.uniform(0.3, 0.8)
            window = random.choice(VALID_EFF_WINDOWS)
            return EfficiencyGene(feature, operator, threshold, window)

        # 5% Divergence Gene (Structure)
        elif rand_val < 0.75:
            f1 = random.choice(pool)
            f2 = random.choice(pool)
            while f1 == f2 and len(pool) > 1: f2 = random.choice(pool)
            window = random.choice(VALID_SLOPE_WINDOWS)
            return DivergenceGene(f1, f2, window)

        # 10% Event Gene (Memory - "Did X happen recently?")
        elif rand_val < 0.85:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = random.uniform(-1.0, 1.0) * stats['std'] + stats['mean']
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return EventGene(feature, operator, threshold, window)

        # 5% Cross Gene (Event - "A crossed B")
        elif rand_val < 0.90:
            feature_left = random.choice(pool)
            feature_right = random.choice(pool)
            while feature_right == feature_left and len(pool) > 1:
                feature_right = random.choice(pool)
            direction = random.choice(['above', 'below'])
            return CrossGene(feature_left, direction, feature_right)

        # 5% Regime Gene (Bounded Metrics)
        elif rand_val < 0.95:
            bounded_pool = [f for f in pool if 'hurst' in f or 'entropy' in f or 'fdi' in f or 'efficiency' in f]
            target_feature = random.choice(bounded_pool) if bounded_pool else random.choice(pool)
            
            op = random.choice(['>', '<'])
            stats = self.feature_stats.get(target_feature, {'mean': 0.5, 'std': 0.1})
            threshold = stats['mean'] + random.choice([-1, 0, 1]) * stats['std']
            window = random.randint(5, 20)
            return PersistenceGene(target_feature, op, threshold, window)

        # 3% Extrema Gene (Breakout)
        elif rand_val < 0.98:
            feature = random.choice(pool)
            mode = random.choice(['max', 'min'])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return ExtremaGene(feature, mode, window)
        
        # 2% Consecutive + Delta (Remaining)
        else:
            if random.random() < 0.5:
                direction = random.choice(['up', 'down'])
                op = random.choice(['>', '=='])
                count = random.randint(2, 6)
                return ConsecutiveGene(direction, op, count)
            else:
                feature = random.choice(pool)
                operator = random.choice(['>', '<'])
                stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
                threshold = random.uniform(-0.5, 0.5) * stats['std']
                lookback = random.choice(VALID_DELTA_LOOKBACKS) 
                return DeltaGene(feature, operator, threshold, lookback)

    def create_random_gene(self):
        # Fallback now uses ZScore instead of Static
        feature = random.choice(self.features)
        operator = random.choice(['>', '<'])
        threshold = random.choice([-2.0, 2.0])
        window = random.choice(VALID_ZSCORE_WINDOWS)
        return ZScoreGene(feature, operator, threshold, window)

    def create_strategy(self, num_genes_range=(2, 4)):
        num_genes = random.randint(num_genes_range[0], num_genes_range[1])
        long_genes = []
        short_genes = []
        regime_genes = []
        
        for _ in range(num_genes):
            pool = self.regime_pool if random.random() < 0.5 else self.trigger_pool
            long_genes.append(self.create_gene_from_pool(pool))
            
        for _ in range(num_genes):
            pool = self.regime_pool if random.random() < 0.5 else self.trigger_pool
            short_genes.append(self.create_gene_from_pool(pool))
            
        # --- REGIME GENE INJECTION (20% Chance) ---
        if random.random() < 0.20:
            # Create a Regime Gene (Gate)
            # Use Range or Persistence on a Regime Feature
            pool = self.regime_pool if self.regime_pool else self.features
            target_feature = random.choice(pool)
            
            stats = self.feature_stats.get(target_feature, {'mean': 0, 'std': 1})
            
            # Persistence Gene (e.g. Volatility > High for 10 bars)
            op = random.choice(['>', '<'])
            threshold = stats['mean'] + random.choice([-1, 1]) * stats['std']
            regime_genes.append(PersistenceGene(target_feature, op, threshold, 10))
        
        # Concordance: For complex strategies, allow 1 outlier (Robustness)
        concordance = None
        if num_genes > 2:
            # Require ~60% agreement (Super-Majority)
            # 3 genes -> 2 (0.6 * 3 = 1.8 -> ceil=2)
            # 4 genes -> 3 (0.6 * 4 = 2.4 -> ceil=3)
            # 5 genes -> 3 (0.6 * 5 = 3.0 -> ceil=3)
            concordance = math.ceil(num_genes * 0.6)

        return Strategy(
            name=f"Strat_{random.randint(1000,9999)}",
            long_genes=long_genes,
            short_genes=short_genes,
            regime_genes=regime_genes,
            min_concordance=concordance
        )
