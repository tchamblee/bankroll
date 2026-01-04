import random
import json
import math
import os
import config
from .constants import (
    VALID_DELTA_LOOKBACKS,
    VALID_ZSCORE_WINDOWS,
    VALID_CORR_WINDOWS,
    VALID_FLUX_LAGS,
    VALID_EFF_WINDOWS,
    VALID_SLOPE_WINDOWS
)
from .genes import (
    ZScoreGene, SoftZScoreGene, RelationalGene, SqueezeGene, CorrelationGene, FluxGene,
    EfficiencyGene, DivergenceGene, EventGene, CrossGene, PersistenceGene,
    ExtremaGene, ConsecutiveGene, DeltaGene, SeasonalityGene, MeanReversionGene, HysteresisGene, gene_from_dict
)
from .strategy import Strategy

class GenomeFactory:
    def __init__(self, survivors_file=None):
        self.features = []
        if survivors_file and os.path.exists(survivors_file):
            try:
                with open(survivors_file, 'r') as f:
                    content = json.load(f)
                    # basic check if it's a list of strings (features) or dicts (strategies)
                    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], str):
                        self.features = content
            except Exception as e:
                print(f"⚠️ Warning: Could not load features from {survivors_file}: {e}")

        self.feature_stats = {} 
        
        # Categorize Features for Gated Logic
        self.regime_keywords = ['hurst', 'volatility', 'efficiency', 'entropy', 'skew', 'trend_strength', 
                               'yang_zhang', 'lambda', 'force', 'fdi',
                               'Vol_Ratio', 'news', 'panic', 'crisis', 'epu', 'total_vol', 'premium']
        
        self.update_pools()

    def update_pools(self):
        # self.features is presumed to be sorted by IC (descending) from purge_features.py
        # Creating subsets using list comprehension preserves this order.
        self.regime_pool = [f for f in self.features if any(k in f for k in self.regime_keywords)]
        self.trigger_pool = [f for f in self.features if f not in self.regime_pool]

    def set_stats(self, df):
        # If no features loaded (first run), populate from dataframe
        if not self.features:
            ignore_cols = {'open', 'high', 'low', 'close', 'volume', 'time_start', 'time_end', 
                          'time_hour', 'time_weekday', 'log_ret', 'target', 'symbol', 'vix_close', 'evz_close'}
            self.features = [c for c in df.columns if c not in ignore_cols and not c.startswith('metadata_')]
            self.update_pools()

        for f in self.features:
            if f in df.columns:
                self.feature_stats[f] = {'mean': df[f].mean(), 'std': df[f].std()}

    def _weighted_choice(self, pool):
        """
        Selects from pool using Rank-Based Probability.
        Assumes 'pool' is sorted by quality (Best -> Worst), which is true
        if inherited from the IC-sorted 'survivors_*.json'.
        
        Uses a Triangular Distribution biased towards index 0 (Top Rank).
        This removes human bias ('boost_keywords') while favoring statistically strong features.
        The 'Long Tail' of weaker features is still accessible for diversity.
        """
        if not pool: return None
        
        # random.triangular(low, high, mode)
        # Mode 0 biases heavily towards the top of the list.
        # Range [0, len(pool)] -> casting to int gives indices 0..len-1
        idx = int(random.triangular(0, len(pool), 0))
        
        # Safety clamp
        idx = min(idx, len(pool) - 1)
        
        return pool[idx]

    def create_gene_from_pool(self, pool):
        if not pool: return self.create_random_gene() # Fallback
        
        rand_val = random.random()
        
        # 15% Mean Reversion Gene (New - Volatility Risk Premium Gated)
        if rand_val < 0.15:
            trigger = self._weighted_choice(self.trigger_pool) if self.trigger_pool else self._weighted_choice(pool)
            
            # Find a Premium feature if possible
            prem_feats = [f for f in self.regime_pool if 'premium' in f]
            if prem_feats:
                regime = random.choice(prem_feats)
                reg_thresh = 0.0 # Premium > 0 = Fear
            else:
                regime = self._weighted_choice(self.regime_pool) if self.regime_pool else self._weighted_choice(pool)
                # Generic regime threshold
                stats = self.feature_stats.get(regime, {'mean': 0, 'std': 1})
                reg_thresh = stats['mean'] + random.choice([-0.5, 0.5]) * stats['std']
            
            threshold = random.choice([2.0, 2.5, 3.0]) # High Sigma for Fade
            direction = random.choice(['long', 'short'])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            
            return MeanReversionGene(trigger, regime, threshold, reg_thresh, direction, window)

        # 20% ZScore Gene (The "Super Gene" - Adaptive, Robust, Statistical)
        elif rand_val < 0.35:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            # Relaxed Thresholds for Higher Frequency (1.25 sigma ~ 20% occurrence)
            threshold = random.choice([-1.5, -1.25, 1.25, 1.5])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return ZScoreGene(feature, operator, threshold, window)
        
        # 10% Soft ZScore Gene (Continuous Confidence)
        elif rand_val < 0.45:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-1.5, -1.0, 1.0, 1.5])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            slope = random.uniform(0.5, 2.0)
            return SoftZScoreGene(feature, operator, threshold, window, slope)

        # 5% Seasonality Gene (Time-Based Alpha)
        elif rand_val < 0.50:
             operator = random.choice(['>', '<'])
             threshold = random.choice([-1.5, -1.0, 1.0, 1.5])
             return SeasonalityGene(operator, threshold)

        # 10% Relational Gene (Context - "Is A > B?")
        elif rand_val < 0.60:
            feature_left = self._weighted_choice(pool)
            # Find compatible features (same root)
            # e.g. 'volatility_100' compatible with 'volatility_200'
            root = feature_left.rsplit('_', 1)[0]
            compatible = [f for f in pool if f.startswith(root) and f != feature_left]
            
            if compatible:
                feature_right = random.choice(compatible)
                operator = random.choice(['>', '<'])
                return RelationalGene(feature_left, operator, feature_right)
            else:
                # Fallback to ZScore if no compatible comparison found
                return self.create_random_gene()

        # 5% Squeeze Gene (Regime Detector - Volatility Compression)
        elif rand_val < 0.65:
            feature_short = self._weighted_choice(pool)
            # Squeeze needs compatible long-term feature
            root = feature_short.rsplit('_', 1)[0]
            compatible = [f for f in pool if f.startswith(root) and f != feature_short]
            
            if compatible:
                feature_long = random.choice(compatible)
                multiplier = random.uniform(0.5, 0.95)
                return SqueezeGene(feature_short, feature_long, multiplier)
            else:
                return self.create_random_gene()

        # 10% Correlation Gene (Synergy - "Are A and B moving together?")
        elif rand_val < 0.75:
            # Correlation can be between ANY two features (that's the point)
            feature_left = self._weighted_choice(pool)
            feature_right = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-0.6, -0.4, 0.4, 0.6])
            window = random.choice(VALID_CORR_WINDOWS)
            return CorrelationGene(feature_left, feature_right, operator, threshold, window)

        # 5% Flux Gene (Acceleration)
        elif rand_val < 0.80:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = random.uniform(-0.1, 0.1) * stats['std']
            lag = random.choice(VALID_FLUX_LAGS)
            return FluxGene(feature, operator, threshold, lag)

        # 5% Efficiency Gene (Path)
        elif rand_val < 0.85:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.uniform(0.3, 0.8)
            window = random.choice(VALID_EFF_WINDOWS)
            return EfficiencyGene(feature, operator, threshold, window)

        # 5% Divergence Gene (Structure)
        elif rand_val < 0.90:
            # Divergence needs compatible features (Price vs Oscillator usually, or Price vs Price)
            # Simplified: Random pair is risky. Let's restrict to same root.
            f1 = self._weighted_choice(pool)
            root = f1.rsplit('_', 1)[0]
            compatible = [f for f in pool if f.startswith(root) and f != f1]
            
            if compatible:
                f2 = random.choice(compatible)
                window = random.choice(VALID_SLOPE_WINDOWS)
                return DivergenceGene(f1, f2, window)
            else:
                return self.create_random_gene()

        # 3% Event Gene (Memory - "Did X happen recently?")
        elif rand_val < 0.93:
            feature = self._weighted_choice(pool)
            
            # Make Event Adaptive: Wrap in Z-Score
            # "Did Z-Score(Feature) > 2.0 happen in last 10 bars?"
            z_window = random.choice(VALID_ZSCORE_WINDOWS)
            adaptive_feature = f"zscore_{feature}_{z_window}"
            
            operator = random.choice(['>', '<'])
            threshold = random.choice([-1.5, -1.0, 1.0, 1.5]) # Relaxed Sigma
            window = random.choice(VALID_ZSCORE_WINDOWS) # Lookback for the event itself
            
            return EventGene(adaptive_feature, operator, threshold, window)

        # 2% Cross Gene (Event - "A crossed B")
        elif rand_val < 0.95:
            feature_left = self._weighted_choice(pool)
            # Enforce Compatibility
            root = feature_left.rsplit('_', 1)[0]
            compatible = [f for f in pool if f.startswith(root) and f != feature_left]
            
            if compatible:
                feature_right = random.choice(compatible)
                direction = random.choice(['above', 'below'])
                return CrossGene(feature_left, direction, feature_right)
            else:
                return self.create_random_gene()

        # 3% Hysteresis Gene (Path Dependency - "Is Price > Price when Feature was last here?")
        elif rand_val < 0.98:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return HysteresisGene(feature, operator, window)

        # Remaining: Persistence, Extrema, Consecutive, Delta
        else:
            dice = random.random()
            if dice < 0.25:
                bounded_pool = [f for f in pool if 'hurst' in f or 'entropy' in f or 'fdi' in f or 'efficiency' in f]
                target_feature = self._weighted_choice(bounded_pool) if bounded_pool else self._weighted_choice(pool)
                
                op = random.choice(['>', '<'])
                stats = self.feature_stats.get(target_feature, {'mean': 0.5, 'std': 0.1})
                threshold = stats['mean'] + random.choice([-1, 0, 1]) * stats['std']
                window = random.randint(3, 8)
                return PersistenceGene(target_feature, op, threshold, window)
            elif dice < 0.50:
                feature = self._weighted_choice(pool)
                mode = random.choice(['max', 'min'])
                window = random.choice(VALID_ZSCORE_WINDOWS)
                return ExtremaGene(feature, mode, window)
            elif dice < 0.75:
                direction = random.choice(['up', 'down'])
                op = random.choice(['>', '=='])
                count = random.randint(2, 6)
                return ConsecutiveGene(direction, op, count)
            else:
                feature = self._weighted_choice(pool)
                operator = random.choice(['>', '<'])
                stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
                threshold = random.uniform(-0.5, 0.5) * stats['std']
                lookback = random.choice(VALID_DELTA_LOOKBACKS) 
                return DeltaGene(feature, operator, threshold, lookback)

    def create_random_gene(self):
        # Fallback now uses ZScore instead of Static
        feature = self._weighted_choice(self.features)
        operator = random.choice(['>', '<'])
        # Lower thresholds to encourage activity (1.5 sigma instead of 2.0)
        threshold = random.choice([-1.5, 1.5, 2.0])
        window = random.choice(VALID_ZSCORE_WINDOWS)
        return ZScoreGene(feature, operator, threshold, window)


    def create_strategy(self, num_genes_range=(config.GENE_COUNT_MIN, config.GENE_COUNT_MAX)):
        # print("DEBUG: Creating strategy...") 
        num_genes = random.randint(max(2, num_genes_range[0]), max(2, num_genes_range[1])) # Ensure at least 2 for Setup+Trigger
        
        long_genes = []
        short_genes = []
        
        # --- ORGANIC GATING ENFORCEMENT ---
        # 1. The Setup (Regime Gene)
        # print("DEBUG: Creating Setup Gene")
        long_genes.append(self.create_gene_from_pool(self.regime_pool))
        short_genes.append(self.create_gene_from_pool(self.regime_pool))
        
        # 2. The Trigger (Action Gene)
        # print("DEBUG: Creating Trigger Gene")
        long_genes.append(self.create_gene_from_pool(self.trigger_pool))
        short_genes.append(self.create_gene_from_pool(self.trigger_pool))
        
        # 3. Filler (Random Mix)
        for _ in range(num_genes - 2):
            # print("DEBUG: Creating Filler Gene")
            pool = self.regime_pool if random.random() < 0.5 else self.trigger_pool
            long_genes.append(self.create_gene_from_pool(pool))
            
            pool = self.regime_pool if random.random() < 0.5 else self.trigger_pool
            short_genes.append(self.create_gene_from_pool(pool))
        
        # Concordance: Majority Rule
        concordance = None
        if num_genes <= 2:
            concordance = 2 # Require BOTH (Setup + Trigger) for small strategies
        else:
            concordance = math.ceil(num_genes * 0.51)

        sl_pct = random.choice(config.STOP_LOSS_OPTIONS)
        tp_pct = random.choice(config.TAKE_PROFIT_OPTIONS)

        return Strategy(
            name=f"Strat_{random.randint(1000,9999)}",
            long_genes=long_genes,
            short_genes=short_genes,
            min_concordance=concordance,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct
        )
