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
    ExtremaGene, ConsecutiveGene, DeltaGene, SeasonalityGene, gene_from_dict
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
                               'Vol_Ratio', 'news', 'panic', 'crisis', 'epu', 'total_vol']
        
        # Boost keywords for Physics/Microstructure (Alpha Refinement)
        self.boost_keywords = ['hurst', 'entropy', 'fdi', 'yang_zhang', 'kyle', 
                              'flow', 'ofi', 'imbalance', 'vpin', 'liquidation', 'force', 'shock', 'seasonal']
        
        self.update_pools()

    def update_pools(self):
        self.regime_pool = [f for f in self.features if any(k in f for k in self.regime_keywords)]
        self.trigger_pool = [f for f in self.features if f not in self.regime_pool]

    def set_stats(self, df):
        # If no features loaded (first run), populate from dataframe
        if not self.features:
            ignore_cols = {'open', 'high', 'low', 'close', 'volume', 'time_start', 'time_end', 
                          'time_hour', 'time_weekday', 'log_ret', 'target', 'symbol'}
            self.features = [c for c in df.columns if c not in ignore_cols and not c.startswith('metadata_')]
            self.update_pools()
            print(f"  GenomeFactory: Auto-detected {len(self.features)} features from Dataframe.")

        for f in self.features:
            if f in df.columns:
                self.feature_stats[f] = {'mean': df[f].mean(), 'std': df[f].std()}

    def _weighted_choice(self, pool):
        """Selects from pool with bias towards physics/microstructure."""
        if not pool: return None
        
        boosted = [f for f in pool if any(k in f for k in self.boost_keywords)]
        regular = [f for f in pool if f not in boosted]
        
        if not boosted: return random.choice(pool)
        if not regular: return random.choice(pool)
            
        # 60% chance to pick from boosted features
        if random.random() < 0.60:
            return random.choice(boosted)
        else:
            return random.choice(regular)

    def create_gene_from_pool(self, pool):
        if not pool: return self.create_random_gene() # Fallback
        
        rand_val = random.random()
        
        # 25% ZScore Gene (The "Super Gene" - Adaptive, Robust, Statistical)
        if rand_val < 0.25:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            # Relaxed Thresholds for Higher Frequency (1.25 sigma ~ 20% occurrence)
            threshold = random.choice([-1.5, -1.25, 1.25, 1.5])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return ZScoreGene(feature, operator, threshold, window)
        
        # 10% Soft ZScore Gene (Continuous Confidence)
        elif rand_val < 0.35:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-1.5, -1.0, 1.0, 1.5])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            slope = random.uniform(0.5, 2.0)
            return SoftZScoreGene(feature, operator, threshold, window, slope)

        # 10% Seasonality Gene (Time-Based Alpha)
        elif rand_val < 0.45:
             operator = random.choice(['>', '<'])
             threshold = random.choice([-1.5, -1.0, 1.0, 1.5])
             return SeasonalityGene(operator, threshold)

        # 10% Relational Gene (Context - "Is A > B?")
        elif rand_val < 0.55:
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
        elif rand_val < 0.50:
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
        elif rand_val < 0.60:
            # Correlation can be between ANY two features (that's the point)
            feature_left = self._weighted_choice(pool)
            feature_right = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-0.6, -0.4, 0.4, 0.6])
            window = random.choice(VALID_CORR_WINDOWS)
            return CorrelationGene(feature_left, feature_right, operator, threshold, window)

        # 5% Flux Gene (Acceleration)
        elif rand_val < 0.65:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = random.uniform(-0.1, 0.1) * stats['std']
            lag = random.choice(VALID_FLUX_LAGS)
            return FluxGene(feature, operator, threshold, lag)

        # 5% Efficiency Gene (Path)
        elif rand_val < 0.70:
            feature = self._weighted_choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.uniform(0.3, 0.8)
            window = random.choice(VALID_EFF_WINDOWS)
            return EfficiencyGene(feature, operator, threshold, window)

        # 5% Divergence Gene (Structure)
        elif rand_val < 0.75:
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

        # 10% Event Gene (Memory - "Did X happen recently?")
        elif rand_val < 0.85:
            feature = self._weighted_choice(pool)
            
            # Make Event Adaptive: Wrap in Z-Score
            # "Did Z-Score(Feature) > 2.0 happen in last 10 bars?"
            z_window = random.choice(VALID_ZSCORE_WINDOWS)
            adaptive_feature = f"zscore_{feature}_{z_window}"
            
            operator = random.choice(['>', '<'])
            threshold = random.choice([-1.5, -1.0, 1.0, 1.5]) # Relaxed Sigma
            window = random.choice(VALID_ZSCORE_WINDOWS) # Lookback for the event itself
            
            return EventGene(adaptive_feature, operator, threshold, window)

        # 5% Cross Gene (Event - "A crossed B")
        elif rand_val < 0.90:
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

        # 5% Regime Gene (Bounded Metrics)
        elif rand_val < 0.95:
            bounded_pool = [f for f in pool if 'hurst' in f or 'entropy' in f or 'fdi' in f or 'efficiency' in f]
            target_feature = self._weighted_choice(bounded_pool) if bounded_pool else self._weighted_choice(pool)
            
            op = random.choice(['>', '<'])
            stats = self.feature_stats.get(target_feature, {'mean': 0.5, 'std': 0.1})
            threshold = stats['mean'] + random.choice([-1, 0, 1]) * stats['std']
            window = random.randint(3, 8)
            return PersistenceGene(target_feature, op, threshold, window)

        # 3% Extrema Gene (Breakout)
        elif rand_val < 0.98:
            feature = self._weighted_choice(pool)
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

    def create_archetype_strategy(self, archetype_name):
        """Creates a structured strategy based on known market phenomena."""
        long_genes = []
        short_genes = []
        
        # Helper to find specific features
        def find_feat(keyword):
            candidates = [f for f in self.features if keyword in f]
            return random.choice(candidates) if candidates else random.choice(self.features)

        if archetype_name == "Trend":
            # Setup: High Hurst or Low FDI (Trending Regime)
            regime_feat = find_feat('hurst')
            long_genes.append(PersistenceGene(regime_feat, '>', 0.5, 5))
            short_genes.append(PersistenceGene(regime_feat, '>', 0.5, 5))
            
            # Trigger: Momentum / Breakout
            mom_feat = find_feat('close')
            long_genes.append(DeltaGene(mom_feat, '>', 0.0, 50)) # Price Up
            short_genes.append(DeltaGene(mom_feat, '<', 0.0, 50)) # Price Down
            
        elif archetype_name == "MeanRev":
            # Setup: Low Hurst or High FDI (Choppy Regime)
            regime_feat = find_feat('hurst')
            long_genes.append(PersistenceGene(regime_feat, '<', 0.5, 5))
            short_genes.append(PersistenceGene(regime_feat, '<', 0.5, 5))
            
            # Trigger: Overextended Price (Z-Score Reversion)
            z_feat = find_feat('close')
            long_genes.append(ZScoreGene(z_feat, '<', -2.0, 100)) # Oversold -> Buy
            short_genes.append(ZScoreGene(z_feat, '>', 2.0, 100)) # Overbought -> Sell

        elif archetype_name == "Breakout":
            # Setup: Volatility Compression (Squeeze)
            vol_feat = find_feat('volatility')
            long_genes.append(ZScoreGene(vol_feat, '<', -1.0, 50)) # Low Vol
            short_genes.append(ZScoreGene(vol_feat, '<', -1.0, 50))
            
            # Trigger: High Volume or Sharp Move
            vol_feat = find_feat('volume') # or trade count
            long_genes.append(ZScoreGene(vol_feat, '>', 2.0, 10)) # Volume Spike
            short_genes.append(ZScoreGene(vol_feat, '>', 2.0, 10))

        # Filler (Optional: 1 extra gene for flavor)
        if random.random() < 0.5:
             long_genes.append(self.create_gene_from_pool(self.trigger_pool))
             short_genes.append(self.create_gene_from_pool(self.trigger_pool))

        sl_pct = random.choice([1.5, 2.0, 2.5])
        tp_pct = random.choice([3.0, 4.0, 5.0, 6.0])

        return Strategy(
            name=f"{archetype_name}_{random.randint(1000,9999)}",
            long_genes=long_genes,
            short_genes=short_genes,
            min_concordance=len(long_genes), # Require ALL conditions for archetypes (Strict)
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct
        )

    def create_strategy(self, num_genes_range=(config.GENE_COUNT_MIN, config.GENE_COUNT_MAX)):
        # 30% Chance to use an Alpha Seed (Archetype)
        if random.random() < 0.30:
            atype = random.choice(["Trend", "MeanRev", "Breakout"])
            return self.create_archetype_strategy(atype)
            
        num_genes = random.randint(max(2, num_genes_range[0]), max(2, num_genes_range[1])) # Ensure at least 2 for Setup+Trigger
        
        long_genes = []
        short_genes = []
        
        # --- ORGANIC GATING ENFORCEMENT ---
        # 1. The Setup (Regime Gene)
        long_genes.append(self.create_gene_from_pool(self.regime_pool))
        short_genes.append(self.create_gene_from_pool(self.regime_pool))
        
        # 2. The Trigger (Action Gene)
        long_genes.append(self.create_gene_from_pool(self.trigger_pool))
        short_genes.append(self.create_gene_from_pool(self.trigger_pool))
        
        # 3. Filler (Random Mix)
        for _ in range(num_genes - 2):
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

        sl_pct = random.choice([1.5, 2.0, 2.5])
        tp_pct = random.choice([3.0, 4.0, 5.0, 6.0])

        return Strategy(
            name=f"Strat_{random.randint(1000,9999)}",
            long_genes=long_genes,
            short_genes=short_genes,
            min_concordance=concordance,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct
        )
