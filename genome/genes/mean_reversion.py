import numpy as np
import random
from ..constants import VALID_ZSCORE_WINDOWS, VALID_SIGMA_THRESHOLDS_POSITIVE, VALID_SIGMA_THRESHOLDS

class MeanReversionGene:
    """
    Regime-Gated Mean Reversion Gene.
    Logic: Enter ONLY when volatility is expensive (Regime > Thresh) AND price is extreme (ZScore > Thresh).
    """
    def __init__(self, feature: str, regime_feature: str, threshold: float, regime_threshold: float, direction: str, window: int):
        self.feature = feature
        self.regime_feature = regime_feature
        self.threshold = abs(threshold) # Magnitude (Sigma)
        self.regime_threshold = regime_threshold
        self.direction = direction # 'long' (Fade Low) or 'short' (Fade High)
        self.window = window
        self.type = 'mean_reversion'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.regime_feature, self.threshold, self.regime_threshold, self.direction, self.window)
            if key in cache: return cache[key]

        # 1. Get Z-Score of Feature
        z_key = f"zscore_{self.feature}_{self.window}"
        
        # 2. Get Regime Feature (Assumed to be pre-calculated in context)
        # e.g. 'vol_premium_z_2000'
        reg_key = self.regime_feature
        
        if z_key not in context or reg_key not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            z_vals = context[z_key]
            reg_vals = context[reg_key]
            
            # Regime Condition: Is Volatility Expensive?
            # (or whatever the regime feature implies, > thresh)
            regime_condition = reg_vals > self.regime_threshold
            
            if self.direction == 'long':
                # Fade Low: Z < -Threshold
                trigger = z_vals < -self.threshold
            else:
                # Fade High: Z > Threshold
                trigger = z_vals > self.threshold
                
            res = regime_condition & trigger
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool, regime_pool=None):
        # 1. Mutate Window
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)

        # 2. Mutate Thresholds (Strict Grid)
        if random.random() < 0.3:
            self.threshold = random.choice(VALID_SIGMA_THRESHOLDS_POSITIVE)

        if random.random() < 0.3:
            self.regime_threshold = random.choice(VALID_SIGMA_THRESHOLDS)

        # 3. Mutate Feature
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

        # 4. Mutate Regime Feature
        if regime_pool and random.random() < 0.1:
            self.regime_feature = random.choice(regime_pool)

    def copy(self):
        return MeanReversionGene(self.feature, self.regime_feature, self.threshold, self.regime_threshold, self.direction, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'regime_feature': self.regime_feature,
            'threshold': self.threshold,
            'regime_threshold': self.regime_threshold,
            'direction': self.direction,
            'window': self.window
        }

    def __repr__(self):
        dir_sym = "📉 BuyDip" if self.direction == 'long' else "📈 SellRip"
        return f"MR({self.feature}, {self.window}) {dir_sym} [Z>{self.threshold:.1f}σ | {self.regime_feature}>{self.regime_threshold:.1f}]"
