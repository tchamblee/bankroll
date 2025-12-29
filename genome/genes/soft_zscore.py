import numpy as np
import random
from ..constants import VALID_ZSCORE_WINDOWS

class SoftZScoreGene:
    """
    'Soft' Statistical Gene.
    Returns a continuous confidence score [0.0, 1.0] instead of a binary boolean.
    Uses a sigmoid function centered at the threshold.
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int, slope: float = 1.0):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold # Sigma value where confidence is 0.5
        self.window = window
        self.slope = slope # Steepness of the sigmoid
        self.type = 'soft_zscore'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window, self.slope)
            if key in cache: return cache[key]

        ctx_key = f"zscore_{self.feature}_{self.window}"
        if ctx_key not in context:
            # Default to 0.0 (No confidence)
            res = np.zeros(context.get('__len__', 0), dtype=np.float32)
        else:
            z_score = context[ctx_key]
            
            # Distance from threshold
            if self.operator == '>':
                delta = z_score - self.threshold
            elif self.operator == '<':
                delta = self.threshold - z_score
            else:
                delta = np.zeros_like(z_score)

            # Sigmoid: 1 / (1 + exp(-slope * delta))
            # Optimization: clip delta to avoid overflow
            clipped_delta = np.clip(delta * self.slope, -10, 10)
            res = 1.0 / (1.0 + np.exp(-clipped_delta))
            
            # Map < 0.5 confidence to 0.0 to act like a filter? 
            # Or keep it fully continuous?
            # The Strategy class currently sums votes. 
            # If we return 0.01 it counts as 0.01 vote.
            # If we return 0.99 it counts as 0.99 vote.
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Window
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)
            
        # 2. Mutate Threshold
        if random.random() < 0.3:
            self.threshold += random.uniform(-0.5, 0.5)
            
        # 3. Mutate Slope
        if random.random() < 0.3:
            self.slope = np.clip(self.slope + random.uniform(-0.5, 0.5), 0.1, 5.0)

        # 4. Mutate Feature
        if random.random() < 0.1: 
            self.feature = random.choice(features_pool)
            
        # 5. Mutate Operator
        if random.random() < 0.2: 
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return SoftZScoreGene(self.feature, self.operator, self.threshold, self.window, self.slope)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window,
            'slope': self.slope
        }

    def __repr__(self):
        return f"SoftZ({self.feature}, {self.window}) {self.operator} {self.threshold:.2f}σ"
