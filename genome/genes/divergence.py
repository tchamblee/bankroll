import numpy as np
import random
from ..constants import VALID_SLOPE_WINDOWS

class DivergenceGene:
    """
    'Physics' Gene. Checks if two features are moving in opposite directions.
    True if Slope(A) * Slope(B) < 0 (i.e. signs are different).
    """
    def __init__(self, feature_a: str, feature_b: str, window: int):
        self.feature_a = feature_a
        self.feature_b = feature_b
        self.window = window
        self.type = 'divergence'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature_a, self.feature_b, self.window)
            if key in cache: return cache[key]

        k_a = f"slope_{self.feature_a}_{self.window}"
        k_b = f"slope_{self.feature_b}_{self.window}"
        
        if k_a not in context or k_b not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            s_a = context[k_a]
            s_b = context[k_b]
            # Divergence = One positive, one negative
            res = (s_a * s_b) < 0
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3: self.window = random.choice(VALID_SLOPE_WINDOWS)
        if random.random() < 0.2: self.feature_a = random.choice(features_pool)
        if random.random() < 0.2: self.feature_b = random.choice(features_pool)

    def copy(self):
        return DivergenceGene(self.feature_a, self.feature_b, self.window)

    def to_dict(self):
        return {'type': self.type, 'feature_a': self.feature_a, 'feature_b': self.feature_b, 'window': self.window}

    def __repr__(self):
        return f"Div({self.feature_a}, {self.feature_b}, {self.window})"
