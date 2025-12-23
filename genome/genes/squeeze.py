import numpy as np
import random

class SqueezeGene:
    """
    'Compression' Gene.
    Checks if a short-term metric is compressed relative to a long-term one.
    Logic: Short < Long * Multiplier
    Example: vol_20 < vol_100 * 0.7 (Squeeze)
    """
    def __init__(self, feature_short: str, feature_long: str, multiplier: float):
        self.feature_short = feature_short
        self.feature_long = feature_long
        self.multiplier = multiplier
        self.type = 'squeeze'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature_short, self.feature_long, self.multiplier)
            if key in cache: return cache[key]

        if self.feature_short not in context or self.feature_long not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            s_data = context[self.feature_short]
            l_data = context[self.feature_long]
            res = s_data < (l_data * self.multiplier)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.multiplier += random.uniform(-0.1, 0.1)
            self.multiplier = max(0.1, min(1.5, self.multiplier))
        if random.random() < 0.3:
            self.feature_short = random.choice(features_pool)
        if random.random() < 0.3:
            self.feature_long = random.choice(features_pool)

    def copy(self):
        return SqueezeGene(self.feature_short, self.feature_long, self.multiplier)

    def to_dict(self):
        return {
            'type': self.type,
            'feature_short': self.feature_short,
            'feature_long': self.feature_long,
            'multiplier': self.multiplier
        }

    def __repr__(self):
        return f"{self.feature_short} < {self.multiplier:.2f} * {self.feature_long}"
