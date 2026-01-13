import numpy as np
import random
from ..constants import VALID_ZSCORE_WINDOWS, VALID_SIGMA_THRESHOLDS

class ZScoreGene:
    """
    'Statistical' Gene.
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold # Sigma value
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
        # 1. Mutate Window (Strict Grid)
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)

        # 2. Mutate Threshold (Strict Grid - prevents magic numbers)
        if random.random() < 0.3:
            self.threshold = random.choice(VALID_SIGMA_THRESHOLDS)

        # 3. Mutate Feature
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

        # 4. Mutate Operator
        if random.random() < 0.2:
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return ZScoreGene(self.feature, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"Z({self.feature}, {self.window}) {self.operator} {self.threshold:.10f}σ"
