import numpy as np
import random
from ..constants import VALID_SIGMA_THRESHOLDS

class SeasonalityGene:
    """
    'Seasonality' Gene.
    Exploits intraday seasonal patterns.
    Uses 'seasonal_deviation' feature (Z-Score of Price vs Expected Path).
    """
    def __init__(self, operator: str, threshold: float):
        self.feature = 'seasonal_deviation'
        self.operator = operator # '>' (Momentum) or '<' (Reversion) usually
        self.threshold = threshold # Sigma
        self.type = 'seasonality'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.operator, self.threshold)
            if key in cache: return cache[key]

        # Check if feature exists (it might not if pipeline failed)
        if self.feature not in context:
            # Fallback: Try to find any seasonal feature
            candidates = [k for k in context.keys() if 'seasonal_deviation' in k]
            if candidates:
                self.feature = candidates[0]
            else:
                return np.zeros(context.get('__len__', 0), dtype=bool)

        data = context[self.feature]
        
        if self.operator == '>': res = data > self.threshold
        elif self.operator == '<': res = data < self.threshold
        else: res = np.zeros(len(data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool=None):
        # 1. Mutate Threshold (Strict Grid)
        if random.random() < 0.5:
            self.threshold = random.choice(VALID_SIGMA_THRESHOLDS)

        # 2. Mutate Operator
        if random.random() < 0.3:
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return SeasonalityGene(self.operator, self.threshold)

    def to_dict(self):
        return {
            'type': self.type,
            'operator': self.operator,
            'threshold': self.threshold
        }

    def __repr__(self):
        return f"SeasDev {self.operator} {self.threshold:.2f}σ"
