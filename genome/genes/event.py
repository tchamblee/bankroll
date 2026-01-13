import numpy as np
import random
from ..constants import VALID_ZSCORE_WINDOWS, VALID_SIGMA_THRESHOLDS

class EventGene:
    """
    'Memory' Gene.
    Checks if a condition was True at least ONCE in the last N bars.
    Format: (Feature > Threshold) occurred in last Window bars
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.window = window
        self.type = 'event'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window)
            if key in cache: return cache[key]

        if self.feature not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[self.feature]
            # 1. Generate Boolean Mask
            if self.operator == '>': mask = data > self.threshold
            elif self.operator == '<': mask = data < self.threshold
            else: mask = np.zeros(len(data), dtype=bool)
            
            # 2. Check Event (Rolling Sum > 0)
            # Cumsum trick: Sum[i] - Sum[i-w] > 0
            mask_int = mask.astype(int)
            csum = np.cumsum(mask_int)
            csum = np.insert(csum, 0, 0) 
            
            rolling_sum = np.zeros(len(data), dtype=int)
            w = self.window
            if len(data) >= w:
                rolling_sum[w-1:] = csum[w:] - csum[:-w]
                
            res = rolling_sum > 0
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)
        if random.random() < 0.3:
            self.threshold = random.choice(VALID_SIGMA_THRESHOLDS)
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return EventGene(self.feature, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"Event({self.feature} {self.operator} {self.threshold:.2f} in last {self.window})"
