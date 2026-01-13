import numpy as np
import random
from ..constants import VALID_PERCENTAGE_THRESHOLDS

class ValidityGene:
    """
    'Density' Gene.
    Checks if a condition was true for a percentage of the last N bars.
    Logic: Count(True) / Window >= Percentage
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int, percentage: float):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.window = window
        self.percentage = percentage # 0.0 to 1.0
        self.type = 'validity'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window, self.percentage)
            if key in cache: return cache[key]

        if self.feature not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[self.feature]
            # 1. Mask
            if self.operator == '>': mask = data > self.threshold
            elif self.operator == '<': mask = data < self.threshold
            else: mask = np.zeros(len(data), dtype=bool)
            
            # 2. Rolling Sum (Cumsum trick)
            mask_int = mask.astype(int)
            csum = np.cumsum(mask_int)
            csum = np.insert(csum, 0, 0)
            
            rolling_sum = np.zeros(len(data), dtype=int)
            w = self.window
            if len(data) >= w:
                rolling_sum[w-1:] = csum[w:] - csum[:-w]
            
            # 3. Check Density
            required_count = int(w * self.percentage)
            res = rolling_sum >= required_count
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.window = random.choice([10, 20, 30, 50])
        if random.random() < 0.3:
            self.percentage = random.choice(VALID_PERCENTAGE_THRESHOLDS)
        # Note: threshold here is feature-specific, keep continuous but round to avoid magic numbers
        if random.random() < 0.3:
            self.threshold = round(self.threshold + random.uniform(-0.1, 0.1), 2)
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return ValidityGene(self.feature, self.operator, self.threshold, self.window, self.percentage)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window,
            'percentage': self.percentage
        }

    def __repr__(self):
        return f"Validity({self.feature} {self.operator} {self.threshold:.2f}) >= {self.percentage*100:.0f}% of {self.window}"
