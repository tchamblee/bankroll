import numpy as np
import random

class PersistenceGene:
    """
    'Filter' Gene.
    Checks if a condition has been True for N consecutive bars.
    Format: (Feature > Threshold) for Window bars
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.window = window
        self.type = 'persistence'

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
            
            # 2. Check Persistence (Rolling Sum of Booleans == Window)
            # Use pandas for easy rolling, or convolution for speed?
            # Convolution is faster for pure numpy:
            # If we convolve mask with a kernel of ones=[1,1,...], output >= window means true
            
            # Standard conv approach for moving sum
            kernel = np.ones(self.window)
            # mode='full' then slice, or manual padding. 
            # Simple manual loop is slow. Let's use cumsum trick.
            
            # Cumsum trick: Sum[i] - Sum[i-w]. If diff == w, then all 1s.
            mask_int = mask.astype(int)
            csum = np.cumsum(mask_int)
            csum = np.insert(csum, 0, 0) # Pad start
            
            # diff[i] = csum[i+1] - csum[i+1-w]
            # We want res[i] corresponding to data[i]
            # rolling sum at i includes i, i-1 ... i-w+1
            
            rolling_sum = np.zeros(len(data), dtype=int)
            w = self.window
            if len(data) >= w:
                rolling_sum[w-1:] = csum[w:] - csum[:-w]
                
            res = rolling_sum == w
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.window = random.choice([3, 5, 8, 10])
        # Note: threshold here is feature-specific, keep continuous but round to avoid magic numbers
        if random.random() < 0.3:
            self.threshold = round(self.threshold + random.uniform(-0.5, 0.5), 2)
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return PersistenceGene(self.feature, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"({self.feature} {self.operator} {self.threshold:.2f}) FOR {self.window} BARS"
