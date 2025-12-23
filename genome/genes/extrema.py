import numpy as np
import pandas as pd
import random
from ..constants import VALID_ZSCORE_WINDOWS

class ExtremaGene:
    """
    'Breakout' Gene.
    Checks if current value is the Max or Min of the last N bars.
    """
    def __init__(self, feature: str, mode: str, window: int):
        self.feature = feature
        self.mode = mode # 'max' or 'min'
        self.window = window
        self.type = 'extrema'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.mode, self.window)
            if key in cache: return cache[key]

        if self.feature not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[self.feature]
            s = pd.Series(data)
            
            if self.mode == 'max':
                rolling = s.rolling(self.window).max().values
                # Use isclose for float safety or >= for breakout logic
                res = data >= rolling 
            else: # min
                rolling = s.rolling(self.window).min().values
                res = data <= rolling
                
            res = np.nan_to_num(res, nan=False).astype(bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)
        if random.random() < 0.3:
            self.mode = 'min' if self.mode == 'max' else 'max'
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return ExtremaGene(self.feature, self.mode, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'mode': self.mode,
            'window': self.window
        }

    def __repr__(self):
        return f"{self.feature} IS {self.mode.upper()}({self.window})"
