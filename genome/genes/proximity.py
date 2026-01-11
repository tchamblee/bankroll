import numpy as np
import pandas as pd
import random
from ..constants import VALID_ZSCORE_WINDOWS

class ProximityGene:
    """
    'Structure' Gene.
    Checks if a feature is close to its recent Max or Min (Support/Resistance).
    Logic: Abs(Val - Extrema) <= Threshold
    """
    def __init__(self, feature: str, mode: str, threshold: float, window: int):
        self.feature = feature
        self.mode = mode # 'max' or 'min'
        self.threshold = threshold # Absolute distance (or sigma if z-score)
        self.window = window
        self.type = 'proximity'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.mode, self.threshold, self.window)
            if key in cache: return cache[key]

        if self.feature not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[self.feature]
            s = pd.Series(data)
            
            if self.mode == 'max':
                ref_val = s.rolling(self.window).max().values
                dist = np.abs(data - ref_val)
            else: # min
                ref_val = s.rolling(self.window).min().values
                dist = np.abs(data - ref_val)
                
            res = dist <= self.threshold
            res = np.nan_to_num(res, nan=False).astype(bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)
        if random.random() < 0.3:
            # Mutate threshold by +/- 10%
            self.threshold = max(0.0001, self.threshold * random.choice([0.9, 1.1]))
        if random.random() < 0.3:
            self.mode = 'min' if self.mode == 'max' else 'max'
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return ProximityGene(self.feature, self.mode, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'mode': self.mode,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"Near {self.mode.upper()}({self.feature}, {self.window}) <= {self.threshold:.4f}"
