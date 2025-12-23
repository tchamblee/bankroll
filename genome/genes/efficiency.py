import numpy as np
import random
from ..constants import VALID_EFF_WINDOWS

class EfficiencyGene:
    """
    'Physics' Gene. Measures Market Efficiency (Kaufman).
    ER = NetChange / TotalPath. 1.0 = Straight Line, 0.0 = Noise.
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.window = window
        self.type = 'efficiency'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window)
            if key in cache: return cache[key]

        ctx_key = f"eff_{self.feature}_{self.window}"
        if ctx_key not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[ctx_key]
            if self.operator == '>': res = data > self.threshold
            elif self.operator == '<': res = data < self.threshold
            else: res = np.zeros(len(data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3: self.window = random.choice(VALID_EFF_WINDOWS)
        if random.random() < 0.3: self.threshold = max(0.0, min(1.0, self.threshold + random.uniform(-0.1, 0.1)))
        if random.random() < 0.1: self.feature = random.choice(features_pool)
        if random.random() < 0.2: self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return EfficiencyGene(self.feature, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {'type': self.type, 'feature': self.feature, 'operator': self.operator, 'threshold': self.threshold, 'window': self.window}

    def __repr__(self):
        return f"Eff({self.feature}, {self.window}) {self.operator} {self.threshold:.2f}"
