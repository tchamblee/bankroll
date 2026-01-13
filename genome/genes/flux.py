import numpy as np
import random
from ..constants import VALID_FLUX_LAGS

class FluxGene:
    """
    'Physics' Gene. Measures Acceleration (Second Derivative).
    Flux = (Val[t] - Val[t-L]) - (Val[t-L] - Val[t-2L])
    """
    def __init__(self, feature: str, operator: str, threshold: float, lag: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.lag = lag
        self.type = 'flux'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.lag)
            if key in cache: return cache[key]

        ctx_key = f"flux_{self.feature}_{self.lag}"
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
        if random.random() < 0.3: self.lag = random.choice(VALID_FLUX_LAGS)
        # Note: threshold here is feature-specific, keep continuous but round to avoid magic numbers
        if random.random() < 0.3: self.threshold = round(self.threshold + random.uniform(-0.1, 0.1), 4)
        if random.random() < 0.1: self.feature = random.choice(features_pool)
        if random.random() < 0.2: self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return FluxGene(self.feature, self.operator, self.threshold, self.lag)

    def to_dict(self):
        return {'type': self.type, 'feature': self.feature, 'operator': self.operator, 'threshold': self.threshold, 'lag': self.lag}

    def __repr__(self):
        return f"Flux({self.feature}, {self.lag}) {self.operator} {self.threshold:.4f}"
