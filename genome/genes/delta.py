import numpy as np
import random
from ..constants import VALID_DELTA_LOOKBACKS

class DeltaGene:
    """
    'Momentum' Gene.
    Checks the change in a feature over time.
    Format: Delta(Feature, Lookback) <Operator> Threshold
    """
    def __init__(self, feature: str, operator: str, threshold: float, lookback: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.lookback = lookback
        self.type = 'delta'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.lookback)
            if key in cache: return cache[key]

        ctx_key = f"delta_{self.feature}_{self.lookback}"
        if ctx_key not in context:
            # Fallback/Lazy Compute if not in context
            # NOTE: BacktestEngine must be updated to support Just-In-Time computation or population scanning
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[ctx_key]
            if self.operator == '>': res = data > self.threshold
            elif self.operator == '<': res = data < self.threshold
            else: res = np.zeros(len(data), dtype=bool)
        
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Lookback (Strict Grid)
        if random.random() < 0.3:
            self.lookback = random.choice(VALID_DELTA_LOOKBACKS)

        # 2. Mutate Threshold (round to avoid magic numbers)
        if random.random() < 0.3:
            change = self.threshold * 0.1 * (1 if random.random() > 0.5 else -1)
            if abs(self.threshold) < 0.001:
                change = 0.001 * (1 if random.random() > 0.5 else -1)
            self.threshold = round(self.threshold + change, 6)

        # 3. Mutate Feature
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

        # 4. Mutate Operator
        if random.random() < 0.2:
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return DeltaGene(self.feature, self.operator, self.threshold, self.lookback)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'lookback': self.lookback
        }

    def __repr__(self):
        return f"Delta({self.feature}, {self.lookback}) {self.operator} {self.threshold:.10f}"
