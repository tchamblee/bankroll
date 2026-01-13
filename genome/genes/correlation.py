import numpy as np
import random
from ..constants import VALID_CORR_WINDOWS, VALID_CORR_THRESHOLDS

class CorrelationGene:
    """
    'Synergy' Gene.
    Checks rolling correlation between two features.
    Format: Corr(A, B, Window) <Operator> Threshold
    """
    def __init__(self, feature_left: str, feature_right: str, operator: str, threshold: float, window: int):
        self.feature_left = feature_left
        self.feature_right = feature_right
        self.operator = operator
        self.threshold = threshold
        self.window = window
        self.type = 'correlation'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature_left, self.feature_right, self.operator, self.threshold, self.window)
            if key in cache: return cache[key]

        f1, f2 = sorted([self.feature_left, self.feature_right])
        ctx_key = f"corr_{f1}_{f2}_{self.window}"
        
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
        if random.random() < 0.3:
            self.window = random.choice(VALID_CORR_WINDOWS)
        if random.random() < 0.3:
            self.threshold = random.choice(VALID_CORR_THRESHOLDS)
        if random.random() < 0.15:
            self.feature_left = random.choice(features_pool)
        if random.random() < 0.15:
            self.feature_right = random.choice(features_pool)
        if random.random() < 0.2:
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return CorrelationGene(self.feature_left, self.feature_right, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature_left': self.feature_left,
            'feature_right': self.feature_right,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"Corr({self.feature_left}, {self.feature_right}, {self.window}) {self.operator} {self.threshold:.2f}"
