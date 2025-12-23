import numpy as np
import random

class CrossGene:
    """
    'Event' Gene.
    Detects when Feature A crosses Feature B.
    Direction: 'above', 'below'
    """
    def __init__(self, feature_left: str, direction: str, feature_right: str):
        self.feature_left = feature_left
        self.direction = direction
        self.feature_right = feature_right
        self.type = 'cross'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature_left, self.direction, self.feature_right)
            if key in cache: return cache[key]

        if self.feature_left not in context or self.feature_right not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            s1 = context[self.feature_left]
            s2 = context[self.feature_right]
            
            # Current diff and Previous diff
            diff = s1 - s2
            prev_diff = np.roll(diff, 1)
            prev_diff[0] = 0 # Prevent wrap-around noise
            
            if self.direction == 'above':
                # Cross Above: Now > 0 AND Prev <= 0
                res = (diff > 0) & (prev_diff <= 0)
            elif self.direction == 'below':
                # Cross Below: Now < 0 AND Prev >= 0
                res = (diff < 0) & (prev_diff >= 0)
            else:
                res = np.zeros(len(s1), dtype=bool)
                
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.direction = 'below' if self.direction == 'above' else 'above'
        if random.random() < 0.3:
            self.feature_left = random.choice(features_pool)
        if random.random() < 0.3:
            self.feature_right = random.choice(features_pool)

    def copy(self):
        return CrossGene(self.feature_left, self.direction, self.feature_right)

    def to_dict(self):
        return {
            'type': self.type,
            'feature_left': self.feature_left,
            'direction': self.direction,
            'feature_right': self.feature_right
        }

    def __repr__(self):
        return f"{self.feature_left} CROSS {self.direction.upper()} {self.feature_right}"
