import numpy as np
import random

class ConsecutiveGene:
    """
    'Pattern' Gene.
    Checks for consecutive bars of a certain type (Up/Down).
    """
    def __init__(self, direction: str, operator: str, count: int):
        self.direction = direction # 'up' or 'down'
        self.operator = operator # '>', '=='
        self.count = count
        self.type = 'consecutive'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.direction, self.operator, self.count)
            if key in cache: return cache[key]

        ctx_key = f"consecutive_{self.direction}"
        if ctx_key not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[ctx_key]
            if self.operator == '>': res = data > self.count
            elif self.operator == '==': res = data == self.count
            else: res = np.zeros(len(data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool=None):
        # 1. Mutate Count
        if random.random() < 0.5:
            self.count = max(1, self.count + random.choice([-1, 1]))
            
        # 2. Mutate Direction
        if random.random() < 0.2:
            self.direction = 'up' if self.direction == 'down' else 'down'
            
    def copy(self):
        return ConsecutiveGene(self.direction, self.operator, self.count)
    
    def to_dict(self):
        return {
            'type': self.type,
            'direction': self.direction,
            'operator': self.operator,
            'count': self.count
        }

    def __repr__(self):
        return f"Consecutive({self.direction}) {self.operator} {self.count}"
