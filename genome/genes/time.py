import numpy as np
import random

class TimeGene:
    """
    'Seasonality' Gene.
    Filters by Hour of Day or Day of Week.
    Type: 'hour' or 'weekday'
    """
    def __init__(self, mode: str, operator: str, value: int):
        self.mode = mode # 'hour' or 'weekday'
        self.operator = operator # '>' , '<', '==', '!='
        self.value = value
        self.type = 'time'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.mode, self.operator, self.value)
            if key in cache: return cache[key]

        ctx_key = f"time_{self.mode}"
        if ctx_key not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[ctx_key]
            if self.operator == '>': res = data > self.value
            elif self.operator == '<': res = data < self.value
            elif self.operator == '==': res = data == self.value
            elif self.operator == '!=': res = data != self.value
            else: res = np.zeros(len(data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool=None):
        # 1. Mutate Value
        if random.random() < 0.5:
            if self.mode == 'hour':
                self.value = (self.value + random.choice([-1, 1])) % 24
            else: # weekday
                self.value = (self.value + random.choice([-1, 1])) % 7
        
        # 2. Mutate Operator
        if random.random() < 0.3:
            self.operator = random.choice(['>', '<', '==', '!='])
            
    def copy(self):
        return TimeGene(self.mode, self.operator, self.value)

    def to_dict(self):
        return {
            'type': self.type,
            'mode': self.mode,
            'operator': self.operator,
            'value': self.value
        }

    def __repr__(self):
        return f"Time({self.mode}) {self.operator} {self.value}"
