import numpy as np
import random

class RelationalGene:
    """
    Sophisticated 'Context' Gene.
    Format: Feature_A <Operator> Feature_B
    Example: volatility_50 > volatility_200 (Volatility Expansion)
    """
    def __init__(self, feature_left: str, operator: str, feature_right: str):
        self.feature_left = feature_left
        self.operator = operator
        self.feature_right = feature_right
        self.type = 'relational'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature_left, self.operator, self.feature_right)
            if key in cache: return cache[key]

        if self.feature_left not in context or self.feature_right not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            left_data = context[self.feature_left]
            right_data = context[self.feature_right]
            
            if self.operator == '>': res = left_data > right_data
            elif self.operator == '<': res = left_data < right_data
            else: res = np.zeros(len(left_data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Operator
        if random.random() < 0.3: 
            self.operator = '>' if self.operator == '<' else '<'
            
        # 2. Mutate Left Feature (and update Right to match)
        if random.random() < 0.3: 
            new_left = random.choice(features_pool)
            root = new_left.rsplit('_', 1)[0]
            compatible = [f for f in features_pool if f.startswith(root) and f != new_left]
            
            if compatible:
                self.feature_left = new_left
                self.feature_right = random.choice(compatible)
            
        # 3. Mutate Right Feature (Must match Left)
        if random.random() < 0.3: 
            root = self.feature_left.rsplit('_', 1)[0]
            compatible = [f for f in features_pool if f.startswith(root) and f != self.feature_left]
            
            if compatible:
                self.feature_right = random.choice(compatible)

    def copy(self):
        return RelationalGene(self.feature_left, self.operator, self.feature_right)

    def to_dict(self):
        return {
            'type': self.type,
            'feature_left': self.feature_left,
            'operator': self.operator,
            'feature_right': self.feature_right
        }

    def __repr__(self):
        return f"{self.feature_left} {self.operator} {self.feature_right}"
