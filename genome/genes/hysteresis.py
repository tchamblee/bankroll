import numpy as np
import random
from numba import jit
from ..constants import VALID_ZSCORE_WINDOWS

@jit(nopython=True)
def _eval_hysteresis(feature_arr, price_arr, window, op_code, epsilon_pct):
    """
    Checks if current price is Higher/Lower than it was the last time
    the feature was at the current level (within epsilon).
    
    op_code: 1 for '>', -1 for '<'
    """
    n = len(feature_arr)
    res = np.zeros(n, dtype=np.bool_)
    
    for i in range(window, n):
        target_val = feature_arr[i]
        tol = abs(target_val * epsilon_pct)
        if tol < 1e-6: tol = 1e-6 # Minimum tolerance
        
        found_idx = -1
        # Scan backwards
        for k in range(1, window):
            if abs(feature_arr[i-k] - target_val) < tol:
                found_idx = i-k
                break
        
        if found_idx != -1:
            p_now = price_arr[i]
            p_prev = price_arr[found_idx]
            
            if op_code == 1: # Current > Prev
                res[i] = p_now > p_prev
            elif op_code == -1: # Current < Prev
                res[i] = p_now < p_prev
                
    return res

class HysteresisGene:
    """
    Path Dependency Gene.
    Checks price context relative to feature history.
    "Is Price higher now than the last time RSI was 70?"
    """
    def __init__(self, feature: str, operator: str, window: int):
        self.feature = feature
        self.operator = operator # '>' (Price Higher) or '<' (Price Lower)
        self.window = window
        self.type = 'hysteresis'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        key = (self.type, self.feature, self.operator, self.window)
        if cache is not None and key in cache:
            return cache[key]

        if self.feature not in context or 'close' not in context:
            return np.zeros(context.get('__len__', 0), dtype=bool)

        f_data = context[self.feature]
        p_data = context['close']
        
        op_code = 1 if self.operator == '>' else -1
        
        # JIT Eval
        res = _eval_hysteresis(f_data, p_data, self.window, op_code, 0.05)
        
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)
            
        if random.random() < 0.3:
            self.operator = '>' if self.operator == '<' else '<'
            
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return HysteresisGene(self.feature, self.operator, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'window': self.window
        }

    def __repr__(self):
        op_str = "Higher" if self.operator == '>' else "Lower"
        return f"Price is {op_str} vs last time {self.feature} was here (Win: {self.window})"
