import numpy as np
import pandas as pd
import random
import json
import os
import config

# Strict grid to prevent overfitting
VALID_DELTA_LOOKBACKS = [5, 10, 20, 50, 100, 200]
VALID_ZSCORE_WINDOWS = [20, 50, 100, 200, 400]
VALID_SLOPE_WINDOWS = [10, 20, 50, 100]
VALID_CORR_WINDOWS = [20, 50, 100, 200]

def gene_from_dict(d):
    """Factory to restore gene from dictionary."""
    if d['type'] == 'static':
        return StaticGene(d['feature'], d['operator'], d['threshold'])
    elif d['type'] == 'relational':
        return RelationalGene(d['feature_left'], d['operator'], d['feature_right'])
    elif d['type'] == 'delta':
        return DeltaGene(d['feature'], d['operator'], d['threshold'], d['lookback'])
    elif d['type'] == 'zscore':
        return ZScoreGene(d['feature'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'slope':
        return SlopeGene(d['feature'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'correlation':
        return CorrelationGene(d['feature_left'], d['feature_right'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'time':
        return TimeGene(d['mode'], d['operator'], d['value'])
    elif d['type'] == 'consecutive':
        return ConsecutiveGene(d['direction'], d['operator'], d['count'])
    elif d['type'] == 'cross':
        return CrossGene(d['feature_left'], d['direction'], d['feature_right'])
    elif d['type'] == 'persistence':
        return PersistenceGene(d['feature'], d['operator'], d['threshold'], d['window'])
    elif d['type'] == 'squeeze':
        return SqueezeGene(d['feature_short'], d['feature_long'], d['multiplier'])
    elif d['type'] == 'range':
        return RangeGene(d['feature'], d['min_val'], d['max_val'])
    return None

class SqueezeGene:
    """
    'Compression' Gene.
    Checks if a short-term metric is compressed relative to a long-term one.
    Logic: Short < Long * Multiplier
    Example: vol_20 < vol_100 * 0.7 (Squeeze)
    """
    def __init__(self, feature_short: str, feature_long: str, multiplier: float):
        self.feature_short = feature_short
        self.feature_long = feature_long
        self.multiplier = multiplier
        self.type = 'squeeze'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature_short, self.feature_long, self.multiplier)
            if key in cache: return cache[key]

        if self.feature_short not in context or self.feature_long not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            s_data = context[self.feature_short]
            l_data = context[self.feature_long]
            res = s_data < (l_data * self.multiplier)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.multiplier += random.uniform(-0.1, 0.1)
            self.multiplier = max(0.1, min(1.5, self.multiplier))
        if random.random() < 0.3:
            self.feature_short = random.choice(features_pool)
        if random.random() < 0.3:
            self.feature_long = random.choice(features_pool)

    def copy(self):
        return SqueezeGene(self.feature_short, self.feature_long, self.multiplier)

    def to_dict(self):
        return {
            'type': self.type,
            'feature_short': self.feature_short,
            'feature_long': self.feature_long,
            'multiplier': self.multiplier
        }

    def __repr__(self):
        return f"{self.feature_short} < {self.multiplier:.2f} * {self.feature_long}"

class RangeGene:
    """
    'Zone' Gene.
    Checks if a feature is inside a specific value range.
    Logic: Min < Feature < Max
    """
    def __init__(self, feature: str, min_val: float, max_val: float):
        self.feature = feature
        self.min_val = min_val
        self.max_val = max_val
        self.type = 'range'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.min_val, self.max_val)
            if key in cache: return cache[key]

        if self.feature not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[self.feature]
            res = (data > self.min_val) & (data < self.max_val)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.4:
            # Shift the range
            shift = (self.max_val - self.min_val) * 0.1 * random.choice([-1, 1])
            self.min_val += shift
            self.max_val += shift
        if random.random() < 0.4:
            # Expand/Contract
            change = (self.max_val - self.min_val) * 0.1
            self.min_val -= change
            self.max_val += change
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return RangeGene(self.feature, self.min_val, self.max_val)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'min_val': self.min_val,
            'max_val': self.max_val
        }

    def __repr__(self):
        return f"{self.min_val:.2f} < {self.feature} < {self.max_val:.2f}"

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

class PersistenceGene:
    """
    'Filter' Gene.
    Checks if a condition has been True for N consecutive bars.
    Format: (Feature > Threshold) for Window bars
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.window = window
        self.type = 'persistence'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window)
            if key in cache: return cache[key]

        if self.feature not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[self.feature]
            # 1. Generate Boolean Mask
            if self.operator == '>': mask = data > self.threshold
            elif self.operator == '<': mask = data < self.threshold
            else: mask = np.zeros(len(data), dtype=bool)
            
            # 2. Check Persistence (Rolling Sum of Booleans == Window)
            # Use pandas for easy rolling, or convolution for speed?
            # Convolution is faster for pure numpy:
            # If we convolve mask with a kernel of ones=[1,1,...], output >= window means true
            
            # Standard conv approach for moving sum
            kernel = np.ones(self.window)
            # mode='full' then slice, or manual padding. 
            # Simple manual loop is slow. Let's use cumsum trick.
            
            # Cumsum trick: Sum[i] - Sum[i-w]. If diff == w, then all 1s.
            mask_int = mask.astype(int)
            csum = np.cumsum(mask_int)
            csum = np.insert(csum, 0, 0) # Pad start
            
            # diff[i] = csum[i+1] - csum[i+1-w]
            # We want res[i] corresponding to data[i]
            # rolling sum at i includes i, i-1 ... i-w+1
            
            rolling_sum = np.zeros(len(data), dtype=int)
            w = self.window
            if len(data) >= w:
                rolling_sum[w-1:] = csum[w:] - csum[:-w]
                
            res = rolling_sum == w
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        if random.random() < 0.3:
            self.window = max(2, self.window + random.choice([-1, 1, 2]))
        if random.random() < 0.3:
            self.threshold += random.uniform(-0.5, 0.5)
        if random.random() < 0.1:
            self.feature = random.choice(features_pool)

    def copy(self):
        return PersistenceGene(self.feature, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"({self.feature} {self.operator} {self.threshold:.2f}) FOR {self.window} BARS"

class StaticGene:
    """
    Classic 'Magic Number' Gene.
    Format: Feature <Operator> Threshold
    Example: frac_diff_02 > 0.45
    """
    def __init__(self, feature: str, operator: str, threshold: float):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.type = 'static'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        # Cache Key
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold)
            if key in cache: return cache[key]

        # Context is a dict of numpy arrays
        if self.feature not in context:
            # Fallback for safety, though precalc should handle this
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            data = context[self.feature]
            if self.operator == '>': res = data > self.threshold
            elif self.operator == '<': res = data < self.threshold
            else: res = np.zeros(len(data), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Threshold
        if random.random() < 0.5:
            change = self.threshold * 0.1 * (1 if random.random() > 0.5 else -1)
            # Handle near-zero thresholds
            if abs(self.threshold) < 0.001: 
                change = 0.001 * (1 if random.random() > 0.5 else -1)
            self.threshold += change
            
        # 2. Mutate Operator
        if random.random() < 0.2: 
            self.operator = '>' if self.operator == '<' else '<'
            
        # 3. Mutate Feature
        if random.random() < 0.1: 
            self.feature = random.choice(features_pool)

    def copy(self):
        return StaticGene(self.feature, self.operator, self.threshold)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold
        }

    def __repr__(self):
        return f"{self.feature} {self.operator} {self.threshold:.10f}"

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
            
        # 2. Mutate Left Feature
        if random.random() < 0.3: 
            self.feature_left = random.choice(features_pool)
            
        # 3. Mutate Right Feature
        if random.random() < 0.3: 
            self.feature_right = random.choice(features_pool)

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
        
        # 2. Mutate Threshold
        if random.random() < 0.3:
            change = self.threshold * 0.1 * (1 if random.random() > 0.5 else -1)
            if abs(self.threshold) < 0.001: 
                change = 0.001 * (1 if random.random() > 0.5 else -1)
            self.threshold += change
            
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

class SlopeGene:
    """
    'Trend' Gene.
    Checks the linear regression slope of a feature.
    Format: Slope(Feature, Window) <Operator> Threshold
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.window = window
        self.type = 'slope'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window)
            if key in cache: return cache[key]

        ctx_key = f"slope_{self.feature}_{self.window}"
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
            self.window = random.choice(VALID_SLOPE_WINDOWS)
        if random.random() < 0.3:
            self.threshold += random.uniform(-0.01, 0.01)
        if random.random() < 0.1: 
            self.feature = random.choice(features_pool)
        if random.random() < 0.2: 
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return SlopeGene(self.feature, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"Slope({self.feature}, {self.window}) {self.operator} {self.threshold:.5f}"

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
            self.threshold += random.uniform(-0.1, 0.1)
            self.threshold = max(-1.0, min(1.0, self.threshold))
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

class ZScoreGene:
    """
    'Statistical' Gene.
    """
    def __init__(self, feature: str, operator: str, threshold: float, window: int):
        self.feature = feature
        self.operator = operator
        self.threshold = threshold # Sigma value
        self.window = window
        self.type = 'zscore'

    def evaluate(self, context: dict, cache: dict = None) -> np.array:
        if cache is not None:
            key = (self.type, self.feature, self.operator, self.threshold, self.window)
            if key in cache: return cache[key]

        ctx_key = f"zscore_{self.feature}_{self.window}"
        if ctx_key not in context:
            res = np.zeros(context.get('__len__', 0), dtype=bool)
        else:
            z_score = context[ctx_key]
            if self.operator == '>': res = z_score > self.threshold
            elif self.operator == '<': res = z_score < self.threshold
            else: res = np.zeros(len(z_score), dtype=bool)
            
        if cache is not None: cache[key] = res
        return res

    def mutate(self, features_pool):
        # 1. Mutate Window (Strict Grid)
        if random.random() < 0.3:
            self.window = random.choice(VALID_ZSCORE_WINDOWS)
            
        # 2. Mutate Threshold (Sigma)
        if random.random() < 0.3:
            self.threshold += random.uniform(-0.5, 0.5)
            
        # 3. Mutate Feature
        if random.random() < 0.1: 
            self.feature = random.choice(features_pool)
            
        # 4. Mutate Operator
        if random.random() < 0.2: 
            self.operator = '>' if self.operator == '<' else '<'

    def copy(self):
        return ZScoreGene(self.feature, self.operator, self.threshold, self.window)

    def to_dict(self):
        return {
            'type': self.type,
            'feature': self.feature,
            'operator': self.operator,
            'threshold': self.threshold,
            'window': self.window
        }

    def __repr__(self):
        return f"Z({self.feature}, {self.window}) {self.operator} {self.threshold:.10f}σ"

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

class Strategy:
    """
    Represents a Bidirectional Trading Strategy.
    """
    def __init__(self, name="Strategy", long_genes=None, short_genes=None, min_concordance=None):
        self.name = name
        self.long_genes = long_genes if long_genes else []
        self.short_genes = short_genes if short_genes else []
        self.min_concordance = min_concordance
        self.fitness = 0.0
        
    def generate_signal(self, context: dict, cache: dict = None) -> np.array:
        n_rows = context.get('__len__', 0)
        if n_rows == 0 and len(context) > 0:
            for val in context.values():
                 if hasattr(val, 'shape'):
                     n_rows = val.shape[0]
                     break
        
        # Helper for Voting Logic
        def get_votes(genes):
            if not genes: return np.zeros(n_rows, dtype=int)
            votes = np.zeros(n_rows, dtype=int)
            for gene in genes:
                # Numpy handles bool -> int addition automatically (True=1, False=0)
                # This avoids allocating a temporary int array for every gene
                votes += gene.evaluate(context, cache)
            return votes

        l_votes = get_votes(self.long_genes)
        s_votes = get_votes(self.short_genes)
        
        # Concordance Logic
        l_thresh = self.min_concordance if self.min_concordance else len(self.long_genes)
        s_thresh = self.min_concordance if self.min_concordance else len(self.short_genes)
        
        if self.long_genes: l_thresh = max(1, min(l_thresh, len(self.long_genes)))
        if self.short_genes: s_thresh = max(1, min(s_thresh, len(self.short_genes)))
        
        go_long = l_votes >= l_thresh if self.long_genes else np.zeros(n_rows, dtype=bool)
        go_short = s_votes >= s_thresh if self.short_genes else np.zeros(n_rows, dtype=bool)
        
        net_signal = go_long.astype(int) - go_short.astype(int)
        return net_signal * config.MAX_LOTS

    def to_dict(self):
        return {
            'name': self.name,
            'long_genes': [g.to_dict() for g in self.long_genes],
            'short_genes': [g.to_dict() for g in self.short_genes],
            'min_concordance': self.min_concordance,
            'fitness': self.fitness
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(
            name=d.get('name', 'Strategy'),
            long_genes=[gene_from_dict(g) for g in d.get('long_genes', [])],
            short_genes=[gene_from_dict(g) for g in d.get('short_genes', [])],
            min_concordance=d.get('min_concordance')
        )
        s.fitness = d.get('fitness', 0.0)
        return s

    def __repr__(self):
        l_str = f" + ".join([str(g) for g in self.long_genes]) if self.long_genes else "None"
        s_str = f" + ".join([str(g) for g in self.short_genes]) if self.short_genes else "None"
        return f"[{self.name}] LONG:({l_str}) | SHORT:({s_str})"

class GenomeFactory:
    def __init__(self, survivors_file):
        with open(survivors_file, 'r') as f:
            self.features = json.load(f)
        self.feature_stats = {} 
        
        # Categorize Features for Gated Logic
        self.regime_keywords = ['hurst', 'volatility', 'efficiency', 'entropy', 'skew', 'trend_strength', 
                               'yang_zhang', 'lambda', 'force', 'fdi',
                               'Vol_Ratio', 'news', 'panic', 'crisis', 'epu', 'total_vol']
        self.regime_pool = [f for f in self.features if any(k in f for k in self.regime_keywords)]
        self.trigger_pool = [f for f in self.features if f not in self.regime_pool]
        
        # print(f"Factory Loaded: {len(self.regime_pool)} Regime Features | {len(self.trigger_pool)} Trigger Features")

    def set_stats(self, df):
        for f in self.features:
            if f in df.columns:
                self.feature_stats[f] = {'mean': df[f].mean(), 'std': df[f].std()}

    def create_gene_from_pool(self, pool):
        if not pool: return self.create_random_gene() # Fallback
        
        rand_val = random.random()
        
        # 5% Consecutive Gene (Pattern)
        if rand_val < 0.05:
            direction = random.choice(['up', 'down'])
            op = random.choice(['>', '=='])
            count = random.randint(2, 6)
            return ConsecutiveGene(direction, op, count)
            
        # 10% Persistence Gene (Filter)
        elif rand_val < 0.15:
            feature = random.choice(pool)
            op = random.choice(['>', '<'])
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = stats['mean'] + random.choice([-1, 0, 1]) * stats['std']
            window = random.randint(3, 10)
            return PersistenceGene(feature, op, threshold, window)

        # 10% Relational Gene (Context)
        elif rand_val < 0.25:
            feature_left = random.choice(pool)
            feature_right = random.choice(pool)
            while feature_right == feature_left and len(pool) > 1:
                feature_right = random.choice(pool)
            operator = random.choice(['>', '<'])
            return RelationalGene(feature_left, operator, feature_right)
            
        # 10% Cross Gene (Event)
        elif rand_val < 0.35:
            feature_left = random.choice(pool)
            feature_right = random.choice(pool)
            while feature_right == feature_left and len(pool) > 1:
                feature_right = random.choice(pool)
            direction = random.choice(['above', 'below'])
            return CrossGene(feature_left, direction, feature_right)

        # 10% Squeeze Gene (Compression)
        elif rand_val < 0.45:
            feature_short = random.choice(pool)
            feature_long = random.choice(pool)
            while feature_long == feature_short and len(pool) > 1:
                feature_long = random.choice(pool)
            multiplier = random.uniform(0.5, 0.95)
            return SqueezeGene(feature_short, feature_long, multiplier)
            
        # 5% Range Gene (Zone) - Reduced
        elif rand_val < 0.50:
            feature = random.choice(pool)
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            center = stats['mean'] + random.uniform(-1, 1) * stats['std']
            width = random.uniform(0.5, 2.0) * stats['std']
            return RangeGene(feature, center - width/2, center + width/2)
            
        # 15% Delta Gene (Momentum)
        elif rand_val < 0.65:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            stats = self.feature_stats.get(feature, {'mean': 0, 'std': 1})
            threshold = random.uniform(-0.5, 0.5) * stats['std']
            lookback = random.choice(VALID_DELTA_LOOKBACKS) 
            return DeltaGene(feature, operator, threshold, lookback)
            
        # 10% Slope Gene (Trend)
        elif rand_val < 0.75:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.uniform(-0.02, 0.02) # Small slope threshold
            window = random.choice(VALID_SLOPE_WINDOWS)
            return SlopeGene(feature, operator, threshold, window)
            
        # 10% Correlation Gene (Synergy)
        elif rand_val < 0.85:
            feature_left = random.choice(pool)
            feature_right = random.choice(pool)
            while feature_right == feature_left and len(pool) > 1:
                feature_right = random.choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-0.8, -0.5, 0.5, 0.8])
            window = random.choice(VALID_CORR_WINDOWS)
            return CorrelationGene(feature_left, feature_right, operator, threshold, window)
            
        # 15% ZScore Gene (Statistical Extreme)
        else:
            feature = random.choice(pool)
            operator = random.choice(['>', '<'])
            threshold = random.choice([-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0])
            window = random.choice(VALID_ZSCORE_WINDOWS)
            return ZScoreGene(feature, operator, threshold, window)

    def create_random_gene(self):
        # Fallback now uses ZScore instead of Static
        feature = random.choice(self.features)
        operator = random.choice(['>', '<'])
        threshold = random.choice([-2.0, 2.0])
        window = random.choice(VALID_ZSCORE_WINDOWS)
        return ZScoreGene(feature, operator, threshold, window)

    def create_strategy(self, num_genes_range=(3, 5)):
        num_genes = random.randint(num_genes_range[0], num_genes_range[1])
        long_genes = []
        short_genes = []
        
        for _ in range(num_genes):
            pool = self.regime_pool if random.random() < 0.5 else self.trigger_pool
            long_genes.append(self.create_gene_from_pool(pool))
            
        for _ in range(num_genes):
            pool = self.regime_pool if random.random() < 0.5 else self.trigger_pool
            short_genes.append(self.create_gene_from_pool(pool))
        
        # Concordance: For complex strategies, allow 1 outlier (Robustness)
        concordance = None
        if num_genes > 2:
            # Require ~70% agreement (Super-Majority)
            concordance = int(max(2, num_genes * 0.7))

        return Strategy(
            name=f"Strat_{random.randint(1000,9999)}",
            long_genes=long_genes,
            short_genes=short_genes,
            min_concordance=concordance
        )