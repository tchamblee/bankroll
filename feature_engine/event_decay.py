import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def _jit_bars_since_true(bool_arr):
    """
    Counts the number of bars since the last True value in the boolean array.
    Returns array of integers.
    If True at index i, result[i] = 0.
    If False, result[i] = result[i-1] + 1.
    """
    n = len(bool_arr)
    res = np.full(n, np.nan) # Initialize with NaN (or -1 or huge number)
    
    # Track the index of the last True
    last_true_idx = -1
    
    for i in range(n):
        if bool_arr[i]:
            last_true_idx = i
            res[i] = 0
        else:
            if last_true_idx != -1:
                res[i] = i - last_true_idx
                
    return res

def add_event_decay_features(df, high_windows=[100, 200, 400], shock_windows=[100], sigma_threshold=3.0):
    """
    Adds 'Time Since Last Event' features (Alpha Decay / Burst Features).
    
    1. bars_since_high_{w}: Bars since a new high in window w.
    2. bars_since_shock_{w}: Bars since a >3 sigma return (using volatility_{w}).
    """
    if df is None: return None
    df = df.copy()
    
    # Ensure log_ret exists
    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
    print("Calculating Event Decay Features (Burst Features)...")
    
    # 1. Bars Since High
    for w in high_windows:
        # Rolling Max High
        roll_max = df['high'].rolling(w).max()
        
        # Identify new highs (where High >= RollingMax)
        # Note: Since roll_max includes current bar, High[t] can equal RollMax[t]
        is_high = (df['high'] >= roll_max - 1e-9) 
        
        # Calculate decay
        decay = _jit_bars_since_true(is_high.values)
        
        # Normalize?
        # Raw bars count is good, but maybe log(bars) or bars/window is better for ML.
        # The prompt asks for "bars_since_high". We'll provide the raw count.
        # But for stability, we might want to fill NaNs (start of series) with window size or -1.
        # Let's keep NaNs for now, or fill with w.
        
        df[f'bars_since_high_{w}'] = decay
        
    # 2. Bars Since Shock
    # "Shock" = |Return| > 3 * Sigma
    abs_ret = df['log_ret'].abs()
    
    for w in shock_windows:
        vol_col = f'volatility_{w}'
        
        # If volatility feature not pre-calculated, calculate it on the fly
        if vol_col in df.columns:
            sigma = df[vol_col]
        else:
            # Fallback: Simple Rolling Std of returns
            sigma = df['log_ret'].rolling(w).std()
            
        is_shock = (abs_ret > sigma_threshold * sigma)
        
        decay = _jit_bars_since_true(is_shock.values)
        df[f'bars_since_shock_{w}'] = decay

    return df
