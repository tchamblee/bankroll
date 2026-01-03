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
    
    # 1. Market Structure (Pullback Depth)
    # Replaces sparse 'bars_since_high'
    
    # Get ATR if available
    atr = df['atr'] if 'atr' in df.columns else df['close'].rolling(50).std()
    
    for w in high_windows:
        # Rolling Max High
        roll_max = df['high'].rolling(w).max()
        
        # Pullback Ratio: How many ATRs are we below the recent high?
        # Continuous metric of "Dip" depth
        pullback = (roll_max - df['close']) / atr.replace(0, 1.0)
        
        df[f'pullback_ratio_{w}'] = pullback.fillna(0)
        
        # Bars Since High (Kept as legacy since it survived some windows, but Pullback is better)
        # Identify new highs
        is_high = (df['high'] >= roll_max - 1e-9) 
        decay = _jit_bars_since_true(is_high.values)
        if w != 400: # 400 failed
            df[f'bars_since_high_{w}'] = decay
        
    # 2. Bars Since Shock (Removed: Failed Triage)
    # "Shock" = |Return| > 3 * Sigma
    # abs_ret = df['log_ret'].abs()
    
    # for w in shock_windows:
    #    pass 

    return df
