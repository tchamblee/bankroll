import numpy as np
import pandas as pd
from numba import jit
from joblib import Parallel, delayed

@jit(nopython=True)
def _jit_fisher_info_rolling(vals, window):
    """
    Calculates the Fisher Information Metric (FIM) for a Gaussian process.
    FIM ~ 1 / sigma^2  (for mean parameter estimation)
    FIM ~ 2 / sigma^2  (for variance parameter estimation)

    We use the rate of change of the statistical state.
    Fisher Information roughly measures how much information the data provides about the parameter.
    If volatility is low, FIM is high (high precision).
    If volatility is high, FIM is low (low precision).

    However, a more useful metric for trading is the 'Fisher Information Distance' (Rao Distance)
    which measures the speed of change of the probability distribution.

    Here we implement a simplified "Information Flow" metric:
    Flow = (d_mean/dt)^2 * FIM_mean + (d_var/dt)^2 * FIM_var

    For Gaussian N(mu, sigma^2):
    ds^2 = (dmu/sigma)^2 + 2*(dsigma/sigma)^2

    This measures the 'statistical velocity' of the market.
    """
    n = len(vals)
    res = np.empty(n)
    res[:] = np.nan

    # Need at least window + 1 values to compute first valid result
    # (window for current slice, +1 for previous slice comparison)
    if n < window + 2 or window < 2:
        return res

    for i in range(window + 1, n):
        # We need local mean/std estimates and their derivatives.
        # Window 1: [i-window : i] (Current)
        # Window 0: [i-window-1 : i-1] (Previous)

        # Current window statistics
        slice_curr = vals[i-window : i]
        mu = np.mean(slice_curr)
        sigma = np.std(slice_curr)

        if sigma < 1e-9:
            res[i] = 0.0 # Degenerate
            continue

        # Previous window statistics
        slice_prev = vals[i-window-1 : i-1]
        sigma_prev = np.std(slice_prev)

        # Guard against zero sigma_prev for dsigma calculation
        if sigma_prev < 1e-9:
            res[i] = 0.0
            continue

        # Derivative of mean (Velocity of Price)
        # dmu ~ mean(t) - mean(t-1)
        # Rolling mean difference is (x[t] - x[t-w])/w
        dmu = (vals[i-1] - vals[i-window-1]) / window

        # Derivative of sigma (Velocity of Volatility)
        dsigma = sigma - sigma_prev

        # Rao Distance (Squared) / dt
        # ds^2 = (dmu/sigma)^2 + 2*(dsigma/sigma)^2
        ds2 = (dmu / sigma)**2 + 2 * (dsigma / sigma)**2

        # Information Velocity = sqrt(ds^2)
        res[i] = np.sqrt(ds2)

    return res

def _calc_fisher_info_window(df_min, w):
    """Worker for parallel calculation."""
    # We calculate on Log Returns
    log_ret = np.log(df_min['close'] / df_min['close'].shift(1)).fillna(0).values
    
    fim_vel = _jit_fisher_info_rolling(log_ret, w)
    
    return {
        f'fisher_info_velocity_{w}': fim_vel
    }

def add_information_geometry_features(df, windows=[50, 100, 200]):
    """
    Adds Information Geometry features:
    1. Fisher Information Velocity (Statistical Speed of the Market).
       High Velocity = Regime Shift / Breakout.
       Low Velocity = Stable / Dead.
       
    2. Shannon Information Rate (Entropy Flow).
    """
    if df is None: return None
    print("Calculating Information Geometry Features (Fisher Information)...")
    df = df.copy()
    
    cols_needed = ['close']
    df_min = df[cols_needed].copy()
    
    # Parallel Calculation
    results = Parallel(n_jobs=-1)(
        delayed(_calc_fisher_info_window)(df_min, w)
        for w in windows
    )
    
    new_cols = {}
    for r in results:
        new_cols.update(r)
        
    # Additional: Information Acceleration (Change in Information Velocity)
    for w in windows:
        vel_col = f'fisher_info_velocity_{w}'
        if vel_col in new_cols:
            vel = pd.Series(new_cols[vel_col], index=df.index)
            acc = vel.diff()
            new_cols[f'fisher_info_accel_{w}'] = acc
            
    df_new = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, df_new], axis=1)
