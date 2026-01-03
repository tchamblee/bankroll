import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import config

def _calc_window_features(w, log_ret, close, open, high, low, gk_var):
    """Worker function for parallel window feature calculation."""
    res = {}
    
    # 1. Velocity
    res[f'velocity_{w}'] = log_ret.rolling(w).sum()
    
    # 2. Volatility (Garman-Klass)
    vol = np.sqrt(gk_var.rolling(w).mean())
    if w != 800: # Match logic in original script
        res[f'volatility_{w}'] = vol
    
    # 3. Efficiency & Trend Strength
    net_change = close.diff(w).abs()
    path = close.diff().abs().rolling(w).sum()
    efficiency = np.where(path == 0, 0, net_change / path)
    
    # Trend Strength needs rolling mean of volatility
    # Note: original used rolling(1000).mean() globally, but it was per window 'vol'
    # Re-evaluating: vol_norm = vol / vol.rolling(1000).mean()
    vol_norm = vol / vol.rolling(config.VOLATILITY_NORMALIZATION_WINDOW, min_periods=100).mean()
    trend_strength = efficiency * vol_norm
    res[f'trend_strength_{w}'] = trend_strength
    
    # 4. Skewness
    res[f'skew_{w}'] = log_ret.rolling(w).skew()
    
    # 5. ROC Features
    res[f'volatility_roc_{w}'] = vol.pct_change(fill_method=None)
    res[f'skew_roc_{w}'] = res[f'skew_{w}'].diff()
    res[f'trend_strength_roc_{w}'] = trend_strength.diff()
    
    return res

def add_features_to_bars(df, windows=[50, 100, 200, 400, 800, 1600]):
    if df is None: return None
    df = df.copy()
    
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = log_ret

    # Pre-calc Garman-Klass Variance components
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    gk_var = gk_var.clip(lower=0)

    # Parallelize calculations
    results = Parallel(n_jobs=-1)(
        delayed(_calc_window_features)(w, log_ret, df['close'], df['open'], df['high'], df['low'], gk_var) 
        for w in windows
    )
    
    # Merge results
    new_cols = {}
    for r in results:
        new_cols.update(r)
        
    # Add ATR (Window 50 for consistency with backtest engine)
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    c_prev = np.roll(c, 1); c_prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    df['atr'] = pd.Series(tr).rolling(config.ATR_WINDOW, min_periods=1).mean().values

    df_new = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, df_new], axis=1)
