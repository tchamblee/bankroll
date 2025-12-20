import numpy as np
import pandas as pd

def add_features_to_bars(df, windows=[50, 100, 200, 400, 800, 1600]):
    if df is None: return None
    df = df.copy() # Avoid SettingWithCopy if necessary
    
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # Pre-calc Garman-Klass Variance components
    # Var_GK = 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    # Ensure non-negative (theoretical issue, practically fine)
    gk_var = gk_var.clip(lower=0)

    for w in windows:
        # 1. Velocity (Cumulative Log Return) - Scale Invariant
        df[f'velocity_{w}'] = df['log_ret'].rolling(w).sum()
        
        # 2. Volatility (Garman-Klass) - More precise, uses OHLC
        # Rolling Vol = sqrt( Rolling Mean of Variance )
        df[f'volatility_{w}'] = np.sqrt(gk_var.rolling(w).mean())
        
        net_change = df['close'].diff(w).abs()
        path = df['close'].diff().abs().rolling(w).sum()
        df[f'efficiency_{w}'] = np.where(path==0, 0, net_change/path)
        
        df[f'autocorr_{w}'] = df['log_ret'].rolling(w).corr(df['log_ret'].shift(1))
        
        # 3. Skewness (Tail Asymmetry) - 3rd Moment
        df[f'skew_{w}'] = df['log_ret'].rolling(w).skew()
        
        # 4. Price Z-Score (Mean Reversion / Deviation) - REDUNDANT with Velocity
        # Distance from moving average, normalized by GK volatility
        # ma = df['close'].rolling(w).mean()
        # GK Volatility is percentage-based (e.g. 0.001 for 0.1%). 
        # To normalize a price difference ($), we need price * vol ($).
        # df[f'price_zscore_{w}'] = (df['close'] - ma) / (df[f'volatility_{w}'] * df['close']).replace(0, 1) 
        
        vol_norm = df[f'volatility_{w}'] / df[f'volatility_{w}'].rolling(1000, min_periods=100).mean()
        df[f'trend_strength_{w}'] = df[f'efficiency_{w}'] * vol_norm
        
        # Drop redundant efficiency column (Survives only as trend_strength)
        df.drop(columns=[f'efficiency_{w}'], inplace=True)

    return df
