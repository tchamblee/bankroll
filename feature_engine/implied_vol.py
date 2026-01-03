import pandas as pd
import numpy as np
import config

def add_implied_vol_features(bars, vix_df):
    """
    Adds Implied Volatility features (Regime).
    Aligns VIX (1-min bars) to the primary Volume Bars.
    """
    print("Calculating Implied Volatility Features (VIX)...")
    
    if bars is None or bars.empty: return bars
    df = bars.copy()
    
    # 1. Prepare External Data
    # Rename columns for clarity before merge
    if vix_df is not None and not vix_df.empty:
        vix_clean = vix_df[['ts_event', 'close']].rename(columns={'close': 'vix_close'})
        vix_clean = vix_clean.sort_values('ts_event')
    else:
        vix_clean = None
        
    # 2. Merge Asof (Align to Volume Bar End Time)
    # We use 'time_end' of volume bars to match the latest available VIX info
    # 'ts_event' in VIX is the timestamp of the 1-min bar.
    
    df = df.sort_values('time_end')
    
    if vix_clean is not None:
        df = pd.merge_asof(df, vix_clean, left_on='time_end', right_on='ts_event', direction='backward')
        # Fill gaps (overnight/weekend)
        df['vix_close'] = df['vix_close'].ffill()
        
    # 3. Compute Features
    
    # A. VIX Trend / Regime
    if 'vix_close' in df.columns:
        # VIX Z-Score (Regime)
        # Using a long window (e.g. 2 weeks ~ 2000 5-min bars? No, VIX moves slow.)
        # Let's use 1 day (approx 400 bars) and 1 week (2000 bars)
        df['vix_trend_400'] = df['vix_close'].diff(400)
        
        roll_mean = df['vix_close'].rolling(2000).mean()
        roll_std = df['vix_close'].rolling(2000).std()
        df['vix_zscore_2000'] = (df['vix_close'] - roll_mean) / (roll_std + 1e-6)

    # Cleanup merge columns
    drop_cols = ['ts_event']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df
