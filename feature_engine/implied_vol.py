import pandas as pd
import numpy as np
import config

def add_implied_vol_features(bars, vix_df, evz_df):
    """
    Adds Implied Volatility features (Regime, Risk Premium).
    Aligns VIX/EVZ (1-min bars) to the primary Volume Bars.
    """
    print("Calculating Implied Volatility Features (VIX/EVZ)...")
    
    if bars is None or bars.empty: return bars
    df = bars.copy()
    
    # 1. Prepare External Data
    # Rename columns for clarity before merge
    if vix_df is not None and not vix_df.empty:
        vix_clean = vix_df[['ts_event', 'close']].rename(columns={'close': 'vix_close'})
        vix_clean = vix_clean.sort_values('ts_event')
    else:
        vix_clean = None
        
    if evz_df is not None and not evz_df.empty:
        evz_clean = evz_df[['ts_event', 'close']].rename(columns={'close': 'evz_close'})
        evz_clean = evz_clean.sort_values('ts_event')
    else:
        evz_clean = None
        
    # 2. Merge Asof (Align to Volume Bar End Time)
    # We use 'time_end' of volume bars to match the latest available VIX/EVZ info
    # 'ts_event' in VIX/EVZ is the timestamp of the 1-min bar.
    
    df = df.sort_values('time_end')
    
    if vix_clean is not None:
        df = pd.merge_asof(df, vix_clean, left_on='time_end', right_on='ts_event', direction='backward')
        # Fill gaps (overnight/weekend)
        df['vix_close'] = df['vix_close'].ffill()
        
    if evz_clean is not None:
        df = pd.merge_asof(df, evz_clean, left_on='time_end', right_on='ts_event', direction='backward', suffixes=('', '_evz'))
        df['evz_close'] = df['evz_close'].ffill()
        
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

    # B. Volatility Risk Premium (EVZ vs Realized)
    # EVZ is Annualized %. Yang-Zhang is Per-Bar Log Ret Std.
    if 'evz_close' in df.columns and 'yang_zhang_vol_100' in df.columns:
        # Annualize Realized Vol
        # Annualization Factor in config is usually for Sharpe (daily/hourly).
        # We need Bars Per Year.
        # config.ANNUALIZATION_FACTOR is ~114408 (5-min bars).
        
        rv_annual = df['yang_zhang_vol_100'] * np.sqrt(config.ANNUALIZATION_FACTOR) * 100
        
        # Premium: Implied - Realized
        df['vol_risk_premium_100'] = df['evz_close'] - rv_annual
        
        # Premium Z-Score (is options expensive relative to recent history?)
        prem_mean = df['vol_risk_premium_100'].rolling(2000).mean()
        prem_std = df['vol_risk_premium_100'].rolling(2000).std()
        df['vol_premium_z_2000'] = (df['vol_risk_premium_100'] - prem_mean) / (prem_std + 1e-6)
        
    # Cleanup merge columns
    drop_cols = ['ts_event', 'ts_event_evz']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    return df
