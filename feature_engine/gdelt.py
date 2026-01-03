import pandas as pd
import numpy as np
import config

def add_gdelt_features(df, gdelt_df):
    """
    Merges GDELT features into Intraday Bars.
    Refactored: Only retains 'news_rvol_usd' (Relative Volume USD) as it is the only robust feature.
    """
    if df is None or gdelt_df is None: return df
    
    df = df.copy()
    
    # Check if GDELT is Intraday or Daily
    is_intraday_gdelt = False
    if isinstance(gdelt_df.index, pd.DatetimeIndex):
        if len(gdelt_df) > 1:
            diff = gdelt_df.index[1] - gdelt_df.index[0]
            if diff < pd.Timedelta(hours=23):
                is_intraday_gdelt = True
    
    if is_intraday_gdelt:
        print("Merging GDELT Features (Intraday - 15min Resolution)...")
        # Ensure sorting for merge_asof
        df = df.sort_values('time_start')
        gdelt_df = gdelt_df.sort_index()
        
        # --- ROLLING TRANSFORMATION ---
        window = 96 # 24 hours
        
        # Only process USD Volume for RVOL calculation
        if 'news_vol_usd' not in gdelt_df.columns:
            return df
            
        gdelt_rolled = gdelt_df[['news_vol_usd']].copy()
        gdelt_rolled['news_vol_usd'] = gdelt_rolled['news_vol_usd'].rolling(window, min_periods=1).sum()
        
        # RVOL (Relative Volume) - Normalize by 30-day average
        rvol_window = 2880 # 30 days * 96
        base_usd = gdelt_rolled['news_vol_usd'].rolling(rvol_window, min_periods=1).mean().replace(0, 1)
        gdelt_rolled['news_rvol_usd'] = gdelt_rolled['news_vol_usd'] / base_usd
        
        # Prepare for merge
        gdelt_reset = gdelt_rolled.reset_index().rename(columns={'index': 'time_start', 'date_utc': 'time_start'})
        
        # Check column name of time in gdelt
        if 'time_start' not in gdelt_reset.columns:
             for c in gdelt_reset.columns:
                if pd.api.types.is_datetime64_any_dtype(gdelt_reset[c]):
                    gdelt_reset = gdelt_reset.rename(columns={c: 'time_start'})
                    break
        
        # Ensure Timezones match
        if 'time_start' in df.columns:
            df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
        if 'time_start' in gdelt_reset.columns:
            gdelt_reset['time_start'] = pd.to_datetime(gdelt_reset['time_start'], utc=True)

        merged = pd.merge_asof(
            df, 
            gdelt_reset[['time_start', 'news_rvol_usd']], 
            on='time_start', 
            direction='backward'
        )
        
    else:
        # Legacy Daily Support (Minimal)
        print("Merging GDELT Features (Daily)...")
        df['date'] = df['time_start'].dt.normalize()
        gdelt_shifted = gdelt_df.shift(1).reset_index()
        # Assume daily gdelt has pre-calc features or just raw
        # We only want news_rvol_usd if possible, but daily logic is legacy.
        # Just merge what we have but drop EPU/Tone later if needed.
        # For safety, we skip extensive logic here as we mostly use intraday now.
        merged = pd.merge(df, gdelt_shifted, on='date', how='left')
        merged.drop(columns=['date'], errors='ignore', inplace=True)
        
    # Fill NaNs
    if 'news_rvol_usd' in merged.columns:
        merged['news_rvol_usd'] = merged['news_rvol_usd'].ffill().fillna(0)
    
    return merged
