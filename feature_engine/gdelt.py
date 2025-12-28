import pandas as pd
import numpy as np

def add_gdelt_features(df, gdelt_df):
    """
    Merges GDELT features into Intraday Bars.
    Supports both Daily (Legacy) and Intraday (V2) GDELT data.
    """
    if df is None or gdelt_df is None: return df
    
    df = df.copy()
    
    # Check if GDELT is Intraday or Daily
    is_intraday_gdelt = False
    if isinstance(gdelt_df.index, pd.DatetimeIndex):
        # Check frequency/resolution roughly
        # If mean diff is < 24 hours, it's intraday
        if len(gdelt_df) > 1:
            diff = gdelt_df.index[1] - gdelt_df.index[0]
            if diff < pd.Timedelta(hours=23):
                is_intraday_gdelt = True
    
    if is_intraday_gdelt:
        print("Merging GDELT Features (Intraday - 15min Resolution)...")
        # Ensure sorting for merge_asof
        df = df.sort_values('time_start')
        gdelt_df = gdelt_df.sort_index()
        
        # Merge AsOf
        # backward: find the last GDELT row where gdelt_time <= bar_time
        # allow_exact_matches=True (assuming GDELT timestamp is publication time)
        # Note: In real live trading, we might have a 15 min lag. 
        # But for backtest/parity, we assume we have the data associated with that 15-min bucket.
        
        # We need to make sure index is a column for merge_asof if it's not
        gdelt_reset = gdelt_df.reset_index().rename(columns={'index': 'time_start', 'date_utc': 'time_start'})
        
        # Check column name of time in gdelt
        if 'time_start' not in gdelt_reset.columns:
            # Maybe it stayed as 'index' or 'date'
            cols = gdelt_reset.columns
            # Try to find datetime col
            for c in cols:
                if pd.api.types.is_datetime64_any_dtype(gdelt_reset[c]):
                    gdelt_reset = gdelt_reset.rename(columns={c: 'time_start'})
                    break
        
        merged = pd.merge_asof(
            df, 
            gdelt_reset, 
            on='time_start', 
            direction='backward'
        )
        
    else:
        print("Merging GDELT Features (Daily - Legacy Lag 1 Day)...")
        # Legacy Daily Logic
        df['date'] = df['time_start'].dt.normalize()
        gdelt_shifted = gdelt_df.shift(1).reset_index()
        merged = pd.merge(df, gdelt_shifted, on='date', how='left')
        merged.drop(columns=['date'], errors='ignore', inplace=True)
        
    # Fill NaNs (Forward Fill)
    cols_to_fill = ['news_vol_eur', 'news_tone_eur', 'news_vol_usd', 'news_tone_usd', 
                    'news_vol_zscore', 'global_polarity', 
                    'epu_total', 'epu_usd', 'epu_eur', 'epu_diff',
                    'central_bank_tone', 'energy_crisis_eur', 'panic_score', 'conflict_intensity']
    
    existing_cols = [c for c in cols_to_fill if c in merged.columns]
    merged[existing_cols] = merged[existing_cols].ffill().fillna(0)
    
    # Derived Physics Features
    if 'news_vol_eur' in merged.columns and 'news_vol_usd' in merged.columns:
        merged['news_vol_diff'] = merged['news_vol_eur'] - merged['news_vol_usd']
        merged['news_vol_ratio'] = merged['news_vol_eur'] / (merged['news_vol_usd'] + 1.0)

    if 'news_tone_eur' in merged.columns and 'news_tone_usd' in merged.columns:
        merged['news_tone_diff'] = merged['news_tone_eur'] - merged['news_tone_usd']
    
    # Cleanup
    drop_cols = ['total_vol', 'inflation_vol', 'central_bank_tone', 
                 'energy_crisis_eur', 'news_vol_zscore']
    merged.drop(columns=drop_cols, errors='ignore', inplace=True)
    
    return merged
