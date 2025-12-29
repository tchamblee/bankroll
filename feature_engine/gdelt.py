import pandas as pd
import numpy as np
import config

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
        
        # --- ROLLING TRANSFORMATION (Bridge Daily vs Intraday) ---
        # Strategies were trained on Daily aggregates (24h sum).
        # Raw 15-min data is too sparse and small-scaled.
        # We apply a 24-hour (96 periods) rolling window to simulate a 
        # "Continuous Daily" value.
        
        window = 96 # 15 min * 96 = 24 hours
        
        # Columns to SUM (Counts)
        sum_cols = ['news_vol_eur', 'news_vol_usd', 'total_count', 
                    'conflict_intensity', 'epu_total', 'epu_usd', 'epu_eur', 
                    'inflation_vol', 'cb_count', 'energy_crisis_eur']
        
        # Columns to MEAN (Scores/Tones)
        # Note: Ideally weighted mean, but simple mean on rolling window is decent proxy
        # provided we assume article volume is somewhat distributed.
        mean_cols = ['news_tone_eur', 'news_tone_usd', 'global_tone', 
                     'global_polarity', 'central_bank_tone']
                     
        # Filter cols that actually exist
        sum_cols = [c for c in sum_cols if c in gdelt_df.columns]
        mean_cols = [c for c in mean_cols if c in gdelt_df.columns]
        
        # Apply Rolling
        # min_periods=1 ensures we get data immediately without waiting 24h (starts noisy, stabilizes)
        gdelt_rolled = gdelt_df.copy()
        gdelt_rolled[sum_cols] = gdelt_rolled[sum_cols].rolling(window, min_periods=1).sum()
        gdelt_rolled[mean_cols] = gdelt_rolled[mean_cols].rolling(window, min_periods=1).mean()
        
        # Recalculate derived columns that depend on rolled values
        # e.g. panic_score, epu_diff
        
        # EPU Diff
        if 'epu_usd' in gdelt_rolled.columns and 'epu_eur' in gdelt_rolled.columns:
            gdelt_rolled['epu_diff'] = gdelt_rolled['epu_usd'] - gdelt_rolled['epu_eur']
            
        # Panic Score (Re-run logic on smoothed data)
        if 'global_tone' in gdelt_rolled.columns and 'global_polarity' in gdelt_rolled.columns:
             # Expanding/Rolling Z-Score on the SMOOTHED tone
             t_mean = gdelt_rolled['global_tone'].expanding(min_periods=96).mean()
             t_std = gdelt_rolled['global_tone'].expanding(min_periods=96).std()
             
             t_mean = t_mean.bfill().fillna(0)
             t_std = t_std.bfill().fillna(1.0)
             
             z = (gdelt_rolled['global_tone'] - t_mean) / t_std
             gdelt_rolled['panic_score'] = np.where(z < config.PANIC_SCORE_THRESHOLD, gdelt_rolled['global_polarity'] * -1, 0)
        
        # We need to make sure index is a column for merge_asof if it's not
        gdelt_reset = gdelt_rolled.reset_index().rename(columns={'index': 'time_start', 'date_utc': 'time_start'})
        
        # Check column name of time in gdelt
        if 'time_start' not in gdelt_reset.columns:
            # Maybe it stayed as 'index' or 'date'
            cols = gdelt_reset.columns
            # Try to find datetime col
            for c in cols:
                if pd.api.types.is_datetime64_any_dtype(gdelt_reset[c]):
                    gdelt_reset = gdelt_reset.rename(columns={c: 'time_start'})
                    break
        
        # Ensure Timezones match (Force UTC)
        if 'time_start' in df.columns:
            df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
        if 'time_start' in gdelt_reset.columns:
            gdelt_reset['time_start'] = pd.to_datetime(gdelt_reset['time_start'], utc=True)

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
        
    # --- SENTIMENT VELOCITY (24h Trend) ---
    # Lag 96 bars (approx 24h)
    vel_lag = 96
    
    if 'global_tone' in merged.columns:
        merged['tone_velocity'] = merged['global_tone'] - merged['global_tone'].shift(vel_lag)
        
    if 'epu_total' in merged.columns:
        # Pct Change for Counts
        merged['epu_velocity'] = merged['epu_total'].pct_change(vel_lag).replace([np.inf, -np.inf], 0).fillna(0)
        
    if 'conflict_intensity' in merged.columns:
        merged['conflict_velocity'] = merged['conflict_intensity'].pct_change(vel_lag).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Cleanup
    drop_cols = ['total_vol', 'inflation_vol', 'central_bank_tone', 
                 'energy_crisis_eur', 'news_vol_zscore']
    merged.drop(columns=drop_cols, errors='ignore', inplace=True)
    
    return merged
