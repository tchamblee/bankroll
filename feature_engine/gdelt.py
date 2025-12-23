import pandas as pd
import numpy as np

def add_gdelt_features(df, gdelt_df):
    """
    Merges Daily GDELT features into Intraday Bars.
    CRITICAL: Uses LAG 1 (Yesterday's News) to avoid Lookahead Bias.
    """
    if df is None or gdelt_df is None: return df
    print("Merging GDELT Features (Lag 1 Day)...")
    df = df.copy()
    
    # Extract Date from Bar Start Time
    df['date'] = df['time_start'].dt.normalize()
    
    # Shift GDELT by 1 Day to represent "Yesterday"
    # Since gdelt_df is indexed by date, we can shift the index or the data.
    # Ideally, for date T, we want GDELT from T-1.
    # So we shift GDELT index forward by 1 day? 
    # No, if we want T's features to be T-1's data, we merge T with T-1.
    # Easier: Shift GDELT dataframe forward by 1 day.
    
    gdelt_shifted = gdelt_df.shift(1) # Row 1 becomes Row 2. Date index stays? No, shift moves data.
    # shift(1) on a DF moves values down.
    # 2025-12-01: Data_A
    # 2025-12-02: Data_B
    # after shift(1):
    # 2025-12-01: NaN
    # 2025-12-02: Data_A
    # This is exactly what we want. On Dec 2nd, we see Dec 1st data.
    
    # Merge
    # We reset index to make 'date' a column
    gdelt_shifted = gdelt_shifted.reset_index()
    
    merged = pd.merge(df, gdelt_shifted, on='date', how='left')
    
    # Fill NaNs (e.g. weekends or missing days) with 0 or forward fill?
    # Forward fill is better for sentiment persistence
    cols_to_fill = ['news_vol_eur', 'news_tone_eur', 'news_vol_usd', 'news_tone_usd', 
                    'news_vol_zscore', 'global_polarity', 
                    'epu_total', 'epu_usd', 'epu_eur',
                    'central_bank_tone', 'energy_crisis_eur', 'panic_score']
    
    # Only fill columns that exist (in case we run old logic)
    existing_cols = [c for c in cols_to_fill if c in merged.columns]
    merged[existing_cols] = merged[existing_cols].ffill().fillna(0)
    
    # Derived Physics Features
    # 1. Attention Divergence (Mass Differential)
    merged['news_vol_diff'] = merged['news_vol_eur'] - merged['news_vol_usd']
    merged['news_vol_ratio'] = merged['news_vol_eur'] / (merged['news_vol_usd'] + 1.0)

    # 2. Sentiment Divergence
    merged['news_tone_diff'] = merged['news_tone_eur'] - merged['news_tone_usd']
    
    # 4. Regime/Panic Features
    # Already calculated in load: news_vol_zscore, epu_diff, conflict_intensity
    
    # Drop temp date column and Redundant Features
    # REMOVED WEAK SIGNALS: central_bank_tone, news_velocity_*, energy_crisis_eur
    # KEPT: news_tone_*, news_vol_* for derived feature calculation downstream if needed
    drop_cols = ['date', 'total_vol', 'conflict_intensity', 'inflation_vol', 'epu_diff', 'global_tone', 
                 'global_polarity', 'panic_score', 'central_bank_tone', 
                 'energy_crisis_eur', 'news_vol_zscore']
    merged.drop(columns=drop_cols, errors='ignore', inplace=True)
    
    return merged
