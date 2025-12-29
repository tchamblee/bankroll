import pandas as pd
import numpy as np

def add_seasonality_features(df, lookback_days=20):
    """
    Computes Intraday Seasonality Features.
    Calculates the 'Expected Return' for the current time of day based on a rolling window of previous days.
    
    Args:
        df: DataFrame with 'log_ret' and 'time_start'
        lookback_days: Number of past days to average for the seasonal baseline.
    """
    if df is None or 'log_ret' not in df.columns or 'time_start' not in df.columns:
        return df
    
    print(f"Calculating Seasonality Features (Lookback: {lookback_days} days)...")
    
    # 1. Extract Time Components
    df = df.copy()
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
        df['time_start'] = pd.to_datetime(df['time_start'])
        
    # Create a 'Time of Day' signature (e.g. Minutes from midnight)
    # 5-minute bars -> ~288 bins
    minutes_from_midnight = df['time_start'].dt.hour * 60 + df['time_start'].dt.minute
    df['_tod'] = minutes_from_midnight
    df['_date'] = df['time_start'].dt.date
    
    # 2. Pivot to Sparse Matrix (Days x TimeOfDay)
    # This aligns all 10:00 AM bars, all 10:05 AM bars, etc.
    pivot = df.pivot_table(index='_date', columns='_tod', values='log_ret')
    
    # 3. Compute Rolling Seasonality (Shift 1 to avoid lookahead!)
    # We want expected return for Today based on Last N days.
    expected_ret_matrix = pivot.rolling(window=lookback_days, min_periods=5).mean().shift(1)
    expected_std_matrix = pivot.rolling(window=lookback_days, min_periods=5).std().shift(1)
    
    # 4. Map back to original DataFrame
    # Stack the matrix to get a Series indexed by (Date, TOD)
    expected_ret_series = expected_ret_matrix.stack()
    expected_ret_series.name = 'expected_seasonal_ret'
    
    expected_std_series = expected_std_matrix.stack()
    expected_std_series.name = 'expected_seasonal_std'
    
    # Join based on keys
    # Create temp index for joining
    df = df.set_index(['_date', '_tod'])
    
    # Join
    df = df.join(expected_ret_series)
    df = df.join(expected_std_series)
    
    # Reset index
    df = df.reset_index(drop=False)
    
    # 5. Compute Deviation Features
    # Fill NaNs (first N days) with 0
    df['expected_seasonal_ret'] = df['expected_seasonal_ret'].fillna(0.0)
    df['expected_seasonal_std'] = df['expected_seasonal_std'].fillna(1.0) # Avoid div/0
    
    # Feature 1: Seasonal Deviation (Z-Score)
    # (Actual - Expected) / Expected_Vol
    # Note: Use current realized volatility for normalization? Or seasonal volatility?
    # Using seasonal volatility tells us if this move is abnormal *for this time of day*.
    df['seasonal_deviation'] = (df['log_ret'] - df['expected_seasonal_ret']) / df['expected_seasonal_std'].replace(0, 1)
    
    # Feature 2: Seasonal Trend Alignment
    # simply the expected return itself (is this time of day usually bullish?)
    df['seasonal_trend'] = df['expected_seasonal_ret']
    
    # Cleanup
    drop_cols = ['_tod', '_date', 'expected_seasonal_ret', 'expected_seasonal_std']
    df.drop(columns=drop_cols, inplace=True)
    
    return df
