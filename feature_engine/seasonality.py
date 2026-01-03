import pandas as pd
import numpy as np

def add_seasonality_features(df, lookback_days=20):
    """
    Computes Cyclical Time Features.
    (Seasonal Deviation/Trend were purged due to low signal).
    
    Args:
        df: DataFrame with 'log_ret' and 'time_start'
        lookback_days: Deprecated/Unused. Kept for signature compatibility.
    """
    if df is None or 'time_start' not in df.columns:
        return df
    
    # print(f"Calculating Seasonality Features...") # Reduced verbosity
    
    df = df.copy()
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
        df['time_start'] = pd.to_datetime(df['time_start'])
        
    # Feature 3: Cyclical Time Encoding (Better than raw hours)
    # 24 hour cycle
    hour_float = df['time_start'].dt.hour + df['time_start'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour_float / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour_float / 24.0)
    
    return df