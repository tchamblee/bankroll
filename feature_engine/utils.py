import numpy as np
import pandas as pd
import json
import os
import config


def rolling_zscore(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Compute rolling z-score with unified handling.

    This centralizes the rolling z-score calculation used across
    many feature modules to ensure consistency.

    Args:
        series: Input pandas Series
        window: Rolling window size
        min_periods: Minimum periods for rolling calculation.
                     Defaults to window // 2 if not specified.

    Returns:
        Series of z-scores
    """
    if min_periods is None:
        min_periods = max(1, window // 2)

    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window, min_periods=min_periods).std()
    # Consistent zero/nan handling - replace 0 std with 1 to avoid div by zero
    rolling_std = rolling_std.replace(0, 1)

    return (series - rolling_mean) / rolling_std


def rolling_zscore_multi(df: pd.DataFrame, columns: list, window: int,
                         min_periods: int = None, suffix: str = None) -> pd.DataFrame:
    """
    Compute rolling z-score for multiple columns at once.

    Args:
        df: Input DataFrame
        columns: List of column names to z-score
        window: Rolling window size
        min_periods: Minimum periods for rolling calculation
        suffix: Suffix to add to column names (e.g., '_z_100')
                If None, uses f'_z_{window}'

    Returns:
        DataFrame with z-scored columns added
    """
    if suffix is None:
        suffix = f'_z_{window}'

    for col in columns:
        if col in df.columns:
            df[f'{col}{suffix}'] = rolling_zscore(df[col], window, min_periods)

    return df


def filter_survivors(df, config_path=None):
    if config_path is None:
        config_path = os.path.join(config.DIRS['FEATURES_DIR'], "survivors.json")

    if not os.path.exists(config_path):
        return df
        
    try:
        with open(config_path, "r") as f:
            survivors = json.load(f)
        
        # Always keep metadata columns
        meta = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume']
        cols_to_keep = meta + [c for c in survivors if c in df.columns]
        
        print(f"Loaded {len(survivors)} features from {config_path}.")
        filtered_df = df[cols_to_keep]
        print(f"Filtered to {len(cols_to_keep)} columns.")
        return filtered_df
        
    except Exception as e:
        print(f"Error loading survivor config: {e}. Keeping all.")
        return df
