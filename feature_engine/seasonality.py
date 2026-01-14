import pandas as pd
import numpy as np

def add_seasonality_features(df, lookback_days=20):
    """
    Computes Cyclical Time Features and Intraday Regime Features.

    Args:
        df: DataFrame with 'log_ret' and 'time_start'
        lookback_days: Deprecated/Unused. Kept for signature compatibility.
    """
    if df is None or 'time_start' not in df.columns:
        return df

    df = df.copy()
    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
        df['time_start'] = pd.to_datetime(df['time_start'])

    # Cyclical Time Encoding (Better than raw hours)
    # 24 hour cycle
    hour_float = df['time_start'].dt.hour + df['time_start'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour_float / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour_float / 24.0)

    # Intraday Regime Feature
    # ES session transitions: Overnight -> Pre-market build -> Open momentum -> Midday chop -> Close rebalancing
    # Combine time bucket with realized volatility to let evolution discover regime-dependent strategies

    # Session buckets (UTC hours) for ES/US equity:
    # 0: Overnight (21-24 UTC and 0-13 UTC = 4pm-8am ET)
    # 1: Pre-market (13-15 UTC = 8-10am ET)
    # 2: Morning (15-18 UTC = 10am-1pm ET)
    # 3: Afternoon/Close (18-21 UTC = 1-4pm ET)
    hour = df['time_start'].dt.hour
    # Map hours to sessions using np.select for wraparound handling
    conditions = [
        (hour >= 21) | (hour < 13),  # Overnight
        (hour >= 13) & (hour < 15),  # Pre-market
        (hour >= 15) & (hour < 18),  # Morning
        (hour >= 18) & (hour < 21),  # Afternoon/Close
    ]
    session = pd.Series(np.select(conditions, [0, 1, 2, 3], default=0), index=df.index).astype(float)

    # Realized volatility percentile over 1 hour (~40 volume bars at ~1.5 min avg)
    if 'log_ret' in df.columns:
        vol_1h = df['log_ret'].abs().rolling(40, min_periods=10).mean()
        vol_percentile_1h = vol_1h.rolling(500, min_periods=100).rank(pct=True)
    else:
        vol_percentile_1h = pd.Series(0.5, index=df.index)

    # Intraday Regime: Session * Volatility Percentile
    # High values = active session + high vol (momentum regime)
    # Low values = quiet session + low vol (mean-reversion/chop regime)
    df['intraday_regime'] = session * vol_percentile_1h

    # Also provide session indicator for simpler genes
    df['session_bucket'] = session

    return df