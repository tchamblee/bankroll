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
    # Markets transition: London Open momentum -> NY session mean-reversion -> late-day chop
    # Combine time bucket with realized volatility to let evolution discover regime-dependent strategies

    # Session buckets (UTC hours):
    # 0: Asia (00-07), 1: London Open (08-12), 2: NY Overlap (13-16), 3: NY Afternoon (17-22), 4: Late (23)
    hour = df['time_start'].dt.hour
    session = pd.cut(hour, bins=[-1, 7, 12, 16, 22, 24], labels=[0, 1, 2, 3, 4]).astype(float)

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