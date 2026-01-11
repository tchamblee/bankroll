"""
Economic Surprise Features

Computes surprise_z = (actual - forecast) / historical_std for economic releases.
Values are forward-filled until the next release of the same event type.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Key events to track (event_key -> feature_name)
KEY_EVENTS = {
    # USD Events
    'core_cpi_m_m': 'usd_core_cpi',
    'cpi_m_m': 'usd_cpi',
    'cpi_y_y': 'usd_cpi_yy',
    'core_ppi_m_m': 'usd_core_ppi',
    'ppi_m_m': 'usd_ppi',
    'non_farm_employment_change': 'usd_nfp',
    'unemployment_rate': 'usd_unemployment',
    'unemployment_claims': 'usd_claims',
    'ism_manufacturing_pmi': 'usd_ism_mfg',
    'ism_services_pmi': 'usd_ism_svc',
    'flash_manufacturing_pmi': 'usd_flash_mfg',
    'flash_services_pmi': 'usd_flash_svc',
    'adp_non_farm_employment_change': 'usd_adp',
    'advance_gdp_q_q': 'usd_gdp',
    'prelim_gdp_q_q': 'usd_gdp_prelim',
    'jolts_job_openings': 'usd_jolts',
    'retail_sales_m_m': 'usd_retail',
    'core_retail_sales_m_m': 'usd_core_retail',
    # EUR Events
    'german_flash_manufacturing_pmi': 'eur_de_mfg_pmi',
    'german_flash_services_pmi': 'eur_de_svc_pmi',
    'flash_manufacturing_pmi': 'eur_flash_mfg',
    'flash_services_pmi': 'eur_flash_svc',
    'german_prelim_gdp_q_q': 'eur_de_gdp',
    'main_refinancing_rate': 'eur_ecb_rate',
}

# Default data path
DEFAULT_EVENTS_PATH = Path(__file__).parent.parent / "data" / "econ_calendar" / "processed" / "econ_events.parquet"


def load_econ_events(path=None):
    """Load economic events from parquet."""
    path = path or DEFAULT_EVENTS_PATH
    if not Path(path).exists():
        print(f"Warning: Economic events file not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)

    # Compute raw surprise
    df['surprise'] = df['actual'] - df['forecast']

    return df


def compute_surprise_z(events_df):
    """
    Compute z-scored surprises for each event type.

    Returns DataFrame with columns: timestamp_utc, event_key, surprise_z
    """
    if events_df.empty:
        return pd.DataFrame()

    # Filter to events with actual and forecast
    df = events_df[events_df['surprise'].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    # Compute rolling std for each event type (use all historical data)
    results = []

    for event_key in df['event_key'].unique():
        event_df = df[df['event_key'] == event_key].sort_values('week')

        if len(event_df) < 2:
            continue

        # Use expanding std (all prior observations)
        surprises = event_df['surprise'].values

        # Compute historical std (excluding current observation)
        stds = []
        for i in range(len(surprises)):
            if i < 2:
                # Not enough history, use a default
                stds.append(np.nan)
            else:
                stds.append(np.std(surprises[:i]))

        event_df = event_df.copy()
        event_df['hist_std'] = stds

        # Z-score: surprise / historical_std
        # Clip to avoid extreme values from low-vol periods
        event_df['surprise_z'] = (event_df['surprise'] / event_df['hist_std']).clip(-5, 5)

        results.append(event_df[['week', 'currency', 'event_key', 'event_name', 'surprise', 'surprise_z']])

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def add_econ_surprise_features(bars_df, events_path=None):
    """
    Add economic surprise features to bars DataFrame.

    Features added:
    - {event}_surprise_z: Z-scored surprise for each key event type
    - {event}_bars_since: Bars since last release of this event

    Values are forward-filled until the next release.
    """
    if 'time_end' not in bars_df.columns:
        print("Warning: bars_df missing 'time_end' column")
        return bars_df

    # Load and process events
    events_df = load_econ_events(events_path)
    if events_df.empty:
        print("Warning: No economic events loaded")
        return bars_df

    surprise_df = compute_surprise_z(events_df)
    if surprise_df.empty:
        print("Warning: No surprise data computed")
        return bars_df

    # Ensure bars are sorted
    bars_df = bars_df.sort_values('time_end').copy()

    # Parse week string to datetime for merging
    # Week format: "jul14.2025" -> use Wednesday of that week as approximate event time
    # (most high-impact events are mid-week)
    def parse_week(week_str):
        try:
            from datetime import datetime, timedelta
            # Parse "jul14.2025" format - this is the Monday
            dt = datetime.strptime(week_str, "%b%d.%Y")
            # Use Wednesday (Monday + 2 days) as typical event day
            dt = dt + timedelta(days=2)
            return pd.Timestamp(dt, tz='UTC')
        except:
            return pd.NaT

    surprise_df['event_time'] = surprise_df['week'].apply(parse_week)
    surprise_df = surprise_df.dropna(subset=['event_time'])

    # For each key event type, create a feature column
    for event_key, feature_name in KEY_EVENTS.items():
        col_name = f"{feature_name}_surprise_z"

        # Get surprises for this event
        event_surprises = surprise_df[surprise_df['event_key'] == event_key].copy()

        if event_surprises.empty:
            bars_df[col_name] = 0.0
            continue

        event_surprises = event_surprises.sort_values('event_time')

        # Merge with bars using asof (forward fill from event time)
        event_for_merge = event_surprises[['event_time', 'surprise_z']].rename(
            columns={'surprise_z': col_name}
        )

        # Use merge_asof to get the most recent event for each bar
        merged = pd.merge_asof(
            bars_df[['time_end']].reset_index(),
            event_for_merge,
            left_on='time_end',
            right_on='event_time',
            direction='backward'
        )

        bars_df[col_name] = merged[col_name].fillna(0.0).values

    # Add aggregate features
    surprise_cols = [c for c in bars_df.columns if c.endswith('_surprise_z')]
    if surprise_cols:
        # Average absolute surprise across all tracked events
        bars_df['econ_surprise_magnitude'] = bars_df[surprise_cols].abs().mean(axis=1)

        # USD vs EUR surprise differential
        usd_cols = [c for c in surprise_cols if c.startswith('usd_')]
        eur_cols = [c for c in surprise_cols if c.startswith('eur_')]

        if usd_cols and eur_cols:
            bars_df['usd_eur_surprise_diff'] = (
                bars_df[usd_cols].mean(axis=1) - bars_df[eur_cols].mean(axis=1)
            )

    return bars_df


def show_surprise_stats(events_path=None):
    """Show statistics about loaded surprise data."""
    events_df = load_econ_events(events_path)
    if events_df.empty:
        print("No events loaded")
        return

    surprise_df = compute_surprise_z(events_df)
    if surprise_df.empty:
        print("No surprise data")
        return

    print("=" * 60)
    print("ECONOMIC SURPRISE DATA STATISTICS")
    print("=" * 60)
    print(f"Total events with surprises: {len(surprise_df)}")
    print(f"Event types tracked: {surprise_df['event_key'].nunique()}")
    print()

    print("Events per type:")
    counts = surprise_df.groupby('event_key').size().sort_values(ascending=False)
    for event_key, count in counts.items():
        if event_key in KEY_EVENTS:
            print(f"  {KEY_EVENTS[event_key]:25} ({event_key}): {count}")

    print()
    print("Sample z-scores:")
    sample = surprise_df[surprise_df['surprise_z'].notna()].head(10)
    for _, row in sample.iterrows():
        print(f"  {row['event_name'][:30]:30} | z={row['surprise_z']:+.2f}")


if __name__ == "__main__":
    show_surprise_stats()
