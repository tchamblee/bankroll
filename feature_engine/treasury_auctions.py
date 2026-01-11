"""
Treasury Auction Features

Adds bid-to-cover and tail z-scores from recent Treasury auctions.
Values are forward-filled from auction date until the next auction of the same tenor.

Key features:
- auction_btc_zscore_10y: Bid-to-cover z-score for 10Y notes (positive = strong demand)
- auction_tail_zscore_10y: Tail z-score (positive = weak auction, had to pay up)
- auction_stress_aggregate: Average tail z-score across recent auctions (market stress)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Default data path
DEFAULT_AUCTIONS_PATH = Path(__file__).parent.parent / "data" / "treasury_auctions.parquet"

# Key tenors to track (most relevant for EUR/USD via rate differentials)
KEY_TENORS = ['2y', '5y', '10y', '30y']


def load_auction_data(path=None):
    """Load Treasury auction data from parquet."""
    path = path or DEFAULT_AUCTIONS_PATH
    if not Path(path).exists():
        print(f"Warning: Treasury auctions file not found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df['auction_date'] = pd.to_datetime(df['auction_date'])

    # Ensure timezone aware (UTC) for merging with bars
    if df['auction_date'].dt.tz is None:
        df['auction_date'] = df['auction_date'].dt.tz_localize('UTC')

    return df


def add_treasury_auction_features(bars_df, auctions_path=None):
    """
    Add Treasury auction features to bars DataFrame.

    Features added per key tenor:
    - auction_btc_zscore_{tenor}: Bid-to-cover z-score (negative = weak demand)
    - auction_tail_zscore_{tenor}: Tail z-score (positive = weak auction)

    Aggregate features:
    - auction_stress_aggregate: Average tail z-score across all recent auctions
    """
    if 'time_end' not in bars_df.columns:
        print("Warning: bars_df missing 'time_end' column")
        return bars_df

    auctions_df = load_auction_data(auctions_path)
    if auctions_df.empty:
        print("Warning: No Treasury auction data loaded")
        return bars_df

    # Ensure bars are sorted
    bars_df = bars_df.sort_values('time_end').copy()

    # For each key tenor, create features
    for tenor in KEY_TENORS:
        btc_col = f"auction_btc_zscore_{tenor}"
        tail_col = f"auction_tail_zscore_{tenor}"

        # Get auctions for this tenor
        tenor_auctions = auctions_df[auctions_df['tenor'] == tenor].copy()

        if tenor_auctions.empty:
            bars_df[btc_col] = 0.0
            bars_df[tail_col] = 0.0
            continue

        tenor_auctions = tenor_auctions.sort_values('auction_date')

        # Prepare for merge_asof
        # Use auction_date as the event time (forward fill from there)
        btc_for_merge = tenor_auctions[['auction_date', 'btc_zscore']].rename(
            columns={'btc_zscore': btc_col}
        )
        tail_for_merge = tenor_auctions[['auction_date', 'tail_zscore']].rename(
            columns={'tail_zscore': tail_col}
        )

        # Merge bid-to-cover
        merged = pd.merge_asof(
            bars_df[['time_end']].reset_index(),
            btc_for_merge,
            left_on='time_end',
            right_on='auction_date',
            direction='backward'
        )
        bars_df[btc_col] = merged[btc_col].fillna(0.0).values

        # Merge tail
        merged = pd.merge_asof(
            bars_df[['time_end']].reset_index(),
            tail_for_merge,
            left_on='time_end',
            right_on='auction_date',
            direction='backward'
        )
        bars_df[tail_col] = merged[tail_col].fillna(0.0).values

    # Aggregate features
    tail_cols = [f"auction_tail_zscore_{t}" for t in KEY_TENORS]
    btc_cols = [f"auction_btc_zscore_{t}" for t in KEY_TENORS]

    # Average tail z-score (higher = more stress from weak auctions)
    bars_df['auction_stress_aggregate'] = bars_df[tail_cols].mean(axis=1)

    # Average bid-to-cover z-score (lower = weaker demand overall)
    bars_df['auction_demand_aggregate'] = bars_df[btc_cols].mean(axis=1)

    return bars_df


def show_auction_stats(auctions_path=None):
    """Show statistics about loaded auction data."""
    df = load_auction_data(auctions_path)
    if df.empty:
        print("No auction data loaded")
        return

    print("=" * 60)
    print("TREASURY AUCTION DATA STATISTICS")
    print("=" * 60)
    print(f"Total auctions: {len(df)}")
    print(f"Date range: {df['auction_date'].min()} to {df['auction_date'].max()}")
    print()

    print("Auctions per tenor:")
    print(df.groupby('tenor').size().sort_index())
    print()

    print("Recent auctions (last 5):")
    recent = df.nlargest(5, 'auction_date')[['auction_date', 'tenor', 'btc_zscore', 'tail_zscore']]
    for _, row in recent.iterrows():
        print(f"  {row['auction_date'].date()} {row['tenor']:>4} | BTC z={row['btc_zscore']:+.2f} | Tail z={row['tail_zscore']:+.2f}")


if __name__ == "__main__":
    show_auction_stats()
