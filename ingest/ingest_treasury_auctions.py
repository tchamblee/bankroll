#!/usr/bin/env python3
"""
Treasury Auction Data Ingest

Fetches auction results from fiscaldata.treasury.gov API.
Tracks bid-to-cover ratio and tail (high_yield - avg_yield) for Notes and Bonds.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import config

# API Configuration
API_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"

# Security terms we care about (most market-moving)
KEY_TENORS = {
    '2-Year': '2y',
    '3-Year': '3y',
    '5-Year': '5y',
    '7-Year': '7y',
    '10-Year': '10y',
    '20-Year': '20y',
    '30-Year': '30y',
    # Handle variations
    '9-Year 10-Month': '10y',
    '9-Year 11-Month': '10y',
    '19-Year 10-Month': '20y',
    '19-Year 11-Month': '20y',
    '29-Year 10-Month': '30y',
    '29-Year 11-Month': '30y',
}

OUTPUT_FILE = Path(config.DIRS['DATA_DIR']) / "treasury_auctions.parquet"


def fetch_auction_data(start_date: str = "2020-01-01") -> pd.DataFrame:
    """Fetch Treasury auction data from fiscal data API."""

    fields = [
        "auction_date", "security_type", "security_term",
        "bid_to_cover_ratio", "high_yield", "avg_med_yield",
        "offering_amt", "total_tendered", "total_accepted"
    ]

    params = {
        "filter": f"auction_date:gte:{start_date},security_type:in:(Note,Bond)",
        "fields": ",".join(fields),
        "page[size]": 1000,
        "sort": "-auction_date",
        "format": "json"
    }

    all_data = []
    page = 1

    while True:
        params["page[number]"] = page
        response = requests.get(API_BASE, params=params)

        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            break

        data = response.json()
        records = data.get("data", [])

        if not records:
            break

        all_data.extend(records)

        # Check if more pages
        meta = data.get("meta", {})
        total_pages = meta.get("total-pages", 1)

        if page >= total_pages:
            break

        page += 1

    return pd.DataFrame(all_data)


def process_auction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw auction data into features."""

    if df.empty:
        return pd.DataFrame()

    # Convert types
    df['auction_date'] = pd.to_datetime(df['auction_date'])

    # Convert numeric columns (API returns strings)
    numeric_cols = ['bid_to_cover_ratio', 'high_yield', 'avg_med_yield',
                    'offering_amt', 'total_tendered', 'total_accepted']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('null', None), errors='coerce')

    # Filter to auctions with results (not just announced)
    df = df[df['bid_to_cover_ratio'].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    # Map security terms to standard tenors
    df['tenor'] = df['security_term'].map(KEY_TENORS)
    df = df[df['tenor'].notna()].copy()

    # Calculate tail (high_yield - avg_yield, in bps)
    # Positive tail = weak auction (had to pay up)
    df['tail_bps'] = (df['high_yield'] - df['avg_med_yield']) * 100

    # Calculate bid-to-cover z-score by tenor
    # (how this auction compared to historical average for same tenor)
    df = df.sort_values('auction_date')

    results = []
    for tenor in df['tenor'].unique():
        tenor_df = df[df['tenor'] == tenor].copy()

        # Rolling stats (use 20 auctions = ~1 year for most tenors)
        tenor_df['btc_mean_20'] = tenor_df['bid_to_cover_ratio'].rolling(20, min_periods=5).mean()
        tenor_df['btc_std_20'] = tenor_df['bid_to_cover_ratio'].rolling(20, min_periods=5).std()
        tenor_df['btc_zscore'] = (tenor_df['bid_to_cover_ratio'] - tenor_df['btc_mean_20']) / tenor_df['btc_std_20'].replace(0, 1)

        # Same for tail
        tenor_df['tail_mean_20'] = tenor_df['tail_bps'].rolling(20, min_periods=5).mean()
        tenor_df['tail_std_20'] = tenor_df['tail_bps'].rolling(20, min_periods=5).std()
        tenor_df['tail_zscore'] = (tenor_df['tail_bps'] - tenor_df['tail_mean_20']) / tenor_df['tail_std_20'].replace(0, 1)

        results.append(tenor_df)

    df = pd.concat(results, ignore_index=True)

    # Select final columns
    output_cols = [
        'auction_date', 'security_type', 'tenor',
        'bid_to_cover_ratio', 'btc_zscore',
        'high_yield', 'tail_bps', 'tail_zscore'
    ]

    return df[output_cols].sort_values('auction_date')


def ingest_treasury_auctions(lookback_years: int = 5):
    """Main ingest function."""

    start_date = (datetime.now() - timedelta(days=365 * lookback_years)).strftime('%Y-%m-%d')

    print(f"Fetching Treasury auction data since {start_date}...")
    raw_df = fetch_auction_data(start_date)

    if raw_df.empty:
        print("No auction data fetched")
        return

    print(f"Fetched {len(raw_df)} raw auction records")

    processed_df = process_auction_data(raw_df)

    if processed_df.empty:
        print("No valid auction data after processing")
        return

    print(f"Processed {len(processed_df)} auctions with results")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

    # Show summary
    print("\nAuction Summary by Tenor:")
    summary = processed_df.groupby('tenor').agg({
        'bid_to_cover_ratio': ['mean', 'std', 'count'],
        'tail_bps': ['mean', 'std']
    }).round(3)
    print(summary)

    print("\nRecent Auctions:")
    print(processed_df.tail(10).to_string())


if __name__ == "__main__":
    ingest_treasury_auctions()
