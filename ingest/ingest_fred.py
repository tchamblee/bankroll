import os
import sys
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from shared import get_logger, save_data

# Setup Logging
logger = get_logger("FRED_Ingest", "ingest_fred.log")

FRED_API_KEY = os.environ.get("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Mapping of Series ID to Friendly Name
SERIES_MAP = {
    'WALCL': 'fed_assets',        # Weekly (Wed)
    'WTREGEN': 'tga_balance',     # Weekly (Wed) - Treasury General Account
    'RRPONTSYD': 'reverse_repo',  # Daily - Overnight Reverse Repurchase Agreements
    'BAMLC0A0CM': 'credit_spread',# Daily - ICE BofA US Corp Master Option-Adjusted Spread
    'T10YIE': 'inflation_breakeven', # Daily - 10-Year Breakeven Inflation Rate
    'VIXCLS': 'vix',              # Daily - CBOE Volatility Index
    'VXVCLS': 'vix3m',            # Daily - CBOE 3-Month Volatility Index
    'NFCI': 'financial_conditions', # Weekly - Chicago Fed National Financial Conditions Index
    'DFF': 'fed_funds_rate',      # Daily - Federal Funds Effective Rate
    'ECBDFR': 'ecb_deposit_rate', # Daily - ECB Deposit Facility Rate
}

def fetch_series(series_id, start_date=None):
    """
    Fetches data for a single series from FRED.
    """
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not found in environment variables. Skipping FRED fetch.")
        return None

    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'sort_order': 'asc'
    }
    
    if start_date:
        params['observation_start'] = start_date

    try:
        response = requests.get(FRED_BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        observations = data.get('observations', [])
        if not observations:
            logger.warning(f"No observations found for {series_id}")
            return None
            
        df = pd.DataFrame(observations)
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df[['date', 'value']].rename(columns={'value': SERIES_MAP[series_id]})
        df = df.dropna()
        return df.set_index('date')
        
    except Exception as e:
        logger.error(f"Error fetching {series_id}: {e}")
        return None

def ingest_fred_data(lookback_days=365*5):
    """
    Fetches all configured FRED series, merges them, handles frequency mismatches,
    calculates derived metrics (Net Liquidity), and saves to parquet.
    """
    if not FRED_API_KEY:
        logger.error("Cannot run ingest_fred_data: FRED_API_KEY missing.")
        return

    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    logger.info(f"Fetching FRED data since {start_date}...")
    
    dfs = []
    for series_id in SERIES_MAP.keys():
        df = fetch_series(series_id, start_date)
        if df is not None:
            dfs.append(df)
            logger.info(f"Fetched {series_id}: {len(df)} rows")
    
    if not dfs:
        logger.warning("No FRED data fetched.")
        return

    # Merge all series on Date index (Outer Join to keep all timestamps)
    merged_df = pd.concat(dfs, axis=1)
    
    # Forward Fill to handle frequency mismatches (Weekly -> Daily)
    # WALCL and WTREGEN are weekly. RRP is daily.
    merged_df = merged_df.sort_index().ffill()
    
    # Calculate Net Liquidity
    # Net Liquidity = Fed Assets - TGA - RRP
    # Note: Units might differ.
    # WALCL: Millions of Dollars
    # WTREGEN: Billions of Dollars -> Convert to Millions (* 1000)
    # RRPONTSYD: Billions of Dollars -> Convert to Millions (* 1000)
    
    # Normalize units to Billions
    if 'fed_assets' in merged_df.columns:
        merged_df['fed_assets_bil'] = merged_df['fed_assets'] / 1000.0
    
    cols_needed = ['fed_assets_bil', 'tga_balance', 'reverse_repo']
    if all(c in merged_df.columns for c in cols_needed):
        merged_df['net_liquidity_bil'] = (
            merged_df['fed_assets_bil'] -
            merged_df['tga_balance'] -
            merged_df['reverse_repo']
        )
        logger.info("Calculated Net Liquidity")
    else:
        logger.warning(f"Could not calculate Net Liquidity. Missing columns. Present: {merged_df.columns}")

    # Calculate USD-EUR Rate Differential (proxy for cross-currency basis)
    if 'fed_funds_rate' in merged_df.columns and 'ecb_deposit_rate' in merged_df.columns:
        merged_df['usd_eur_rate_diff'] = merged_df['fed_funds_rate'] - merged_df['ecb_deposit_rate']
        logger.info("Calculated USD-EUR Rate Differential")

    # Reset Index for saving
    merged_df = merged_df.reset_index().rename(columns={'index': 'date'})
    
    # Save to Parquet
    save_data(merged_df, "fred_macro_daily.parquet", logger)

if __name__ == "__main__":
    # Test Run
    ingest_fred_data()
