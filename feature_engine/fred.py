import pandas as pd
import numpy as np
import os
import config

def add_fred_features(bars_df):
    """
    Merges FRED macro data (Net Liquidity, Credit Spreads, Breakevens) into the intraday feature matrix.
    Since FRED data is daily/weekly, we forward fill it to match the intraday bars.
    """
    fred_path = os.path.join(config.DIRS['DATA_DIR'], "fred_macro_daily.parquet")
    if not os.path.exists(fred_path):
        print(f"⚠️ FRED data not found at {fred_path}. Skipping FRED features.")
        return bars_df

    print("Loading FRED Macro Data...")
    fred_df = pd.read_parquet(fred_path)
    
    # Ensure datetime
    fred_df['date'] = pd.to_datetime(fred_df['date'])
    
    # Prepare bars_df for merge
    # We merge on DATE.
    # bars_df should have 'time_start'
    if 'time_start' not in bars_df.columns:
        print("⚠️ bars_df missing 'time_start'. Skipping FRED merge.")
        return bars_df

    # Extract date from bars
    bars_df['_merge_date'] = bars_df['time_start'].dt.normalize()
    
    # Merge
    # FRED data is End-of-Day usually.
    # ideally we should use yesterday's FRED data for today's trading to avoid lookahead bias?
    # FRED release times vary.
    # For weekly data (WALCL), it's released on Thurs for Wed data.
    # Lagging by 1 day is safe for Daily data.
    
    fred_df['date_shifted'] = fred_df['date'] + pd.Timedelta(days=1)
    
    merged = pd.merge(bars_df, fred_df, left_on='_merge_date', right_on='date_shifted', how='left')
    
    # Forward fill missing values (weekends, holidays, intraday)
    # Since we merged on date, all intraday bars for same day get same value (or NaN if missing).
    # But if FRED has gaps, we need to fill.
    # Actually, pd.merge 'left' keeps bars structure.
    # If a day is missing in FRED, we get NaNs. We should ffill.
    
    cols_to_fill = [c for c in fred_df.columns if c not in ['date', 'date_shifted']]
    merged[cols_to_fill] = merged[cols_to_fill].ffill()
    
    # Feature Engineering
    # 1. Net Liquidity Trend (20-day Change)
    if 'net_liquidity_bil' in merged.columns:
        # We need to be careful computing rolling stats on intraday data using daily values.
        # It's better to compute derived features on daily FRED df BEFORE merge, or use diff logic here.
        # Let's compute daily changes in FRED df first? No, we are already here.
        
        # Calculate changes based on the daily values distributed to bars.
        # Since values are constant intraday, standard rolling diff works but effectively measures day-to-day changes spread out?
        # No, rolling(window) on 5-min bars with daily data is weird.
        # Better: Calculate rolling features in FRED DF, then merge.
        pass
    
    # Cleanup
    merged.drop(columns=['_merge_date', 'date', 'date_shifted'], inplace=True, errors='ignore')
    
    return merged

def precompute_fred_derived(fred_df):
    """
    Computes rolling metrics on the daily FRED data.
    """
    df = fred_df.copy()
    df = df.sort_values('date')
    
    # 1. Net Liquidity Z-Score (Regime)
    if 'net_liquidity_bil' in df.columns:
        df['net_liq_zscore_60d'] = (df['net_liquidity_bil'] - df['net_liquidity_bil'].rolling(60).mean()) / df['net_liquidity_bil'].rolling(60).std()
        df['net_liq_trend_20d'] = df['net_liquidity_bil'].diff(20)

    # 2. Credit Stress Z-Score
    if 'credit_spread' in df.columns:
        df['credit_stress_zscore_60d'] = (df['credit_spread'] - df['credit_spread'].rolling(60).mean()) / df['credit_spread'].rolling(60).std()
        
    # 3. Inflation Expectations Trend
    if 'inflation_breakeven' in df.columns:
        df['inflation_expectations_trend'] = df['inflation_breakeven'].diff(20)

    return df

def add_fred_features_v2(bars_df):
    """
    Robust Version: Computes features on Daily FRED data first, then merges.
    """
    fred_path = os.path.join(config.DIRS['DATA_DIR'], "fred_macro_daily.parquet")
    if not os.path.exists(fred_path):
        return bars_df

    fred_df = pd.read_parquet(fred_path)
    # Ensure datetime and UTC timezone to match IBKR bars
    fred_df['date'] = pd.to_datetime(fred_df['date'])
    if fred_df['date'].dt.tz is None:
        fred_df['date'] = fred_df['date'].dt.tz_localize('UTC')
    else:
        fred_df['date'] = fred_df['date'].dt.tz_convert('UTC')
    
    # Compute derived features on DAILY data
    fred_df = precompute_fred_derived(fred_df)
    
    # Lag by 1 day to prevent lookahead
    fred_df['merge_date'] = fred_df['date'] + pd.Timedelta(days=1)
    
    # Prepare Bars
    if 'time_start' not in bars_df.columns: return bars_df
    # Normalize bar times to midnight UTC for merging
    bars_df['_date_only'] = bars_df['time_start'].dt.normalize()
    
    # Merge
    # Select only feature columns + merge key
    cols_to_use = ['merge_date', 'net_liquidity_bil', 'net_liq_zscore_60d', 'net_liq_trend_20d', 
                   'credit_spread', 'credit_stress_zscore_60d', 'inflation_breakeven', 'inflation_expectations_trend']
    cols_present = [c for c in cols_to_use if c in fred_df.columns]
    
    merged = pd.merge(bars_df, fred_df[cols_present], left_on='_date_only', right_on='merge_date', how='left')
    
    # Forward Fill (propagate last known daily value to today's intraday bars)
    # Note: If today is Monday, we merge with Sunday(shifted Sat data) -> might be NaN if FRED is M-F.
    # So we need to ffill across the merged result to handle weekends/holidays gaps.
    feat_cols = [c for c in cols_present if c != 'merge_date']
    merged[feat_cols] = merged[feat_cols].ffill()
    
    merged.drop(columns=['_date_only', 'merge_date'], inplace=True, errors='ignore')
    return merged
