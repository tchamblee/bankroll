import pandas as pd
import numpy as np
import glob
import os

def check_ticker_health(ticker_pattern, name):
    print(f"\n--- Checking Health: {name} ({ticker_pattern}) ---")
    files = glob.glob(ticker_pattern)
    if not files:
        print("  ❌ No files found.")
        return

    try:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values("ts_event").reset_index(drop=True)
    except Exception as e:
        print(f"  ❌ Error reading files: {e}")
        return

    # 1. Basic Stats
    n = len(df)
    start = df['ts_event'].min()
    end = df['ts_event'].max()
    duration = end - start
    print(f"  ✅ Count: {n:,} ticks")
    print(f"  ✅ Range: {start} to {end} ({duration})")

    # 2. Price Integrity
    # Handle different column names (FX vs Stock)
    if 'mid_price' in df.columns:
        price = df['mid_price']
    elif 'pricebid' in df.columns:
        price = (df['pricebid'] + df['priceask']) / 2
    elif 'last_price' in df.columns:
        price = df['last_price']
    else:
        print("  ❌ No price column found!")
        return

    zeros = (price == 0).sum()
    negs = (price < 0).sum()
    nans = price.isna().sum()
    
    if zeros + negs + nans > 0:
        print(f"  ⚠️ Bad Prices: NaNs={nans}, Zeros={zeros}, Negs={negs}")
    else:
        print("  ✅ Price Integrity: Clean")

    # 3. Gap Detection (Time)
    # Calc diff in seconds
    time_diffs = df['ts_event'].diff().dt.total_seconds()
    
    # Define a "Gap" as > 1 hour (3600s) for active markets
    gaps = time_diffs[time_diffs > 3600]
    
    if len(gaps) > 0:
        print(f"  ⚠️ Gap Warning: Found {len(gaps)} gaps > 1 hour.")
        print(f"     Max Gap: {gaps.max()/3600:.2f} hours")
    else:
        print("  ✅ Continuity: No major gaps (>1h)")

    # 4. Outlier Detection (Flash Crashes/Bad Ticks)
    # Return > 5% in 1 tick?
    pct_change = price.pct_change().abs()
    outliers = pct_change[pct_change > 0.05] # 5% move in 1 tick is likely bad data
    
    if len(outliers) > 0:
        print(f"  ❌ FATAL: Found {len(outliers)} massive outliers (>5% 1-tick move).")
        print(f"     Example indices: {outliers.index[:5].tolist()}")
    else:
        print("  ✅ Volatility Check: No massive outliers")

if __name__ == "__main__":
    base_dir = "data/raw_ticks"
    
    targets = [
        (os.path.join(base_dir, "RAW_TICKS_EURUSD*.parquet"), "EUR/USD"),
        (os.path.join(base_dir, "RAW_TICKS_TNX*.parquet"), "TNX (US 10Y)"),
        (os.path.join(base_dir, "RAW_TICKS_DXY*.parquet"), "DXY (Dollar)"),
        (os.path.join(base_dir, "RAW_TICKS_BUND*.parquet"), "BUND (DE 10Y)"),
        (os.path.join(base_dir, "RAW_TICKS_SPY*.parquet"), "SPY (S&P 500)"),
    ]
    
    for pattern, name in targets:
        check_ticker_health(pattern, name)
