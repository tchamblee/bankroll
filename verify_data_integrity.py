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
    # Handle different column names with validity check
    price = None
    
    # Helper to check validity
    def has_data(col):
        return col in df.columns and df[col].notna().sum() > 0
    
    # Try Mid Price
    if has_data('mid_price'):
        price = df['mid_price']
    # Try Bid/Ask
    elif has_data('pricebid') and has_data('priceask'):
        price = (df['pricebid'] + df['priceask']) / 2
    # Try Last Price
    elif has_data('last_price'):
        price = df['last_price']
        
    if price is None:
        print("  ❌ No valid price column found!")
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
    
    # Define thresholds
    GAP_THRESHOLD = 3600 # 1 hour
    OVERNIGHT_MAX = 20 * 3600 # 20 hours (Standard daily close)
    CLOSURE_MAX = 100 * 3600 # 100 hours (Long weekends / Holidays)
    
    # Find all gaps > 1 hour
    large_gaps = time_diffs[time_diffs > GAP_THRESHOLD]
    
    if len(large_gaps) > 0:
        overnights = 0
        closures = 0
        anomalies = []
        
        for idx, duration in large_gaps.items():
            if duration < OVERNIGHT_MAX:
                overnights += 1
            elif duration < CLOSURE_MAX:
                closures += 1
            else:
                timestamp = df.loc[idx, 'ts_event']
                anomalies.append((timestamp, duration))
        
        # Report
        print(f"  ℹ️  Gaps > 1h: {len(large_gaps)} (Overnight: {overnights}, Closures: {closures})")
        
        if len(anomalies) > 0:
            print(f"  ⚠️ UNEXPECTED GAPS (>100h): Found {len(anomalies)} anomalies!")
            for ts, dur in anomalies[:3]:
                print(f"     - {ts}: {dur/3600:.2f} hours")
            if len(anomalies) > 3: print(f"     ... (+ {len(anomalies)-3} more)")
        else:
            print("  ✅ Continuity: All gaps are expected closures.")
    else:
        print("  ✅ Continuity: Continuous stream.")

    # 4. Outlier Detection (Flash Crashes/Bad Ticks)
    # Return > 5% in 1 tick?
    pct_change = price.pct_change(fill_method=None).abs()
    outliers = pct_change[pct_change > 0.05] # 5% move in 1 tick is likely bad data
    
    if len(outliers) > 0:
        print(f"  ❌ FATAL: Found {len(outliers)} massive outliers (>5% 1-tick move).")
        print(f"     Example indices: {outliers.index[:5].tolist()}")
    else:
        print("  ✅ Volatility Check: No massive outliers")

if __name__ == "__main__":
    # Point to CLEAN ticks now
    base_dir = "data/clean_ticks"
    
    targets = [
        (os.path.join(base_dir, "CLEAN_EURUSD.parquet"), "EUR/USD"),
        (os.path.join(base_dir, "CLEAN_TNX.parquet"), "TNX (US 10Y)"),
        (os.path.join(base_dir, "CLEAN_DXY.parquet"), "DXY (Dollar)"),
        (os.path.join(base_dir, "CLEAN_BUND.parquet"), "BUND (DE 10Y)"),
        (os.path.join(base_dir, "CLEAN_SPY.parquet"), "SPY (S&P 500)"),
    ]
    
    for pattern, name in targets:
        check_ticker_health(pattern, name)
