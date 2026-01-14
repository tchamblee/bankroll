import pandas as pd
import numpy as np
import glob
import os
import config
from joblib import Parallel, delayed

def _process_ticker(ticker, raw_dir, clean_dir):
    try:
        out_name = f"CLEAN_{ticker}.parquet"
        out_path = os.path.join(clean_dir, out_name)
        
        # Find Source Files first to check timestamps
        # Prefer BARS (1-min aggregated data) over TICKS (raw tick data)
        # This matches the default config mode (BARS_TRADES_1MIN)
        bars_pattern = os.path.join(raw_dir, f"{config.RAW_DATA_PREFIX_BARS}_{ticker}*.parquet")
        ticks_pattern = os.path.join(raw_dir, f"{config.RAW_DATA_PREFIX_TICKS}_{ticker}*.parquet")

        bars_files = glob.glob(bars_pattern)
        ticks_files = glob.glob(ticks_pattern)

        # Use BARS if available, otherwise fall back to TICKS
        files = bars_files if bars_files else ticks_files
        if not files: return

        # Check for existing and freshness
        if os.path.exists(out_path):
            clean_mtime = os.path.getmtime(out_path)
            
            # Get max mtime of source files
            src_mtime = max([os.path.getmtime(f) for f in files])
            
            if clean_mtime > src_mtime:
                print(f"  ⏭️ {ticker}: Cleaned data is up to date. Skipping.")
                return
            else:
                print(f"  🔄 {ticker}: Source data detected as newer. Re-cleaning...")

        # Load All
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values("ts_event").reset_index(drop=True)
        
        # Identify Price
        price_col = None
        
        # Helper to check validity
        def has_data(col):
            return df[col].notna().sum() > 0

        if 'mid_price' in df.columns and has_data('mid_price'):
            price_col = 'mid_price'
        
        # Prioritize Bid/Ask if they have data (FX)
        elif 'pricebid' in df.columns and 'priceask' in df.columns and has_data('pricebid') and has_data('priceask'):
            df['mid_price'] = (df['pricebid'] + df['priceask']) / 2
            price_col = 'mid_price'
            
        # Fallback to Last Price (Stocks/Futures)
        elif 'last_price' in df.columns and has_data('last_price'):
            price_col = 'last_price'
        
        if not price_col:
            print(f"  ❌ No price column for {ticker}")
            return
            
        # Filter Bad Prices (Zero/Neg/NaN)
        valid_mask = (df[price_col] > 0) & df[price_col].notna()
        if (~valid_mask).sum() > 0:
            print(f"  🧹 {ticker}: Removing {(~valid_mask).sum()} invalid price rows (NaN/<=0)")
            df = df[valid_mask]
        
        # Filter Outliers (> 2% jump)
        pct_change = df[price_col].pct_change(fill_method=None).abs()
        outliers = pct_change > 0.02
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            print(f"  🧹 {ticker}: Removing {outlier_count} outliers (>2%)")
            df = df[~outliers]
        
        # Save Consolidated
        out_name = f"CLEAN_{ticker}.parquet"
        out_path = os.path.join(clean_dir, out_name)
        df.to_parquet(out_path, index=False)
        print(f"  ✅ {ticker}: Saved {out_name} ({len(df):,} ticks)")
        
    except Exception as e:
        print(f"  ❌ Error processing {ticker}: {e}")

def clean_data_lake():
    raw_dir = config.DIRS['DATA_RAW_TICKS']
    clean_dir = config.DIRS['DATA_CLEAN_TICKS']
    
    os.makedirs(clean_dir, exist_ok=True)
    
    print(f"Starting Parallel Data Lake Cleaning...")
    print(f"Source: {raw_dir}")
    print(f"Dest:   {clean_dir}")
    
    # Group by Ticker
    all_files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    tickers = sorted(list({os.path.basename(f).split('_')[2] for f in all_files}))
        
    print(f"Found Tickers: {tickers}")
    
    Parallel(n_jobs=-1)(
        delayed(_process_ticker)(ticker, raw_dir, clean_dir) for ticker in tickers
    )

if __name__ == "__main__":
    clean_data_lake()
