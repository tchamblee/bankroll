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
        
        # Skip if already exists
        if os.path.exists(out_path):
            print(f"  ⏭️ {ticker}: Cleaned data already exists. Skipping.")
            return

        if ticker == 'EURUSD':
            pattern = os.path.join(raw_dir, f"{config.RAW_DATA_PREFIX_TICKS}_{ticker}*.parquet")
        else:
            pattern = os.path.join(raw_dir, f"{config.RAW_DATA_PREFIX_BARS}_{ticker}*.parquet")
            
        files = glob.glob(pattern)
        
        if not files: return
        
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
