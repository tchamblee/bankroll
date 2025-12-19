import pandas as pd
import numpy as np
import glob
import os

def clean_file(file_path, output_dir, threshold=0.02):
    try:
        df = pd.read_parquet(file_path)
        original_len = len(df)
        
        # Identify price column
        price_col = None
        if 'mid_price' in df.columns:
            price_col = 'mid_price'
        elif 'last_price' in df.columns:
            price_col = 'last_price'
        elif 'pricebid' in df.columns and 'priceask' in df.columns:
            # Create mid_price if missing for cleaning
            df['mid_price'] = (df['pricebid'] + df['priceask']) / 2
            price_col = 'mid_price'
            
        if price_col:
            # 1. Filter Outliers (> threshold jump)
            pct_change = df[price_col].pct_change(fill_method=None).abs()
            outliers = pct_change > threshold
            outlier_count = outliers.sum()
            
            if "EURUSD" in file_path and outlier_count > 0:
                print(f"  🚨 Found {outlier_count} outliers in EURUSD file: {file_path}")
            
            if outlier_count > 0:
                print(f"  🧹 Removing {outlier_count} outliers from {os.path.basename(file_path)}")
                df = df[~outliers]
                
            # 2. Filter Bad Prices (Zero/Negative)
            bad_prices = (df[price_col] <= 0)
            bad_count = bad_prices.sum()
            if bad_count > 0:
                print(f"  🧹 Removing {bad_count} zero/neg prices")
                df = df[~bad_prices]
        
        # Save to Clean Dir
        filename = os.path.basename(file_path)
        out_path = os.path.join(output_dir, filename)
        df.to_parquet(out_path, index=False)
        
        # print(f"  ✅ Saved {filename} ({len(df)} ticks)")
        
    except Exception as e:
        print(f"  ❌ Error cleaning {file_path}: {e}")

def clean_data_lake():
    raw_dir = "data/raw_ticks"
    clean_dir = "data/clean_ticks"
    
    os.makedirs(clean_dir, exist_ok=True)
    
    print(f"Starting Data Lake Cleaning (Aggregated)...")
    print(f"Source: {raw_dir}")
    print(f"Dest:   {clean_dir}")
    
    # Group by Ticker
    all_files = glob.glob(os.path.join(raw_dir, "*.parquet"))
    tickers = set()
    for f in all_files:
        # Filename: RAW_TICKS_NAME_DATE.parquet
        name = os.path.basename(f).split('_')[2] # EURUSD, TNX, etc
        tickers.add(name)
        
    print(f"Found Tickers: {tickers}")
    
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        pattern = os.path.join(raw_dir, f"RAW_TICKS_{ticker}*.parquet")
        files = glob.glob(pattern)
        
        if not files: continue
        
        try:
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
                continue
                
            # Filter Bad Prices (Zero/Neg/NaN)
            # Important: Filter NaNs first so they don't break pct_change logic implicitly
            valid_mask = (df[price_col] > 0) & df[price_col].notna()
            if (~valid_mask).sum() > 0:
                print(f"  🧹 Removing {(~valid_mask).sum()} invalid price rows (NaN/<=0)")
                df = df[valid_mask]
            
            # Filter Outliers (> 2% jump)
            # fill_method=None to allow calc, but we just removed NaNs so it's safe
            pct_change = df[price_col].pct_change(fill_method=None).abs()
            outliers = pct_change > 0.02
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"  🧹 Removing {outlier_count} outliers (>2%)")
                df = df[~outliers]
            
            # Save Consolidated
            out_name = f"CLEAN_{ticker}.parquet"
            out_path = os.path.join(clean_dir, out_name)
            df.to_parquet(out_path, index=False)
            print(f"  ✅ Saved {out_name}: {len(df):,} ticks")
            
        except Exception as e:
            print(f"  ❌ Error processing {ticker}: {e}")

if __name__ == "__main__":
    clean_data_lake()
