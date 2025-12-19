import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_price_series(pattern, name, resample='1h'):
    print(f"Loading {name}...")
    files = glob.glob(pattern)
    if not files:
        print(f"  ❌ No files for {name}")
        return None
    
    try:
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.sort_values("ts_event").set_index("ts_event")
        
        # Handle price column with data validity check
        price = None
        
        # 1. Try Mid Price
        if 'mid_price' in df.columns and not df['mid_price'].isna().all():
            price = df['mid_price']
            
        # 2. Try Bid/Ask (if Mid failed)
        elif 'pricebid' in df.columns and 'priceask' in df.columns:
            # Check if they have data
            if not df['pricebid'].isna().all() and not df['priceask'].isna().all():
                price = (df['pricebid'] + df['priceask']) / 2
        
        # 3. Try Last Price (Fallback or if others failed)
        if price is None and 'last_price' in df.columns and not df['last_price'].isna().all():
            price = df['last_price']
            
        if price is None:
            print(f"  ❌ No valid price data for {name}")
            return None
            
        # Resample to common grid
        resampled = price.resample(resample).last().ffill()
        return resampled
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def plot_health_check():
    base_dir = "data/raw_ticks"
    assets = {
        "EUR/USD": os.path.join(base_dir, "RAW_TICKS_EURUSD*.parquet"),
        "TNX (Yields)": os.path.join(base_dir, "RAW_TICKS_TNX*.parquet"),
        "DXY (Dollar)": os.path.join(base_dir, "RAW_TICKS_DXY*.parquet"),
        "BUND (DE 10Y)": os.path.join(base_dir, "RAW_TICKS_BUND*.parquet"),
        "SPY (Risk)": os.path.join(base_dir, "RAW_TICKS_SPY*.parquet")
    }
    
    # Find common start date (Max of all start dates)
    loaded_data = {}
    start_dates = []
    
    for name, pattern in assets.items():
        data = load_price_series(pattern, name)
        if data is not None:
            loaded_data[name] = data
            start_dates.append(data.index.min())
            
    if not start_dates:
        print("No data loaded.")
        return

    common_start = max(start_dates)
    print(f"\nZooming into common period starting: {common_start}")
    
    plt.figure(figsize=(15, 10))
    
    for name, data in loaded_data.items():
        # Slice to common period
        subset = data[data.index >= common_start]
        print(f"  {name}: {len(data)} -> {len(subset)} points")
        
        # Normalize to 0 at start of this window
        if len(subset) > 0:
            first_valid = subset.first_valid_index()
            if first_valid is None:
                print(f"  ⚠️ {name} skipped: All NaNs in slice")
                continue
                
            start_price = subset.loc[first_valid]
            
            if start_price > 0:
                # We align visually to the first valid point
                subset = (subset / start_price) - 1
                plt.plot(subset, label=name, linewidth=1.5)
            else:
                print(f"  ⚠️ {name} skipped: Start price is {start_price}")
            
    plt.title(f"Data Lake Health Check: Normalized Returns (Zoomed from {common_start.date()})", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = os.path.join("data", "data_health_check.png")
    plt.savefig(output_file)
    print(f"\n📸 Chart saved to {output_file}")

if __name__ == "__main__":
    plot_health_check()