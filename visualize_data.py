import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import config
from feature_engine.loader import load_price_series

def plot_health_check():
    base_dir = config.DIRS['DATA_RAW_TICKS']
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
        # Pass data_dir explicity or let it fallback. 
        # Since pattern contains full path now (os.path.join above), we should pass data_dir=None or handle it.
        # Wait, load_price_series takes (pattern, name, resample, data_dir).
        # In loader.py: files = glob.glob(os.path.join(search_dir, pattern))
        # So it expects pattern to be relative to data_dir if data_dir is set, or relative to CLEAN/RAW defaults.
        
        # If I pass full path in pattern, and data_dir is None, it tries to join default dirs with full path pattern?
        # glob.glob(os.path.join(dir, /abs/path)) -> /abs/path (usually).
        # Let's check os.path.join behavior. os.path.join("foo", "/bar") -> "/bar".
        # So if I pass full path as pattern, it should work even if search_dir is something else.
        
        # However, loader.py logic:
        # if data_dir is None: search_dirs = [CLEAN, RAW]
        # for d in search_dirs: glob.glob(os.path.join(d, pattern))
        
        # If pattern is absolute (which it is here), os.path.join ignores d.
        # So it should work.
        
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