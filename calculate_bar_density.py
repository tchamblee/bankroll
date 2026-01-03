import pandas as pd
import numpy as np
import os
import config

def calculate_density():
    path = config.DIRS['FEATURE_MATRIX']
    print(f"📂 Loading: {path}")
    
    if not os.path.exists(path):
        print("❌ Error: Feature Matrix not found.")
        return
        
    # Read only time column for speed
    df = pd.read_parquet(path, columns=['time_start'])
    
    if 'time_start' not in df.columns:
        print("❌ Error: 'time_start' column missing.")
        return
        
    # Convert to datetime if needed
    times = pd.to_datetime(df['time_start'], utc=True)
    
    total_bars = len(times)
    start_date = times.min()
    end_date = times.max()
    
    # Calculate unique trading days
    # We use .date to ignore time components
    unique_days = times.dt.date.nunique()
    
    print(f"\n📊 DATASET STATS")
    print(f"=================")
    print(f"Start Date: {start_date}")
    print(f"End Date:   {end_date}")
    print(f"Total Bars: {total_bars}")
    print(f"Unique Trading Days: {unique_days}")
    
    # METRIC 1: Bars per Active Trading Day
    if unique_days == 0:
        print("❌ Error: No unique days found.")
        return

    bars_per_trading_day = total_bars / unique_days
    
    # METRIC 2: Average Duration
    diffs = times.diff().dropna()
    avg_duration_sec = diffs.dt.total_seconds().mean()
    median_duration_sec = diffs.dt.total_seconds().median()
    
    print(f"\n⏱️  BAR DURATION")
    print(f"=================")
    print(f"Average: {avg_duration_sec:.2f} seconds ({avg_duration_sec/60:.2f} mins)")
    print(f"Median:  {median_duration_sec:.2f} seconds ({median_duration_sec/60:.2f} mins)")
    
    # CALCULATION
    factor = bars_per_trading_day * 252
    
    print(f"\n🧮 CALCULATION")
    print(f"==============")
    print(f"Bars Per Trading Day: {bars_per_trading_day:.2f}")
    print(f"Trading Days / Year:  252")
    print(f"Calculated Factor:    {factor:.2f}")
    print(f"Current Config:       {config.ANNUALIZATION_FACTOR}")
    
    print(f"\n👉 RECOMMENDED ACTION:")
    print(f"Set ANNUALIZATION_FACTOR = {int(factor)}")

if __name__ == "__main__":
    calculate_density()