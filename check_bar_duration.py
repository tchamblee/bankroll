import pandas as pd
import numpy as np
from feature_engine import FeatureEngine
import config

def check_duration():
    engine = FeatureEngine(config.DIRS['DATA_RAW_TICKS'])
    
    # Load EUR/USD
    df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    
    # Create 500-tick bars
    print("Generating 500-tick bars...")
    engine.create_volume_bars(df, volume_threshold=500)
    bars = engine.bars.copy()
    
    # Calculate Duration
    bars['duration'] = (bars['time_end'] - bars['time_start']).dt.total_seconds() / 60 # Minutes
    
    print(f"\n--- Bar Duration Analysis (500 Ticks) ---")
    print(f"Count: {len(bars)}")
    print(f"Min Duration: {bars['duration'].min():.2f} min")
    print(f"Median Duration: {bars['duration'].median():.2f} min")
    print(f"Mean Duration: {bars['duration'].mean():.2f} min")
    print(f"Max Duration: {bars['duration'].max():.2f} min")
    
    # Check Active Session (Durations < 10 mins)
    active = bars[bars['duration'] < 10]
    print(f"\n--- Active Market Stats (Excluding quiet hours) ---")
    print(f"Active Bars: {len(active)} ({len(active)/len(bars):.1%})")
    print(f"Median Active Duration: {active['duration'].median():.2f} min")
    
    # Project Horizons
    median_min = bars['duration'].median()
    active_min = active['duration'].median()
    
    print(f"\n--- Projected Horizons (Median / Active) ---")
    for h in [60, 90, 120]:
        print(f"Horizon {h}: {h*median_min/60:.1f}h (Median) / {h*active_min/60:.1f}h (Active)")

if __name__ == "__main__":
    check_duration()
