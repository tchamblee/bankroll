import os
import pandas as pd
import config
from feature_engine import create_full_feature_engine

def generate_feature_matrix():
    out_path = config.DIRS['FEATURE_MATRIX']
    if os.path.exists(out_path):
        print(f"⏩ Feature Matrix already exists at {out_path}. Skipping generation.")
        return

    # Remove verified marker if it exists (force re-verification)
    verified_marker = out_path + ".verified"
    if os.path.exists(verified_marker):
        os.remove(verified_marker)

    print("==============================================")
    print("🏭 GENERATING FULL FEATURE MATRIX")
    print("==============================================")
    
    # 1. Create Engine (Expensive Step)
    # Using 1 Billion units as Volume Threshold for EURUSD Volume Bars
    engine = create_full_feature_engine(config.DIRS['DATA_CLEAN_TICKS'], volume_threshold=1_000_000_000)
    
    if engine is None or engine.bars is None or len(engine.bars) == 0:
        print("❌ Error: Feature Engine failed to generate data.")
        exit(1)
        
    df = engine.bars
    print(f"\n📊 Generated Matrix Shape: {df.shape}")
    print(f"📅 Date Range: {df['time_start'].min()} to {df['time_start'].max()}")
    
    # 2. Save to Parquet
    out_path = config.DIRS['FEATURE_MATRIX']
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    print(f"💾 Saving to {out_path}...")
    df.to_parquet(out_path, index=False)
    print("✅ Feature Matrix Saved Successfully.")

if __name__ == "__main__":
    generate_feature_matrix()
