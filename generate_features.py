import os
import pandas as pd
import config
from feature_engine import create_full_feature_engine

def generate_feature_matrix():
    print("==============================================")
    print("🏭 GENERATING FULL FEATURE MATRIX")
    print("==============================================")
    
    # 1. Create Engine (Expensive Step)
    # Note: We pass the RAW_TICKS dir because the loader looks for "RAW_..." files 
    # but internally redirects to "processed_data/clean_ticks" if they exist.
    engine = create_full_feature_engine(config.DIRS['DATA_RAW_TICKS'])
    
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
