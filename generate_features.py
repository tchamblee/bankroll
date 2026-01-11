import os
import pandas as pd
import config
import argparse
from feature_engine import create_full_feature_engine

def generate_feature_matrix(force=False):
    out_path = config.DIRS['FEATURE_MATRIX']
    if os.path.exists(out_path) and not force:
        print(f"⏩ Feature Matrix already exists at {out_path}. Skipping generation.")
        print("   (Use '--force' to override)")
        return

    # Remove verified marker if it exists (force re-verification)
    verified_marker = out_path + ".verified"
    if os.path.exists(verified_marker):
        os.remove(verified_marker)
        
    if force and os.path.exists(out_path):
        print(f"🔥 Force Mode: Removing existing matrix at {out_path}")
        os.remove(out_path)

    print("==============================================")
    print("🏭 GENERATING FULL FEATURE MATRIX")
    print("==============================================")
    
    # 1. Create Engine (Expensive Step)
    engine = create_full_feature_engine(config.DIRS['DATA_CLEAN_TICKS'], volume_threshold=config.VOLUME_THRESHOLD)
    
    if engine is None or engine.bars is None or len(engine.bars) == 0:
        print("❌ Error: Feature Engine failed to generate data.")
        exit(1)
        
    df = engine.bars

    # Ensure chronological order and remove zero-duration bars
    df = df.sort_values('time_start').reset_index(drop=True)
    if 'time_end' in df.columns:
        zero_duration = df['time_start'] == df['time_end']
        if zero_duration.sum() > 0:
            print(f"⚠️  Removing {zero_duration.sum()} zero-duration bars")
            df = df[~zero_duration].reset_index(drop=True)

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
    parser = argparse.ArgumentParser(description="Generate Full Feature Matrix")
    parser.add_argument("--force", action="store_true", help="Force regeneration of the feature matrix")
    args = parser.parse_args()
    
    generate_feature_matrix(force=args.force)
