import pandas as pd
import numpy as np
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def verify_top5_features():
    # print("\n🔍 STARTING TOP 5 FEATURE AUDIT 🔍")
    
    # Check if Feature Matrix exists
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print(f"❌ Feature Matrix not found at {config.DIRS['FEATURE_MATRIX']}")
        sys.exit(1)
        
    # print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
    try:
        df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
        # print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Failed to load Feature Matrix: {e}")
        sys.exit(1)

    for horizon in config.PREDICTION_HORIZONS:
        # print(f"\n--- Auditing Horizon: {horizon} ---")
        
        survivors_file = os.path.join(config.DIRS['FEATURES_DIR'], f"survivors_{horizon}.json")
        if not os.path.exists(survivors_file):
            print(f"⚠️ Survivors file not found: {survivors_file}. Skipping.")
            continue
            
        try:
            with open(survivors_file, 'r') as f:
                survivors = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {survivors_file}: {e}")
            continue
            
        if not survivors:
            print("⚠️ No survivors found.")
            continue
            
        top5 = survivors[:5]
        # print(f"Top 5 Features: {top5}")
        
        for feature in top5:
            if feature not in df.columns:
                print(f"❌ Feature '{feature}' missing from matrix!")
                continue
                
            series = df[feature]
            
            # Basic Stats
            n_nans = series.isna().sum()
            std_val = series.std()
            
            # Simple Logic Checks
            if std_val == 0 or np.isnan(std_val):
                print(f"     ❌ {feature} (H{horizon}): Zero Variance or NaN Std (Dead Feature)")
            elif n_nans > len(df) * 0.1:
                print(f"     ⚠️ {feature} (H{horizon}): High NaN count (>10%)")
            # else:
            #    print("     ✅ Feature looks healthy")

if __name__ == "__main__":
    verify_top5_features()
