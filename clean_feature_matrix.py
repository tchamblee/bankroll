import pandas as pd
import json
import os
import config
import glob

def clean_feature_matrix():
    print("🧹 Starting Feature Matrix Cleanup...")
    
    # 1. Load all survivor lists
    survivor_files = glob.glob(os.path.join(config.DIRS['FEATURES_DIR'], "survivors_*.json"))
    if not survivor_files:
        print("❌ No survivor files found. Run purge_features.py first.")
        return

    all_survivors = set()
    for f in survivor_files:
        with open(f, 'r') as json_file:
            features = json.load(json_file)
            all_survivors.update(features)
            
    print(f"✅ Loaded {len(survivor_files)} survivor lists.")
    print(f"🛡️  Union of Survivors: {len(all_survivors)} unique features.")
    
    # 2. Load Feature Matrix
    print(f"Loading {config.DIRS['FEATURE_MATRIX']}...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    original_count = len(df.columns)
    
    # 3. Define Metadata columns to always keep
    metadata = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 
                'net_aggressor_vol', 'cum_vol', 'vol_proxy', 'bar_id', 'log_ret', 
                'avg_bid_price', 'avg_ask_price', 'avg_bid_size', 'avg_ask_size', 
                'avg_spread', 'ticket_imbalance', 'timestamp']
    
    # Also keep any columns that are in the dataframe but not in our "candidate" set 
    # (just to be safe, though the purge script defines candidates by exclusion)
    # Ideally, we just keep Metadata + Survivors.
    
    # Check which metadata columns actually exist in df
    existing_metadata = [c for c in metadata if c in df.columns]
    
    # Final Keep List
    keep_cols = list(set(existing_metadata).union(all_survivors))
    
    # 4. Filter
    clean_df = df[keep_cols].copy()
    
    print(f"✂️  Removed {original_count - len(clean_df.columns)} features.")
    print(f"✨ New Feature Count: {len(clean_df.columns)}")
    
    # 5. Save
    # We'll overwrite the existing file so purge_features.py uses the clean version next time
    clean_df.to_parquet(config.DIRS['FEATURE_MATRIX'])
    print(f"💾 Saved cleaned matrix to {config.DIRS['FEATURE_MATRIX']}")

if __name__ == "__main__":
    clean_feature_matrix()