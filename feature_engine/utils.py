import os
import json
import pandas as pd

def filter_survivors(df, config_path="data/survivors.json"):
    """
    Retains only the Elite Survivors identified by the Purge (from JSON).
    """
    if df is None: return None
    
    # Default: Keep everything
    cols_to_keep = df.columns.tolist()
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                survivors = json.load(f)
            
            # Always keep metadata columns
            meta = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume']
            cols_to_keep = meta + [c for c in survivors if c in df.columns]
            
            print(f"Loaded {len(survivors)} features from {config_path}.")
        except Exception as e:
            print(f"Error loading survivor config: {e}. Keeping all.")
    else:
        print(f"Config {config_path} not found. Keeping all features.")

    filtered_df = df[cols_to_keep]
    print(f"Filtered to {len(cols_to_keep)} columns.")
    return filtered_df
