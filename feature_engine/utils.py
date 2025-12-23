import numpy as np
import pandas as pd
import json
import os
import config

def filter_survivors(df, config_path=None):
    if config_path is None:
        config_path = os.path.join(config.DIRS['FEATURES_DIR'], "survivors.json")

    if not os.path.exists(config_path):
        return df
        
    try:
        with open(config_path, "r") as f:
            survivors = json.load(f)
        
        # Always keep metadata columns
        meta = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume']
        cols_to_keep = meta + [c for c in survivors if c in df.columns]
        
        print(f"Loaded {len(survivors)} features from {config_path}.")
        filtered_df = df[cols_to_keep]
        print(f"Filtered to {len(cols_to_keep)} columns.")
        return filtered_df
        
    except Exception as e:
        print(f"Error loading survivor config: {e}. Keeping all.")
        return df
