import pandas as pd

def add_delta_features(df, lookback=10):
    if df is None: return None
    print(f"Calculating Delta Features (Lookback={lookback})...")
    df = df.copy()
    
    exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 
               'cum_vol', 'vol_proxy', 'bar_id', 'log_ret',
               'avg_bid_price', 'avg_ask_price', 'avg_bid_size', 'avg_ask_size',
               'ticket_imbalance', 'residual_bund', 'residual_tnx', 'residual_usdchf', 'residual_spy',
               'IBIT_Lag2_Return', 'us_curve', 'bund']

    def is_valid_feature(c):
        if c in exclude: return False
        if c.startswith('delta_'): return False
        if '_roc_' in c: return False
        if 'residual' in c: return False
        
        # NOISE FILTER: Exclude features where Delta is just double-noise or redundant
        noise_keywords = [
            'rel_vol', 'news', 'velocity', 
            'alignment', 'trin', 'vpin', 'voltage', 'epu', 'ibit', 
            'divergence', 'efficiency', 'atr',
            'order_book', 'smoothed_level'
        ]
        
        c_lower = c.lower()
        if any(k in c_lower for k in noise_keywords): return False
        
        return True

    # Also exclude existing delta columns to avoid delta_delta...
    feature_cols = [c for c in df.columns if is_valid_feature(c)]
    
    # To avoid fragmentation warnings, collect new columns in a list and concat at once
    new_cols = {}
    for col in feature_cols:
        new_cols[f'delta_{col}_{lookback}'] = df[col].diff(lookback)
    
    # Concat all new columns at once
    new_df = pd.DataFrame(new_cols, index=df.index)
    
    # Return new DF with columns attached
    return pd.concat([df, new_df], axis=1)
