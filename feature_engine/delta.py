import pandas as pd

def add_delta_features(df, lookback=10):
    if df is None: return None
    print(f"Calculating Delta Features (Lookback={lookback})...")
    df = df.copy()
    
    exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 
               'cum_vol', 'vol_proxy', 'bar_id', 'log_ret',
               'avg_bid_price', 'avg_ask_price', 'avg_bid_size', 'avg_ask_size',
               'ticket_imbalance', 'residual_bund']
    # Also exclude existing delta columns to avoid delta_delta...
    feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('delta_')]
    
    # To avoid fragmentation warnings, collect new columns in a list and concat at once
    new_cols = {}
    for col in feature_cols:
        new_cols[f'delta_{col}_{lookback}'] = df[col].diff(lookback)
    
    # Concat all new columns at once
    new_df = pd.DataFrame(new_cols, index=df.index)
    
    # Return new DF with columns attached
    return pd.concat([df, new_df], axis=1)
