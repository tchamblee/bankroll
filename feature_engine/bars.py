import pandas as pd
import numpy as np

def create_volume_bars(primary_df, volume_threshold=1000):
    """
    Creates Volume Bars from the PRIMARY asset.
    Captures Start/End times for Exact-Interval analysis.
    """
    if primary_df is None: return None

    print(f"Generating Volume Bars (Threshold: {volume_threshold})...")
    
    df = primary_df.copy()
    
    # Volume Proxy Logic - Forcing Tick Bars (1 per row) for robust sampling
    vol_series = pd.Series(1, index=df.index)

    df['vol_proxy'] = vol_series
    df['cum_vol'] = vol_series.cumsum()
    df['bar_id'] = (df['cum_vol'] // volume_threshold).astype(int)
    
    # Aggressor Logic
    delta = df['mid_price'].diff().fillna(0)
    direction = np.sign(delta)
    df['aggressor_vol'] = direction * vol_series
    
    # Aggregation
    # We need first ts_event (start) and last ts_event (end)
    agg_dict = {
        'ts_event': ['first', 'last'],
        'mid_price': ['first', 'max', 'min', 'last'],
        'vol_proxy': 'sum',
        'aggressor_vol': 'sum'
    }
    
    # Add Bid/Ask Size aggregation if available
    if 'sizebid' in df.columns and 'sizeask' in df.columns:
        agg_dict['sizebid'] = 'mean'
        agg_dict['sizeask'] = 'mean'
        
    # Add Bid/Ask Price aggregation for Spread analysis
    if 'pricebid' in df.columns and 'priceask' in df.columns:
        agg_dict['pricebid'] = 'mean'
        agg_dict['priceask'] = 'mean'
        
    bars = df.groupby('bar_id').agg(agg_dict)
    
    # Flatten Columns
    # The order depends on the keys. Groupby sorts keys? No, usually follows order.
    # Safest way is to reconstruct based on what we added
    
    flat_cols = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol']
    if 'sizebid' in agg_dict:
        flat_cols.extend(['avg_bid_size', 'avg_ask_size'])
    if 'pricebid' in agg_dict:
        flat_cols.extend(['avg_bid_price', 'avg_ask_price'])
        
    bars.columns = flat_cols
    bars.reset_index(drop=True, inplace=True)
    
    # Basic Features
    bars['ticket_imbalance'] = np.where(bars['volume'] == 0, 0, bars['net_aggressor_vol'] / bars['volume'])
    
    print(f"Generated {len(bars)} Volume Bars.")
    return bars
