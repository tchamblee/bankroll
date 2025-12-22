import pandas as pd
import numpy as np
from . import ticks

def create_volume_bars(primary_df, volume_threshold=1000):
    """
    Creates Volume Bars from the PRIMARY asset.
    Captures Start/End times for Exact-Interval analysis.
    """
    if primary_df is None: return None

    print(f"Generating Volume Bars (Threshold: {volume_threshold})...")
    
    df = primary_df.copy()

    # Pre-process Ticks for Mid Price and Microstructure
    if 'mid_price' not in df.columns and 'pricebid' in df.columns and 'priceask' in df.columns:
        df['mid_price'] = (df['pricebid'] + df['priceask']) / 2

    # Add Tick-Level Features (OFI, Spread, Mid-Vols)
    if 'pricebid' in df.columns and 'priceask' in df.columns:
        df = ticks.add_tick_microstructure_features(df)
    
    # Volume Proxy Logic - Use Actual Size if Available
    if 'sizebid' in df.columns and 'sizeask' in df.columns:
        # FX Tick Data: Sum of Bid/Ask sizes as proxy for liquidity/volume at that tick
        vol_series = (df['sizebid'].fillna(0) + df['sizeask'].fillna(0))
    elif 'last_size' in df.columns:
        vol_series = df['last_size'].fillna(1)
    elif 'volume' in df.columns:
        vol_series = df['volume'].fillna(1)
    else:
        # Fallback to Tick Count
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
    
    # Add Tick Aggregations if available
    if 'ofi' in df.columns:
        agg_dict['ofi'] = 'sum'
    if 'spread' in df.columns:
        agg_dict['spread'] = 'mean'
    
    # Add Bid/Ask Size aggregation if available
    if 'sizebid' in df.columns and 'sizeask' in df.columns:
        agg_dict['sizebid'] = 'mean'
        agg_dict['sizeask'] = 'mean'
        
    # Add Bid/Ask Price aggregation for Spread analysis
    if 'pricebid' in df.columns and 'priceask' in df.columns:
        agg_dict['pricebid'] = 'mean'
        agg_dict['priceask'] = 'mean'
        
    bars_df = df.groupby('bar_id').agg(agg_dict)
    
    # Flatten Columns
    # Map back to standard names
    new_cols = []
    for col, func in bars_df.columns:
        if col == 'ts_event':
            new_cols.append('time_start' if func == 'first' else 'time_end')
        elif col == 'mid_price':
            if func == 'first': new_cols.append('open')
            elif func == 'max': new_cols.append('high')
            elif func == 'min': new_cols.append('low')
            elif func == 'last': new_cols.append('close')
        elif col == 'vol_proxy': new_cols.append('volume')
        elif col == 'aggressor_vol': new_cols.append('net_aggressor_vol')
        elif col == 'ofi': new_cols.append('tick_ofi')
        elif col == 'spread' and func == 'mean': new_cols.append('tick_spread')
        elif col == 'sizebid': new_cols.append('avg_bid_size')
        elif col == 'sizeask': new_cols.append('avg_ask_size')
        elif col == 'pricebid': new_cols.append('avg_bid_price')
        elif col == 'priceask': new_cols.append('avg_ask_price')
        else: new_cols.append(f'{col}_{func}')
        
    bars_df.columns = new_cols
    bars_df.reset_index(drop=True, inplace=True)
    
    # Basic Features
    bars_df['ticket_imbalance'] = np.where(bars_df['volume'] == 0, 0, bars_df['net_aggressor_vol'] / bars_df['volume'])
    
    print(f"Generated {len(bars_df)} Volume Bars.")
    return bars_df
