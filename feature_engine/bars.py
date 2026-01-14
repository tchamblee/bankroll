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
    # Helper to check if column has meaningful data (not all NaN/zero)
    def has_data(col):
        return col in df.columns and df[col].notna().any() and (df[col].fillna(0) != 0).any()

    if has_data('sizebid') and has_data('sizeask'):
        # FX Tick Data: Sum of Bid/Ask sizes as proxy for liquidity/volume at that tick
        vol_series = (df['sizebid'].fillna(0) + df['sizeask'].fillna(0))
    elif has_data('last_size'):
        vol_series = df['last_size'].fillna(1)
    elif has_data('volume'):
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


def create_volume_bars_from_1min(df, volume_threshold=1000):
    """
    Creates Volume Bars from 1-minute OHLCV bars.
    This is used for consistency between backfill (1-min bars) and live trading.

    Args:
        df: DataFrame with columns [ts_event, open, high, low, close, volume]
        volume_threshold: Contracts per volume bar

    Returns:
        DataFrame with volume bars
    """
    if df is None or len(df) == 0:
        return None

    print(f"Generating Volume Bars from 1-min data (Threshold: {volume_threshold})...")

    df = df.copy()

    # Ensure ts_event column exists
    if 'ts_event' not in df.columns:
        if 'time' in df.columns:
            df['ts_event'] = df['time']
        elif 'datetime' in df.columns:
            df['ts_event'] = df['datetime']
        else:
            print("ERROR: No timestamp column found in 1-min bars")
            return None

    # Ensure volume column exists
    if 'volume' not in df.columns:
        print("ERROR: No volume column found in 1-min bars")
        return None

    # Sort by time
    df = df.sort_values('ts_event').reset_index(drop=True)

    # Calculate cumulative volume and bar assignments
    df['cum_vol'] = df['volume'].cumsum()
    df['bar_id'] = (df['cum_vol'] // volume_threshold).astype(int)

    # Calculate net aggressor volume from price direction
    df['price_change'] = df['close'].diff().fillna(0)
    df['aggressor_vol'] = np.sign(df['price_change']) * df['volume']

    # Aggregate 1-minute bars into volume bars
    bars_df = df.groupby('bar_id').agg({
        'ts_event': ['first', 'last'],
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'aggressor_vol': 'sum'
    })

    # Flatten columns
    bars_df.columns = [
        'time_start', 'time_end',
        'open', 'high', 'low', 'close',
        'volume', 'net_aggressor_vol'
    ]
    bars_df.reset_index(drop=True, inplace=True)

    # Add derived features
    bars_df['ticket_imbalance'] = np.where(
        bars_df['volume'] == 0, 0,
        bars_df['net_aggressor_vol'] / bars_df['volume']
    )

    # Count 1-min bars per volume bar
    tick_counts = df.groupby('bar_id').size()
    bars_df['tick_count'] = tick_counts.values

    print(f"Generated {len(bars_df)} Volume Bars from {len(df)} 1-min bars.")
    return bars_df
