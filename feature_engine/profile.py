import pandas as pd
import numpy as np
import datetime as dt
from numba import jit

@jit(nopython=True)
def _jit_value_area(hist, bin_edges, max_idx, total_vol, value_area_pct):
    """
    Numba-optimized Value Area expansion.
    """
    target_vol = total_vol * value_area_pct
    current_vol = hist[max_idx]
    
    low_idx = max_idx
    high_idx = max_idx
    
    n_bins = len(hist)
    
    while current_vol < target_vol:
        can_go_lower = low_idx > 0
        can_go_higher = high_idx < n_bins - 1
        
        if not can_go_lower and not can_go_higher:
            break
            
        vol_lower = hist[low_idx - 1] if can_go_lower else -1.0
        vol_higher = hist[high_idx + 1] if can_go_higher else -1.0
        
        # Greedy expansion: Pick the side with higher volume
        # If one side is out of bounds (vol = -1), we pick the other.
        if vol_higher > vol_lower:
            high_idx += 1
            current_vol += vol_higher
        else:
            low_idx -= 1
            current_vol += vol_lower
            
    vpoc = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0
    val = (bin_edges[low_idx] + bin_edges[low_idx+1]) / 2.0
    vah = (bin_edges[high_idx] + bin_edges[high_idx+1]) / 2.0
    
    return vpoc, val, vah

def calculate_profiles_fast(dates, prices, volumes, bins=200):
    """
    Fast day-by-day profile calculation avoiding pandas groupby overhead.
    """
    # Find day boundaries
    unique_dates, indices = np.unique(dates, return_index=True)
    
    # Indices are start indices. 
    # Zip with sliced indices
    starts = indices
    ends = np.roll(indices, -1)
    ends[-1] = len(dates)
    
    results = []
    
    # Loop over days (Pure Numpy slicing)
    for i, date in enumerate(unique_dates):
        start = starts[i]
        end = ends[i]
        
        p_slice = prices[start:end]
        v_slice = volumes[start:end]
        
        if len(p_slice) == 0: continue
            
        low_p = p_slice.min()
        high_p = p_slice.max()
        
        if low_p == high_p:
            results.append({
                'date': date,
                'vpoc': low_p, 'val': low_p, 'vah': high_p,
                'total_vol': v_slice.sum()
            })
            continue
            
        # Histogram (Fast in Numpy)
        hist, bin_edges = np.histogram(p_slice, bins=bins, weights=v_slice, range=(low_p, high_p))
        
        # Value Area (Fast in Numba)
        max_idx = np.argmax(hist)
        total_vol = hist.sum()
        
        vpoc, val, vah = _jit_value_area(hist, bin_edges, max_idx, total_vol, 0.70)
        
        results.append({
            'date': date,
            'vpoc': vpoc, 'val': val, 'vah': vah,
            'total_vol': total_vol
        })
        
    return results

def add_market_profile_features(df):
    """
    Generates Market Profile features based on the PREVIOUS day's structure.
    Optimized for speed.
    """
    if df is None: return None
    print("Calculating Market Profile (Volume Profile) Features...")
    df = df.copy()
    
    # Ensure Datetime
    if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
        df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
        
    # Prepare Arrays for Fast Calculation
    # Convert dates to int64 (ns) or just use unique
    dates = df['time_start'].dt.date.values
    prices = df['close'].values
    volumes = df['volume'].values
    
    # Run Fast Calc
    daily_stats = calculate_profiles_fast(dates, prices, volumes)
            
    if not daily_stats:
        print("Warning: No daily stats generated.")
        return df
        
    stats_df = pd.DataFrame(daily_stats)
    stats_df['date'] = pd.to_datetime(stats_df['date']).dt.date
    
    # SHIFT logic remains same
    stats_df['date'] = stats_df['date'] + dt.timedelta(days=1)
    
    # Rename
    stats_df = stats_df.rename(columns={
        'vpoc': 'prev_vpoc',
        'vah': 'prev_vah',
        'val': 'prev_val',
        'total_vol': 'prev_day_vol'
    })
    
    # Merge using Date object (Pandas merge is efficient enough for ~200 rows)
    # Ensure robust datetime matching by converting to datetime64[ns] (normalized to midnight)
    df['date_temp'] = pd.to_datetime(df['time_start'].dt.date)
    stats_df['date'] = pd.to_datetime(stats_df['date'])
    
    merged = pd.merge(df, stats_df, left_on='date_temp', right_on='date', how='left')
    
    # Fill Logic
    cols_to_fill = ['prev_vpoc', 'prev_vah', 'prev_val', 'prev_day_vol']
    merged[cols_to_fill] = merged[cols_to_fill].ffill()
    
    mask = merged['prev_vpoc'].isna()
    merged.loc[mask, 'prev_vpoc'] = merged.loc[mask, 'close']
    merged.loc[mask, 'prev_vah'] = merged.loc[mask, 'close']
    merged.loc[mask, 'prev_val'] = merged.loc[mask, 'close']
    
    # --- New Signal Features ---
    # 1. Identify Day Open (First bar of the day)
    # We can group by date_temp and transform 'first'
    merged['day_open'] = merged.groupby('date_temp')['open'].transform('first')
    
    # 2. Open Zone (Where did we open relative to YESTERDAY's Value?)
    # 1: Above VAH (Strong Bull)
    # 0: Inside Value (Balance)
    # -1: Below VAL (Strong Bear)
    merged['open_zone'] = 0
    merged.loc[merged['day_open'] > merged['prev_vah'], 'open_zone'] = 1
    merged.loc[merged['day_open'] < merged['prev_val'], 'open_zone'] = -1
    
    # 3. Failed Re-entry (Rejection) / Breakout Continuation
    # Create masks for conditions
    opened_above = (merged['open_zone'] == 1)
    opened_below = (merged['open_zone'] == -1)
    
    # Check current bar status
    # If Opened Above, we re-entered if Low <= VAH
    reentered_from_above = opened_above & (merged['low'] <= merged['prev_vah'])
    
    # If Opened Below, we re-entered if High >= VAL
    reentered_from_below = opened_below & (merged['high'] >= merged['prev_val'])
    
    # Cumulative check within the day
    # We need to propagate "True" forward for the rest of the day once it happens
    merged['reentry_event'] = reentered_from_above | reentered_from_below
    merged['reentered_value'] = merged.groupby('date_temp')['reentry_event'].cummax().astype(int)
    
    merged = merged.drop(columns=['reentry_event'])
    
    # Boolean: Currently inside value area
    merged['in_value_area'] = np.where(
        (merged['close'] >= merged['prev_val']) & (merged['close'] <= merged['prev_vah']), 
        1.0, 0.0
    )
    
    # Distance Features (Standard)
    merged['dist_to_vpoc'] = (merged['close'] - merged['prev_vpoc']) / merged['prev_vpoc']
    merged['dist_to_vah'] = (merged['close'] - merged['prev_vah']) / merged['prev_vah']
    merged['dist_to_val'] = (merged['close'] - merged['prev_val']) / merged['prev_val']
    
    # Relative Volume at Price (How "thick" is the market here?)
    # This is hard without full profile, but we can use dist_to_vpoc as a proxy for "Magnetism"
    # Closer to VPOC = Higher expected volume/friction.
    
    merged = merged.drop(columns=['date_temp', 'date'])
    
    return merged
