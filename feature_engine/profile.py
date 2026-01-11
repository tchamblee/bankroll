import pandas as pd
import numpy as np
import datetime as dt
from numba import jit, prange

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


@jit(nopython=True)
def _jit_developing_va(prices, volumes, n_bars, bins, value_area_pct):
    """
    Calculate developing value area at each bar within a day.
    Returns arrays of developing VPOC, VAL, VAH.
    """
    dev_vpoc = np.zeros(n_bars, dtype=np.float64)
    dev_val = np.zeros(n_bars, dtype=np.float64)
    dev_vah = np.zeros(n_bars, dtype=np.float64)

    if n_bars == 0:
        return dev_vpoc, dev_val, dev_vah

    # For first bar, use its price as all values
    dev_vpoc[0] = prices[0]
    dev_val[0] = prices[0]
    dev_vah[0] = prices[0]

    if n_bars == 1:
        return dev_vpoc, dev_val, dev_vah

    # Iteratively build histogram
    for i in range(1, n_bars):
        # Get cumulative prices and volumes up to bar i
        p_slice = prices[:i+1]
        v_slice = volumes[:i+1]

        low_p = p_slice.min()
        high_p = p_slice.max()

        if low_p == high_p:
            dev_vpoc[i] = low_p
            dev_val[i] = low_p
            dev_vah[i] = high_p
            continue

        # Build histogram manually (np.histogram not available in numba)
        bin_width = (high_p - low_p) / bins
        hist = np.zeros(bins, dtype=np.float64)

        for j in range(len(p_slice)):
            bin_idx = int((p_slice[j] - low_p) / bin_width)
            if bin_idx >= bins:
                bin_idx = bins - 1
            hist[bin_idx] += v_slice[j]

        # Find VPOC (max volume bin)
        max_idx = 0
        max_vol = hist[0]
        for k in range(1, bins):
            if hist[k] > max_vol:
                max_vol = hist[k]
                max_idx = k

        # Value Area expansion
        total_vol = hist.sum()
        target_vol = total_vol * value_area_pct
        current_vol = hist[max_idx]

        low_idx = max_idx
        high_idx = max_idx

        while current_vol < target_vol:
            can_go_lower = low_idx > 0
            can_go_higher = high_idx < bins - 1

            if not can_go_lower and not can_go_higher:
                break

            vol_lower = hist[low_idx - 1] if can_go_lower else -1.0
            vol_higher = hist[high_idx + 1] if can_go_higher else -1.0

            if vol_higher > vol_lower:
                high_idx += 1
                current_vol += vol_higher
            else:
                low_idx -= 1
                current_vol += vol_lower

        # Calculate prices from bin indices
        dev_vpoc[i] = low_p + (max_idx + 0.5) * bin_width
        dev_val[i] = low_p + (low_idx + 0.5) * bin_width
        dev_vah[i] = low_p + (high_idx + 0.5) * bin_width

    return dev_vpoc, dev_val, dev_vah


def calculate_developing_profiles(dates, prices, volumes, bins=50):
    """
    Calculate developing value area for each bar within each day.
    Returns arrays aligned with input data.
    """
    n_total = len(dates)
    dev_vpoc = np.zeros(n_total, dtype=np.float64)
    dev_val = np.zeros(n_total, dtype=np.float64)
    dev_vah = np.zeros(n_total, dtype=np.float64)

    # Find day boundaries
    unique_dates, indices = np.unique(dates, return_index=True)
    starts = indices
    ends = np.roll(indices, -1)
    ends[-1] = n_total

    for i in range(len(unique_dates)):
        start = starts[i]
        end = ends[i]

        p_slice = prices[start:end].astype(np.float64)
        v_slice = volumes[start:end].astype(np.float64)
        n_bars = end - start

        if n_bars == 0:
            continue

        d_vpoc, d_val, d_vah = _jit_developing_va(
            p_slice, v_slice, n_bars, bins, 0.70
        )

        dev_vpoc[start:end] = d_vpoc
        dev_val[start:end] = d_val
        dev_vah[start:end] = d_vah

    return dev_vpoc, dev_val, dev_vah


def calculate_composite_profiles(dates, prices, volumes, lookback_days=5, bins=100):
    """
    Calculate composite profile over multiple days using rolling window.
    Returns VPOC, VAL, VAH for composite profile.
    """
    n_total = len(dates)
    comp_vpoc = np.full(n_total, np.nan, dtype=np.float64)
    comp_val = np.full(n_total, np.nan, dtype=np.float64)
    comp_vah = np.full(n_total, np.nan, dtype=np.float64)

    # Get unique dates and their boundaries
    unique_dates, indices = np.unique(dates, return_index=True)
    starts = indices
    ends = np.roll(indices, -1)
    ends[-1] = n_total

    n_days = len(unique_dates)

    # Build daily histograms first (we'll aggregate them)
    # Store daily price ranges and histograms
    daily_low = np.zeros(n_days)
    daily_high = np.zeros(n_days)
    daily_hists = []

    for i in range(n_days):
        start = starts[i]
        end = ends[i]
        p_slice = prices[start:end]
        v_slice = volumes[start:end]

        if len(p_slice) == 0:
            daily_low[i] = np.nan
            daily_high[i] = np.nan
            daily_hists.append(None)
            continue

        daily_low[i] = p_slice.min()
        daily_high[i] = p_slice.max()

        if daily_low[i] == daily_high[i]:
            daily_hists.append(None)
        else:
            hist, bin_edges = np.histogram(
                p_slice, bins=bins, weights=v_slice,
                range=(daily_low[i], daily_high[i])
            )
            daily_hists.append((hist, bin_edges))

    # For each day, calculate composite from past N days
    for day_idx in range(lookback_days, n_days):
        # Get price range over lookback period
        lookback_slice = slice(day_idx - lookback_days, day_idx)
        valid_lows = daily_low[lookback_slice]
        valid_highs = daily_high[lookback_slice]

        # Skip if any day has no data
        if np.any(np.isnan(valid_lows)) or np.any(np.isnan(valid_highs)):
            continue

        composite_low = valid_lows.min()
        composite_high = valid_highs.max()

        if composite_low == composite_high:
            continue

        # Aggregate histograms onto common price grid
        composite_hist = np.zeros(bins, dtype=np.float64)
        bin_width = (composite_high - composite_low) / bins

        for past_day in range(day_idx - lookback_days, day_idx):
            if daily_hists[past_day] is None:
                continue

            hist, bin_edges = daily_hists[past_day]

            # Map each bin from daily histogram to composite grid
            for b in range(len(hist)):
                if hist[b] == 0:
                    continue
                bin_center = (bin_edges[b] + bin_edges[b+1]) / 2
                comp_bin = int((bin_center - composite_low) / bin_width)
                if comp_bin >= bins:
                    comp_bin = bins - 1
                if comp_bin < 0:
                    comp_bin = 0
                composite_hist[comp_bin] += hist[b]

        # Calculate value area from composite histogram
        if composite_hist.sum() == 0:
            continue

        max_idx = np.argmax(composite_hist)
        total_vol = composite_hist.sum()

        # Create bin edges for composite
        comp_edges = np.linspace(composite_low, composite_high, bins + 1)

        vpoc, val, vah = _jit_value_area(
            composite_hist, comp_edges, max_idx, total_vol, 0.70
        )

        # Apply to all bars in current day
        start = starts[day_idx]
        end = ends[day_idx]
        comp_vpoc[start:end] = vpoc
        comp_val[start:end] = val
        comp_vah[start:end] = vah

    return comp_vpoc, comp_val, comp_vah


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
    merged['dist_to_day_open'] = (merged['close'] - merged['day_open']) / merged['day_open']
    
    # Relative Volume at Price (How "thick" is the market here?)
    # This is hard without full profile, but we can use dist_to_vpoc as a proxy for "Magnetism"
    # Closer to VPOC = Higher expected volume/friction.

    # --- Developing Value Area Features ---
    print("  Calculating Developing Value Area...")
    dev_vpoc, dev_val, dev_vah = calculate_developing_profiles(dates, prices, volumes, bins=50)
    merged['dev_vpoc'] = dev_vpoc
    merged['dev_val'] = dev_val
    merged['dev_vah'] = dev_vah

    # Developing VA width (normalized by price)
    merged['dev_va_width'] = (merged['dev_vah'] - merged['dev_val']) / merged['close']

    # Price position relative to developing profile
    merged['price_vs_dev_vpoc'] = (merged['close'] - merged['dev_vpoc']) / merged['dev_vpoc']

    # Binary: inside developing value area
    merged['in_dev_value'] = np.where(
        (merged['close'] >= merged['dev_val']) & (merged['close'] <= merged['dev_vah']),
        1.0, 0.0
    )

    # --- Composite Profile Features (Weekly = 5 days, Monthly = 20 days) ---
    print("  Calculating Weekly Composite Profile...")
    weekly_vpoc, weekly_val, weekly_vah = calculate_composite_profiles(
        dates, prices, volumes, lookback_days=5, bins=100
    )
    merged['weekly_vpoc'] = weekly_vpoc
    merged['weekly_val'] = weekly_val
    merged['weekly_vah'] = weekly_vah

    # Fill NaN with forward fill then price fallback
    for col in ['weekly_vpoc', 'weekly_val', 'weekly_vah']:
        merged[col] = merged[col].ffill()
        mask = merged[col].isna()
        merged.loc[mask, col] = merged.loc[mask, 'close']

    # Weekly composite features
    merged['weekly_in_value'] = np.where(
        (merged['close'] >= merged['weekly_val']) & (merged['close'] <= merged['weekly_vah']),
        1.0, 0.0
    )
    merged['dist_to_weekly_vpoc'] = (merged['close'] - merged['weekly_vpoc']) / merged['weekly_vpoc']

    print("  Calculating Monthly Composite Profile...")
    monthly_vpoc, monthly_val, monthly_vah = calculate_composite_profiles(
        dates, prices, volumes, lookback_days=20, bins=100
    )
    merged['monthly_vpoc'] = monthly_vpoc
    merged['monthly_val'] = monthly_val
    merged['monthly_vah'] = monthly_vah

    # Fill NaN with forward fill then price fallback
    for col in ['monthly_vpoc', 'monthly_val', 'monthly_vah']:
        merged[col] = merged[col].ffill()
        mask = merged[col].isna()
        merged.loc[mask, col] = merged.loc[mask, 'close']

    # Monthly composite features
    merged['monthly_in_value'] = np.where(
        (merged['close'] >= merged['monthly_val']) & (merged['close'] <= merged['monthly_vah']),
        1.0, 0.0
    )
    merged['dist_to_monthly_vpoc'] = (merged['close'] - merged['monthly_vpoc']) / merged['monthly_vpoc']

    merged = merged.drop(columns=['date_temp', 'date'])

    return merged
