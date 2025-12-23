import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def _calc_micro_window_features(w, ticket_imbalance, log_ret, bar_duration, pres_imbalance, normalized_ofi):
    """Worker function for parallel window microstructure feature calculation."""
    res = {}
    
    # 1. Flow Trend
    flow_trend = ticket_imbalance.rolling(w).mean()
    res[f'flow_trend_{w}'] = flow_trend
    
    # 2. Price-Flow Correlation
    res[f'price_flow_corr_{w}'] = log_ret.rolling(w).corr(ticket_imbalance)
    
    # 3. Flow Shock
    flow_std = ticket_imbalance.rolling(w).std()
    res[f'flow_shock_{w}'] = (ticket_imbalance - flow_trend) / flow_std.replace(0, 1)
    
    # 3b. OFI Trend & Shock (Higher Fidelity)
    if normalized_ofi is not None:
        ofi_trend = normalized_ofi.rolling(w).mean()
        res[f'ofi_trend_{w}'] = ofi_trend
        
        ofi_std = normalized_ofi.rolling(w).std()
        res[f'ofi_shock_{w}'] = (normalized_ofi - ofi_trend) / ofi_std.replace(0, 1)

    # 4. Duration Trend
    if bar_duration is not None:
        res[f'duration_trend_{w}'] = bar_duration.rolling(w).mean()
        
    # 5. Pressure Trend
    if pres_imbalance is not None:
        res[f'pres_trend_{w}'] = pres_imbalance.rolling(w).mean()
        res[f'price_pressure_corr_{w}'] = log_ret.rolling(w).corr(pres_imbalance)

    # 6. Order Book Alignment
    if pres_imbalance is not None:
        alignment = ticket_imbalance * pres_imbalance
        res[f'order_book_alignment_{w}'] = alignment.rolling(w).mean()

    return res

def add_microstructure_features(df, windows=[50, 100]):
    if df is None: return None
    print(f"Calculating Microstructure (Order Flow) Features...")
    df = df.copy()
    
    # Ensure base inputs
    vol = df['volume'].replace(0, 1)
    ticket_imbalance = df['net_aggressor_vol'] / vol
    df['ticket_imbalance'] = ticket_imbalance
    
    # Prepare OFI
    normalized_ofi = None
    if 'tick_ofi' in df.columns:
        # Normalize OFI by volume to make it comparable to ticket_imbalance
        normalized_ofi = df['tick_ofi'] / vol
    
    log_ret = np.log(df['close'] / df['close'].shift(1))
    df['log_ret'] = log_ret
    
    bar_duration = None
    if 'time_end' in df.columns and 'time_start' in df.columns:
        bar_duration = (df['time_end'] - df['time_start']).dt.total_seconds()
        df['bar_duration'] = bar_duration
        
    pres_imbalance = None
    if 'avg_bid_size' in df.columns and 'avg_ask_size' in df.columns:
        total_size = df['avg_bid_size'] + df['avg_ask_size']
        pres_imbalance = (df['avg_bid_size'] - df['avg_ask_size']) / total_size.replace(0, 1)
        df['pres_imbalance'] = pres_imbalance
        
    if 'avg_bid_price' in df.columns and 'avg_ask_price' in df.columns:
        mid_price = (df['avg_ask_price'] + df['avg_bid_price']) / 2
        df['avg_spread'] = (df['avg_ask_price'] - df['avg_bid_price']) / mid_price.replace(0, 1)
        
    # Parallelize
    results = Parallel(n_jobs=-1)(
        delayed(_calc_micro_window_features)(w, ticket_imbalance, log_ret, bar_duration, pres_imbalance, normalized_ofi)
        for w in windows
    )
    
    new_cols = {}
    for r in results:
        new_cols.update(r)
        
    df_new = pd.DataFrame(new_cols, index=df.index)
    return pd.concat([df, df_new], axis=1)
