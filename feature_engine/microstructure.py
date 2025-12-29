import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from .event_decay import _jit_bars_since_true

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
    
    # --- NEW: Event-Driven Microstructure ---
    # 1. Liquidation Event (Large Volume + Large Move)
    # Using Rolling 100-bar stats for Z-Score
    w_event = 100
    
    # Volume Z-Score
    vol_mean = vol.rolling(w_event).mean()
    vol_std = vol.rolling(w_event).std().replace(0, 1)
    vol_z = (vol - vol_mean) / vol_std
    
    # Return Z-Score
    ret_std = log_ret.rolling(w_event).std().replace(0, 1)
    ret_z = log_ret.abs() / ret_std
    
    # Liquidation = High Vol (>2.5) AND High Move (>2.5)
    is_liquidation = (vol_z > 2.5) & (ret_z > 2.5)
    df_new['bars_since_liquidation'] = _jit_bars_since_true(is_liquidation.values)
    df_new['decay_liquidation'] = np.exp(-5.0 * df_new['bars_since_liquidation'] / 100.0).fillna(0)

    # 2. Imbalance Spike
    # Ticket Imbalance Z-Score
    imb_mean = ticket_imbalance.rolling(w_event).mean()
    imb_std = ticket_imbalance.rolling(w_event).std().replace(0, 1)
    imb_z = (ticket_imbalance - imb_mean) / imb_std
    
    # Spike = |Imbalance| > 2.5 sigma
    is_imb_spike = (imb_z.abs() > 2.5)
    df_new['bars_since_imbalance_spike'] = _jit_bars_since_true(is_imb_spike.values)
    df_new['decay_imbalance_spike'] = np.exp(-5.0 * df_new['bars_since_imbalance_spike'] / 100.0).fillna(0)

    return pd.concat([df, df_new], axis=1)
