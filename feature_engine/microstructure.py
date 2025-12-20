import numpy as np
import pandas as pd

def add_microstructure_features(df, windows=[50, 100]):
    if df is None: return None
    print(f"Calculating Microstructure (Order Flow) Features...")
    df = df.copy()
    
    # Ensure we have the base inputs
    if 'ticket_imbalance' not in df.columns:
        # Re-calculate if missing (Net / Vol)
        # Avoid div by zero
        vol = df['volume'].replace(0, 1)
        df['ticket_imbalance'] = df['net_aggressor_vol'] / vol
        
    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
    # --- NEW: Bar Duration (Time Dilation) ---
    if 'time_end' in df.columns and 'time_start' in df.columns:
        # Calculate duration in seconds
        df['bar_duration'] = (df['time_end'] - df['time_start']).dt.total_seconds()
        # Normalize duration (optional, but raw seconds is fine for trees)
        
    # --- NEW: Pressure Imbalance (L1 Order Book) ---
    if 'avg_bid_size' in df.columns and 'avg_ask_size' in df.columns:
        total_size = df['avg_bid_size'] + df['avg_ask_size']
        df['pres_imbalance'] = (df['avg_bid_size'] - df['avg_ask_size']) / total_size.replace(0, 1)
        
    # --- NEW: Spread Intensity (Cost of Liquidity) ---
    if 'avg_bid_price' in df.columns and 'avg_ask_price' in df.columns:
        df['avg_spread'] = df['avg_ask_price'] - df['avg_bid_price']
        
    for w in windows:
        # 1. Flow Trend (Persistence of Buying/Selling pressure)
        df[f'flow_trend_{w}'] = df['ticket_imbalance'].rolling(w).mean()
        
        # 2. Price-Flow Correlation (Wyckoff / Divergence)
        # High +Corr = Healthy Trend. Low/Neg Corr = Absorption/Divergence.
        df[f'price_flow_corr_{w}'] = df['log_ret'].rolling(w).corr(df['ticket_imbalance'])
        
        # 3. Flow Shock (Z-Score of Flow)
        # How unusual is this buying/selling relative to recent history?
        flow_std = df['ticket_imbalance'].rolling(w).std()
        df[f'flow_shock_{w}'] = (df['ticket_imbalance'] - df[f'flow_trend_{w}']) / flow_std.replace(0, 1)

        # 4. Duration Trend (Acceleration/Deceleration)
        if 'bar_duration' in df.columns:
            df[f'duration_trend_{w}'] = df['bar_duration'].rolling(w).mean()
            
        # 5. Pressure Trend (Persistent Support/Resistance)
        if 'pres_imbalance' in df.columns:
            df[f'pres_trend_{w}'] = df['pres_imbalance'].rolling(w).mean()
            # Price-Pressure Correlation (Magnet vs Wall)
            df[f'price_pressure_corr_{w}'] = df['log_ret'].rolling(w).corr(df['pres_imbalance'])

        # 6. Flow Autocorrelation (Herding)
        # Persistence of the order flow itself
        df[f'flow_autocorr_{w}'] = df['ticket_imbalance'].rolling(w).corr(df['ticket_imbalance'].shift(1))
        
        # 7. Spread Intensity (Spread / Volatility) - REDUNDANT with avg_spread
        # if 'avg_spread' in df.columns and f'volatility_{w}' in df.columns:
            # Normalize spread by price to match log-return volatility units
            # spread_pct = df['avg_spread'] / df['close']
            # df[f'spread_intensity_{w}'] = spread_pct / df[f'volatility_{w}'].replace(0, 1e-6)

        # 8. Order Book Alignment (Flow * Pressure)
        # Do buyers have support? Or are they hitting walls?
        if 'pres_imbalance' in df.columns:
            # Instantaneous alignment
            alignment = df['ticket_imbalance'] * df['pres_imbalance']
            df[f'order_book_alignment_{w}'] = alignment.rolling(w).mean()

    return df
