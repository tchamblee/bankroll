import pandas as pd
import numpy as np
from .loader import load_ticker_data

def add_macro_voltage_features(df, data_dir, windows=[50, 100]):
    """
    Adds Transatlantic Voltage features (US vs DE Policy Spreads).
    
    1. US Policy: US 2-Year Yield (US2Y)
    2. DE Policy: German 2-Year Yield (SCHATZ)
    3. Voltage: US2Y - SCHATZ (Differential)
    4. Curve Slopes: 10Y - 2Y for both.
    
    Updated to use merge_asof for precise alignment without resampling.
    """
    if df is None: return None
    print("Calculating Transatlantic Voltage Features (US2Y vs SCHATZ)...")
    df = df.copy()
    
    # 1. Load Macro Tickers
    us2y_df = load_ticker_data(data_dir, "CLEAN_US2Y.parquet")
    schatz_df = load_ticker_data(data_dir, "CLEAN_SCHATZ.parquet")
    tnx_df = load_ticker_data(data_dir, "CLEAN_TNX.parquet")
    bund_df = load_ticker_data(data_dir, "CLEAN_BUND.parquet")
    
    if us2y_df is None or schatz_df is None or tnx_df is None or bund_df is None:
        print("Skipping Voltage Features: Missing Macro Data.")
        return df

    # 2. Prepare Macro Data
    # Merge all macro series into one "State" DataFrame first? 
    # Or just merge them one by one into the main DF? 
    # Merging one by one is safer with merge_asof if timestamps differ slightly between macro feeds.
    
    def prepare_macro(macro_df, name):
        # Ensure UTC
        if macro_df['ts_event'].dt.tz is None:
            macro_df['ts_event'] = macro_df['ts_event'].dt.tz_localize('UTC')
        else:
            macro_df['ts_event'] = macro_df['ts_event'].dt.tz_convert('UTC')
            
        macro_df = macro_df.sort_values('ts_event')
        # Rename mid_price to the feature name
        macro_df = macro_df[['ts_event', 'mid_price']].rename(columns={'mid_price': name})
        return macro_df

    us2y_df = prepare_macro(us2y_df, 'us2y')
    schatz_df = prepare_macro(schatz_df, 'schatz')
    tnx_df = prepare_macro(tnx_df, 'tnx')
    bund_df = prepare_macro(bund_df, 'bund')
    
    # 3. Merge Asof
    # Ensure main DF is sorted by time_end (the time we know the bar is complete)
    # We use time_end to look back at the latest known macro price.
    
    # Ensure UTC for main DF
    if df['time_end'].dt.tz is None:
        df['time_end'] = df['time_end'].dt.tz_localize('UTC')
    else:
        df['time_end'] = df['time_end'].dt.tz_convert('UTC')
        
    df = df.sort_values('time_end')
    
    # Helper to merge
    def merge_macro(base, macro, col_name):
        return pd.merge_asof(
            base,
            macro,
            left_on='time_end',
            right_on='ts_event',
            direction='backward'
        ).drop(columns=['ts_event'])

    df = merge_macro(df, us2y_df, 'us2y')
    df = merge_macro(df, schatz_df, 'schatz')
    df = merge_macro(df, tnx_df, 'tnx')
    df = merge_macro(df, bund_df, 'bund')
    
    # Fill NaNs (Forward fill equivalent is implicit in merge_asof, 
    # but initial rows might be NaN if macro data starts later)
    cols_to_fill = ['us2y', 'schatz', 'tnx', 'bund']
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill().fillna(0)
    
    # 4. Calculate Voltage Features
    # Raw Spreads (Price Differentials)
    df['voltage_diff'] = df['us2y'] - df['schatz']
    
    # Curve Slopes (10Y - 2Y) -> Actually Price(10Y) - Price(2Y)
    df['us_curve'] = df['tnx'] - df['us2y']
    df['de_curve'] = df['bund'] - df['schatz']
    
    # 5. Derived Features (Velocity of Voltage)
    for w in windows:
        # Change in Voltage
        df[f'voltage_vel_{w}'] = df['voltage_diff'].diff(w)
        
        # Divergence: Price Trend vs Voltage Trend
        # Correlation between EURUSD Returns and Voltage Changes
        df[f'voltage_corr_{w}'] = df['log_ret'].rolling(w).corr(df['voltage_diff'].diff()).fillna(0)

    # Clean up raw macro columns (only keep derived features)
    df.drop(columns=['us2y', 'schatz', 'tnx', 'bund'], errors='ignore', inplace=True)

    return df
