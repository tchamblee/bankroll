import pandas as pd
import numpy as np
from .loader import load_ticker_data

def add_macro_voltage_features(df, us2y_df, schatz_df, tnx_df, bund_df, windows=[50, 100]):
    """
    Adds Transatlantic Voltage features (US vs DE Policy Spreads).
    
    1. US Policy: US 2-Year Yield (US2Y)
    2. DE Policy: German 2-Year Yield (SCHATZ)
    3. Voltage: US2Y - SCHATZ (Differential)
    4. Curve Slopes: 10Y - 2Y for both.
    """
    if df is None: return None
    print("Calculating Transatlantic Voltage Features (US2Y vs SCHATZ)...")
    df = df.copy()
    
    if us2y_df is None or schatz_df is None or tnx_df is None or bund_df is None:
        print("Skipping Voltage Features: Missing Macro Data.")
        return df

    # 2. Prepare Macro Data
    def prepare_macro(macro_df, name):
        # Handle Column Names (last_price if it's a bar, mid_price if it's tick/processed)
        price_col = 'last_price' if 'last_price' in macro_df.columns else 'mid_price'
        if price_col not in macro_df.columns:
            if 'close' in macro_df.columns: price_col = 'close'
            else: return None

        # Ensure UTC
        if macro_df['ts_event'].dt.tz is None:
            macro_df['ts_event'] = macro_df['ts_event'].dt.tz_localize('UTC')
        else:
            macro_df['ts_event'] = macro_df['ts_event'].dt.tz_convert('UTC')
            
        macro_df = macro_df.sort_values('ts_event')
        macro_df = macro_df[['ts_event', price_col]].dropna().rename(columns={price_col: name})
        return macro_df

    us2y_df = prepare_macro(us2y_df, 'us2y')
    schatz_df = prepare_macro(schatz_df, 'schatz')
    tnx_df = prepare_macro(tnx_df, 'tnx')
    bund_df = prepare_macro(bund_df, 'bund')
    
    if any(m is None for m in [us2y_df, schatz_df, tnx_df, bund_df]):
        print("Skipping Voltage Features: Missing Price Columns in Macro Data.")
        return df
    
    # 3. Merge Asof
    if df['time_end'].dt.tz is None:
        df['time_end'] = df['time_end'].dt.tz_localize('UTC')
    else:
        df['time_end'] = df['time_end'].dt.tz_convert('UTC')
        
    df = df.sort_values('time_end')
    
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
    
    cols_to_fill = ['us2y', 'schatz', 'tnx', 'bund']
    df[cols_to_fill] = df[cols_to_fill].ffill().bfill().fillna(0)
    
    # 4. Calculate Voltage Features
    df['voltage_diff'] = df['us2y'] - df['schatz']
    df['us_curve'] = df['tnx'] - df['us2y']
    df['de_curve'] = df['bund'] - df['schatz']
    
    # REMOVED: Derivative/Velocity features (diff) as they result in zero variance 
    # on high-frequency bars due to slow-updating macro data.
    # We keep the levels (voltage_diff, curves) which represent the macro regime.

    df.drop(columns=['us2y', 'schatz', 'tnx', 'bund'], errors='ignore', inplace=True)
    return df