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

    # 2. Resample all to 1-Minute Grid to align timestamps
    # We use the time range of the primary DF
    start_time = df['time_start'].min().floor('1min')
    end_time = df['time_end'].max().ceil('1min')
    
    # Create the grid
    # freq='1min' is a safe enough approximation for macro shifts
    time_grid = pd.date_range(start_time, end_time, freq='1min', tz='UTC')
    macro_grid = pd.DataFrame(index=time_grid)
    
    def resample_ticker(ticker_df, name):
        ticker_df['ts_event'] = pd.to_datetime(ticker_df['ts_event'])
        # Sort and deduplicate
        ticker_df = ticker_df.sort_values('ts_event')
        # Resample to 1min, forward fill
        resampled = ticker_df.set_index('ts_event')[['mid_price']].resample('1min').last().ffill()
        resampled.columns = [name]
        return resampled

    # Resample all
    us2y_s = resample_ticker(us2y_df, 'us2y')
    schatz_s = resample_ticker(schatz_df, 'schatz')
    tnx_s = resample_ticker(tnx_df, 'tnx')
    bund_s = resample_ticker(bund_df, 'bund')
    
    # Merge into grid
    macro_grid = macro_grid.join([us2y_s, schatz_s, tnx_s, bund_s], how='left')
    
    # Forward fill gaps (macro data is sparse/slower)
    macro_grid = macro_grid.ffill().bfill()
    
    # 3. Calculate Voltage Features on the Grid
    # Note: Yields are often prices in futures. 
    # US 2Y Futures (ZT): Price = 100 - Yield (roughly)? No, ZT trades as price.
    # TNX (CBOE Index): Is Yield * 10.
    # We assume 'mid_price' reflects the Yield or a proxy.
    # IMPORTANT: If these are FUTURES PRICES, higher price = lower yield.
    # Voltage = US Yield - DE Yield.
    # If using Futures Price: Voltage ~ (DE Price - US Price)?
    # Let's assume standard "Risk On/Off" correlation.
    # We will just use the raw prices and let the model figure out the sign.
    
    # Raw Spreads (Price Differentials)
    macro_grid['voltage_diff'] = macro_grid['us2y'] - macro_grid['schatz']
    
    # Curve Slopes (10Y - 2Y) -> Actually Price(10Y) - Price(2Y)
    macro_grid['us_curve'] = macro_grid['tnx'] - macro_grid['us2y']
    macro_grid['de_curve'] = macro_grid['bund'] - macro_grid['schatz']
    
    # 4. Merge back to Volume Bars
    # Use merge_asof on time_end
    macro_grid.index.name = 'timestamp'
    macro_grid = macro_grid.reset_index()
    
    # Select cols
    cols_to_add = ['timestamp', 'voltage_diff', 'us_curve', 'de_curve']
    
    df = df.sort_values('time_end')
    macro_grid = macro_grid.sort_values('timestamp')
    
    merged = pd.merge_asof(
        df,
        macro_grid[cols_to_add],
        left_on='time_end',
        right_on='timestamp',
        direction='backward'
    )
    
    merged.drop(columns=['timestamp'], inplace=True)
    
    # Fill any remaining NaNs
    merged[['voltage_diff', 'us_curve', 'de_curve']] = \
        merged[['voltage_diff', 'us_curve', 'de_curve']].fillna(0)
        
    # 5. Derived Features (Velocity of Voltage)
    for w in windows:
        # Change in Voltage
        merged[f'voltage_vel_{w}'] = merged['voltage_diff'].diff(w)
        
        # Divergence: Price Trend vs Voltage Trend
        # Correlation between EURUSD Returns and Voltage Changes
        # If Voltage (US Rates UP relative to DE) goes UP, EURUSD should go DOWN.
        # So correlation should be negative.
        # If correlation breaks, it's a signal.
        merged[f'voltage_corr_{w}'] = merged['log_ret'].rolling(w).corr(merged['voltage_diff'].diff())
        
        # Z-Score of Voltage (Regime)
        vol_mean = merged['voltage_diff'].rolling(w * 10).mean() # Long baseline
        vol_std = merged['voltage_diff'].rolling(w * 10).std()
        merged[f'voltage_zscore_{w}'] = (merged['voltage_diff'] - vol_mean) / vol_std.replace(0, 1)

    return merged
