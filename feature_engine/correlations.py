import pandas as pd
import numpy as np
from .loader import load_ticker_data

def add_correlator_residual(primary_df, correlator_df, suffix="_corr", window=100):
    """
    Calculates the Residual (Alpha) of the Primary asset relative to the Correlator.
    1. Resamples Correlator to the EXACT start/end times of the Primary Bars.
    2. Calculates Returns for both over that interval.
    3. Computes Rolling Beta and Residual.
    """
    if primary_df is None or correlator_df is None: return primary_df
    
    print(f"Calculating Residuals for {suffix} (Window={window})...")
    df = primary_df.copy()
    
    # 1. Align Correlator Prices to Bar Start/End
    # We use merge_asof to find the Correlator price at the exact moment the bar started and ended.
    # This effectively 'resamples' the Correlator to the Primary's Volume Clock.
    
    corr_clean = correlator_df[['ts_event', 'mid_price']].sort_values('ts_event').dropna()
    
    # Get Start Prices
    # We merge onto df. We need to ensure types match.
    start_prices = pd.merge_asof(
        df[['time_start']], 
        corr_clean, 
        left_on='time_start', 
        right_on='ts_event', 
        direction='backward'
    )['mid_price']
    
    # Get End Prices
    end_prices = pd.merge_asof(
        df[['time_end']], 
        corr_clean, 
        left_on='time_end', 
        right_on='ts_event', 
        direction='backward'
    )['mid_price']
    
    # 2. Calculate Returns over the Interval
    # Handle gaps where correlator might not have data (NaNs)
    # Using logarithmic returns for better additivity
    corr_ret = np.log(end_prices / start_prices)
    prim_ret = np.log(df['close'] / df['open']) # Log return of the bar itself
    
    # 3. Rolling Beta and Residual
    # We construct a temporary DF to do rolling calc
    tmp = pd.DataFrame({'prim': prim_ret, 'corr': corr_ret})
    
    # Rolling Covariance and Variance
    # Beta = Cov(P, C) / Var(C)
    rolling_cov = tmp['prim'].rolling(window).cov(tmp['corr'])
    rolling_var = tmp['corr'].rolling(window).var()
    
    beta = rolling_cov / rolling_var
    
    # Expected Return based on Correlator
    expected_ret = beta * tmp['corr']
    
    # Residual = Actual - Expected
    df[f'residual{suffix}'] = tmp['prim'] - expected_ret
    df[f'beta{suffix}'] = beta
    
    # Fill NaNs (early window)
    df[f'residual{suffix}'] = df[f'residual{suffix}'].fillna(0)
    
    return df

def add_crypto_features(df, data_dir, ibit_pattern="CLEAN_IBIT.parquet"):
    """
    Adds Crypto-Lead features using IBIT data.
    1. Resamples EURUSD (df) and IBIT to 1-min fixed intervals.
    2. Calculates Time-Based correlations and lags.
    3. Merges back to Volume Clock.
    """
    if df is None: return None
    
    # Load IBIT
    ibit_df = load_ticker_data(data_dir, ibit_pattern)
    if ibit_df is None:
        print("Skipping Crypto Features (IBIT not found).")
        return df
        
    print("Calculating Crypto Features (IBIT/EURUSD)...")
    df = df.copy()
    
    # 1. Resample to 1-Minute Grid
    # EURUSD (From Bars - approximate but good enough if bars are granular)
    # We use time_end as the timestamp
    eur_1m = df.set_index('time_end')[['close']].resample('1min').last().ffill()
    eur_1m.columns = ['eur_close']
    
    # IBIT
    ibit_df['ts_event'] = pd.to_datetime(ibit_df['ts_event'])
    ibit_1m = ibit_df.set_index('ts_event')[['mid_price']].resample('1min').last().ffill()
    ibit_1m.columns = ['ibit_close']
    
    # Align
    joined = pd.concat([eur_1m, ibit_1m], axis=1).dropna()
    
    # 2. Calculate Features on Time Grid
    # Returns
    joined['eur_ret'] = np.log(joined['eur_close'] / joined['eur_close'].shift(1))
    joined['ibit_ret'] = np.log(joined['ibit_close'] / joined['ibit_close'].shift(1))
    
    # A. The "Crypto Lead" (2-Minute Lag)
    # Logic: ln(Price_{t-2} / Price_{t-3})
    # This is the return of the bar 2 minutes ago?
    # If t is current time, t-1 is 1 min ago, t-2 is 2 min ago.
    # Return at t-2 is ln(P_{t-2}/P_{t-3}). 
    # So we shift the return series by 2.
    joined['IBIT_Lag2_Return'] = joined['ibit_ret'].shift(2)
    
    # B. Dynamic Correlation Regime (60m)
    joined['Corr_Regime_60m'] = joined['eur_ret'].rolling(60).corr(joined['ibit_ret'])
    
    # C. Volatility Ratio (30m)
    # Ratio of StdDevs
    vol_eur = joined['eur_ret'].rolling(30).std()
    vol_ibit = joined['ibit_ret'].rolling(30).std()
    joined['Vol_Ratio_30m'] = vol_eur / vol_ibit.replace(0, np.nan)
    
    # 3. Merge back to Volume Bars
    # We use merge_asof on the Bar's time_end matching the 1-min timestamp
    # Reset index to make 'timestamp' a column
    joined.index.name = 'timestamp'
    joined = joined.reset_index()
    
    # Sort for asof merge
    joined = joined.sort_values('timestamp')
    df = df.sort_values('time_end')
    
    # We only want the new columns
    cols_to_add = ['timestamp', 'IBIT_Lag2_Return', 'Corr_Regime_60m', 'Vol_Ratio_30m']
    
    merged = pd.merge_asof(
        df,
        joined[cols_to_add],
        left_on='time_end',
        right_on='timestamp',
        direction='backward' # Use latest known 1-min data
    )
    
    # Drop temp timestamp
    merged.drop(columns=['timestamp'], inplace=True)
    
    # Fill NaNs (early windows)
    merged[['IBIT_Lag2_Return', 'Corr_Regime_60m', 'Vol_Ratio_30m']] = \
        merged[['IBIT_Lag2_Return', 'Corr_Regime_60m', 'Vol_Ratio_30m']].fillna(0)
        
    return merged
