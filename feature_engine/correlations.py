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
    
    price_col = 'mid_price'
    if 'mid_price' not in correlator_df.columns:
        if 'close' in correlator_df.columns: price_col = 'close'
        elif 'last_price' in correlator_df.columns: price_col = 'last_price'
        else: return primary_df

    corr_clean = correlator_df[['ts_event', price_col]].sort_values('ts_event').dropna()
    
    # Get Start Prices
    # We merge onto df. We need to ensure types match.
    start_prices = pd.merge_asof(
        df[['time_start']], 
        corr_clean, 
        left_on='time_start', 
        right_on='ts_event', 
        direction='backward'
    )[price_col]
    
    # Get End Prices
    end_prices = pd.merge_asof(
        df[['time_end']], 
        corr_clean, 
        left_on='time_end', 
        right_on='ts_event', 
        direction='backward'
    )[price_col]
    
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
    
    # FIX: Lag the Beta to avoid look-ahead bias.
    # We must use the Beta estimated from [t-window, t-1] to predict t.
    # Current 'beta' at index t includes data at t.
    beta_lagged = beta.shift(1)
    
    # Expected Return based on Correlator (using OUT-OF-SAMPLE Beta)
    expected_ret = beta_lagged * tmp['corr']
    
    # Residual = Actual - Expected
    df[f'residual{suffix}'] = tmp['prim'] - expected_ret
    df[f'beta{suffix}'] = beta_lagged.fillna(0) # Store the beta actually used
    
    # Fill NaNs (early window)
    df[f'residual{suffix}'] = df[f'residual{suffix}'].fillna(0)
    
    return df

def add_crypto_features(df, ibit_df):
    """
    Adds Crypto-Lead features using IBIT data.
    1. Resamples Primary Ticker (df) and IBIT to 1-min fixed intervals.
    2. Calculates Time-Based correlations and lags.
    3. Merges back to Volume Clock.
    """
    if df is None: return None
    
    if ibit_df is None:
        print("Skipping Crypto Features (IBIT not found).")
        return df
        
    print(f"Calculating Crypto Features (IBIT/{config.PRIMARY_TICKER})...")
    df = df.copy()
    
    # 1. Resample to 1-Minute Grid
    # Primary (From Bars - approximate but good enough if bars are granular)
    # We use time_end as the timestamp
    eur_1m = df.set_index('time_end')[['close']].resample('1min').last().ffill()
    eur_1m.columns = ['eur_close']
    
    # IBIT
    ibit_df['ts_event'] = pd.to_datetime(ibit_df['ts_event'])
    
    price_col = 'mid_price'
    if 'mid_price' not in ibit_df.columns:
        if 'close' in ibit_df.columns: price_col = 'close'
        elif 'last_price' in ibit_df.columns: price_col = 'last_price'
        else: 
            print("Skipping Crypto Features: IBIT price column not found.")
            return df

    ibit_1m = ibit_df.set_index('ts_event')[[price_col]].resample('1min').last().ffill()
    ibit_1m.columns = ['ibit_close']
    
    # Align
    joined = pd.concat([eur_1m, ibit_1m], axis=1).dropna()
    
    # 2. Calculate Features on Time Grid
    # Returns
    joined['eur_ret'] = np.log(joined['eur_close'] / joined['eur_close'].shift(1))
    joined['ibit_ret'] = np.log(joined['ibit_close'] / joined['ibit_close'].shift(1))
    
    # A. Bitcoin Trend (30m) - Replaces Failed Lag2
    # Simple Return over last 30 minutes
    joined['IBIT_Trend_30m'] = joined['ibit_close'].pct_change(30)
    
    # B. Dynamic Correlation Regime (60m)
    joined['Corr_Regime_60m'] = joined['eur_ret'].rolling(60).corr(joined['ibit_ret'])
    
    # NEW: Crypto Beta (60m)
    cov_60 = joined['eur_ret'].rolling(60).cov(joined['ibit_ret'])
    var_60 = joined['ibit_ret'].rolling(60).var()
    joined['IBIT_Beta_60m'] = cov_60 / var_60.replace(0, np.nan)
    
    # C. Volatility Ratio (30m)
    # Ratio of StdDevs
    vol_eur = joined['eur_ret'].rolling(30).std()
    vol_ibit = joined['ibit_ret'].rolling(30).std()
    joined['Vol_Ratio_30m'] = vol_eur / vol_ibit.replace(0, np.nan)
    
    # NEW: Momentum Divergence (30m)
    # Normalized Trend Difference
    eur_trend = joined['eur_close'].pct_change(30)
    # IBIT_Trend_30m is already pct_change(30)
    joined['IBIT_Divergence_30m'] = (eur_trend / vol_eur.replace(0, np.nan)) - (joined['IBIT_Trend_30m'] / vol_ibit.replace(0, np.nan))
    
    # 3. Merge back to Volume Bars
    # We use merge_asof on the Bar's time_end matching the 1-min timestamp
    # Reset index to make 'timestamp' a column
    joined.index.name = 'timestamp'
    joined = joined.reset_index()
    
    # Sort for asof merge
    joined = joined.sort_values('timestamp')
    df = df.sort_values('time_end')
    
    # We only want the new columns
    cols_to_add = ['timestamp', 'IBIT_Trend_30m', 'Corr_Regime_60m', 'Vol_Ratio_30m', 'IBIT_Beta_60m', 'IBIT_Divergence_30m']
    
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
    merged[['IBIT_Trend_30m', 'Corr_Regime_60m', 'Vol_Ratio_30m', 'IBIT_Beta_60m', 'IBIT_Divergence_30m']] = \
        merged[['IBIT_Trend_30m', 'Corr_Regime_60m', 'Vol_Ratio_30m', 'IBIT_Beta_60m', 'IBIT_Divergence_30m']].fillna(0)
        
    return merged
