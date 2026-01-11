import pandas as pd
import numpy as np
import config
from .loader import load_ticker_data


def rolling_hurst(series, window):
    """
    Calculate rolling Hurst Exponent using the R/S (Rescaled Range) method.

    H > 0.5: Trend-persistent (momentum)
    H < 0.5: Mean-reverting (anti-persistent)
    H = 0.5: Random walk

    Args:
        series: Price or spread series
        window: Rolling window size (recommend 100-200 for stability)

    Returns:
        Series of Hurst exponent values
    """
    def calc_hurst(x):
        if len(x) < 20 or np.std(x) < 1e-10:
            return np.nan

        # Use returns for stationarity
        returns = np.diff(x)
        n = len(returns)

        if n < 10:
            return np.nan

        # Calculate R/S for multiple sub-periods
        # Simplified: use full window R/S
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret < 1e-10:
            return np.nan

        # Cumulative deviation from mean
        cum_dev = np.cumsum(returns - mean_ret)

        # Range
        R = np.max(cum_dev) - np.min(cum_dev)

        # Rescaled range
        RS = R / std_ret

        # Hurst exponent: H = log(R/S) / log(n)
        if RS <= 0:
            return np.nan

        H = np.log(RS) / np.log(n)

        # Clamp to reasonable range [0, 1]
        return np.clip(H, 0.0, 1.0)

    return series.rolling(window).apply(calc_hurst, raw=True)

def add_intermarket_features(primary_df, correlator_dfs):
    """
    Adds robust Intermarket relationships using ES (Equities), ZN (Rates), and 6E (FX).
    
    Args:
        primary_df (pd.DataFrame): The main strategy dataframe (Volume Clock).
        correlator_dfs (dict): Dictionary of {suffix: DataFrame} for ES, ZN, 6E.
    
    Returns:
        pd.DataFrame: Enriched dataframe.
    """
    if primary_df is None: return None
    
    print(f"Calculating Intermarket Features (ES, ZN, 6E)...")
    df = primary_df.copy()
    
    # Ensure sorted by time_end for merge_asof
    if 'time_end' in df.columns:
        df = df.sort_values('time_end')
    
    # We need to create a unified time index to align these assets
    # Since primary_df is Volume Bars, we use 'time_end' as the sync point.
    
    for suffix, corr_df in correlator_dfs.items():
        if corr_df is None or corr_df.empty:
            print(f"  Warning: Data for {suffix} is empty or None. Skipping.")
            continue
            
        print(f"  Processing {suffix}...")
        
        # Ensure DateTime
        if 'ts_event' in corr_df.columns:
            corr_df['ts_event'] = pd.to_datetime(corr_df['ts_event'])
            corr_df = corr_df.sort_values('ts_event')
            
            # CRITICAL FIX: Look-Ahead Bias Prevention
            # IBKR Historical Data uses Bar Start Time. Close price is known at End Time.
            # If we merge on Start Time, we peek 1 minute into future.
            # Heuristic: If median delta is ~60s, it's 1-min bars. Shift +60s.
            if len(corr_df) > 100:
                sample_diffs = corr_df['ts_event'].diff().dropna().iloc[:1000]
                median_diff = sample_diffs.median().total_seconds()
                
                if 58 <= median_diff <= 62:
                    print(f"  ⚠️  Detected 1-min Bars for {suffix}. Shifting timestamps +60s to prevent look-ahead.")
                    corr_df['ts_event'] = corr_df['ts_event'] + pd.Timedelta(seconds=60)
        else:
             print(f"  Warning: No 'ts_event' in {suffix}. Skipping.")
             continue

        # 1. Price Alignment (Merge AsOf)
        # Find the Price of the Correlator at the time the Primary Bar ENDED
        # Using 'last_price' if available, else 'close' or 'mid_price'
        price_col = 'last_price' if 'last_price' in corr_df.columns else 'close'
        if price_col not in corr_df.columns:
            # Fallback
            if 'mid_price' in corr_df.columns: price_col = 'mid_price'
            else: 
                print(f"  Warning: No price column in {suffix}. Skipping.")
                continue
                
        # Clean Correlator
        target = corr_df[['ts_event', price_col]].dropna().rename(columns={price_col: f'price{suffix}'})
        
        # Merge
        merged = pd.merge_asof(
            df[['time_end']],
            target,
            left_on='time_end',
            right_on='ts_event',
            direction='backward'
        )

        # Persist aligned price for later calculations (Spreads/Curves)
        if f'price{suffix}' in merged.columns:
            df[f'price{suffix}'] = merged[f'price{suffix}']

        # Special Handling for Market Internals (TICK/TRIN) which are Stationary Levels
        if 'tick' in suffix.lower() or 'trin' in suffix.lower():
            # Just add the Level and a Z-Score
            df[f'level{suffix}'] = merged[f'price{suffix}'].ffill()
            
            # Refactor: Use Smoothed Levels (Trend) instead of volatile Z-scores
            df[f'smoothed_level_100{suffix}'] = df[f'level{suffix}'].rolling(100).mean()
            df[f'smoothed_level_400{suffix}'] = df[f'level{suffix}'].rolling(400).mean()
            
            # Simple Z-Score (Regime) - Short term (50) and Long term (200)
            roll_mean = df[f'level{suffix}'].rolling(50).mean()
            roll_std = df[f'level{suffix}'].rolling(50).std().replace(0, 1)
            df[f'zscore_50{suffix}'] = (df[f'level{suffix}'] - roll_mean) / roll_std
            
            continue # Skip the Returns/Correlation logic for these

        # Calculate Returns of Correlator aligned to Primary Bars
        # Ret = ln(P_t / P_{t-1})
        merged[f'ret{suffix}'] = np.log(merged[f'price{suffix}'] / merged[f'price{suffix}'].shift(1))
        
        # 2. Correlation Regime (Rolling 100 bars)
        # Correlation of Primary Returns vs Correlator Returns
        # Note: Primary Returns need to be calculated here if not present, but usually are.
        # We'll re-calculate temp returns just to be safe.
        prim_ret = np.log(df['close'] / df['close'].shift(1))
        
        df[f'corr_100{suffix}'] = prim_ret.rolling(100).corr(merged[f'ret{suffix}'])
        
        # 3. Relative Strength (Ratio Trend)
        # Ratio = Primary / Correlator
        # We want the Trend of this Ratio (Z-Score of Ratio)
        ratio = df['close'] / merged[f'price{suffix}']
        ratio_mean = ratio.rolling(200).mean()
        ratio_std = ratio.rolling(200).std()
        df[f'rel_strength_z{suffix}'] = (ratio - ratio_mean) / ratio_std.replace(0, 1)
        
        # 4. Lead-Lag / Divergence
        # If Correlation is HIGH, but Returns Diverge -> Signal?
        # Divergence = (Prim_Ret_Norm - Corr_Ret_Norm)
        # We normalize returns by volatility to compare apples to oranges
        prim_vol = prim_ret.rolling(50).std()
        corr_vol = merged[f'ret{suffix}'].rolling(50).std()
        
        norm_prim = prim_ret / prim_vol.replace(0, 1)
        norm_corr = merged[f'ret{suffix}'] / corr_vol.replace(0, 1)
        
        df[f'divergence_50{suffix}'] = norm_prim - norm_corr
        
        # Fill NaNs
        cols = [f'corr_100{suffix}', f'rel_strength_z{suffix}', f'divergence_50{suffix}']
        df[cols] = df[cols].fillna(0)

    # --- COMPUTE FEATURES (The Matrix) ---
    # "USD Coherence": Are Majors moving together?
    # {config.PRIMARY_TICKER} and GBPUSD should be POSITIVELY correlated (both /USD)
    # {config.PRIMARY_TICKER} and USDJPY should be NEGATIVELY correlated (USD/ vs /USD)
    # If USD is driving: corr(EUR, GBP) ~ 1.0, corr(EUR, JPY) ~ -1.0
    # Coherence = (corr_GBP - corr_JPY) / 2.0  -> Target 1.0
    if 'corr_100_gbpusd' in df.columns and 'corr_100_usdjpy' in df.columns:
        df['usd_coherence_100'] = (df['corr_100_gbpusd'] - df['corr_100_usdjpy']) / 2.0
    
    # --- EXPLICIT SPREAD FEATURES (Rates) ---
    # We look for specific suffixes to construct physically meaningful spreads
    # Spread = Yield A - Yield B
    
    # TNX (US 10Y) - BUND (EU 10Y)
    if 'price_tnx' in df.columns and 'price_bund' in df.columns:
        df['spread_tnx_bund'] = df['price_tnx'] - df['price_bund']
        # Z-Score of Spread (Regime)
        spread_mean = df['spread_tnx_bund'].rolling(400).mean()
        spread_std = df['spread_tnx_bund'].rolling(400).std()
        df['spread_tnx_bund_z_400'] = (df['spread_tnx_bund'] - spread_mean) / spread_std.replace(0, 1)
        # New Feature: Spread Momentum
        df['spread_tnx_bund_slope_100'] = df['spread_tnx_bund'].diff(100)

    # US2Y (US 2Y) - SCHATZ (EU 2Y)
    if 'price_us2y' in df.columns and 'price_schatz' in df.columns:
        df['spread_us2y_schatz'] = df['price_us2y'] - df['price_schatz']
        # Z-Score
        spread_mean = df['spread_us2y_schatz'].rolling(400).mean()
        spread_std = df['spread_us2y_schatz'].rolling(400).std()
        df['spread_us2y_schatz_z_400'] = (df['spread_us2y_schatz'] - spread_mean) / spread_std.replace(0, 1)

    # --- DOMESTIC YIELD CURVES ---
    # US Curve: TNX (10Y) - US2Y (2Y)
    if 'price_tnx' in df.columns and 'price_us2y' in df.columns:
        df['curve_us_10y_2y'] = df['price_tnx'] - df['price_us2y']
        df['curve_us_10y_2y_z_400'] = (df['curve_us_10y_2y'] - df['curve_us_10y_2y'].rolling(400).mean()) / df['curve_us_10y_2y'].rolling(400).std().replace(0, 1)
        # New Feature: Curve Slope (Steepening/Flattening Momentum)
        df['curve_us_10y_2y_slope_100'] = df['curve_us_10y_2y'].diff(100)

    # EU Curve: BUND (10Y) - SCHATZ (2Y)
    if 'price_bund' in df.columns and 'price_schatz' in df.columns:
        df['curve_eu_10y_2y'] = df['price_bund'] - df['price_schatz']
        df['curve_eu_10y_2y_z_400'] = (df['curve_eu_10y_2y'] - df['curve_eu_10y_2y'].rolling(400).mean()) / df['curve_eu_10y_2y'].rolling(400).std().replace(0, 1)

    # --- BTP/BUND SPREAD (Eurozone Fragility) ---
    # BTP (Italian 10Y) - BUND (German 10Y) = Peripheral risk premium
    # Widening spread = Eurozone stress, narrowing = calm
    if 'price_btp' in df.columns and 'price_bund' in df.columns:
        df['spread_btp_bund'] = df['price_btp'] - df['price_bund']

        # Z-Score of Spread (Regime)
        spread_mean = df['spread_btp_bund'].rolling(400).mean()
        spread_std = df['spread_btp_bund'].rolling(400).std()
        df['spread_btp_bund_z_400'] = (df['spread_btp_bund'] - spread_mean) / spread_std.replace(0, 1)

        # Spread Momentum
        df['spread_btp_bund_slope_100'] = df['spread_btp_bund'].diff(100)

        # Hurst Exponent of Spread (Market Structure)
        # H > 0.5: Trend-persistent (directional move in progress)
        # H < 0.5: Mean-reverting (choppy, uncertain)
        # H ~ 0.5: Random walk
        # Regime changes in H often precede price moves
        df['spread_btp_bund_hurst_100'] = rolling_hurst(df['spread_btp_bund'], 100)
        df['spread_btp_bund_hurst_200'] = rolling_hurst(df['spread_btp_bund'], 200)

        # Hurst Change (Acceleration of regime shift)
        df['spread_btp_bund_hurst_delta_50'] = df['spread_btp_bund_hurst_100'].diff(50)

        # Fill NaNs from rolling calculations
        hurst_cols = ['spread_btp_bund_hurst_100', 'spread_btp_bund_hurst_200', 'spread_btp_bund_hurst_delta_50']
        df[hurst_cols] = df[hurst_cols].fillna(0.5)  # Default to random walk

    # 3. Vol-of-Vol (Second Derivative of Fear)
    if 'price_vix' in df.columns:
        vix_ret = np.log(df['price_vix'] / df['price_vix'].shift(1))
        df['vol_of_vol_vix_100'] = vix_ret.rolling(100).std()

    # 4. Cross-Asset Momentum Confirmation (Risk-On/Risk-Off Score)
    # EURUSD long is higher-confidence when ES rallying AND ZN falling (risk-on)
    # Risk-on = equities up, bonds down. Risk-off = opposite.
    # +2 = strong risk-on, -2 = strong risk-off, 0 = mixed
    if 'price_es' in df.columns and 'price_zn' in df.columns:
        # Calculate returns over 50 bars (approx 1.5 hours at avg bar duration)
        ret_50_es = np.log(df['price_es'] / df['price_es'].shift(50))
        ret_50_zn = np.log(df['price_zn'] / df['price_zn'].shift(50))

        # Sign of momentum: +1 if up, -1 if down, 0 if flat
        es_momentum = np.sign(ret_50_es)
        zn_momentum = np.sign(ret_50_zn)

        # Risk-on: ES up (+1) and ZN down (-1) = +1 - (-1) = +2
        # Risk-off: ES down (-1) and ZN up (+1) = -1 - 1 = -2
        # Mixed: 0
        df['risk_on_momentum'] = es_momentum - zn_momentum

    return df
