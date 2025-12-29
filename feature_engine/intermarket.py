import pandas as pd
import numpy as np
from .loader import load_ticker_data

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

        # Special Handling for Market Internals (TICK/TRIN) which are Stationary Levels
        if 'tick' in suffix.lower() or 'trin' in suffix.lower():
            # Just add the Level and a Z-Score
            df[f'level{suffix}'] = merged[f'price{suffix}'].fillna(method='ffill')
            
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

    return df
