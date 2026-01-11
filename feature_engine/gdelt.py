import pandas as pd
import numpy as np
import config

def add_gdelt_features(df, gdelt_df):
    """
    Merges GDELT features into Intraday Bars.
    Refactored: Only retains 'news_rvol_usd' (Relative Volume USD) as it is the only robust feature.
    """
    if df is None or gdelt_df is None: return df
    
    df = df.copy()
    
    # Check if GDELT is Intraday or Daily
    is_intraday_gdelt = False
    if isinstance(gdelt_df.index, pd.DatetimeIndex):
        if len(gdelt_df) > 1:
            diff = gdelt_df.index[1] - gdelt_df.index[0]
            if diff < pd.Timedelta(hours=23):
                is_intraday_gdelt = True
    
    if is_intraday_gdelt:
        print("Merging GDELT Features (Intraday - 15min Resolution)...")
        # Ensure sorting for merge_asof
        df = df.sort_values('time_start')
        gdelt_df = gdelt_df.sort_index()
        
        # --- ROLLING TRANSFORMATION ---
        window = 96 # 24 hours
        
        # Only process USD Volume for RVOL calculation
        if 'news_vol_usd' not in gdelt_df.columns:
            return df
            
        cols_to_use = ['news_vol_usd']
        if 'news_tone_usd' in gdelt_df.columns:
            cols_to_use.append('news_tone_usd')
            
        gdelt_rolled = gdelt_df[cols_to_use].copy()
        
        # 1. Volume Rolling
        gdelt_rolled['vol_sum_96'] = gdelt_rolled['news_vol_usd'].rolling(window, min_periods=1).sum()
        
        # RVOL (Relative Volume) - Normalize by 30-day average
        rvol_window = 2880 # 30 days * 96
        base_usd = gdelt_rolled['news_vol_usd'].rolling(rvol_window, min_periods=1).mean().replace(0, 1)
        # Use sum of last 24h volume / expected 24h volume? 
        # Or just current vol / expected?
        # Existing logic: gdelt_rolled['news_vol_usd'] is 15-min vol.
        # rvol_window mean is avg 15-min vol.
        # So it's 15-min RVOL.
        # Refactor: We want 24h accumulated Volume RVOL?
        # Let's keep existing 15-min RVOL logic but maybe smooth it.
        # Actually, let's stick to the existing working RVOL logic.
        gdelt_rolled['news_rvol_usd'] = gdelt_rolled['news_vol_usd'].rolling(window, min_periods=1).mean() / base_usd
        
        # 2. Sentiment (Volume-Weighted)
        if 'news_tone_usd' in gdelt_rolled.columns:
            # Weighted Tone = Sum(Tone * Vol) / Sum(Vol)
            tone_vol = gdelt_rolled['news_tone_usd'] * gdelt_rolled['news_vol_usd']
            gdelt_rolled['news_sentiment_trend'] = tone_vol.rolling(window, min_periods=1).sum() / gdelt_rolled['vol_sum_96'].replace(0, 1)

            # Sentiment Impact = Sentiment * RVOL (Magnitude * Intensity)
            gdelt_rolled['news_sentiment_impact'] = gdelt_rolled['news_sentiment_trend'] * gdelt_rolled['news_rvol_usd']

            # 3. News Sentiment Decay Features
            # News impact decays exponentially (half-life ~2-4 hours for FX)
            # EWM with span converts half-life to decay factor: span = half_life / ln(2)
            # For 15-min data: 2h = 8 periods, 4h = 16 periods, 8h = 32 periods
            for hl_hours in [2, 4, 8]:
                hl_periods = int(hl_hours * 4)  # 4 periods per hour (15-min)
                span = hl_periods / np.log(2)
                gdelt_rolled[f'news_impact_decay_{hl_hours}h'] = gdelt_rolled['news_sentiment_impact'].ewm(span=span, min_periods=1).mean()

        # Prepare for merge
        gdelt_reset = gdelt_rolled.reset_index().rename(columns={'index': 'time_start', 'date_utc': 'time_start'})
        
        # Check column name of time in gdelt
        if 'time_start' not in gdelt_reset.columns:
             for c in gdelt_reset.columns:
                if pd.api.types.is_datetime64_any_dtype(gdelt_reset[c]):
                    gdelt_reset = gdelt_reset.rename(columns={c: 'time_start'})
                    break
        
        # Ensure Timezones match
        if 'time_start' in df.columns:
            df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
        if 'time_start' in gdelt_reset.columns:
            gdelt_reset['time_start'] = pd.to_datetime(gdelt_reset['time_start'], utc=True)

        # Merge Columns
        merge_cols = ['time_start', 'news_rvol_usd']
        if 'news_sentiment_trend' in gdelt_reset.columns:
            merge_cols.extend(['news_sentiment_trend', 'news_sentiment_impact'])
            # Add decay features
            for hl in [2, 4, 8]:
                decay_col = f'news_impact_decay_{hl}h'
                if decay_col in gdelt_reset.columns:
                    merge_cols.append(decay_col)

        merged = pd.merge_asof(
            df, 
            gdelt_reset[merge_cols], 
            on='time_start', 
            direction='backward'
        )
        
    else:
        # Legacy Daily Support (Minimal)
        print("Merging GDELT Features (Daily)...")
        df['date'] = df['time_start'].dt.normalize()
        gdelt_shifted = gdelt_df.shift(1).reset_index()
        # Assume daily gdelt has pre-calc features or just raw
        # We only want news_rvol_usd if possible, but daily logic is legacy.
        # Just merge what we have but drop EPU/Tone later if needed.
        # For safety, we skip extensive logic here as we mostly use intraday now.
        merged = pd.merge(df, gdelt_shifted, on='date', how='left')
        merged.drop(columns=['date'], errors='ignore', inplace=True)
        
    # Fill NaNs
    if 'news_rvol_usd' in merged.columns:
        merged['news_rvol_usd'] = merged['news_rvol_usd'].ffill().fillna(0)
    
    return merged
