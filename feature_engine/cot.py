import pandas as pd
import numpy as np
import os
import config

# Mapping of IBKR Symbol/Instrument to COT Prefix
# Accessing config.TARGETS could be complex if we just have a dataframe.
# We will try to infer or use a fixed map.
SYMBOL_TO_COT_MAP = {
    'ES': 'cot_es',
    'MES': 'cot_es', # Micro maps to E-Mini
    'ZN': 'cot_zn',
    'EUR': 'cot_6e', # FX EUR
    'BTC': 'cot_btc',
    'MBT': 'cot_btc', # Micro Bitcoin
    'UBT': 'cot_btc'  # 
}

def get_cot_prefix(bars_df):
    """Tries to guess the COT prefix based on column names or metadata."""
    # This is tricky because bars_df is usually generic.
    # We rely on the calling pipeline to know what symbol it is processing.
    # However, feature_engine functions usually just take a DF.
    # 
    # HACK: Look at the global config or assume the caller handles filtering.
    # BUT, 'add_cot_features' will likely be called in a loop where we know the symbol.
    # 
    # For now, let's assume we pass the symbol or we return ALL COT columns
    # and let the feature selection figure it out? No, that's wasteful.
    #
    # Better: Inspect 'config.py' active target? No, pipeline runs for all.
    #
    # Let's check 'bars_df' for a 'symbol' column? 
    # Usually my bars_df doesn't have a symbol column.
    
    return None

def precompute_cot_features(cot_df):
    """
    Computes Z-Scores and Rolling Metrics on Weekly COT Data.
    """
    df = cot_df.copy()
    df = df.sort_values('date')
    
    # Identify prefixes present
    prefixes = set()
    for col in df.columns:
        if '_net' in col:
            prefixes.add(col.split('_net')[0])
            
    for p in prefixes:
        net_col = f"{p}_net"
        sent_col = f"{p}_sentiment"
        oi_col = f"{p}_oi"
        
        if net_col in df.columns:
            # 1. Net Positioning Z-Score (1-Year / 52-Week Lookback)
            # This identifies "Crowded Trades" relative to recent history
            roll_mean = df[net_col].rolling(52, min_periods=10).mean()
            roll_std = df[net_col].rolling(52, min_periods=10).std()
            df[f'{p}_net_zscore_1y'] = (df[net_col] - roll_mean) / roll_std.replace(0, 1)
            
            # 2. Net Positioning Z-Score (3-Year / 156-Week Lookback) - Structural
            roll_mean_3y = df[net_col].rolling(156, min_periods=30).mean()
            roll_std_3y = df[net_col].rolling(156, min_periods=30).std()
            df[f'{p}_net_zscore_3y'] = (df[net_col] - roll_mean_3y) / roll_std_3y.replace(0, 1)
            
            # 3. Positioning Change (Delta)
            df[f'{p}_net_change_1w'] = df[net_col].diff()
            
        if sent_col in df.columns:
            # Sentiment Extremes
            df[f'{p}_sentiment_extreme'] = np.where(df[sent_col] > 0.8, 1, 
                                           np.where(df[sent_col] < 0.2, -1, 0))

    return df

def add_cot_features(bars_df, symbol=None):
    """
    Merges COT data.
    
    Args:
        bars_df (pd.DataFrame): Intraday bars.
        symbol (str): The ticker symbol (e.g. 'ES'). If None, tries to merge all (messy).
    """
    cot_path = os.path.join(config.DIRS['DATA_DIR'], "cot_weekly.parquet")
    if not os.path.exists(cot_path):
        return bars_df
        
    cot_df = pd.read_parquet(cot_path)
    # Ensure datetime UTC
    cot_df['date'] = pd.to_datetime(cot_df['date'])
    if cot_df['date'].dt.tz is None:
        cot_df['date'] = cot_df['date'].dt.tz_localize('UTC')
    else:
        cot_df['date'] = cot_df['date'].dt.tz_convert('UTC')
        
    # Precompute
    cot_df = precompute_cot_features(cot_df)
    
    # Determine which columns to keep based on Symbol
    # If symbol is provided, filter for relevant columns
    cols_to_merge = ['merge_date'] # We will create merge_date
    
    target_prefix = None
    if symbol:
        # Normalize symbol
        sym_clean = symbol.replace("1!", "") # generic cleanup
        target_prefix = SYMBOL_TO_COT_MAP.get(sym_clean)
        
        # Also try exact match
        if not target_prefix and sym_clean in SYMBOL_TO_COT_MAP:
             target_prefix = SYMBOL_TO_COT_MAP[sym_clean]
             
    if target_prefix:
        # Only keep columns for this prefix
        relevant_cols = [c for c in cot_df.columns if c.startswith(target_prefix)]
        if not relevant_cols:
            return bars_df
        cols_to_merge.extend(relevant_cols)
    else:
        # Keep all? Might be too much noise. 
        # But if we don't know the symbol, maybe we shouldn't merge.
        # However, cross-asset COT might be useful (e.g. Bond positioning for Stocks).
        # Let's merge ALL valid feature columns.
        cols_to_merge.extend([c for c in cot_df.columns if '_zscore' in c or '_change' in c or '_sentiment' in c])

    # Lagging Strategy:
    # Report Date = Tuesday
    # Release Date = Friday (3 days later)
    # Safe availability = Saturday (4 days later)
    # So we map Report_Date + 4 Days -> Available Date
    cot_df['merge_date'] = cot_df['date'] + pd.Timedelta(days=4)
    
    # Prepare Bars
    if 'time_start' not in bars_df.columns: return bars_df
    bars_df['_date_only'] = bars_df['time_start'].dt.normalize()
    
    # Merge
    # Since COT is weekly, we have gaps. We need to forward fill.
    # Using merge_asof is better than merge on date for sparse data?
    # No, merge on date + ffill is standard for daily/weekly to intraday.
    
    # Filter COT df to needed cols
    cols_to_merge = list(set(cols_to_merge)) # dedup
    # Ensure merge_date is in it
    if 'merge_date' not in cols_to_merge: cols_to_merge.append('merge_date')
    
    # Check if cols exist
    available_cols = [c for c in cols_to_merge if c in cot_df.columns]
    
    merged = pd.merge(bars_df, cot_df[available_cols], left_on='_date_only', right_on='merge_date', how='left')
    
    # Forward Fill features
    feat_cols = [c for c in available_cols if c != 'merge_date']
    merged[feat_cols] = merged[feat_cols].ffill()
    
    # Cleanup
    merged.drop(columns=['_date_only', 'merge_date'], inplace=True, errors='ignore')
    
    return merged
