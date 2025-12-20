import pandas as pd
import glob
import os

def load_ticker_data(data_dir, pattern):
    """Loads and returns a sorted DataFrame for a specific ticker pattern."""
    # Point to Clean Data Lake
    # Pattern usually passed as "RAW_TICKS_..." so we join with clean dir
    clean_dir = os.path.join(os.path.dirname(data_dir), "clean_ticks")
    if os.path.exists(clean_dir):
        search_dir = clean_dir
    else:
        search_dir = data_dir # Fallback to raw if clean doesn't exist
        
    files = glob.glob(os.path.join(search_dir, pattern))
    if not files:
        # Try raw dir if clean failed
        files = glob.glob(os.path.join(data_dir, pattern))
        if not files:
            print(f"No files found for {pattern} in {search_dir} or {data_dir}")
            return None
    
    print(f"Loading {len(files)} files for {pattern} from {os.path.dirname(files[0])}...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    if not dfs: return None

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("ts_event").reset_index(drop=True)
    
    # Calculate Mid Price logic
    if 'mid_price' not in df.columns:
        if 'pricebid' in df.columns and 'priceask' in df.columns:
            if df['pricebid'].isna().mean() > 0.9 and 'last_price' in df.columns:
                    df['mid_price'] = df['last_price']
            else:
                df['mid_price'] = (df['pricebid'] + df['priceask']) / 2
        elif 'last_price' in df.columns:
            df['mid_price'] = df['last_price']
        elif 'price' in df.columns:
            df['mid_price'] = df['price']
    
    # Ensure we have a valid mid_price
    df = df.dropna(subset=['mid_price'])
    
    return df

def load_gdelt_data(data_dir, pattern="GDELT_GKG_*.parquet"):
    """
    Loads GDELT GKG data and aggregates it to Daily resolution.
    Extracts Sentiment (Tone) and Attention (Volume) for EUR vs USD.
    """
    search_dir = os.path.join(os.path.dirname(data_dir), "gdelt")
    if not os.path.exists(search_dir):
        # Try raw dir
        search_dir = os.path.join(data_dir, "gdelt")
        
    files = glob.glob(os.path.join(search_dir, pattern))
    if not files:
        print(f"No GDELT files found in {search_dir}")
        return None
        
    print(f"Loading {len(files)} GDELT files...")
    dfs = []
    for f in files:
        try:
            # Read only necessary columns
            dfs.append(pd.read_parquet(f, columns=['date_str', 'tone_raw', 'LOCATIONS', 'THEMES']))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs: return None
    
    raw_df = pd.concat(dfs, ignore_index=True)
    
    # Parse Date
    # Assuming date_str is YYYYMMDD
    raw_df['date'] = pd.to_datetime(raw_df['date_str'], format='%Y%m%d', errors='coerce').dt.tz_localize('UTC')
    raw_df = raw_df.dropna(subset=['date'])
    
    # Parse Tone and Polarity
    # Tone Raw: Tone, Pos, Neg, Polarity, ARD, SGRD
    tone_data = raw_df['tone_raw'].astype(str).str.split(',', expand=True)
    raw_df['tone'] = tone_data[0].astype(float)
    raw_df['polarity'] = tone_data[3].astype(float)
    
    # Define Keywords
    eur_locs = ['Europe', 'Brussels', 'Germany', 'France', 'Italy', 'Spain', 'EUR', 'Euro']
    usd_locs = ['United States', 'US', 'Washington', 'New York', 'America', 'Fed']
    
    # Optimized String Matching
    raw_df['loc_str'] = raw_df['LOCATIONS'].fillna("").astype(str).str.upper()
    raw_df['theme_str'] = raw_df['THEMES'].fillna("").astype(str)
    
    # Location Masks
    eur_mask = raw_df['loc_str'].str.contains('|'.join([x.upper() for x in eur_locs]))
    usd_mask = raw_df['loc_str'].str.contains('|'.join([x.upper() for x in usd_locs]))
    de_mask = raw_df['loc_str'].str.contains('GERMANY|BERLIN|DE') # Specific for EPU comparison
    
    # Aggregation by Date
    agg_data = []
    
    for date, group in raw_df.groupby('date'):
        # 1. Base Stats
        eur_group = group[eur_mask[group.index]]
        usd_group = group[usd_mask[group.index]]
        
        eur_vol = len(eur_group)
        usd_vol = len(usd_group)
        
        # 2. Panic Index
        # Global Tone/Polarity
        global_tone = group['tone'].mean()
        global_polarity = group['polarity'].mean()
        
        # Conflict Intensity
        conflict_mask = group['theme_str'].str.contains('ARMEDCONFLICT|CRISISLEX|UNREST')
        conflict_intensity = conflict_mask.sum()
        
        # 3. EPU (Economic Policy Uncertainty)
        epu_mask = group['theme_str'].str.contains('EPU')
        epu_all = epu_mask.sum()
        epu_usd = (epu_mask & usd_mask[group.index]).sum()
        epu_eur = (epu_mask & de_mask[group.index]).sum() # Using Germany as proxy for EUR Policy Core
        
        # 4. Inflation/Yields
        inflation_mask = group['theme_str'].str.contains('ECON_INFLATION|TAX_FNCACT')
        inflation_chatter = inflation_mask.sum()
        
        cb_mask = group['theme_str'].str.contains('CENTRAL_BANK')
        cb_tone = group.loc[cb_mask, 'tone'].mean() if cb_mask.any() else 0
        
        # 5. Asset Specific
        # Energy Crisis in Europe
        energy_mask = group['theme_str'].str.contains('ENV_OIL|ECON_ENERGY_PRICES')
        energy_crisis_eur = (energy_mask & eur_mask[group.index]).sum()
        
        agg_data.append({
            'date': date,
            'news_vol_eur': eur_vol,
            'news_tone_eur': eur_group['tone'].mean() if eur_vol > 0 else 0,
            'news_vol_usd': usd_vol,
            'news_tone_usd': usd_group['tone'].mean() if usd_vol > 0 else 0,
            
            # New Features
            'panic_score': (global_polarity * -1) if global_tone < -5 else 0, # Synthetic Panic Signal
            'global_tone': global_tone,
            'global_polarity': global_polarity,
            'conflict_intensity': conflict_intensity,
            
            'epu_total': epu_all,
            'epu_usd': epu_usd,
            'epu_eur': epu_eur, # Germany
            'epu_diff': epu_usd - epu_eur, # Policy Premium
            
            'inflation_vol': inflation_chatter,
            'central_bank_tone': cb_tone,
            
            'energy_crisis_eur': energy_crisis_eur
        })
        
    gdelt_daily = pd.DataFrame(agg_data).sort_values('date').set_index('date')
    
    # 6. Volume Anomalies (Z-Score) - Time Series Calculation
    # We need a rolling window (e.g., 30 days)
    # Total Volume
    gdelt_daily['total_vol'] = gdelt_daily['news_vol_eur'] + gdelt_daily['news_vol_usd'] # Proxy
    roll_mean = gdelt_daily['total_vol'].rolling(30, min_periods=5).mean()
    roll_std = gdelt_daily['total_vol'].rolling(30, min_periods=5).std()
    gdelt_daily['news_vol_zscore'] = (gdelt_daily['total_vol'] - roll_mean) / roll_std.replace(0, 1)
    
    print(f"Processed GDELT data: {len(gdelt_daily)} days.")
    return gdelt_daily
