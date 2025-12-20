import pandas as pd
import glob
import os
import config
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def load_ticker_data(data_dir, pattern):
    """Loads and returns a sorted DataFrame for a specific ticker pattern."""
    # Point to Clean Data Lake
    clean_dir = config.DIRS['DATA_CLEAN_TICKS']
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

from concurrent.futures import ProcessPoolExecutor

def process_gdelt_file(f):
    try:
        # Read only necessary columns
        df = pd.read_parquet(f, columns=['date_str', 'tone_raw', 'LOCATIONS', 'THEMES'])
        if df.empty: return None
        
        # Parse Date
        df['date'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce').dt.tz_localize('UTC')
        df = df.dropna(subset=['date'])
        if df.empty: return None

        # Parse Tone and Polarity
        tone_data = df['tone_raw'].astype(str).str.split(',', expand=True)
        df['tone'] = tone_data[0].astype(float)
        df['polarity'] = tone_data[3].astype(float)
        
        # Define Keywords
        eur_locs = ['Europe', 'Brussels', 'Germany', 'France', 'Italy', 'Spain', 'EUR', 'Euro']
        usd_locs = ['United States', 'US', 'Washington', 'New York', 'America', 'Fed']
        
        # Optimized String Matching
        # FillNa to avoid errors
        loc_str = df['LOCATIONS'].fillna("").astype(str).str.upper()
        theme_str = df['THEMES'].fillna("").astype(str)
        
        # Location Masks
        eur_mask = loc_str.str.contains('|'.join([x.upper() for x in eur_locs]), regex=True)
        usd_mask = loc_str.str.contains('|'.join([x.upper() for x in usd_locs]), regex=True)
        de_mask = loc_str.str.contains('GERMANY|BERLIN|DE', regex=True)
        
        # Pre-calculate boolean masks for themes to speed up grouping
        conflict_mask = theme_str.str.contains('ARMEDCONFLICT|CRISISLEX|UNREST', regex=True)
        epu_mask = theme_str.str.contains('EPU', regex=True)
        inflation_mask = theme_str.str.contains('ECON_INFLATION|TAX_FNCACT', regex=True)
        cb_mask = theme_str.str.contains('CENTRAL_BANK', regex=True)
        energy_mask = theme_str.str.contains('ENV_OIL|ECON_ENERGY_PRICES', regex=True)
        
        # Aggregate by Date within this file
        agg_data = []
        for date, group in df.groupby('date'):
            # Indices for this group
            idx = group.index
            
            # Group masks (slicing boolean arrays by index)
            g_eur_mask = eur_mask.loc[idx]
            g_usd_mask = usd_mask.loc[idx]
            g_de_mask = de_mask.loc[idx]
            
            eur_group = group[g_eur_mask]
            usd_group = group[g_usd_mask]
            
            eur_vol = len(eur_group)
            usd_vol = len(usd_group)
            
            # Base Stats
            agg_data.append({
                'date': date,
                'news_vol_eur': eur_vol,
                'news_tone_eur_sum': eur_group['tone'].sum(), # Sum for now, divide by vol later
                'news_vol_usd': usd_vol,
                'news_tone_usd_sum': usd_group['tone'].sum(),
                
                'global_tone_sum': group['tone'].sum(),
                'global_polarity_sum': group['polarity'].sum(),
                'total_count': len(group),
                
                'conflict_intensity': conflict_mask.loc[idx].sum(),
                'epu_total': epu_mask.loc[idx].sum(),
                'epu_usd': (epu_mask.loc[idx] & g_usd_mask).sum(),
                'epu_eur': (epu_mask.loc[idx] & g_de_mask).sum(),
                
                'inflation_vol': inflation_mask.loc[idx].sum(),
                
                # For CB Tone, we need the mean of a subset
                'cb_tone_sum': group.loc[cb_mask.loc[idx], 'tone'].sum(),
                'cb_count': cb_mask.loc[idx].sum(),
                
                'energy_crisis_eur': (energy_mask.loc[idx] & g_eur_mask).sum()
            })
            
        return pd.DataFrame(agg_data)
        
    except Exception as e:
        print(f"Error processing {f}: {e}")
        return None

def load_gdelt_data(data_dir, pattern="GDELT_GKG_*.parquet"):
    """
    Loads GDELT GKG data and aggregates it to Daily resolution.
    Extracts Sentiment (Tone) and Attention (Volume) for EUR vs USD.
    """
    search_dir = config.DIRS['DATA_GDELT']
    if not os.path.exists(search_dir):
        # Try raw dir
        search_dir = os.path.join(data_dir, "gdelt")
        
    files = glob.glob(os.path.join(search_dir, pattern))
    if not files:
        print(f"No GDELT files found in {search_dir}")
        return None
        
    print(f"Loading {len(files)} GDELT files (Parallel)...")
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_gdelt_file, files))
    
    # Filter Nones
    dfs = [r for r in results if r is not None]
    
    if not dfs: return None
    
    # Concat all partial aggregations
    combined = pd.concat(dfs, ignore_index=True)
    
    # Final Aggregation by Date (summing up partials)
    final_agg = combined.groupby('date').sum()
    
    # Calculate weighted means
    gdelt_daily = pd.DataFrame(index=final_agg.index)
    
    # 1. Base Stats
    gdelt_daily['news_vol_eur'] = final_agg['news_vol_eur']
    gdelt_daily['news_tone_eur'] = final_agg['news_tone_eur_sum'] / final_agg['news_vol_eur'].replace(0, 1)
    
    gdelt_daily['news_vol_usd'] = final_agg['news_vol_usd']
    gdelt_daily['news_tone_usd'] = final_agg['news_tone_usd_sum'] / final_agg['news_vol_usd'].replace(0, 1)
    
    # 2. Panic Index
    gdelt_daily['global_tone'] = final_agg['global_tone_sum'] / final_agg['total_count'].replace(0, 1)
    gdelt_daily['global_polarity'] = final_agg['global_polarity_sum'] / final_agg['total_count'].replace(0, 1)
    
    # Panic Score Logic (re-implemented)
    # 'panic_score': (global_polarity * -1) if global_tone < -5 else 0
    gdelt_daily['panic_score'] = np.where(gdelt_daily['global_tone'] < -5, gdelt_daily['global_polarity'] * -1, 0)
    
    gdelt_daily['conflict_intensity'] = final_agg['conflict_intensity']
    
    # 3. EPU
    gdelt_daily['epu_total'] = final_agg['epu_total']
    gdelt_daily['epu_usd'] = final_agg['epu_usd']
    gdelt_daily['epu_eur'] = final_agg['epu_eur']
    gdelt_daily['epu_diff'] = gdelt_daily['epu_usd'] - gdelt_daily['epu_eur']
    
    # 4. Inflation/Yields
    gdelt_daily['inflation_vol'] = final_agg['inflation_vol']
    gdelt_daily['central_bank_tone'] = final_agg['cb_tone_sum'] / final_agg['cb_count'].replace(0, 1)
    
    # 5. Asset Specific
    gdelt_daily['energy_crisis_eur'] = final_agg['energy_crisis_eur']
    
    # 6. Volume Anomalies (Z-Score)
    gdelt_daily['total_vol'] = gdelt_daily['news_vol_eur'] + gdelt_daily['news_vol_usd']
    roll_mean = gdelt_daily['total_vol'].rolling(30, min_periods=5).mean()
    roll_std = gdelt_daily['total_vol'].rolling(30, min_periods=5).std()
    gdelt_daily['news_vol_zscore'] = (gdelt_daily['total_vol'] - roll_mean) / roll_std.replace(0, 1)
    
    print(f"Processed GDELT data: {len(gdelt_daily)} days.")
    return gdelt_daily
