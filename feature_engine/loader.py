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
        
        # Define Keywords from Config
        eur_locs = config.GDELT_KEYWORDS['EUR_LOCS']
        usd_locs = config.GDELT_KEYWORDS['USD_LOCS']
        
        # Optimized String Matching
        # FillNa to avoid errors
        loc_str = df['LOCATIONS'].fillna("").astype(str).str.upper()
        theme_str = df['THEMES'].fillna("").astype(str)
        
        # Location Masks
        eur_mask = loc_str.str.contains('|'.join([x.upper() for x in eur_locs]), regex=True)
        usd_mask = loc_str.str.contains('|'.join([x.upper() for x in usd_locs]), regex=True)
        de_mask = loc_str.str.contains('GERMANY|BERLIN|DE', regex=True)
        
        # Pre-calculate boolean masks for themes to speed up grouping
        conflict_mask = theme_str.str.contains(config.GDELT_KEYWORDS['CONFLICT_THEMES'], regex=True)
        epu_mask = theme_str.str.contains(config.GDELT_KEYWORDS['EPU_THEMES'], regex=True)
        inflation_mask = theme_str.str.contains(config.GDELT_KEYWORDS['INFLATION_THEMES'], regex=True)
        cb_mask = theme_str.str.contains(config.GDELT_KEYWORDS['CB_THEMES'], regex=True)
        energy_mask = theme_str.str.contains(config.GDELT_KEYWORDS['ENERGY_THEMES'], regex=True)
        
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
    Uses chunked processing to avoid OOM on large datasets.
    """
    search_dir = config.DIRS['DATA_GDELT']
    if not os.path.exists(search_dir):
        # Try raw dir
        search_dir = os.path.join(data_dir, "gdelt")
        
    files = glob.glob(os.path.join(search_dir, pattern))
    if not files:
        print(f"No GDELT files found in {search_dir}")
        return None
        
    print(f"Loading {len(files)} GDELT files (Parallel - Chunked)...")
    
    # Chunking Configuration
    chunk_size = 100 # Number of files per chunk
    max_workers = 8  # Limit concurrent workers to save RAM
    intermediate_aggs = []
    
    # Process in Chunks
    for i in range(0, len(files), chunk_size):
        chunk_files = files[i : i + chunk_size]
        print(f"  Processing chunk {i // chunk_size + 1} / {(len(files) - 1) // chunk_size + 1}...")
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_gdelt_file, chunk_files))
            
            # Filter Nones
            dfs = [r for r in results if r is not None]
            
            if dfs:
                # Concat and Aggregate this chunk immediately to save memory
                chunk_df = pd.concat(dfs, ignore_index=True)
                chunk_agg = chunk_df.groupby('date').sum()
                intermediate_aggs.append(chunk_agg)
                
            # Explicit clean up
            del results
            del dfs
            
        except Exception as e:
            print(f"  ⚠️ Error processing chunk {i}: {e}")
            continue

    if not intermediate_aggs:
        print("No valid GDELT data processed.")
        return None
    
    # Final Aggregation
    print("Aggregating final results...")
    try:
        final_agg = pd.concat(intermediate_aggs).groupby('date').sum()
        print(f"  Final Agg Shape: {final_agg.shape}")
    except Exception as e:
        print(f"  ❌ Aggregation Error: {e}")
        return None
    
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
    # Use Z-Score for robustness
    # FIX: Use Expanding Window to avoid Lookahead Bias
    tone_mean = gdelt_daily['global_tone'].expanding(min_periods=5).mean()
    tone_std = gdelt_daily['global_tone'].expanding(min_periods=5).std()
    
    # Fill early NaNs with first valid or 0/1 defaults
    tone_mean = tone_mean.bfill().fillna(0)
    tone_std = tone_std.bfill().fillna(1.0)
    
    # gdelt_daily['tone_zscore'] = (gdelt_daily['global_tone'] - tone_mean) / tone_std
    tone_zscore = (gdelt_daily['global_tone'] - tone_mean) / tone_std
    
    gdelt_daily['panic_score'] = np.where(tone_zscore < -2.0, gdelt_daily['global_polarity'] * -1, 0)
    
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
    # gdelt_daily['news_vol_zscore'] = (gdelt_daily['total_vol'] - roll_mean) / roll_std.replace(0, 1)
    
def process_gdelt_v2_file(f):
    """
    Process a single 15-min GDELT V2 GKG parquet file.
    Returns aggregated stats for that 15-min interval.
    """
    try:
        # Load Parquet
        # Expected cols: GKGRECORDID, DATE, SourceCommonName, DocumentIdentifier, Counts, Themes, Locations, V2Tone, batch_ts, date_utc
        df = pd.read_parquet(f)
        if df.empty: return None

        # Ensure date_utc is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date_utc']):
            df['date_utc'] = pd.to_datetime(df['date_utc'])
            
        # We aggregate everything to the distinct timestamp of this file (usually just one)
        # But let's group by date_utc just in case multiple timestamps exist (rare in batch file)
        
        # Parse V2Tone (already parsed in ingest as tone_mean, tone_polarity)
        # Check if columns exist
        if 'tone_mean' not in df.columns:
            # Fallback if raw V2Tone exists
            if 'V2Tone' in df.columns:
                 tone_data = df['V2Tone'].astype(str).str.split(',', expand=True)
                 df['tone_mean'] = tone_data[0].astype(float)
                 df['tone_polarity'] = tone_data[3].astype(float)
            else:
                return None

        # Keyword Definitions (Same as Daily)
        eur_locs = config.GDELT_KEYWORDS['EUR_LOCS']
        usd_locs = config.GDELT_KEYWORDS['USD_LOCS']
        
        # Optimize matching
        loc_str = df['Locations'].fillna("").astype(str).str.upper()
        theme_str = df['Themes'].fillna("").astype(str)
        
        eur_mask = loc_str.str.contains('|'.join([x.upper() for x in eur_locs]), regex=True)
        usd_mask = loc_str.str.contains('|'.join([x.upper() for x in usd_locs]), regex=True)
        de_mask = loc_str.str.contains('GERMANY|BERLIN|DE', regex=True)
        
        conflict_mask = theme_str.str.contains(config.GDELT_KEYWORDS['CONFLICT_THEMES'], regex=True)
        epu_mask = theme_str.str.contains(config.GDELT_KEYWORDS['EPU_THEMES'], regex=True)
        inflation_mask = theme_str.str.contains(config.GDELT_KEYWORDS['INFLATION_THEMES'], regex=True)
        cb_mask = theme_str.str.contains(config.GDELT_KEYWORDS['CB_THEMES'], regex=True)
        energy_mask = theme_str.str.contains(config.GDELT_KEYWORDS['ENERGY_THEMES'], regex=True)
        
        agg_data = []
        for date, group in df.groupby('date_utc'):
            idx = group.index
            g_eur_mask = eur_mask.loc[idx]
            g_usd_mask = usd_mask.loc[idx]
            g_de_mask = de_mask.loc[idx]
            
            eur_group = group[g_eur_mask]
            usd_group = group[g_usd_mask]
            
            agg_data.append({
                'time_start': date, # Using time_start to match bar convention
                'news_vol_eur': len(eur_group),
                'news_tone_eur_sum': eur_group['tone_mean'].sum(),
                'news_vol_usd': len(usd_group),
                'news_tone_usd_sum': usd_group['tone_mean'].sum(),
                
                'global_tone_sum': group['tone_mean'].sum(),
                'global_polarity_sum': group['tone_polarity'].sum(),
                'total_count': len(group),
                
                'conflict_intensity': conflict_mask.loc[idx].sum(),
                'epu_total': epu_mask.loc[idx].sum(),
                'epu_usd': (epu_mask.loc[idx] & g_usd_mask).sum(),
                'epu_eur': (epu_mask.loc[idx] & g_de_mask).sum(),
                
                'inflation_vol': inflation_mask.loc[idx].sum(),
                'cb_tone_sum': group.loc[cb_mask.loc[idx], 'tone_mean'].sum(),
                'cb_count': cb_mask.loc[idx].sum(),
                'energy_crisis_eur': (energy_mask.loc[idx] & g_eur_mask).sum()
            })
            
        return pd.DataFrame(agg_data)
        
    except Exception as e:
        print(f"Error processing V2 file {f}: {e}")
        return None

def load_gdelt_v2_data(data_dir=None):
    """
    Loads GDELT V2 GKG (15-min) data and aggregates it to Intraday resolution.
    Returns a DataFrame indexed by time_start (UTC).
    """
    if data_dir is None:
        data_dir = config.DIRS.get("DATA_GDELT", "data/gdelt")
        
    v2_dir = os.path.join(data_dir, "v2_gkg")
    
    # Check if dir exists, if not, try config.DIRS['DATA_GDELT']
    if not os.path.exists(v2_dir):
        alt_dir = os.path.join(config.DIRS.get("DATA_GDELT", "data/gdelt"), "v2_gkg")
        if os.path.exists(alt_dir):
            v2_dir = alt_dir
    
    if not os.path.exists(v2_dir):
        print(f"No V2 GKG dir found at {v2_dir}")
        return None
        
    files = glob.glob(os.path.join(v2_dir, "gdelt_v2_gkg_*.parquet"))
    files.sort() # Ensure time order
    
    if not files:
        print(f"No GDELT V2 files found in {v2_dir}")
        return None
        
    print(f"Loading {len(files)} GDELT V2 files (Intraday)...")
    
    # Process in chunks to be safe
    chunk_size = 200
    max_workers = 8
    intermediate_aggs = []
    
    for i in range(0, len(files), chunk_size):
        chunk_files = files[i : i + chunk_size]
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_gdelt_v2_file, chunk_files))
            
            dfs = [r for r in results if r is not None]
            if dfs:
                chunk_df = pd.concat(dfs, ignore_index=True)
                # Group by time in case of overlaps, though unlikely with 15-min batches
                chunk_agg = chunk_df.groupby('time_start').sum().reset_index()
                intermediate_aggs.append(chunk_agg)
                
            del results, dfs
            
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            
    if not intermediate_aggs:
        return None
        
    final_agg = pd.concat(intermediate_aggs).groupby('time_start').sum()
    
    # Calculate derived metrics (Means, Ratios)
    gdelt_intraday = pd.DataFrame(index=final_agg.index)
    
    # 1. Base Stats
    gdelt_intraday['news_vol_eur'] = final_agg['news_vol_eur']
    gdelt_intraday['news_tone_eur'] = final_agg['news_tone_eur_sum'] / final_agg['news_vol_eur'].replace(0, 1)
    
    gdelt_intraday['news_vol_usd'] = final_agg['news_vol_usd']
    gdelt_intraday['news_tone_usd'] = final_agg['news_tone_usd_sum'] / final_agg['news_vol_usd'].replace(0, 1)
    
    # 2. Panic Index (Intraday)
    gdelt_intraday['global_tone'] = final_agg['global_tone_sum'] / final_agg['total_count'].replace(0, 1)
    gdelt_intraday['global_polarity'] = final_agg['global_polarity_sum'] / final_agg['total_count'].replace(0, 1)
    
    # Expanding Z-Score for Panic (Intraday Adaptation)
    # Using smaller window for intraday reaction? No, expanding is safer.
    tone_mean = gdelt_intraday['global_tone'].expanding(min_periods=10).mean()
    tone_std = gdelt_intraday['global_tone'].expanding(min_periods=10).std()
    
    tone_mean = tone_mean.bfill().fillna(0)
    tone_std = tone_std.bfill().fillna(1.0)
    
    tone_zscore = (gdelt_intraday['global_tone'] - tone_mean) / tone_std
    gdelt_intraday['panic_score'] = np.where(tone_zscore < -2.0, gdelt_intraday['global_polarity'] * -1, 0)
    
    gdelt_intraday['conflict_intensity'] = final_agg['conflict_intensity']
    
    # 3. EPU
    gdelt_intraday['epu_total'] = final_agg['epu_total']
    gdelt_intraday['epu_usd'] = final_agg['epu_usd']
    gdelt_intraday['epu_eur'] = final_agg['epu_eur']
    gdelt_intraday['epu_diff'] = gdelt_intraday['epu_usd'] - gdelt_intraday['epu_eur']
    
    # 4. Inflation/Yields
    gdelt_intraday['inflation_vol'] = final_agg['inflation_vol']
    gdelt_intraday['central_bank_tone'] = final_agg['cb_tone_sum'] / final_agg['cb_count'].replace(0, 1)
    
    # 5. Asset Specific
    gdelt_intraday['energy_crisis_eur'] = final_agg['energy_crisis_eur']
    
    # 6. Volume
    gdelt_intraday['total_vol'] = gdelt_intraday['news_vol_eur'] + gdelt_intraday['news_vol_usd']
    
    # Enforce UTC Index
    gdelt_intraday.index = pd.to_datetime(gdelt_intraday.index, utc=True)
    
    print(f"Processed GDELT V2 Intraday data: {len(gdelt_intraday)} periods.")
    return gdelt_intraday
