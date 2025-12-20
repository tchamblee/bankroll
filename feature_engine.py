import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import timedelta
import physics_features as phys

class FeatureEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bars = None
        # We don't store raw ticks in self anymore to save memory, 
        # but we need them temporarily for correlator processing.

    def load_ticker_data(self, pattern):
        """Loads and returns a sorted DataFrame for a specific ticker pattern."""
        # Point to Clean Data Lake
        # Pattern usually passed as "RAW_TICKS_..." so we join with clean dir
        # We need to handle if pattern is absolute or relative
        # Assuming pattern is just the filename pattern like "RAW_TICKS_SPY*.parquet"
        
        # If the user passes a full path, we might break.
        # But in our scripts we pass "RAW_TICKS_...".
        # Let's override the data_dir to clean_ticks if not already set
        clean_dir = os.path.join(os.path.dirname(self.data_dir), "clean_ticks")
        if os.path.exists(clean_dir):
            search_dir = clean_dir
        else:
            search_dir = self.data_dir # Fallback to raw if clean doesn't exist
            
        files = glob.glob(os.path.join(search_dir, pattern))
        if not files:
            # Try raw dir if clean failed
            files = glob.glob(os.path.join(self.data_dir, pattern))
            if not files:
                print(f"No files found for {pattern} in {search_dir} or {self.data_dir}")
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

    def create_volume_bars(self, primary_df, volume_threshold=1000):
        """
        Creates Volume Bars from the PRIMARY asset.
        Captures Start/End times for Exact-Interval analysis.
        """
        if primary_df is None: return None

        print(f"Generating Volume Bars (Threshold: {volume_threshold})...")
        
        df = primary_df.copy()
        
        # Volume Proxy Logic - Forcing Tick Bars (1 per row) for robust sampling
        vol_series = pd.Series(1, index=df.index)

        df['vol_proxy'] = vol_series
        df['cum_vol'] = vol_series.cumsum()
        df['bar_id'] = (df['cum_vol'] // volume_threshold).astype(int)
        
        # Aggressor Logic
        delta = df['mid_price'].diff().fillna(0)
        direction = np.sign(delta)
        df['aggressor_vol'] = direction * vol_series
        
        # Aggregation
        # We need first ts_event (start) and last ts_event (end)
        agg_dict = {
            'ts_event': ['first', 'last'],
            'mid_price': ['first', 'max', 'min', 'last'],
            'vol_proxy': 'sum',
            'aggressor_vol': 'sum'
        }
        
        # Add Bid/Ask Size aggregation if available
        if 'sizebid' in df.columns and 'sizeask' in df.columns:
            agg_dict['sizebid'] = 'mean'
            agg_dict['sizeask'] = 'mean'
            
        # Add Bid/Ask Price aggregation for Spread analysis
        if 'pricebid' in df.columns and 'priceask' in df.columns:
            agg_dict['pricebid'] = 'mean'
            agg_dict['priceask'] = 'mean'
            
        bars = df.groupby('bar_id').agg(agg_dict)
        
        # Flatten Columns
        # The order depends on the keys. Groupby sorts keys? No, usually follows order.
        # Safest way is to reconstruct based on what we added
        
        flat_cols = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol']
        if 'sizebid' in agg_dict:
            flat_cols.extend(['avg_bid_size', 'avg_ask_size'])
        if 'pricebid' in agg_dict:
            flat_cols.extend(['avg_bid_price', 'avg_ask_price'])
            
        bars.columns = flat_cols
        bars.reset_index(drop=True, inplace=True)
        
        # Basic Features
        bars['ticket_imbalance'] = np.where(bars['volume'] == 0, 0, bars['net_aggressor_vol'] / bars['volume'])
        
        self.bars = bars
        print(f"Generated {len(bars)} Volume Bars.")
        return bars

    def add_correlator_residual(self, correlator_df, suffix="_corr", window=100):
        """
        Calculates the Residual (Alpha) of the Primary asset relative to the Correlator.
        1. Resamples Correlator to the EXACT start/end times of the Primary Bars.
        2. Calculates Returns for both over that interval.
        3. Computes Rolling Beta and Residual.
        """
        if self.bars is None or correlator_df is None: return
        
        print(f"Calculating Residuals for {suffix} (Window={window})...")
        
        # 1. Align Correlator Prices to Bar Start/End
        # We use merge_asof to find the Correlator price at the exact moment the bar started and ended.
        # This effectively 'resamples' the Correlator to the Primary's Volume Clock.
        
        corr_clean = correlator_df[['ts_event', 'mid_price']].sort_values('ts_event').dropna()
        
        # Get Start Prices
        # We merge onto self.bars. We need to ensure types match.
        start_prices = pd.merge_asof(
            self.bars[['time_start']], 
            corr_clean, 
            left_on='time_start', 
            right_on='ts_event', 
            direction='backward'
        )['mid_price']
        
        # Get End Prices
        end_prices = pd.merge_asof(
            self.bars[['time_end']], 
            corr_clean, 
            left_on='time_end', 
            right_on='ts_event', 
            direction='backward'
        )['mid_price']
        
        # 2. Calculate Returns over the Interval
        # Handle gaps where correlator might not have data (NaNs)
        # Using logarithmic returns for better additivity
        corr_ret = np.log(end_prices / start_prices)
        prim_ret = np.log(self.bars['close'] / self.bars['open']) # Log return of the bar itself
        
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
        self.bars[f'residual{suffix}'] = tmp['prim'] - expected_ret
        self.bars[f'beta{suffix}'] = beta
        
        # Fill NaNs (early window)
        self.bars[f'residual{suffix}'] = self.bars[f'residual{suffix}'].fillna(0)

    def load_gdelt_data(self, pattern="GDELT_GKG_*.parquet"):
        """
        Loads GDELT GKG data and aggregates it to Daily resolution.
        Extracts Sentiment (Tone) and Attention (Volume) for EUR vs USD.
        """
        search_dir = os.path.join(os.path.dirname(self.data_dir), "gdelt")
        if not os.path.exists(search_dir):
            # Try raw dir
            search_dir = os.path.join(self.data_dir, "gdelt")
            
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

    def add_gdelt_features(self, gdelt_df):
        """
        Merges Daily GDELT features into Intraday Bars.
        CRITICAL: Uses LAG 1 (Yesterday's News) to avoid Lookahead Bias.
        """
        if not hasattr(self, 'bars') or gdelt_df is None: return
        print("Merging GDELT Features (Lag 1 Day)...")
        
        df = self.bars.copy()
        
        # Extract Date from Bar Start Time
        df['date'] = df['time_start'].dt.normalize()
        
        # Shift GDELT by 1 Day to represent "Yesterday"
        # Since gdelt_df is indexed by date, we can shift the index or the data.
        # Ideally, for date T, we want GDELT from T-1.
        # So we shift GDELT index forward by 1 day? 
        # No, if we want T's features to be T-1's data, we merge T with T-1.
        # Easier: Shift GDELT dataframe forward by 1 day.
        
        gdelt_shifted = gdelt_df.shift(1) # Row 1 becomes Row 2. Date index stays? No, shift moves data.
        # shift(1) on a DF moves values down.
        # 2025-12-01: Data_A
        # 2025-12-02: Data_B
        # after shift(1):
        # 2025-12-01: NaN
        # 2025-12-02: Data_A
        # This is exactly what we want. On Dec 2nd, we see Dec 1st data.
        
        # Merge
        # We reset index to make 'date' a column
        gdelt_shifted = gdelt_shifted.reset_index()
        
        merged = pd.merge(df, gdelt_shifted, on='date', how='left')
        
        # Fill NaNs (e.g. weekends or missing days) with 0 or forward fill?
        # Forward fill is better for sentiment persistence
        cols_to_fill = ['news_vol_eur', 'news_tone_eur', 'news_vol_usd', 'news_tone_usd', 
                        'global_tone', 'global_polarity', 'conflict_intensity', 
                        'epu_total', 'epu_usd', 'epu_eur', 'epu_diff',
                        'inflation_vol', 'central_bank_tone', 'energy_crisis_eur', 'news_vol_zscore']
        
        # Only fill columns that exist (in case we run old logic)
        existing_cols = [c for c in cols_to_fill if c in merged.columns]
        merged[existing_cols] = merged[existing_cols].ffill().fillna(0)
        
        # Derived Physics Features
        # 1. Sentiment Divergence (Force Differential)
        merged['news_tone_diff'] = merged['news_tone_eur'] - merged['news_tone_usd']
        
        # 2. Attention Divergence (Mass Differential)
        merged['news_vol_diff'] = merged['news_vol_eur'] - merged['news_vol_usd']
        
        # 3. News Velocity (Rate of Change of Attention)
        # This is essentially Delta of Volume
        merged['news_velocity_eur'] = merged['news_vol_eur'].diff().fillna(0)
        merged['news_velocity_usd'] = merged['news_vol_usd'].diff().fillna(0)
        
        # 4. Regime/Panic Features
        # Already calculated in load: news_vol_zscore, epu_diff, conflict_intensity
        
        # Drop temp date column
        merged.drop(columns=['date'], inplace=True)
        
        self.bars = merged

    def add_features_to_bars(self, windows=[50, 100, 200, 400]):
        if not hasattr(self, 'bars'): return
        df = self.bars
        
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # Pre-calc Garman-Klass Variance components
        # Var_GK = 0.5 * (ln(H/L))^2 - (2*ln(2)-1) * (ln(C/O))^2
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        # Ensure non-negative (theoretical issue, practically fine)
        gk_var = gk_var.clip(lower=0)

        for w in windows:
            # 1. Velocity (Cumulative Log Return) - Scale Invariant
            df[f'velocity_{w}'] = df['log_ret'].rolling(w).sum()
            
            # 2. Volatility (Garman-Klass) - More precise, uses OHLC
            # Rolling Vol = sqrt( Rolling Mean of Variance )
            df[f'volatility_{w}'] = np.sqrt(gk_var.rolling(w).mean())
            
            net_change = df['close'].diff(w).abs()
            path = df['close'].diff().abs().rolling(w).sum()
            df[f'efficiency_{w}'] = np.where(path==0, 0, net_change/path)
            
            df[f'autocorr_{w}'] = df['log_ret'].rolling(w).corr(df['log_ret'].shift(1))
            
            # 3. Skewness (Tail Asymmetry) - 3rd Moment
            df[f'skew_{w}'] = df['log_ret'].rolling(w).skew()
            
            # 4. Price Z-Score (Mean Reversion / Deviation)
            # Distance from moving average, normalized by GK volatility
            ma = df['close'].rolling(w).mean()
            # GK Volatility is percentage-based (e.g. 0.001 for 0.1%). 
            # To normalize a price difference ($), we need price * vol ($).
            df[f'price_zscore_{w}'] = (df['close'] - ma) / (df[f'volatility_{w}'] * df['close']).replace(0, 1) 
            
            vol_norm = df[f'volatility_{w}'] / df[f'volatility_{w}'].rolling(1000, min_periods=100).mean()
            df[f'trend_strength_{w}'] = df[f'efficiency_{w}'] * vol_norm

        self.bars = df

    def add_physics_features(self):
        if not hasattr(self, 'bars'): return
        df = self.bars
        print("Calculating Physics Features...")
        
        # Ensure log_ret exists for Entropy calc
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
        df['frac_diff_04'] = phys.frac_diff_ffd(df['close'], d=0.4)
        df['frac_diff_02'] = phys.frac_diff_ffd(df['close'], d=0.2)
        df['hurst_100'] = phys.get_hurst_exponent(df['close'], window=100)
        df['hurst_200'] = phys.get_hurst_exponent(df['close'], window=200)
        
        # Shannon Entropy (Disorder)
        # Using log_ret (stationarized) is better for distribution analysis than raw price
        df['entropy_100'] = phys.get_shannon_entropy(df['log_ret'], window=100)
        df['entropy_200'] = phys.get_shannon_entropy(df['log_ret'], window=200)
        
        self.bars = df

    def add_microstructure_features(self, windows=[50, 100]):
        if not hasattr(self, 'bars'): return
        print(f"Calculating Microstructure (Order Flow) Features...")
        df = self.bars
        
        # Ensure we have the base inputs
        if 'ticket_imbalance' not in df.columns:
            # Re-calculate if missing (Net / Vol)
            # Avoid div by zero
            vol = df['volume'].replace(0, 1)
            df['ticket_imbalance'] = df['net_aggressor_vol'] / vol
            
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            
        # --- NEW: Bar Duration (Time Dilation) ---
        if 'time_end' in df.columns and 'time_start' in df.columns:
            # Calculate duration in seconds
            df['bar_duration'] = (df['time_end'] - df['time_start']).dt.total_seconds()
            # Normalize duration (optional, but raw seconds is fine for trees)
            
        # --- NEW: Pressure Imbalance (L1 Order Book) ---
        if 'avg_bid_size' in df.columns and 'avg_ask_size' in df.columns:
            total_size = df['avg_bid_size'] + df['avg_ask_size']
            df['pres_imbalance'] = (df['avg_bid_size'] - df['avg_ask_size']) / total_size.replace(0, 1)
            
        # --- NEW: Spread Intensity (Cost of Liquidity) ---
        if 'avg_bid_price' in df.columns and 'avg_ask_price' in df.columns:
            df['avg_spread'] = df['avg_ask_price'] - df['avg_bid_price']
            
        for w in windows:
            # 1. Flow Trend (Persistence of Buying/Selling pressure)
            df[f'flow_trend_{w}'] = df['ticket_imbalance'].rolling(w).mean()
            
            # 2. Price-Flow Correlation (Wyckoff / Divergence)
            # High +Corr = Healthy Trend. Low/Neg Corr = Absorption/Divergence.
            df[f'price_flow_corr_{w}'] = df['log_ret'].rolling(w).corr(df['ticket_imbalance'])
            
            # 3. Flow Shock (Z-Score of Flow)
            # How unusual is this buying/selling relative to recent history?
            flow_std = df['ticket_imbalance'].rolling(w).std()
            df[f'flow_shock_{w}'] = (df['ticket_imbalance'] - df[f'flow_trend_{w}']) / flow_std.replace(0, 1)

            # 4. Duration Trend (Acceleration/Deceleration)
            if 'bar_duration' in df.columns:
                df[f'duration_trend_{w}'] = df['bar_duration'].rolling(w).mean()
                
            # 5. Pressure Trend (Persistent Support/Resistance)
            if 'pres_imbalance' in df.columns:
                df[f'pres_trend_{w}'] = df['pres_imbalance'].rolling(w).mean()
                # Price-Pressure Correlation (Magnet vs Wall)
                df[f'price_pressure_corr_{w}'] = df['log_ret'].rolling(w).corr(df['pres_imbalance'])

            # 6. Flow Autocorrelation (Herding)
            # Persistence of the order flow itself
            df[f'flow_autocorr_{w}'] = df['ticket_imbalance'].rolling(w).corr(df['ticket_imbalance'].shift(1))
            
            # 7. Spread Intensity (Spread / Volatility)
            if 'avg_spread' in df.columns and f'volatility_{w}' in df.columns:
                # Normalize spread by price to match log-return volatility units
                spread_pct = df['avg_spread'] / df['close']
                df[f'spread_intensity_{w}'] = spread_pct / df[f'volatility_{w}'].replace(0, 1e-6)

            # 8. Order Book Alignment (Flow * Pressure)
            # Do buyers have support? Or are they hitting walls?
            if 'pres_imbalance' in df.columns:
                # Instantaneous alignment
                alignment = df['ticket_imbalance'] * df['pres_imbalance']
                df[f'order_book_alignment_{w}'] = alignment.rolling(w).mean()

        self.bars = df

    def add_crypto_features(self, ibit_pattern="CLEAN_IBIT.parquet"):
        """
        Adds Crypto-Lead features using IBIT data.
        1. Resamples EURUSD (self.bars) and IBIT to 1-min fixed intervals.
        2. Calculates Time-Based correlations and lags.
        3. Merges back to Volume Clock.
        """
        if not hasattr(self, 'bars'): return
        
        # Load IBIT
        ibit_df = self.load_ticker_data(ibit_pattern)
        if ibit_df is None:
            print("Skipping Crypto Features (IBIT not found).")
            return
            
        print("Calculating Crypto Features (IBIT/EURUSD)...")
        
        # 1. Resample to 1-Minute Grid
        # EURUSD (From Bars - approximate but good enough if bars are granular)
        # We use time_end as the timestamp
        eur_1m = self.bars.set_index('time_end')[['close']].resample('1min').last().ffill()
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
        self.bars = self.bars.sort_values('time_end')
        
        # We only want the new columns
        cols_to_add = ['timestamp', 'IBIT_Lag2_Return', 'Corr_Regime_60m', 'Vol_Ratio_30m']
        
        merged = pd.merge_asof(
            self.bars,
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
            
        self.bars = merged

    def add_monster_features(self, windows=[50, 100]):
        if not hasattr(self, 'bars'): return
        print("Calculating MONSTER Features (YZ Vol, Kyle's Lambda, Force, FDI)...")
        df = self.bars
        
        # Yang-Zhang Volatility (Best Open-Close Estimator)
        for w in windows:
            df[f'yang_zhang_vol_{w}'] = phys.calc_yang_zhang_volatility(df, window=w)
            
            # Kyle's Lambda (Liquidity Cost)
            df[f'kyle_lambda_{w}'] = phys.calc_kyle_lambda(df, window=w)
            
            # Market Force (Physics)
            # Force is instantaneous, but we smooth it
            df[f'market_force_{w}'] = phys.calc_market_force(df, window=w)
            
            # Fractal Dimension Index (Roughness/Complexity)
            df[f'fdi_{w}'] = phys.calc_fractal_dimension(df['close'], window=w)
            
        self.bars = df

    def add_delta_features(self, lookback=10):
        if not hasattr(self, 'bars'): return
        print(f"Calculating Delta Features (Lookback={lookback})...")
        df = self.bars
        exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 
                   'cum_vol', 'vol_proxy', 'bar_id', 'log_ret',
                   'avg_bid_price', 'avg_ask_price', 'avg_bid_size', 'avg_ask_size',
                   'ticket_imbalance', 'residual_bund']
        # Also exclude existing delta columns to avoid delta_delta...
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('delta_')]
        
        # To avoid fragmentation warnings, collect new columns in a list and concat at once
        new_cols = {}
        for col in feature_cols:
            new_cols[f'delta_{col}_{lookback}'] = df[col].diff(lookback)
        
        # Concat all new columns at once
        new_df = pd.DataFrame(new_cols, index=df.index)
        self.bars = pd.concat([df, new_df], axis=1)
        
        # Defragment
        self.bars = self.bars.copy()

    def filter_survivors(self, config_path="data/survivors.json"):
        """
        Retains only the Elite Survivors identified by the Purge (from JSON).
        """
        if not hasattr(self, 'bars'): return
        
        # Default: Keep everything
        cols_to_keep = self.bars.columns.tolist()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    survivors = json.load(f)
                
                # Always keep metadata columns
                meta = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume']
                cols_to_keep = meta + [c for c in survivors if c in self.bars.columns]
                
                print(f"Loaded {len(survivors)} features from {config_path}.")
            except Exception as e:
                print(f"Error loading survivor config: {e}. Keeping all.")
        else:
            print(f"Config {config_path} not found. Keeping all features.")

        self.bars = self.bars[cols_to_keep]
        print(f"Filtered to {len(cols_to_keep)} columns.")

if __name__ == "__main__":
    DATA_PATH = "/home/tony/bankroll/data/raw_ticks"
    engine = FeatureEngine(DATA_PATH)
    spy_df = engine.load_ticker_data("RAW_TICKS_SPY*.parquet")
    tnx_df = engine.load_ticker_data("RAW_TICKS_TNX*.parquet")
    
    if spy_df is not None:
        engine.create_volume_bars(spy_df, volume_threshold=50000)
        if tnx_df is not None:
            engine.add_correlator_residual(tnx_df, suffix="_tnx")
        
        engine.add_features_to_bars()
        
        # Crypto Features (IBIT)
        engine.add_crypto_features("CLEAN_IBIT.parquet")

        # GDELT Integration
        gdelt_df = engine.load_gdelt_data()
        if gdelt_df is not None:
            engine.add_gdelt_features(gdelt_df)
            
        engine.add_physics_features()
        engine.add_microstructure_features()
        engine.add_monster_features()
        engine.add_delta_features()
        
        # Filter using the JSON config we just generated
        engine.filter_survivors()
        
        print("\nFeatures Head (Filtered):")
        # Print whatever remains
        print(engine.bars.tail())
        
        