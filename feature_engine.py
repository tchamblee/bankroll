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
        files = glob.glob(os.path.join(self.data_dir, pattern))
        if not files:
            print(f"No files found for {pattern}")
            return None
        
        print(f"Loading {len(files)} files for {pattern}...")
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
        
        # Outlier Removal: Filter out bad ticks (e.g. > 2% move in one tick)
        # This fixes the 'FATAL Outliers' identified in integrity checks
        if len(df) > 1:
            pct_change = df['mid_price'].pct_change().abs()
            # 0.02 (2%) is huge for a single tick in FX
            outliers = pct_change > 0.02
            outlier_count = outliers.sum()
            if outlier_count > 0:
                print(f"  ⚠️  Removed {outlier_count} price outliers from {pattern}")
                df = df[~outliers]
        
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
        bars = df.groupby('bar_id').agg({
            'ts_event': ['first', 'last'],
            'mid_price': ['first', 'max', 'min', 'last'],
            'vol_proxy': 'sum',
            'aggressor_vol': 'sum'
        })
        
        # Flatten Columns
        bars.columns = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol']
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

    def add_features_to_bars(self, windows=[50, 100, 200, 400]):
        if not hasattr(self, 'bars'): return
        df = self.bars
        
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        for w in windows:
            df[f'velocity_{w}'] = df['close'].diff(w)
            df[f'volatility_{w}'] = df['log_ret'].rolling(w).std()
            
            net_change = df['close'].diff(w).abs()
            path = df['close'].diff().abs().rolling(w).sum()
            df[f'efficiency_{w}'] = np.where(path==0, 0, net_change/path)
            
            df[f'autocorr_{w}'] = df['log_ret'].rolling(w).corr(df['log_ret'].shift(1))
            
            vol_norm = df[f'volatility_{w}'] / df[f'volatility_{w}'].rolling(1000, min_periods=100).mean()
            df[f'trend_strength_{w}'] = df[f'efficiency_{w}'] * vol_norm

        self.bars = df

    def add_physics_features(self):
        if not hasattr(self, 'bars'): return
        df = self.bars
        print("Calculating Physics Features...")
        df['frac_diff_04'] = phys.frac_diff_ffd(df['close'], d=0.4)
        df['frac_diff_02'] = phys.frac_diff_ffd(df['close'], d=0.2)
        df['hurst_100'] = phys.get_hurst_exponent(df['close'], window=100)
        df['hurst_200'] = phys.get_hurst_exponent(df['close'], window=200)
        self.bars = df
        
    def add_delta_features(self, lookback=10):
        if not hasattr(self, 'bars'): return
        print(f"Calculating Delta Features (Lookback={lookback})...")
        df = self.bars
        exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 'cum_vol', 'vol_proxy', 'bar_id', 'log_ret']
        # Also exclude existing delta columns to avoid delta_delta...
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('delta_')]
        
        for col in feature_cols:
            df[f'delta_{col}_{lookback}'] = df[col].diff(lookback)
        
        # Defragment
        self.bars = df.copy()

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
        engine.add_physics_features()
        engine.add_delta_features()
        
        # Filter using the JSON config we just generated
        engine.filter_survivors()
        
        print("\nFeatures Head (Filtered):")
        # Print whatever remains
        print(engine.bars.tail())
        
        