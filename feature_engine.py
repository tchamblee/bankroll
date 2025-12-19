import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta
import physics_features as phys

class FeatureEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df = None

    def load_data(self, pattern="*.parquet"):
        """Loads and sorts all parquet files matching the pattern."""
        files = glob.glob(os.path.join(self.data_dir, pattern))
        if not files:
            print(f"No files found in {self.data_dir}")
            return
        
        print(f"Loading {len(files)} files...")
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
        if not dfs:
            return

        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.sort_values("ts_event").reset_index(drop=True)
        
        # Calculate Mid Price if not present
        if 'mid_price' not in self.df.columns:
            # Check for Bid/Ask first
            if 'pricebid' in self.df.columns and 'priceask' in self.df.columns:
                # If mostly NaNs, try last_price
                if self.df['pricebid'].isna().mean() > 0.9 and 'last_price' in self.df.columns:
                     self.df['mid_price'] = self.df['last_price']
                else:
                    self.df['mid_price'] = (self.df['pricebid'] + self.df['priceask']) / 2
            elif 'last_price' in self.df.columns:
                self.df['mid_price'] = self.df['last_price']
            elif 'price' in self.df.columns:
                self.df['mid_price'] = self.df['price']
            else:
                raise ValueError("Could not determine mid_price from columns.")

        print(f"Loaded {len(self.df)} ticks.")

    def create_volume_bars(self, volume_threshold=1000):
        """
        Aggregates ticks into Volume Bars.
        Calculates Ticket Imbalance during aggregation.
        """
        if self.df is None: return None

        print(f"Generating Volume Bars (Threshold: {volume_threshold})...")
        
        # 1. Prepare Tick Data
        # Volume Proxy
        if 'volume' in self.df.columns and self.df['volume'].isna().mean() < 0.5:
            vol_series = self.df['volume'].fillna(0)
        else:
            if 'sizebid' in self.df.columns:
                 vol_series = (self.df['sizebid'].fillna(0) + self.df['sizeask'].fillna(0)) / 2
            else:
                 vol_series = pd.Series(1, index=self.df.index)

        self.df['vol_proxy'] = vol_series
        self.df['cum_vol'] = vol_series.cumsum()
        self.df['bar_id'] = (self.df['cum_vol'] // volume_threshold).astype(int)
        
        # 2. Aggressor Logic (Ticket Imbalance)
        # Direction: 1 (Up), -1 (Down), 0 (Same)
        delta = self.df['mid_price'].diff().fillna(0)
        direction = np.sign(delta)
        
        # Aggressor Volume = Direction * Volume
        self.df['aggressor_vol'] = direction * vol_series
        
        # 3. Aggregation
        bars = self.df.groupby('bar_id').agg({
            'ts_event': 'last',
            'mid_price': ['first', 'max', 'min', 'last'],
            'vol_proxy': 'sum',
            'aggressor_vol': 'sum'
        })
        
        bars.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol']
        bars.reset_index(drop=True, inplace=True)
        
        # Calculate Imbalance Ratio (-1 to 1)
        # Avoid div by zero
        bars['ticket_imbalance'] = np.where(bars['volume'] == 0, 0, bars['net_aggressor_vol'] / bars['volume'])
        
        self.bars = bars
        print(f"Generated {len(bars)} Volume Bars.")
        return bars

import pandas as pd
import numpy as np
import os
import glob
from datetime import timedelta
import physics_features as phys

class FeatureEngine:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.bars = None
        self.raw_ticks = None

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
        
        return df

    def create_volume_bars(self, primary_df, volume_threshold=1000):
        """
        Creates Volume Bars from the PRIMARY asset.
        """
        if primary_df is None: return None

        print(f"Generating Volume Bars (Threshold: {volume_threshold})...")
        
        df = primary_df.copy()
        
        # Volume Proxy Logic
        if 'volume' in df.columns and df['volume'].isna().mean() < 0.5:
            vol_series = df['volume'].fillna(0)
        else:
            if 'sizebid' in df.columns:
                 vol_series = (df['sizebid'].fillna(0) + df['sizeask'].fillna(0)) / 2
            else:
                 vol_series = pd.Series(1, index=df.index)

        df['vol_proxy'] = vol_series
        df['cum_vol'] = vol_series.cumsum()
        df['bar_id'] = (df['cum_vol'] // volume_threshold).astype(int)
        
        # Aggressor Logic
        delta = df['mid_price'].diff().fillna(0)
        direction = np.sign(delta)
        df['aggressor_vol'] = direction * vol_series
        
        bars = df.groupby('bar_id').agg({
            'ts_event': 'last',
            'mid_price': ['first', 'max', 'min', 'last'],
            'vol_proxy': 'sum',
            'aggressor_vol': 'sum'
        })
        
        bars.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol']
        bars.reset_index(drop=True, inplace=True)
        bars['ticket_imbalance'] = np.where(bars['volume'] == 0, 0, bars['net_aggressor_vol'] / bars['volume'])
        
        self.bars = bars
        print(f"Generated {len(bars)} Volume Bars.")
        return bars

    def merge_correlator(self, correlator_df, suffix="_corr"):
        """
        Merges a secondary ticker (correlator) onto the volume bars using merge_asof.
        """
        if self.bars is None or correlator_df is None: return

        print(f"Merging Correlator data ({suffix})...")
        
        # Ensure datetimes are sorted and valid
        self.bars = self.bars.sort_values('time')
        correlator_df = correlator_df.sort_values('ts_event')
        
        # Merge closest PRIOR tick from correlator to the bar close time
        merged = pd.merge_asof(
            self.bars, 
            correlator_df[['ts_event', 'mid_price']], 
            left_on='time', 
            right_on='ts_event', 
            direction='backward',
            tolerance=pd.Timedelta('1 hour') # Don't merge if data is stale
        )
        
        merged.rename(columns={'mid_price': f'close{suffix}'}, inplace=True)
        merged.drop(columns=['ts_event'], inplace=True) # Drop correlator timestamp
        
        self.bars = merged

    def add_features_to_bars(self, windows=[50, 100, 200, 400]):
        """
        Calculates standard features (Velocity, Volatility, etc.) on the BAR dataframe.
        """
        if not hasattr(self, 'bars'): 
            print("No bars generated. Call create_volume_bars() first.")
            return

        df = self.bars
        
        # Log Returns
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        for w in windows:
            # Velocity: Return over window
            df[f'velocity_{w}'] = df['close'].diff(w)
            
            # Volatility: Std Dev of returns
            df[f'volatility_{w}'] = df['log_ret'].rolling(w).std()
            
            # Efficiency: Net Change / Sum of High-Low Ranges
            net_change = df['close'].diff(w).abs()
            path = df['close'].diff().abs().rolling(w).sum()
            df[f'efficiency_{w}'] = np.where(path==0, 0, net_change/path)
            
            # Cyclicality: Autocorrelation
            df[f'autocorr_{w}'] = df['log_ret'].rolling(w).corr(df['log_ret'].shift(1))
            
            # Regime: Trend Strength
            vol_norm = df[f'volatility_{w}'] / df[f'volatility_{w}'].rolling(1000, min_periods=100).mean()
            df[f'trend_strength_{w}'] = df[f'efficiency_{w}'] * vol_norm

        self.bars = df

    def add_physics_features(self):
        """
        Calculates advanced physics features: FracDiff, Hurst, Efficiency.
        """
        if not hasattr(self, 'bars'): 
            print("No bars generated.")
            return

        df = self.bars
        
        # 1. Fractional Differentiation (Memory)
        print("Calculating Fractional Differentiation (d=0.4, Short Memory)...")
        df['frac_diff_04'] = phys.frac_diff_ffd(df['close'], d=0.4)
        
        print("Calculating Fractional Differentiation (d=0.2, Long Memory)...")
        df['frac_diff_02'] = phys.frac_diff_ffd(df['close'], d=0.2)

        # 2. Hurst Exponent (Regime)
        print("Calculating Hurst Exponent (Window=100)...")
        df['hurst_100'] = phys.get_hurst_exponent(df['close'], window=100)
        
        print("Calculating Hurst Exponent (Window=200)...")
        df['hurst_200'] = phys.get_hurst_exponent(df['close'], window=200)
        
        # 3. Efficiency Ratio (Entropy) - Redundant with add_features_to_bars but useful for standalone physics usage
        # Skipping to avoid duplication

        self.bars = df

    def add_divergence_features(self, suffix="_corr", windows=[50, 200]):
        """
        Calculates divergence between Primary and Correlator (Lead-Lag).
        Feature: Z-Score(Primary) - Z-Score(Correlator)
        """
        col = f'close{suffix}'
        if col not in self.bars.columns: return

        print(f"Calculating Divergence for {suffix}...")
        df = self.bars
        
        for w in windows:
            # Calculate Z-Score for both
            z_prim = (df['close'] - df['close'].rolling(w).mean()) / df['close'].rolling(w).std()
            z_corr = (df[col] - df[col].rolling(w).mean()) / df[col].rolling(w).std()
            
            # Clean naming: strip leading underscore from suffix if present for cleaner col name
            clean_suffix = suffix.lstrip('_')
            df[f'divergence_{clean_suffix}_{w}'] = z_prim - z_corr

        self.bars = df

    def add_delta_features(self, lookback=10):
        """
        Calculates the Rate of Change (Delta) for all existing features.
        This captures the 'Flow' or 'Acceleration' of the metric.
        """
        if not hasattr(self, 'bars'): return
        
        print(f"Calculating Delta Features (Lookback={lookback})...")
        df = self.bars
        
        # Identify feature columns (exclude time, OHLVC, and aux columns)
        exclude = ['time', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 'cum_vol', 'vol_proxy', 'bar_id', 'log_ret']
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith('delta_')]
        
        for col in feature_cols:
            # We calculate the delta (diff)
            # Normalizing by the value itself (pct_change) can be unstable for features oscillating around 0 (like divergence or imbalances)
            # So we stick to simple diff: X[t] - X[t-n]
            df[f'delta_{col}'] = df[col].diff(lookback)
            
        self.bars = df

if __name__ == "__main__":
    DATA_PATH = "/home/tony/bankroll/data/raw_ticks"
    
    engine = FeatureEngine(DATA_PATH)
    
    # 1. Load Primary (SPY)
    spy_df = engine.load_ticker_data("RAW_TICKS_SPY*.parquet")
    
    # 2. Load Correlator (TNX)
    tnx_df = engine.load_ticker_data("RAW_TICKS_TNX*.parquet")
    
    if spy_df is not None:
        engine.create_volume_bars(spy_df, volume_threshold=50000)
        
        if tnx_df is not None:
            engine.merge_correlator(tnx_df, suffix="_tnx")
            engine.add_divergence_features(suffix="_tnx")
        
        engine.add_features_to_bars()
        engine.add_physics_features()
        
        # Add Deltas
        engine.add_delta_features(lookback=10)
        
        print(f"Total Columns: {len(engine.bars.columns)}")
        print("\nFeatures Head (Delta examples):")
        cols = [c for c in engine.bars.columns if 'delta_' in c][:5]
        print(engine.bars.tail()[cols])

if __name__ == "__main__":
    # Example Usage
    DATA_PATH = "/home/tony/bankroll/data/raw_ticks"
    
    engine = FeatureEngine(DATA_PATH)
    engine.load_data("RAW_TICKS_SPY*.parquet") 
    
    if engine.df is not None:
        engine.create_volume_bars(volume_threshold=50000)
        
        engine.add_physics_features()
        
        print("\nPhysics Features Head:")
        cols = ['time', 'close', 'ticket_imbalance', 'frac_diff_04', 'hurst_100', 'efficiency_100']
        print(engine.bars.tail()[cols])
