import os
import config
from . import loader
from . import bars
from . import standard
from . import physics
from . import microstructure
from . import correlations
from . import gdelt
from . import delta
from . import utils
from . import macro_voltage
from . import intermarket
from . import ticks
from . import event_decay
from . import implied_vol


class FeatureEngine:
    def __init__(self, data_dir, use_survivors=False):
        self.data_dir = data_dir
        self.use_survivors = use_survivors
        self.bars = None
        # We don't store raw ticks in self anymore to save memory, 
        # but we need them temporarily for correlator processing.

    def load_ticker_data(self, pattern):
        """Loads and returns a sorted DataFrame for a specific ticker pattern."""
        return loader.load_ticker_data(self.data_dir, pattern)

    def load_gdelt_data(self, pattern="GDELT_GKG_*.parquet"):
        """
        Loads GDELT data. Tries V2 (Intraday) first, falls back to V1 (Daily).
        """
        # Try V2 (Intraday)
        v2_df = loader.load_gdelt_v2_data(self.data_dir)
        if v2_df is not None and not v2_df.empty:
            print("✅ Loaded GDELT V2 (Intraday) data.")
            return v2_df
            
        print("⚠️ GDELT V2 not found. Falling back to Legacy (Daily)...")
        return loader.load_gdelt_data(self.data_dir, pattern)

    def create_volume_bars(self, primary_df, volume_threshold=1000):
        """
        Creates Volume Bars from the PRIMARY asset.
        """
        self.bars = bars.create_volume_bars(primary_df, volume_threshold)
        return self.bars

    def add_correlator_residual(self, correlator_df, suffix="_corr", window=100):
        """
        Calculates the Residual (Alpha) of the Primary asset relative to the Correlator.
        """
        self.bars = correlations.add_correlator_residual(self.bars, correlator_df, suffix, window)

    def add_gdelt_features(self, gdelt_df):
        """
        Merges Daily GDELT features into Intraday Bars.
        """
        self.bars = gdelt.add_gdelt_features(self.bars, gdelt_df)

    def add_features_to_bars(self, windows=[50, 100, 200, 400]):
        self.bars = standard.add_features_to_bars(self.bars, windows)

    def add_physics_features(self):
        self.bars = physics.add_physics_features(self.bars)

    def add_microstructure_features(self, windows=[50, 100]):
        self.bars = microstructure.add_microstructure_features(self.bars, windows)

    def add_crypto_features(self, ibit_df):
        """
        Adds Crypto-Lead features using IBIT data.
        """
        self.bars = correlations.add_crypto_features(self.bars, ibit_df)

    def add_macro_voltage_features(self, us2y_df, schatz_df, tnx_df, bund_df, windows=[50, 100]):
        """
        Adds Transatlantic Voltage features (US2Y vs SCHATZ).
        """
        self.bars = macro_voltage.add_macro_voltage_features(self.bars, us2y_df, schatz_df, tnx_df, bund_df, windows)

    def add_advanced_physics_features(self, windows=[50, 100]):
        self.bars = physics.add_advanced_physics_features(self.bars, windows)

    def add_delta_features(self, lookback=10):
        self.bars = delta.add_delta_features(self.bars, lookback)

    def add_event_decay_features(self, high_windows=[100, 200, 400], shock_windows=[100]):
        self.bars = event_decay.add_event_decay_features(self.bars, high_windows, shock_windows)

    def add_implied_vol_features(self, vix_df, evz_df):
        self.bars = implied_vol.add_implied_vol_features(self.bars, vix_df, evz_df)

    def add_intermarket_features(self, correlator_dfs):
        self.bars = intermarket.add_intermarket_features(self.bars, correlator_dfs)

        # 12. Purge (Survival of the Fittest)
        # ----------------------------------------------------
        if self.use_survivors:
             self.filter_survivors()
        
        return self.bars

    def filter_survivors(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(config.DIRS['FEATURES_DIR'], "survivors.json")

        if not os.path.exists(config_path):
            return
            
        print(f"⚔️ Filtering Features based on {config_path}...")
