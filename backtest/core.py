import pandas as pd
import numpy as np
import tempfile
import os
import shutil
import config
import warnings
from .feature_computation import precompute_base_features

class BacktestEngineBase:
    """
    High-Performance Backtester Base Class.
    Handles data preparation, context management, and resource cleanup.
    """
    def __init__(self, data: pd.DataFrame, cost_bps=None, fixed_cost=config.MIN_COMMISSION, spread_bps=None, account_size=None, target_col='log_ret', annualization_factor=None):
        self.raw_data = data
        self.target_col = target_col
        self.annualization_factor = annualization_factor if annualization_factor else config.ANNUALIZATION_FACTOR
        
        # Temp dir for individual npy files
        # Optimization: Prefer /dev/shm (RAM Disk) on Linux to avoid Disk I/O bottlenecks
        if os.path.exists('/dev/shm') and os.access('/dev/shm', os.W_OK):
            base_temp = '/dev/shm'
        else:
            base_temp = config.DIRS.get('TEMP_DIR', tempfile.gettempdir())
            
        os.makedirs(base_temp, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="backtest_ctx_", dir=base_temp)
        
        self.existing_keys = set()
        
        # --- CONFIGURATION ---
        self.account_size = account_size if account_size else config.ACCOUNT_SIZE
        self.standard_lot = config.STANDARD_LOT_SIZE
        
        # --- COST MODELING ---
        self.cost_bps = cost_bps if cost_bps is not None else config.COST_BPS
        self.spread_bps = spread_bps if spread_bps is not None else config.SPREAD_BPS
        self.min_comm = fixed_cost # "fixed_cost" argument acts as min_comm
        
        self.effective_cost_bps = self.cost_bps
        
        if target_col == 'log_ret' and 'log_ret' not in self.raw_data.columns:
            self.raw_data['log_ret'] = np.log(self.raw_data['close'] / self.raw_data['close'].shift(1))
            
        if target_col in self.raw_data.columns:
            original_len = len(self.raw_data)
            valid_mask = ~self.raw_data[target_col].isna()
            self.raw_data = self.raw_data[valid_mask].reset_index(drop=True)
            self.data_len = len(self.raw_data)
        else:
            self.data_len = len(self.raw_data)

        # Prepare Data Matrices
        self.returns_vec = self.raw_data[self.target_col].values.reshape(-1, 1).astype(np.float32)
        self.close_vec = self.raw_data['close'].values.astype(np.float64) 
        self.open_vec = self.raw_data['open'].values.astype(np.float64)
        self.high_vec = self.raw_data['high'].values.astype(np.float64)
        self.low_vec = self.raw_data['low'].values.astype(np.float64)
        
        if 'time_start' in self.raw_data.columns:
            self.times_vec = self.raw_data['time_start']
        else:
            self.times_vec = pd.Series(self.raw_data.index)
        
        self.precompute_context()
        
        atr_path = os.path.join(self.temp_dir, 'atr_base.npy')
        if os.path.exists(atr_path):
            self.atr_vec = np.load(atr_path).astype(np.float64)
        else:
            self.atr_vec = np.zeros_like(self.close_vec)
        
        n = len(self.raw_data)
        self.train_idx = int(n * config.TRAIN_SPLIT_RATIO)
        self.val_idx = int(n * config.VAL_SPLIT_RATIO)

    def shutdown(self):
        """Explicitly cleans up resources."""
        try:
            from joblib.externals.loky import stop_reusable_executor
            stop_reusable_executor()
        except:
            pass

        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not fully clean temp dir {self.temp_dir}: {e}")

    def __del__(self):
        self.shutdown()

    def precompute_context(self):
        precompute_base_features(self.raw_data, self.temp_dir, self.existing_keys)
        self.base_keys = set(self.existing_keys)

    def reset_jit_context(self):
        pass
