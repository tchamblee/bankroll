import os
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import config
from ..workers import _worker_generate_signals
from ..feature_computation import ensure_feature_context

class SignalGenerationMixin:
    def ensure_context(self, population):
        ensure_feature_context(population, self.temp_dir, self.existing_keys)

    def generate_signal_matrix(self, population: list, horizon=None) -> np.array:
        self.ensure_context(population)
        num_strats = len(population)
        if num_strats == 0:
            return np.zeros((self.data_len, 0), dtype=np.int8)

        # Parallelize Signal Generation
        n_jobs = min(6, multiprocessing.cpu_count())
        batch_size = max(1, num_strats // (16 if num_strats > 100 else 4))
        
        chunks = [population[i:i + batch_size] for i in range(0, num_strats, batch_size)]
        
        # Pass directory path, NOT dictionary or single file
        results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_worker_generate_signals)(chunk, self.temp_dir) for chunk in chunks
        )
        
        signal_matrix = np.hstack(results)
        
        # --- TIME FILTER & SAFE ENTRY ---
        def load_vec(name):
             path = os.path.join(self.temp_dir, f"{name}.npy")
             if os.path.exists(path): return np.load(path)
             return None

        time_hour = load_vec('time_hour')
        time_weekday = load_vec('time_weekday')
        
        if time_hour is not None:
            # 1. Basic Market Hours
            market_open = (time_hour >= config.TRADING_START_HOUR) & (time_hour < config.TRADING_END_HOUR)
            if time_weekday is not None:
                market_open = market_open & (time_weekday < 5)
            
            # 2. Safe Entry Logic (Don't enter if we can't finish before close)
            # Assumption: 2 minutes per bar for duration estimation
            
            if horizon is not None:
                # Homogenous Batch (Training)
                est_duration_hours = (horizon * 3.2) / 60.0
                cutoff_hour = config.TRADING_END_HOUR - est_duration_hours
                safe_mask = (time_hour < cutoff_hour) & market_open
                signal_matrix = signal_matrix * safe_mask[:, np.newaxis]
            
            else:
                # Heterogenous Batch (Mutex / Portfolio)
                # Apply per-strategy mask
                for i, strat in enumerate(population):
                    h = getattr(strat, 'horizon', 0)
                    if h > 0:
                        est_duration = (h * 3.2) / 60.0
                        cutoff = config.TRADING_END_HOUR - est_duration
                        strat_mask = (time_hour < cutoff) & market_open
                        signal_matrix[:, i] *= strat_mask
                    else:
                        # Default to just market hours if horizon unknown
                        signal_matrix[:, i] *= market_open
        
        return signal_matrix
