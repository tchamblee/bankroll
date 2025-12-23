import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump, load
from genome import Strategy
from trade_simulator import TradeSimulator
import config
import os
import shutil
import tempfile

class LazyMMapContext:
    """
    Acts like a dictionary but lazy-loads numpy arrays from a directory using mmap.
    This prevents loading the entire dataset into RAM and allows efficient parallel access.
    """
    def __init__(self, directory):
        self.directory = directory
        self.cache = {}
        # Pre-scan directory to know available keys
        self.available_keys = {f.replace('.npy', '') for f in os.listdir(directory) if f.endswith('.npy')}

    def __getitem__(self, key):
        if key in self.cache:
            return self.cache[key]
            
        if key not in self.available_keys:
            # Check if it's special key like __len__
            if key == '__len__':
                # Try to infer length from any file
                if not self.available_keys: return 0
                first_key = next(iter(self.available_keys))
                arr = self[first_key]
                return len(arr)
            raise KeyError(f"Feature '{key}' not found in context directory.")

        path = os.path.join(self.directory, f"{key}.npy")
        # Load in mmap mode 'r' (Read-Only, Shared)
        arr = np.load(path, mmap_mode='r')
        self.cache[key] = arr
        return arr

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
            
    def __contains__(self, key):
        return key in self.cache or key in self.available_keys

    def __len__(self):
        return self['__len__']

    def keys(self):
        return self.available_keys

    def values(self):
        for k in self.available_keys:
            yield self[k]

def _worker_generate_signals(strategies, context_dir):
    """
    Worker function to generate signals for a chunk of strategies.
    Executed in parallel processes.
    """
    # Use Lazy Context
    context = LazyMMapContext(context_dir)
    
    n_rows = len(context)
    n_strats = len(strategies)
    chunk_matrix = np.zeros((n_rows, n_strats), dtype=np.int8)
    gene_cache = {} 
    
    for i, strat in enumerate(strategies):
        chunk_matrix[:, i] = strat.generate_signal(context, cache=gene_cache)
        
    return chunk_matrix

def _worker_simulate(signals_chunk, params_chunk, prices, times, spread_bps, effective_cost_bps, standard_lot, account_size, time_limit, hours, weekdays, highs, lows):
    """
    Worker function to simulate a chunk of strategy signals.
    """
    simulator = TradeSimulator(
        prices=prices,
        times=times,
        spread_bps=spread_bps,
        cost_bps=effective_cost_bps,
        lot_size=standard_lot,
        account_size=account_size
    )
    
    n_bars, n_strats = signals_chunk.shape
    net_returns = np.zeros((n_bars, n_strats), dtype=np.float32)
    trades_count = np.zeros(n_strats, dtype=int)
    
    for i in range(n_strats):
        sl = params_chunk[i].get('sl', 0.005)
        tp = params_chunk[i].get('tp', 0.0)
        
        net_rets, t_count = simulator.simulate_fast(
            signals_chunk[:, i], 
            stop_loss_pct=sl, 
            take_profit_pct=tp,
            time_limit_bars=time_limit,
            hours=hours,
            weekdays=weekdays,
            highs=highs,
            lows=lows
        )
        net_returns[:, i] = net_rets
        trades_count[i] = t_count
        
    return net_returns, trades_count

class BacktestEngine:
    """
    High-Performance Backtester using Centralized Trade Simulator.
    Ensures 100% consistency in trade logic (Barriers, Costs) across all system components.
    """
    def __init__(self, data: pd.DataFrame, cost_bps=None, fixed_cost=2.0, spread_bps=None, account_size=None, target_col='log_ret', annualization_factor=None):
        self.raw_data = data
        self.target_col = target_col
        self.annualization_factor = annualization_factor if annualization_factor else config.ANNUALIZATION_FACTOR
        
        # Temp dir for individual npy files
        self.temp_dir = tempfile.mkdtemp(prefix="backtest_ctx_")
        self.existing_keys = set()
        
        # --- CONFIGURATION ---
        self.account_size = account_size if account_size else config.ACCOUNT_SIZE
        self.standard_lot = config.STANDARD_LOT_SIZE
        
        # --- COST MODELING ---
        # Stored for passed to TradeSimulator
        self.cost_bps = cost_bps if cost_bps is not None else config.COST_BPS
        self.spread_bps = spread_bps if spread_bps is not None else config.SPREAD_BPS
        
        # Variable Commission: 0.2 bps (0.00002)
        var_comm_pct = self.cost_bps / 10000.0
        # Fixed Commission approximation
        fixed_comm_pct = fixed_cost / 100000.0 
        effective_comm_pct = max(var_comm_pct, fixed_comm_pct)
        
        self.effective_cost_bps = effective_comm_pct * 10000.0
        
        # Pre-calculate log returns if needed (mostly for reference now)
        if target_col == 'log_ret' and 'log_ret' not in self.raw_data.columns:
            self.raw_data['log_ret'] = np.log(self.raw_data['close'] / self.raw_data['close'].shift(1))
            
        # Clean Data
        if target_col in self.raw_data.columns:
            original_len = len(self.raw_data)
            valid_mask = ~self.raw_data[target_col].isna()
            self.raw_data = self.raw_data[valid_mask].reset_index(drop=True)
            self.data_len = len(self.raw_data)
        else:
            self.data_len = len(self.raw_data)

        # Prepare Data Matrices (Keep in RAM for Simulation, but also save to context if needed by genes)
        self.returns_vec = self.raw_data[self.target_col].values.reshape(-1, 1).astype(np.float32)
        self.close_vec = self.raw_data['close'].values.astype(np.float64) 
        self.open_vec = self.raw_data['open'].values.astype(np.float64) # Added for Next-Open Execution
        self.high_vec = self.raw_data['high'].values.astype(np.float64)
        self.low_vec = self.raw_data['low'].values.astype(np.float64)
        
        if 'time_start' in self.raw_data.columns:
            self.times_vec = self.raw_data['time_start']
        else:
            self.times_vec = pd.Series(self.raw_data.index)
        
        # Precompute Derived Features (Saved to disk)
        self.precompute_context()
        
        # Split Indices
        n = len(self.raw_data)
        self.train_idx = int(n * 0.6)
        self.val_idx = int(n * 0.8)

    def __del__(self):
        # Cleanup temp dir
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def _save_feature(self, name, arr):
        path = os.path.join(self.temp_dir, f"{name}.npy")
        np.save(path, arr)
        self.existing_keys.add(name)

    def precompute_context(self):
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self._save_feature(col, self.raw_data[col].values.astype(np.float32))
        
        if 'time_start' in self.raw_data.columns:
            dt = self.raw_data['time_start'].dt
            self._save_feature('time_hour', dt.hour.values.astype(np.float32))
            self._save_feature('time_weekday', dt.dayofweek.values.astype(np.float32))
            
        close = self.raw_data['close'].values
        up_mask = (close > np.roll(close, 1)); up_mask[0] = False
        
        def get_streak(mask):
            streaks = np.zeros(len(mask), dtype=np.float32)
            current = 0
            for i in range(len(mask)):
                current = (current + 1) if mask[i] else 0
                streaks[i] = current
            return streaks

        self._save_feature('consecutive_up', get_streak(up_mask))
        self._save_feature('consecutive_down', get_streak(~up_mask & (close < np.roll(close, 1))))
        
        self.base_keys = set(self.existing_keys)

    def reset_jit_context(self):
        # No-op for now to persist features
        pass

    def ensure_context(self, population: list[Strategy]):
        needed = set()
        for strat in population:
            all_genes = strat.long_genes + strat.short_genes
            for gene in all_genes:
                if gene.type == 'delta': needed.add(('delta', gene.feature, gene.lookback))
                elif gene.type == 'zscore': needed.add(('zscore', gene.feature, gene.window))
                elif gene.type == 'correlation': needed.add(('correlation', gene.feature_left, gene.feature_right, gene.window))
                elif gene.type == 'flux': needed.add(('flux', gene.feature, gene.lag))
                elif gene.type == 'divergence': 
                    needed.add(('slope', gene.feature_a, gene.window))
                    needed.add(('slope', gene.feature_b, gene.window))
                elif gene.type == 'efficiency': needed.add(('efficiency', gene.feature, gene.window))
        
        # Only calculate what's missing
        
        # Helper to load feature from disk for calculation
        # This uses simple load (RAM) because we need it for computation
        def get_data(key):
            if key not in self.existing_keys: return None
            return np.load(os.path.join(self.temp_dir, f"{key}.npy"))

        for item in needed:
            type_ = item[0]
            
            if type_ == 'delta':
                feature, param = item[1], item[2]
                key = f"delta_{feature}_{param}"
                if key in self.existing_keys: continue
                
                arr = get_data(feature)
                if arr is not None:
                    w = param
                    diff = np.zeros_like(arr)
                    if w < len(arr): diff[w:] = arr[w:] - arr[:-w]
                    self._save_feature(key, diff)

            elif type_ == 'flux':
                feature, lag = item[1], item[2]
                key = f"flux_{feature}_{lag}"
                if key in self.existing_keys: continue
                
                arr = get_data(feature)
                if arr is not None:
                    # Flux = Delta(Delta(X, lag), lag)
                    d1 = np.zeros_like(arr)
                    if len(arr) > lag: d1[lag:] = arr[lag:] - arr[:-lag]
                    
                    flux = np.zeros_like(arr)
                    if len(arr) > lag: flux[lag:] = d1[lag:] - d1[:-lag]
                    self._save_feature(key, flux)

            elif type_ == 'slope':
                feature, w = item[1], item[2]
                key = f"slope_{feature}_{w}"
                if key in self.existing_keys: continue
                
                arr = get_data(feature)
                if arr is not None:
                    y = arr
                    n = w
                    if len(y) > w:
                        sum_x = (n * (n - 1)) / 2
                        sum_x2 = (n * (n - 1) * (2 * n - 1)) / 6
                        divisor = n * sum_x2 - sum_x ** 2
                        
                        kernel_xy = np.arange(n)[::-1]
                        sum_xy = np.convolve(y, kernel_xy, mode='full')[:len(y)]
                        sum_y = np.convolve(y, np.ones(n), mode='full')[:len(y)]
                        
                        slope = (n * sum_xy - sum_x * sum_y) / (divisor + 1e-9)
                        slope[:n] = 0.0
                        self._save_feature(key, slope.astype(np.float32))

            elif type_ == 'efficiency':
                feature, w = item[1], item[2]
                key = f"eff_{feature}_{w}"
                if key in self.existing_keys: continue
                
                arr = get_data(feature)
                if arr is not None:
                    # ER = Change / Path
                    change = np.zeros_like(arr)
                    if len(arr) > w: change[w:] = np.abs(arr[w:] - arr[:-w])
                    
                    diff1 = np.zeros_like(arr)
                    diff1[1:] = np.abs(arr[1:] - arr[:-1])
                    
                    kernel = np.ones(w)
                    path = np.convolve(diff1, kernel, mode='full')[:len(arr)]
                    
                    er = np.divide(change, path, out=np.zeros_like(change), where=path!=0)
                    er[:w] = 0.0
                    self._save_feature(key, er.astype(np.float32))

            elif type_ == 'zscore':
                feature, param = item[1], item[2]
                key = f"zscore_{feature}_{param}"
                if key in self.existing_keys: continue

                # Vectorized Numpy Rolling Z-Score
                arr = get_data(feature)
                if arr is not None:
                    w = param
                    n_len = len(arr)
                    if n_len >= w:
                        kernel = np.ones(w)
                        
                        # Rolling Sums
                        sum_x = np.convolve(arr, kernel, 'full')[:n_len]
                        sum_x2 = np.convolve(arr * arr, kernel, 'full')[:n_len]
                        
                        # Rolling Mean and Variance
                        # Mean = Sum / w
                        # Var = E[x^2] - (E[x])^2
                        
                        mean = sum_x / w
                        # To avoid precision issues with Mean(x^2) - Mean(x)^2, use robust variance if needed
                        # But standard fast method:
                        mean_x2 = sum_x2 / w
                        var = mean_x2 - mean**2
                        var = np.maximum(var, 0) # Clip negative 0
                        
                        std = np.sqrt(var)
                        
                        z = np.divide(arr - mean, std, out=np.zeros_like(arr), where=std!=0)
                        
                        # Fix startup (first w-1 are invalid due to growing window in 'full' convolution)
                        z[:w-1] = 0.0
                        
                        self._save_feature(key, z.astype(np.float32))
            
            elif type_ == 'correlation':
                f1, f2, w = item[1], item[2], item[3]
                f1, f2 = sorted([f1, f2])
                key = f"corr_{f1}_{f2}_{w}"
                if key in self.existing_keys: continue
                
                a = get_data(f1)
                b = get_data(f2)
                
                if a is not None and b is not None:
                    n = w
                    kernel = np.ones(n)
                    
                    aa = a * a
                    bb = b * b
                    ab = a * b
                    
                    sum_a = np.convolve(a, kernel, 'full')[:len(a)]
                    sum_b = np.convolve(b, kernel, 'full')[:len(b)]
                    sum_ab = np.convolve(ab, kernel, 'full')[:len(ab)]
                    sum_aa = np.convolve(aa, kernel, 'full')[:len(aa)]
                    sum_bb = np.convolve(bb, kernel, 'full')[:len(bb)]
                    
                    var_a_term = np.maximum(n * sum_aa - sum_a**2, 0)
                    var_b_term = np.maximum(n * sum_bb - sum_b**2, 0)
                    
                    numerator = n * sum_ab - sum_a * sum_b
                    denominator = np.sqrt(var_a_term * var_b_term)
                    
                    corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
                    corr[:n-1] = 0.0
                    corr = np.clip(corr, -1.0, 1.0)
                    
                    self._save_feature(key, corr.astype(np.float32))

    def generate_signal_matrix(self, population: list[Strategy]) -> np.array:
        self.ensure_context(population)
        num_strats = len(population)
        if num_strats == 0:
            return np.zeros((self.data_len, 0), dtype=np.int8)

        # Parallelize Signal Generation
        import multiprocessing
        n_jobs = min(6, multiprocessing.cpu_count())
        batch_size = max(1, num_strats // (16 if num_strats > 100 else 4))
        
        chunks = [population[i:i + batch_size] for i in range(0, num_strats, batch_size)]
        
        # Pass directory path, NOT dictionary or single file
        results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_worker_generate_signals)(chunk, self.temp_dir) for chunk in chunks
        )
        
        signal_matrix = np.hstack(results)
        
        # --- TIME FILTER ---
        # Need to load these from disk for mask
        def load_vec(name):
             path = os.path.join(self.temp_dir, f"{name}.npy")
             if os.path.exists(path): return np.load(path)
             return None

        time_hour = load_vec('time_hour')
        if time_hour is not None:
            market_open = (time_hour >= config.TRADING_START_HOUR) & (time_hour < config.TRADING_END_HOUR)
            time_weekday = load_vec('time_weekday')
            if time_weekday is not None:
                is_weekday = time_weekday < 5
                market_open = market_open & is_weekday
                
            signal_matrix = signal_matrix * market_open[:, np.newaxis]
        
        return signal_matrix

    def run_simulation_batch(self, signals_matrix, strategies, prices, times, time_limit=None, highs=None, lows=None):
        """
        Runs TradeSimulator for each strategy in the batch.
        Returns: net_returns matrix (Bars x Strats), trades_count array
        """
        n_bars, n_strats = signals_matrix.shape
        if n_strats == 0:
            return np.zeros((n_bars, 0)), np.zeros(0)

        # 1. Extract Time Features for Slice
        # Ensure 'times' aligns with 'signals_matrix' rows
        if hasattr(times, 'dt'):
            hours = times.dt.hour.values.astype(np.int8)
            weekdays = times.dt.dayofweek.values.astype(np.int8)
        else:
            # Check if it's already numpy array of datetime or pandas Timestamps
            dt_idx = pd.to_datetime(times)
            hours = dt_idx.hour.values.astype(np.int8)
            weekdays = dt_idx.dayofweek.values.astype(np.int8)

        # 2. Lookahead Entry Filter
        # Prevent entry if we are too close to Market Close
        if time_limit:
            # Estimate: 2 mins per bar (Conservative)
            est_duration_hours = (time_limit * 2.0) / 60.0
            cutoff_hour = config.TRADING_END_HOUR - est_duration_hours
            
            # Entry Mask: (Hour < Cutoff) AND (Weekday < 5)
            # This masks out new entries that would likely be forced closed prematurely.
            safe_entry_mask = (hours < cutoff_hour) & (weekdays < 5)
            
            # Apply to signals
            signals_matrix = signals_matrix * safe_entry_mask[:, np.newaxis]
            
        # Parallelize Simulation
        import multiprocessing
        n_jobs = min(6, multiprocessing.cpu_count())
        batch_size = max(1, n_strats // (16 if n_strats > 100 else 4))
        
        params_list = [{'sl': getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS), 'tp': getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT)} for s in strategies]
        
        # Split signals matrix into chunks along columns (strategies)
        # signals_matrix[:, i:i+batch_size]
        chunks_sig = [signals_matrix[:, i:i + batch_size] for i in range(0, n_strats, batch_size)]
        chunks_params = [params_list[i:i + batch_size] for i in range(0, n_strats, batch_size)]
        
        # Run in parallel using threads for Numba JIT (which releases GIL)
        # max_nbytes=None prevents joblib from memmapping arguments to temp files (Leaking folders fix)
        results = Parallel(n_jobs=n_jobs, max_nbytes=None)(
            delayed(_worker_simulate)(
                chunk_s,
                chunk_p,
                prices, 
                times, 
                self.spread_bps, 
                self.effective_cost_bps, 
                self.standard_lot, 
                self.account_size,
                time_limit,
                hours, # Pass slice-aligned hours
                weekdays, # Pass slice-aligned weekdays
                highs,
                lows
            ) for chunk_s, chunk_p in zip(chunks_sig, chunks_params)
        )
        
        # Reassemble results
        all_net_returns = np.hstack([r[0] for r in results])
        all_trades_count = np.concatenate([r[1] for r in results])
            
        return all_net_returns, all_trades_count

    def evaluate_population(self, population: list[Strategy], set_type='train', return_series=False, prediction_mode=False, time_limit=None):
        if not population: 
            return (pd.DataFrame(), np.array([])) if return_series else pd.DataFrame()
        
        # --- CHUNKING TO PREVENT OOM ---
        # Process strategies in batches of 2000
        BATCH_SIZE = 2000
        n_strats = len(population)
        
        all_results = []
        all_net_returns = []
        
        # Determine index ranges once
        if set_type == 'train':
            start, end = 0, self.train_idx
        elif set_type == 'validation':
            start, end = self.train_idx, self.val_idx
        elif set_type == 'test':
            start, end = self.val_idx, len(self.raw_data)
        else: raise ValueError("set_type must be 'train', 'validation', or 'test'")
        
        prices = self.open_vec[start:end]
        highs = self.high_vec[start:end]
        lows = self.low_vec[start:end]
        times = self.times_vec.iloc[start:end] if hasattr(self.times_vec, 'iloc') else self.times_vec[start:end]
        
        # Loop through chunks
        for i in range(0, n_strats, BATCH_SIZE):
            chunk_pop = population[i:i + BATCH_SIZE]
            
            # 1. Generate Signals for Chunk
            full_signal_matrix = self.generate_signal_matrix(chunk_pop)
            signals = full_signal_matrix[start:end]
            
            # --- FIX: LOOKAHEAD BIAS (Next Open Execution) ---
            # Signals generated at Close[t] must be executed at Open[t+1].
            # We shift signals forward by 1.
            signals = np.vstack([np.zeros((1, signals.shape[1]), dtype=signals.dtype), signals[:-1]])
            
            # 2. Simulate Chunk
            if prediction_mode:
                returns = self.returns_vec[start:end]
                strat_returns = signals * returns
                net_returns = strat_returns
                trades_count = np.sum(np.abs(signals), axis=0) # Approx
            else:
                # Full Simulation Mode
                limit = time_limit if time_limit else 120
                net_returns, trades_count = self.run_simulation_batch(signals, chunk_pop, prices, times, time_limit=limit, highs=highs, lows=lows)
            
            # 3. Calculate Metrics for Chunk
            total_ret = np.sum(net_returns, axis=0)
            stdev = np.std(net_returns, axis=0) + 1e-9
            avg_ret = np.mean(net_returns, axis=0)
            sharpe = (avg_ret / stdev) * np.sqrt(self.annualization_factor)
            
            downside = np.minimum(net_returns, 0)
            downside_std = np.std(downside, axis=0) + 1e-9
            sortino = (avg_ret / downside_std) * np.sqrt(self.annualization_factor)
            
            max_win = np.max(net_returns, axis=0)
            stability_ratio = max_win / (total_ret + 1e-9)
            
            chunk_results = []
            for j, strat in enumerate(chunk_pop):
                final_sortino = sortino[j]
                
                # Soft Penalty for Low Trades (Create Gradient)
                # Increased to 25 to match Prop Desk Mandate (Moderate Stat Sig)
                if trades_count[j] < 25:
                    penalty = (25 - trades_count[j]) * 0.1
                    final_sortino -= penalty
                
                if trades_count[j] == 0:
                    final_sortino = -5.0
                    
                # Stability Penalty
                # Aligned with WFV: Relaxed to 0.6
                if stability_ratio[j] > 0.6 and total_ret[j] > 0: final_sortino *= 0.6
                
                # --- COST COVERAGE PENALTY (Aligned with WFV) ---
                avg_trade_ret = total_ret[j] / (trades_count[j] + 1e-9)
                cost_threshold = (self.effective_cost_bps / 10000.0) * 1.1
                if avg_trade_ret < cost_threshold:
                     final_sortino -= 5.0
                
                # Volume Bonus
                if trades_count[j] > 25:
                    final_sortino *= np.log10(max(trades_count[j], 1))
                
                strat.fitness = final_sortino
                chunk_results.append({
                    'id': strat.name,
                    'sharpe': sharpe[j],
                    'sortino': final_sortino,
                    'total_return': total_ret[j],
                    'trades': trades_count[j],
                    'stability': stability_ratio[j]
                })
            
            all_results.extend(chunk_results)
            
            if return_series:
                all_net_returns.append(net_returns)
                
            # Explicit Cleanup
            del full_signal_matrix, signals, net_returns
            
        results_df = pd.DataFrame(all_results)
        
        if return_series:
            # Concatenate along columns (axis 1)
            if all_net_returns:
                combined_returns = np.hstack(all_net_returns)
            else:
                combined_returns = np.array([])
            return results_df, combined_returns
        else:
            return results_df

    def evaluate_walk_forward(self, population: list[Strategy], folds=4, time_limit=None):
        if not population: return []

        full_signal_matrix = self.generate_signal_matrix(population)
        n_bars = len(self.raw_data)
        dev_end_idx = int(n_bars * 0.8)
        
        window_size = int(dev_end_idx * 0.55)
        step_size = int(dev_end_idx * 0.11)
        
        fold_scores = np.zeros((len(population), folds))
        
        limit = time_limit if time_limit else 120
        
        for f in range(folds):
            start = f * step_size
            train_end = start + window_size
            test_end = train_end + step_size
            
            if test_end > dev_end_idx:
                test_end = dev_end_idx
                train_end = test_end - step_size 
            
            signals = full_signal_matrix[train_end:test_end]
            
            # --- FIX: LOOKAHEAD BIAS (Next Open Execution) ---
            signals = np.vstack([np.zeros((1, signals.shape[1]), dtype=signals.dtype), signals[:-1]])
            prices = self.open_vec[train_end:test_end]
            highs = self.high_vec[train_end:test_end]
            lows = self.low_vec[train_end:test_end]
            
            times = self.times_vec.iloc[train_end:test_end] if hasattr(self.times_vec, 'iloc') else self.times_vec[train_end:test_end]
            
            # Use Simulator
            net_returns, trades_count = self.run_simulation_batch(signals, population, prices, times, time_limit=limit, highs=highs, lows=lows)
            
            total_ret = np.sum(net_returns, axis=0)
            max_win = np.max(net_returns, axis=0)
            stability_ratio = max_win / (total_ret + 1e-9)
            
            avg = np.mean(net_returns, axis=0)
            # Robust Downside Dev: Floor at 1e-6 to prevent explosion on "zero loss" strategies
            downside_std = np.std(np.minimum(net_returns, 0), axis=0)
            downside_std = np.maximum(downside_std, 1e-6)
            
            sortino = (avg / downside_std) * np.sqrt(self.annualization_factor)
            
            # Soft Penalty for WFV (Min Trades)
            # Increased to 10 to force statistical significance (Prop Desk Mandate)
            sortino = np.nan_to_num(sortino, nan=-2.0)
            penalty = np.maximum(0, (30 - trades_count) * 0.2)
            sortino -= penalty
            
            # Stability Penalty: If one trade is > 60% of total return, penalize moderately
            # Relaxed from 0.5/0.5 to 0.6/0.6 to allow for big trend days
            unstable_mask = (stability_ratio > 0.6) & (total_ret > 0)
            sortino[unstable_mask] *= 0.6
            
            # --- COST COVERAGE PENALTY ---
            # Force strategies to target moves significantly larger than the spread/commissions
            avg_trade_ret = total_ret / (trades_count + 1e-9)
            # Threshold: 1.1x the cost (Relaxed from 1.5x to increase frequency)
            cost_threshold = (self.effective_cost_bps / 10000.0) * 1.1
            
            # Vectorized penalty application
            low_profit_mask = avg_trade_ret < cost_threshold
            sortino[low_profit_mask] -= 5.0
            
            # --- VOLUME BONUS ---
            # Reward strategies that trade more frequently to improve statistical significance
            # Safe log10 calculation
            safe_trades = np.maximum(trades_count, 1)
            volume_bonus = np.where(trades_count > 30, np.log10(safe_trades), 1.0)
            sortino *= volume_bonus
            
            # Cap Sortino to prevent infinite skew (e.g. 24.0 -> 10.0)
            sortino = np.minimum(sortino, 15.0) # Increased cap for bonus
            
            fold_scores[:, f] = sortino
            
        avg_sortino = np.mean(fold_scores, axis=1)
        min_sortino = np.min(fold_scores, axis=1)
        fold_std = np.std(fold_scores, axis=1)
        
        # Moderate Robustness
        # Reverted penalty from 1.0 back to 0.5
        robust_score = avg_sortino - (fold_std * 0.5)
        
        results = []
        for i, strat in enumerate(population):
            strat.fitness = robust_score[i]
            results.append({
                'id': strat.name,
                'sortino': robust_score[i],
                'avg_sortino': avg_sortino[i],
                'min_sortino': min_sortino[i],
                'fold_std': fold_std[i]
            })
            
        return pd.DataFrame(results)