import pandas as pd
import numpy as np
import tempfile
import os
import shutil
import config
import warnings
from joblib import Parallel, delayed

# Suppress annoying joblib/loky cleanup warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")

from genome import Strategy
from .workers import _worker_generate_signals, _worker_simulate
from .feature_computation import precompute_base_features, ensure_feature_context
from .statistics import combinatorial_purged_cv

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
        # We use the project's configured temp dir to keep everything contained
        base_temp = config.DIRS.get('TEMP_DIR', tempfile.gettempdir())
        os.makedirs(base_temp, exist_ok=True)
        self.temp_dir = tempfile.mkdtemp(prefix="backtest_ctx_", dir=base_temp)
        
        # Force Joblib to use this directory for its own temp files (memmapping)
        # This ensures that when we delete self.temp_dir, we delete joblib's mess too.
        os.environ['JOBLIB_TEMP_FOLDER'] = self.temp_dir
        
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
        fixed_comm_pct = fixed_cost / config.STANDARD_LOT_SIZE
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
        
        # Load ATR Base (Essential for Dynamic Simulation)
        atr_path = os.path.join(self.temp_dir, 'atr_base.npy')
        if os.path.exists(atr_path):
            self.atr_vec = np.load(atr_path).astype(np.float64)
        else:
            # Fallback if computation failed (should not happen)
            self.atr_vec = np.zeros_like(self.close_vec)
        
        # Split Indices
        n = len(self.raw_data)
        self.train_idx = int(n * config.TRAIN_SPLIT_RATIO)
        self.val_idx = int(n * config.VAL_SPLIT_RATIO)

    def shutdown(self):
        """Explicitly cleans up resources."""
        # 0. Shut down parallel workers to release semaphores/locks
        try:
            from joblib.externals.loky import stop_reusable_executor
            stop_reusable_executor()
        except:
            pass

        # 1. Unset env var to avoid side effects
        if 'JOBLIB_TEMP_FOLDER' in os.environ and os.environ['JOBLIB_TEMP_FOLDER'] == self.temp_dir:
            del os.environ['JOBLIB_TEMP_FOLDER']

        # Cleanup temp dir
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
        # No-op for now to persist features
        pass

    def ensure_context(self, population: list[Strategy]):
        ensure_feature_context(population, self.temp_dir, self.existing_keys)

    def generate_signal_matrix(self, population: list[Strategy], horizon=None) -> np.array:
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

    def run_simulation_batch(self, signals_matrix, strategies, prices, times, time_limit=None, highs=None, lows=None, atr=None):
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

        # ATR Fallback
        if atr is None:
            if highs is not None and lows is not None:
                atr = np.maximum(highs - lows, prices * 0.0005) # Min 5 bps
            else:
                atr = prices * 0.001 # 10 bps fallback

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
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
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
                lows,
                atr # Pass ATR
            ) for chunk_s, chunk_p in zip(chunks_sig, chunks_params)
        )
        
        # Reassemble results
        all_net_returns = np.hstack([r[0] for r in results])
        all_trades_count = np.concatenate([r[1] for r in results])
            
        return all_net_returns, all_trades_count

    def evaluate_population(self, population: list[Strategy], set_type='train', return_series=False, prediction_mode=False, time_limit=None, min_trades=None):
        if not population: 
            return (pd.DataFrame(), np.array([])) if return_series else pd.DataFrame()
        
        # target_min_trades = min_trades if min_trades is not None else config.MIN_TRADES_FOR_METRICS
        if min_trades is not None:
            target_min_trades = min_trades
        else:
            h_ref = time_limit if time_limit else config.DEFAULT_TIME_LIMIT
            target_min_trades = max(10, int(config.MIN_TRADES_COEFFICIENT / h_ref + 5))
        
        # --- CHUNKING TO PREVENT OOM ---
        BATCH_SIZE = config.EVO_BATCH_SIZE
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
        atr = self.atr_vec[start:end]
        times = self.times_vec.iloc[start:end] if hasattr(self.times_vec, 'iloc') else self.times_vec[start:end]
        
        # Loop through chunks
        for i in range(0, n_strats, BATCH_SIZE):
            chunk_pop = population[i:i + BATCH_SIZE]
            
            # 1. Generate Signals for Chunk
            full_signal_matrix = self.generate_signal_matrix(chunk_pop, horizon=time_limit)
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
                limit = time_limit if time_limit else config.DEFAULT_TIME_LIMIT
                net_returns, trades_count = self.run_simulation_batch(signals, chunk_pop, prices, times, time_limit=limit, highs=highs, lows=lows, atr=atr)
            
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
                
                # Hard Cut-off for Low Trades (Kill "Sniper" strategies)
                if trades_count[j] < target_min_trades:
                     final_sortino = -10.0
                
                # Stability Penalty
                elif stability_ratio[j] > 0.6 and total_ret[j] > 0: 
                    final_sortino *= 0.6
                
                # --- COST COVERAGE PENALTY ---
                # Removed: Accurate cost model naturally penalizes low-profit trades.
                # avg_trade_ret = total_ret[j] / (trades_count[j] + 1e-9)
                # cost_threshold = (self.effective_cost_bps / 10000.0) * 1.1
                # if avg_trade_ret < cost_threshold:
                #      final_sortino -= 5.0
                
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
            if all_net_returns:
                combined_returns = np.hstack(all_net_returns)
            else:
                combined_returns = np.array([])
            return results_df, combined_returns
        else:
            return results_df

    def evaluate_walk_forward(self, population: list[Strategy], folds=config.WFV_FOLDS, time_limit=None, min_trades=None):
        if not population: return []
        
        # target_min_trades = min_trades if min_trades is not None else config.MIN_TRADES_FOR_METRICS
        if min_trades is not None:
            target_min_trades = min_trades
        else:
            h_ref = time_limit if time_limit else config.DEFAULT_TIME_LIMIT
            target_min_trades = max(10, int(config.MIN_TRADES_COEFFICIENT / h_ref + 5))

        full_signal_matrix = self.generate_signal_matrix(population, horizon=time_limit)
        n_bars = len(self.raw_data)
        dev_end_idx = int(n_bars * config.VAL_SPLIT_RATIO)
        
        window_size = int(dev_end_idx * 0.55)
        step_size = int(dev_end_idx * 0.11)
        
        fold_scores = np.zeros((len(population), folds))
        
        limit = time_limit if time_limit else config.DEFAULT_TIME_LIMIT
        
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
            atr = self.atr_vec[train_end:test_end]
            
            times = self.times_vec.iloc[train_end:test_end] if hasattr(self.times_vec, 'iloc') else self.times_vec[train_end:test_end]
            
            # Use Simulator
            net_returns, trades_count = self.run_simulation_batch(signals, population, prices, times, time_limit=limit, highs=highs, lows=lows, atr=atr)
            
            total_ret = np.sum(net_returns, axis=0)
            max_win = np.max(net_returns, axis=0)
            stability_ratio = max_win / (total_ret + 1e-9)
            
            avg = np.mean(net_returns, axis=0)
            downside_std = np.std(np.minimum(net_returns, 0), axis=0)
            downside_std = np.maximum(downside_std, 1e-6)
            
            sortino = (avg / downside_std) * np.sqrt(self.annualization_factor)
            
            # Hard Cut-off for Low Trades (Kill "Sniper" strategies)
            sortino = np.nan_to_num(sortino, nan=-10.0)
            
            low_trades_mask = trades_count < target_min_trades
            sortino[low_trades_mask] = -10.0
            
            unstable_mask = (stability_ratio > 0.6) & (total_ret > 0)
            sortino[unstable_mask] *= 0.6
            
            # --- COST COVERAGE PENALTY ---
            # Removed: Accurate cost model naturally penalizes low-profit trades.
            # avg_trade_ret = total_ret / (trades_count + 1e-9)
            # cost_threshold = (self.effective_cost_bps / 10000.0) * 1.1
            
            # low_profit_mask = avg_trade_ret < cost_threshold
            # sortino[low_profit_mask] -= 5.0
            
            # --- OVERFITTING CAP ---
            # Removed: With accurate costs, high Sortino scores are likely legitimate or naturally limited.
            # overfit_mask = sortino > 20.0
            # sortino[overfit_mask] = -10.0
            
            # Cap Sortino to prevent infinite skew
            # sortino = np.minimum(sortino, 15.0)
            
            fold_scores[:, f] = sortino
            
        avg_sortino = np.mean(fold_scores, axis=1)
        min_sortino = np.min(fold_scores, axis=1)
        fold_std = np.std(fold_scores, axis=1)
        
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

    def evaluate_combinatorial_purged_cv(self, population: list[Strategy], n_folds=6, n_test_folds=2, time_limit=None):
        """
        Runs Combinatorial Purged Cross-Validation (CPCV).
        Generates multiple 'alternative history' paths by splicing test blocks.
        Returns detailed statistics for each strategy across all paths.
        """
        if not population: return pd.DataFrame()

        # 1. Setup Data & Signals
        # We use the full available dataset (Train + Val + potentially Test if verifying final)
        # For rigorous testing, we usually use the entire dataset to see stability across ALL regimes.
        n_bars = len(self.raw_data)
        
        # Pre-compute all signals once
        full_signal_matrix = self.generate_signal_matrix(population, horizon=time_limit)
        
        # Shift for execution (Next-Open)
        full_signal_matrix = np.vstack([np.zeros((1, full_signal_matrix.shape[1]), dtype=full_signal_matrix.dtype), full_signal_matrix[:-1]])

        # 2. Generate Splits
        splits = combinatorial_purged_cv(n_bars, n_folds=n_folds, n_test_folds=n_test_folds)
        print(f"  CPCV: Evaluating {len(splits)} combinatorial paths...")
        
        path_scores = np.zeros((len(population), len(splits)))
        
        limit = time_limit if time_limit else config.DEFAULT_TIME_LIMIT

        for i, (train_idx, test_idx) in enumerate(splits):
            # We evaluate on the TEST portion of the split
            # CPCV uses the 'Test' blocks as the OOS performance proxy for that specific path combination
            
            # Slice Data
            # Note: test_idx might not be contiguous, but simulation supports arrays if we handle it carefully.
            # However, run_simulation_batch expects contiguous arrays usually for speed/ATR calculation if passed.
            # But since we pass full ATR vectors sliced by index, it works fine mathematically 
            # as long as we treat them as a sequence of trades.
            # The simulator iterates 0..N, so we construct a 'compressed' time series representing the path.
            
            signals = full_signal_matrix[test_idx]
            prices = self.open_vec[test_idx]
            highs = self.high_vec[test_idx]
            lows = self.low_vec[test_idx]
            atr = self.atr_vec[test_idx]
            
            # Handle timestamps (pandas vs numpy)
            if hasattr(self.times_vec, 'iloc'):
                times = self.times_vec.iloc[test_idx]
            else:
                times = self.times_vec[test_idx]
            
            # Run Simulation
            net_returns, trades_count = self.run_simulation_batch(
                signals, population, prices, times, 
                time_limit=limit, highs=highs, lows=lows, atr=atr
            )
            
            # Metrics
            avg = np.mean(net_returns, axis=0)
            downside = np.minimum(net_returns, 0)
            downside_std = np.std(downside, axis=0) + 1e-9
            
            sortino = (avg / downside_std) * np.sqrt(self.annualization_factor)
            
            # Penalties
            # If a strategy fails in this specific market combo, it gets punished
            # Low trade count in this slice?
            # Adjust target trades based on slice size relative to full data
            slice_ratio = len(test_idx) / n_bars
            min_trades_slice = max(3, int(10 * slice_ratio)) # Very lax, just needs to trade
            
            low_trades_mask = trades_count < min_trades_slice
            sortino[low_trades_mask] = -1.0
            
            path_scores[:, i] = sortino
            
        # 3. Aggregate Results
        # We want strategies that perform well across MANY paths.
        # Metric: 5th Percentile Sortino (Worst 5% of market conditions)
        
        p5_sortino = np.percentile(path_scores, 5, axis=1)
        median_sortino = np.median(path_scores, axis=1)
        std_sortino = np.std(path_scores, axis=1)
        
        results = []
        for i, strat in enumerate(population):
            results.append({
                'id': strat.name,
                'cpcv_p5_sortino': p5_sortino[i],
                'cpcv_median': median_sortino[i],
                'cpcv_std': std_sortino[i],
                'cpcv_min': np.min(path_scores[i])
            })
            
        return pd.DataFrame(results)
