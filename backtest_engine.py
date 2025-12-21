import pandas as pd
import numpy as np
from joblib import Parallel, delayed, dump, load
from strategy_genome import Strategy
from trade_simulator import TradeSimulator
import config
import os
import shutil
import tempfile

def _worker_generate_signals(strategies, context_path):
    """
    Worker function to generate signals for a chunk of strategies.
    Executed in parallel processes.
    """
    # Load context from memmapped file (Shared Memory, Zero Copy)
    # mmap_mode='r' ensures we don't copy data to RAM, just page it.
    context = load(context_path, mmap_mode='r')
    
    n_rows = context.get('__len__', 0)
    # Fallback if __len__ missing but arrays present
    if n_rows == 0 and len(context) > 0:
         for val in context.values():
             if hasattr(val, 'shape'):
                 n_rows = val.shape[0]
                 break
                 
    n_strats = len(strategies)
    chunk_matrix = np.zeros((n_rows, n_strats), dtype=np.int8)
    gene_cache = {} 
    
    for i, strat in enumerate(strategies):
        chunk_matrix[:, i] = strat.generate_signal(context, cache=gene_cache)
        
    return chunk_matrix

def _worker_simulate(signals_chunk, prices, times, spread_bps, effective_cost_bps, standard_lot, account_size, time_limit):
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
        net_rets, t_count = simulator.simulate_fast(
            signals_chunk[:, i], 
            stop_loss_pct=0.005, 
            time_limit_bars=time_limit
        )
        net_returns[:, i] = net_rets
        trades_count[i] = t_count
        
    return net_returns, trades_count

class BacktestEngine:
    """
    High-Performance Backtester using Centralized Trade Simulator.
    Ensures 100% consistency in trade logic (Barriers, Costs) across all system components.
    """
    def __init__(self, data: pd.DataFrame, cost_bps=0.2, fixed_cost=2.0, spread_bps=1.0, account_size=None, target_col='log_ret', annualization_factor=1.0):
        self.raw_data = data
        self.target_col = target_col
        self.annualization_factor = annualization_factor
        self.context = {}
        
        # Temp dir for memmap
        self.temp_dir = tempfile.mkdtemp(prefix="backtest_memmap_")
        self.context_path = os.path.join(self.temp_dir, "context.joblib")
        
        # --- CONFIGURATION ---
        self.account_size = account_size if account_size else config.ACCOUNT_SIZE
        self.standard_lot = config.STANDARD_LOT_SIZE
        
        # --- COST MODELING ---
        # Stored for passed to TradeSimulator
        self.cost_bps = cost_bps
        # self.fixed_cost = fixed_cost # Simulator currently uses bps, might need update if fixed cost crucial
        # Using effective cost bps passed to simulator
        
        # Variable Commission: 0.2 bps (0.00002)
        var_comm_pct = cost_bps / 10000.0
        # Fixed Commission approximation
        fixed_comm_pct = fixed_cost / 100000.0 
        effective_comm_pct = max(var_comm_pct, fixed_comm_pct)
        
        self.effective_cost_bps = effective_comm_pct * 10000.0
        self.spread_bps = spread_bps
        
        # Pre-calculate log returns if needed (mostly for reference now)
        if target_col == 'log_ret' and 'log_ret' not in self.raw_data.columns:
            self.raw_data['log_ret'] = np.log(self.raw_data['close'] / self.raw_data['close'].shift(1))
            
        # Precompute Derived Features
        self.precompute_context()

        # Clean Data
        if target_col in self.raw_data.columns:
            original_len = len(self.raw_data)
            valid_mask = ~self.raw_data[target_col].isna()
            self.raw_data = self.raw_data[valid_mask].reset_index(drop=True)
            for key in self.context:
                if isinstance(self.context[key], np.ndarray) and len(self.context[key]) == original_len:
                    self.context[key] = self.context[key][valid_mask]
            self.context['__len__'] = len(self.raw_data)

        # Prepare Data Matrices
        self.returns_vec = self.raw_data[self.target_col].values.reshape(-1, 1).astype(np.float32)
        self.close_vec = self.raw_data['close'].values.astype(np.float64) # 1D for simulator
        
        if 'time_start' in self.raw_data.columns:
            self.times_vec = self.raw_data['time_start']
        else:
            self.times_vec = pd.Series(self.raw_data.index)
        
        # Split Indices
        n = len(self.raw_data)
        self.train_idx = int(n * 0.6)
        self.val_idx = int(n * 0.8)
        
        # Initial sync
        self._sync_context()

    def __del__(self):
        # Cleanup temp dir
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def precompute_context(self):
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.context[col] = self.raw_data[col].values.astype(np.float32)
        self.context['__len__'] = len(self.raw_data)
        
        if 'time_start' in self.raw_data.columns:
            dt = self.raw_data['time_start'].dt
            self.context['time_hour'] = dt.hour.values.astype(np.float32)
            self.context['time_weekday'] = dt.dayofweek.values.astype(np.float32)
            
        close = self.raw_data['close'].values
        up_mask = (close > np.roll(close, 1)); up_mask[0] = False
        
        def get_streak(mask):
            streaks = np.zeros(len(mask), dtype=np.float32)
            current = 0
            for i in range(len(mask)):
                current = (current + 1) if mask[i] else 0
                streaks[i] = current
            return streaks

        self.context['consecutive_up'] = get_streak(up_mask)
        self.context['consecutive_down'] = get_streak(~up_mask & (close < np.roll(close, 1)))
        
        self.base_keys = set(self.context.keys())

    def reset_jit_context(self):
        keys_to_remove = [k for k in self.context.keys() if k not in self.base_keys]
        for k in keys_to_remove:
            del self.context[k]
        self._sync_context() # Sync deletions

    def ensure_context(self, population: list[Strategy]):
        needed = set()
        for strat in population:
            all_genes = strat.long_genes + strat.short_genes
            for gene in all_genes:
                if gene.type == 'delta': needed.add(('delta', gene.feature, gene.lookback))
                elif gene.type == 'zscore': needed.add(('zscore', gene.feature, gene.window))
        
        modified = False
        for type_, feature, param in needed:
            key = f"{type_}_{feature}_{param}"
            if key in self.context: continue
            
            modified = True
            if type_ == 'delta':
                arr = self.context.get(feature)
                if arr is not None:
                    w = param
                    diff = np.zeros_like(arr)
                    if w < len(arr): diff[w:] = arr[w:] - arr[:-w]
                    self.context[key] = diff
            elif type_ == 'zscore':
                series = self.raw_data[feature]
                z = (series - series.rolling(param).mean()) / (series.rolling(param).std() + 1e-9)
                self.context[key] = z.fillna(0).values.astype(np.float32)
        
        if modified:
            self._sync_context()

    def _sync_context(self):
        """Dump context to disk for shared memory access by workers."""
        dump(self.context, self.context_path)

    def generate_signal_matrix(self, population: list[Strategy]) -> np.array:
        self.ensure_context(population)
        num_strats = len(population)
        if num_strats == 0:
            return np.zeros((self.context['__len__'], 0), dtype=np.int8)

        # Parallelize Signal Generation
        # Determine number of jobs and batch size
        n_jobs = -1 
        batch_size = max(1, num_strats // (16 if num_strats > 100 else 4))
        
        # Split population into chunks
        chunks = [population[i:i + batch_size] for i in range(0, num_strats, batch_size)]
        
        # Run in parallel using Process backend (default) but reading from Shared Memmap
        # Removed prefer="threads" to use processes for GIL bypass
        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker_generate_signals)(chunk, self.context_path) for chunk in chunks
        )
        
        # Concatenate results
        # Each result is (Rows, Chunk_Size)
        signal_matrix = np.hstack(results)
        
        # --- TIME FILTER (User Request) ---
        # Force flat (0) outside of liquid hours and on weekends
        if 'time_hour' in self.context:
            hours = self.context['time_hour']
            # London Open (8) to NY Close (22)
            market_open = (hours >= config.TRADING_START_HOUR) & (hours <= config.TRADING_END_HOUR)
            
            # Weekend Filter (0=Mon, 4=Fri, 5=Sat, 6=Sun)
            if 'time_weekday' in self.context:
                is_weekday = self.context['time_weekday'] < 5
                market_open = market_open & is_weekday
                
            # Apply mask: Any signal outside valid hours becomes 0
            signal_matrix = signal_matrix * market_open[:, np.newaxis]
        
        return signal_matrix

    def run_simulation_batch(self, signals_matrix, prices, times, time_limit=None):
        """
        Runs TradeSimulator for each strategy in the batch.
        Returns: net_returns matrix (Bars x Strats), trades_count array
        """
        n_bars, n_strats = signals_matrix.shape
        if n_strats == 0:
            return np.zeros((n_bars, 0)), np.zeros(0)
            
        # Parallelize Simulation
        n_jobs = -1
        batch_size = max(1, n_strats // (16 if n_strats > 100 else 4))
        
        # Split signals matrix into chunks along columns (strategies)
        # signals_matrix[:, i:i+batch_size]
        chunks = [signals_matrix[:, i:i + batch_size] for i in range(0, n_strats, batch_size)]
        
        # Run in parallel using threads for Numba JIT (which releases GIL)
        # We can keep threading here because Numba bypasses GIL efficiently
        # But to be consistent and safe, we can use processes too if needed. 
        # However, passing signals_matrix (numpy) to processes is cheap if not too huge.
        # Let's use Threads for simulation as it was intended and confirmed to be GIL-free by Numba.
        # Wait, user said "CPU not utilized". Threading might not be scaling. 
        # Let's switch simulation to processes too, just to be sure. 
        results = Parallel(n_jobs=n_jobs)(
            delayed(_worker_simulate)(
                chunk, 
                prices, 
                times, 
                self.spread_bps, 
                self.effective_cost_bps, 
                self.standard_lot, 
                self.account_size,
                time_limit
            ) for chunk in chunks
        )
        
        # Reassemble results
        all_net_returns = np.hstack([r[0] for r in results])
        all_trades_count = np.concatenate([r[1] for r in results])
            
        return all_net_returns, all_trades_count

    def evaluate_population(self, population: list[Strategy], set_type='train', return_series=False, prediction_mode=False, time_limit=None):
        if not population: return []
        
        full_signal_matrix = self.generate_signal_matrix(population)
        
        if set_type == 'train':
            start, end = 0, self.train_idx
        elif set_type == 'validation':
            start, end = self.train_idx, self.val_idx
        elif set_type == 'test':
            start, end = self.val_idx, len(self.raw_data)
        else: raise ValueError("set_type must be 'train', 'validation', or 'test'")
            
        signals = full_signal_matrix[start:end]
        prices = self.close_vec[start:end]
        times = self.times_vec.iloc[start:end] if hasattr(self.times_vec, 'iloc') else self.times_vec[start:end]
        
        # Prediction Mode override (Fast Vectorized)
        if prediction_mode:
            returns = self.returns_vec[start:end]
            strat_returns = signals * returns
            net_returns = strat_returns
            trades_count = np.sum(np.abs(signals), axis=0) # Approx
        else:
            # Full Simulation Mode
            # Use passed time_limit or default 120
            limit = time_limit if time_limit else 120
            
            net_returns, trades_count = self.run_simulation_batch(signals, prices, times, time_limit=limit)
        
        # --- METRICS ---
        total_ret = np.sum(net_returns, axis=0)
        stdev = np.std(net_returns, axis=0) + 1e-9
        avg_ret = np.mean(net_returns, axis=0)
        sharpe = (avg_ret / stdev) * np.sqrt(self.annualization_factor)
        
        downside = np.minimum(net_returns, 0)
        downside_std = np.std(downside, axis=0) + 1e-9
        sortino = (avg_ret / downside_std) * np.sqrt(self.annualization_factor)
        
        max_win = np.max(net_returns, axis=0)
        stability_ratio = max_win / (total_ret + 1e-9)
        
        results = []
        for i, strat in enumerate(population):
            final_sortino = sortino[i]
            if trades_count[i] < 5: final_sortino = -1.0
            if stability_ratio[i] > 0.5 and total_ret[i] > 0: final_sortino *= 0.5
            
            strat.fitness = final_sortino
            results.append({
                'id': strat.name,
                'sharpe': sharpe[i],
                'sortino': final_sortino,
                'total_return': total_ret[i],
                'trades': trades_count[i],
                'stability': stability_ratio[i]
            })
            
        results_df = pd.DataFrame(results)
        return (results_df, net_returns) if return_series else results_df

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
            prices = self.close_vec[train_end:test_end]
            times = self.times_vec.iloc[train_end:test_end] if hasattr(self.times_vec, 'iloc') else self.times_vec[train_end:test_end]
            
            # Use Simulator
            net_returns, trades_count = self.run_simulation_batch(signals, prices, times, time_limit=limit)
            
            avg = np.mean(net_returns, axis=0)
            downside = np.std(np.minimum(net_returns, 0), axis=0) + 1e-9
            sortino = (avg / downside) * np.sqrt(self.annualization_factor)
            
            sortino[trades_count < 8] = -5.0
            fold_scores[:, f] = sortino
            
        avg_sortino = np.mean(fold_scores, axis=1)
        min_sortino = np.min(fold_scores, axis=1)
        fold_std = np.std(fold_scores, axis=1)
        
        # Stricter Robustness: High penalty for variance
        robust_score = avg_sortino - (fold_std * 1.0)
        
        # Hard Kill for blowing up in any fold
        # If any fold has Sortino < -0.5, massive penalty
        for i in range(len(robust_score)):
            if min_sortino[i] < -1.5:
                robust_score[i] -= 10.0
        
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