import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
import config
from ..workers import _worker_simulate

class SimulationMixin:
    def run_simulation_batch(self, signals_matrix, strategies, prices, times, time_limit=None, highs=None, lows=None, atr=None):
        """
        Runs TradeSimulator for each strategy in the batch.
        Returns: net_returns matrix (Bars x Strats), trades_count array
        """
        n_bars, n_strats = signals_matrix.shape
        if n_strats == 0:
            return np.zeros((n_bars, 0)), np.zeros(0)

        # 1. Extract Time Features for Slice
        if hasattr(times, 'dt'):
            hours = times.dt.hour.values.astype(np.int8)
            weekdays = times.dt.dayofweek.values.astype(np.int8)
        else:
            dt_idx = pd.to_datetime(times)
            hours = dt_idx.hour.values.astype(np.int8)
            weekdays = dt_idx.dayofweek.values.astype(np.int8)

        # ATR Fallback
        if atr is None:
            if highs is not None and lows is not None:
                atr = np.maximum(highs - lows, prices * 0.0005) # Min 5 bps
            else:
                atr = prices * 0.001 # 10 bps fallback

        # CRITICAL FIX: Shift ATR by 1 to prevent Look-Ahead Bias
        # Simulator executes at Open[t], so it must use Volatility[t-1]
        if atr is not None and len(atr) > 1:
            atr = np.roll(atr, 1)
            atr[0] = atr[1] # Backfill first element

        # Parallelize Simulation
        n_jobs = min(6, multiprocessing.cpu_count())
        batch_size = max(1, n_strats // (16 if n_strats > 100 else 4))
        
        params_list = [{'sl': getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS), 'tp': getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT)} for s in strategies]
        
        chunks_sig = [signals_matrix[:, i:i + batch_size] for i in range(0, n_strats, batch_size)]
        chunks_params = [params_list[i:i + batch_size] for i in range(0, n_strats, batch_size)]
        
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_worker_simulate)(
                chunk_s,
                chunk_p,
                prices, 
                times, 
                self.spread_bps, 
                self.effective_cost_bps, 
                self.min_comm,
                self.standard_lot, 
                self.account_size,
                time_limit,
                hours, 
                weekdays, 
                highs,
                lows,
                atr
            ) for chunk_s, chunk_p in zip(chunks_sig, chunks_params)
        )
        
        all_net_returns = np.hstack([r[0] for r in results])
        all_trades_count = np.concatenate([r[1] for r in results])
            
        return all_net_returns, all_trades_count
