import pandas as pd
import numpy as np
import config
from ..statistics import combinatorial_purged_cv

class EvaluationMixin:
    def evaluate_population(self, population: list, set_type='train', return_series=False, prediction_mode=False, time_limit=None, min_trades=None):
        if not population: 
            return (pd.DataFrame(), np.array([])) if return_series else pd.DataFrame()
        
        if min_trades is not None:
            target_min_trades = min_trades
        else:
            h_ref = time_limit if time_limit else config.DEFAULT_TIME_LIMIT
            target_min_trades = max(10, int(config.MIN_TRADES_COEFFICIENT / h_ref + 5))
        
        BATCH_SIZE = config.EVO_BATCH_SIZE
        n_strats = len(population)
        
        all_results = []
        all_net_returns = []
        
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
        
        for i in range(0, n_strats, BATCH_SIZE):
            chunk_pop = population[i:i + BATCH_SIZE]
            
            # 1. Generate Signals
            full_signal_matrix = self.generate_signal_matrix(chunk_pop, horizon=time_limit)
            signals = full_signal_matrix[start:end]
            
            # Fix Lookahead: Next Open Execution
            signals = np.vstack([np.zeros((1, signals.shape[1]), dtype=signals.dtype), signals[:-1]])
            
            # 2. Simulate
            if prediction_mode:
                returns = self.returns_vec[start:end]
                strat_returns = signals * returns
                net_returns = strat_returns
                trades_count = np.sum(np.abs(signals), axis=0)
            else:
                limit = time_limit if time_limit else config.DEFAULT_TIME_LIMIT
                net_returns, trades_count = self.run_simulation_batch(signals, chunk_pop, prices, times, time_limit=limit, highs=highs, lows=lows, atr=atr)
            
            # 3. Calculate Metrics
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
                
                if trades_count[j] < target_min_trades:
                     final_sortino = -10.0
                elif stability_ratio[j] > 0.6 and total_ret[j] > 0: 
                    final_sortino *= 0.6
                
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
            
            del full_signal_matrix, signals, net_returns
            
        results_df = pd.DataFrame(all_results)
        
        if return_series:
            combined_returns = np.hstack(all_net_returns) if all_net_returns else np.array([])
            return results_df, combined_returns
        else:
            return results_df

    def evaluate_walk_forward(self, population: list, folds=config.WFV_FOLDS, time_limit=None, min_trades=None):
        if not population: return []
        
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
        fold_trades = np.zeros((len(population), folds))
        
        limit = time_limit if time_limit else config.DEFAULT_TIME_LIMIT
        
        for f in range(folds):
            start = f * step_size
            train_end = start + window_size
            test_end = train_end + step_size
            
            if test_end > dev_end_idx:
                test_end = dev_end_idx
                train_end = test_end - step_size 
            
            signals = full_signal_matrix[train_end:test_end]
            signals = np.vstack([np.zeros((1, signals.shape[1]), dtype=signals.dtype), signals[:-1]])
            
            prices = self.open_vec[train_end:test_end]
            highs = self.high_vec[train_end:test_end]
            lows = self.low_vec[train_end:test_end]
            atr = self.atr_vec[train_end:test_end]
            times = self.times_vec.iloc[train_end:test_end] if hasattr(self.times_vec, 'iloc') else self.times_vec[train_end:test_end]
            
            net_returns, trades_count = self.run_simulation_batch(signals, population, prices, times, time_limit=limit, highs=highs, lows=lows, atr=atr)
            
            fold_trades[:, f] = trades_count

            total_ret = np.sum(net_returns, axis=0)
            max_win = np.max(net_returns, axis=0)
            stability_ratio = max_win / (total_ret + 1e-9)
            
            avg = np.mean(net_returns, axis=0)
            downside_std = np.std(np.minimum(net_returns, 0), axis=0)
            downside_std = np.maximum(downside_std, 1e-6)
            
            sortino = (avg / downside_std) * np.sqrt(self.annualization_factor)
            sortino = np.nan_to_num(sortino, nan=-10.0)
            
            low_trades_mask = trades_count < target_min_trades
            sortino[low_trades_mask] = -10.0
            
            unstable_mask = (stability_ratio > 0.6) & (total_ret > 0)
            sortino[unstable_mask] *= 0.6
            
            fold_scores[:, f] = sortino
            
        avg_sortino = np.mean(fold_scores, axis=1)
        min_sortino = np.min(fold_scores, axis=1)
        fold_std = np.std(fold_scores, axis=1)
        avg_trades = np.mean(fold_trades, axis=1)
        
        robust_score = avg_sortino - (fold_std * 0.5)
        
        results = []
        for i, strat in enumerate(population):
            strat.fitness = robust_score[i]
            results.append({
                'id': strat.name,
                'sortino': robust_score[i],
                'avg_sortino': avg_sortino[i],
                'min_sortino': min_sortino[i],
                'fold_std': fold_std[i],
                'avg_trades': avg_trades[i]
            })
            
        return pd.DataFrame(results)

    def evaluate_combinatorial_purged_cv(self, population: list, n_folds=6, n_test_folds=2, time_limit=None):
        if not population: return pd.DataFrame()

        n_bars = len(self.raw_data)
        full_signal_matrix = self.generate_signal_matrix(population, horizon=time_limit)
        full_signal_matrix = np.vstack([np.zeros((1, full_signal_matrix.shape[1]), dtype=full_signal_matrix.dtype), full_signal_matrix[:-1]])

        splits = combinatorial_purged_cv(n_bars, n_folds=n_folds, n_test_folds=n_test_folds)
        print(f"  CPCV: Evaluating {len(splits)} combinatorial paths...")
        
        path_scores = np.zeros((len(population), len(splits)))
        limit = time_limit if time_limit else config.DEFAULT_TIME_LIMIT

        for i, (train_idx, test_idx) in enumerate(splits):
            signals = full_signal_matrix[test_idx]
            prices = self.open_vec[test_idx]
            highs = self.high_vec[test_idx]
            lows = self.low_vec[test_idx]
            atr = self.atr_vec[test_idx]
            
            if hasattr(self.times_vec, 'iloc'):
                times = self.times_vec.iloc[test_idx]
            else:
                times = self.times_vec[test_idx]
            
            net_returns, trades_count = self.run_simulation_batch(
                signals, population, prices, times, 
                time_limit=limit, highs=highs, lows=lows, atr=atr
            )
            
            avg = np.mean(net_returns, axis=0)
            downside = np.minimum(net_returns, 0)
            downside_std = np.std(downside, axis=0) + 1e-9
            
            sortino = (avg / downside_std) * np.sqrt(self.annualization_factor)
            
            slice_ratio = len(test_idx) / n_bars
            min_trades_slice = max(3, int(10 * slice_ratio))
            
            low_trades_mask = trades_count < min_trades_slice
            sortino[low_trades_mask] = -1.0
            
            path_scores[:, i] = sortino
            
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
