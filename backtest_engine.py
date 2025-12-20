import pandas as pd
import numpy as np
from strategy_genome import Strategy

class BacktestEngine:
    """
    High-Performance Vectorized Backtester.
    Capable of evaluating thousands of strategies simultaneously via Matrix operations.
    """
    def __init__(self, data: pd.DataFrame, cost_bps=0.2, fixed_cost=2.0, spread_bps=1.0, account_size=100000.0, target_col='log_ret', annualization_factor=1.0):
        self.raw_data = data
        self.target_col = target_col
        self.annualization_factor = annualization_factor
        self.context = {}
        
        # --- COST MODELING (Reality Check) ---
        # Variable Commission: 0.2 bps (0.00002)
        var_comm_pct = cost_bps / 10000.0
        
        # Fixed Commission: $2.00 min per trade
        # We must convert this to a percentage of the assumed account size
        # $2 / $100,000 = 0.00002 (2 bps)
        fixed_comm_pct = fixed_cost / account_size
        
        # Effective Commission: Max of Variable vs Fixed
        effective_comm_pct = max(var_comm_pct, fixed_comm_pct)
        
        # Spread & Slippage: Always paid (e.g. 1.0 bps)
        spread_slippage_pct = spread_bps / 10000.0
        
        # Total Cost Per Side
        self.total_cost_pct = effective_comm_pct + spread_slippage_pct
        
        print(f"💰 Cost Model Updated:")
        print(f"   - Account Size: ${account_size:,.0f}")
        print(f"   - Base Comm: {cost_bps} bps | Min: ${fixed_cost}")
        print(f"   - Spread/Slip: {spread_bps} bps")
        print(f"   - Effective Cost Per Side: {self.total_cost_pct*10000:.2f} bps")
        
        # Pre-calculate log returns for the vectorizer if standard mode
        if target_col == 'log_ret' and 'log_ret' not in self.raw_data.columns:
            self.raw_data['log_ret'] = np.log(self.raw_data['close'] / self.raw_data['close'].shift(1))
            
        # Precompute Derived Features for Fast Evaluation
        # CRITICAL FIX: Compute features on FULL continuous data BEFORE dropping NaNs.
        # Otherwise, rolling windows (ZScores) skip time gaps and corrupt data.
        self.precompute_context()

        # Clean Data (Post-Feature Generation)
        if target_col in self.raw_data.columns:
            original_len = len(self.raw_data)
            
            # Identify valid rows
            valid_mask = ~self.raw_data[target_col].isna()
            
            # Slice Raw Data
            self.raw_data = self.raw_data[valid_mask].reset_index(drop=True)
            
            # Slice Context Arrays to match new raw_data
            # Context arrays are numpy arrays aligned with original raw_data
            for key in self.context:
                if isinstance(self.context[key], np.ndarray) and len(self.context[key]) == original_len:
                    self.context[key] = self.context[key][valid_mask]
            
            self.context['__len__'] = len(self.raw_data)

            # Check if we dropped too much
            if len(self.raw_data) < original_len * 0.9:
                print(f"⚠️ Warning: Dropped {original_len - len(self.raw_data)} rows due to NaNs in {target_col}")
            
        # Prepare Data Matrices (Numpy is faster than Pandas for Dot Products)
        self.returns_vec = self.raw_data[self.target_col].values.reshape(-1, 1).astype(np.float32)
        self.close_vec = self.raw_data['close'].values.astype(np.float32)
        
        # Split Indices (60% Train, 20% Val, 20% Test)
        n = len(self.raw_data)
        self.train_idx = int(n * 0.6)
        self.val_idx = int(n * 0.8)
        
        print(f"Engine Initialized. Data: {n} bars.")
        print(f"Train: 0-{self.train_idx} | Val: {self.train_idx}-{self.val_idx} | Test: {self.val_idx}-{n}")
        
    def precompute_context(self):
        """
        Generates base features (Time, Consecutive) and converts raw data to Numpy.
        Dynamic features (Delta/ZScore) are now handled Just-In-Time by ensure_context().
        """
        print("⚡ Pre-computing Base Context (Time/Consecutive) [float32]...")
        # 1. Base Features
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.context[col] = self.raw_data[col].values.astype(np.float32)
            
        self.context['__len__'] = len(self.raw_data)
        
        # 2. Time Features
        if 'time_start' in self.raw_data.columns:
            dt = self.raw_data['time_start'].dt
            self.context['time_hour'] = dt.hour.values.astype(np.float32)
            self.context['time_weekday'] = dt.dayofweek.values.astype(np.float32)
            
        # 3. Consecutive Bars (Pattern)
        # Vectorized calculation of consecutive runs
        close = self.raw_data['close'].values
        # Up Bars
        up_mask = (close > np.roll(close, 1))
        up_mask[0] = False
        # Calculate run lengths
        # Trick: Group by change in value
        # This is hard to fully vector-precompute efficiently for every bar in Python without a loop or specialized func
        # Simpler approximation: Rolling sum of boolean? No, that's count in window.
        # We need "current streak".
        # Fast iterative solution for now (N is usually < 1M, so 0.1s)
        
        def get_streak(mask):
            streaks = np.zeros(len(mask), dtype=np.float32)
            current = 0
            for i in range(len(mask)):
                if mask[i]:
                    current += 1
                else:
                    current = 0
                streaks[i] = current
            return streaks

        self.context['consecutive_up'] = get_streak(up_mask)
        self.context['consecutive_down'] = get_streak(~up_mask & (close < np.roll(close, 1)))

        print(f"⚡ Base Context Ready. Total Arrays: {len(self.context)}")

    def ensure_context(self, population: list[Strategy]):
        """
        Scans population for required Delta/ZScore features and computes them if missing.
        This enables infinite gene diversity without pre-computing everything.
        """
        needed = set()
        for strat in population:
            all_genes = strat.long_genes + strat.short_genes
            for gene in all_genes:
                if gene.type == 'delta':
                    needed.add(('delta', gene.feature, gene.lookback))
                elif gene.type == 'zscore':
                    needed.add(('zscore', gene.feature, gene.window))
                    
        # Calculate missing
        calc_count = 0
        for type_, feature, param in needed:
            key = f"{type_}_{feature}_{param}"
            if key in self.context: continue
            
            calc_count += 1
            if type_ == 'delta':
                # Delta: arr[i] - arr[i-w]
                arr = self.context.get(feature)
                if arr is None: continue
                w = param
                diff = np.zeros_like(arr)
                if w < len(arr):
                    diff[w:] = arr[w:] - arr[:-w]
                self.context[key] = diff
                
            elif type_ == 'zscore':
                # ZScore: (x - mean) / std
                # Use pandas rolling for speed/correctness
                w = param
                series = self.raw_data[feature] # Use raw pandas series
                r_mean = series.rolling(w).mean()
                r_std = series.rolling(w).std()
                z = (series - r_mean) / (r_std + 1e-9)
                self.context[key] = z.fillna(0).values.astype(np.float32)
                
        if calc_count > 0:
            print(f"   > JIT: Computed {calc_count} new features.")

    def generate_signal_matrix(self, population: list[Strategy]) -> np.array:
        """
        Converts a population of Strategy objects into a Boolean Signal Matrix (Time x Strategies).
        Optimized with Gene Caching.
        """
        # Ensure context exists for all genes
        self.ensure_context(population)
        
        num_strats = len(population)
        num_bars = len(self.raw_data)
        
        # Pre-allocate matrix (int8 to save memory)
        signal_matrix = np.zeros((num_bars, num_strats), dtype=np.int8)
        
        # Gene Cache (Memoization)
        # Shared across all strategies in this generation.
        # Key: (GeneType, Feature, Op, Threshold...)
        # Value: Boolean Mask (np.array)
        gene_cache = {}
        
        for i, strat in enumerate(population):
            # Pass the pre-computed Context AND the Cache
            signal_matrix[:, i] = strat.generate_signal(self.context, cache=gene_cache)
            
        return signal_matrix

    def evaluate_population(self, population: list[Strategy], set_type='train', return_series=False, prediction_mode=False):
        """
        Runs the full backtest math on the entire population matrix.
        Returns a list of metrics dictionaries.
        
        :param prediction_mode: If True, assumes 'returns' are pre-calculated labels (e.g., Triple Barrier)
                                aligned with the signal time. No shifting is performed.
        """
        if not population: return []
        
        # 1. Generate Signals
        # (This is the heavy lifting)
        full_signal_matrix = self.generate_signal_matrix(population)
        
        # 2. Slice Data based on Set Type
        if set_type == 'train':
            signals = full_signal_matrix[:self.train_idx]
            returns = self.returns_vec[:self.train_idx]
        elif set_type == 'validation':
            signals = full_signal_matrix[self.train_idx:self.val_idx]
            returns = self.returns_vec[self.train_idx:self.val_idx]
        elif set_type == 'test':
            signals = full_signal_matrix[self.val_idx:]
            returns = self.returns_vec[self.val_idx:]
        else:
            raise ValueError("set_type must be 'train', 'validation', or 'test'")
            
        # 3. Vectorized PnL Calculation
        if prediction_mode:
            # PREDICTION MODE (Triple Barrier / Label Match)
            # Signal(t) predicts Label(t). No shift needed.
            # "Returns" here effectively acts as the PnL of the specific outcome associated with that bar.
            strat_returns = signals * returns
        else:
            # TRADING MODE (Next-Bar Return)
            # Strategy Returns = Signal(t) * Market_Return(t+1)
            # Shift signals forward by 1 to align with NEXT bar's return (No lookahead bias)
            signals_shifted = np.roll(signals, 1, axis=0)
            signals_shifted[0, :] = 0 # Zero out first row after shift
            
            # Gross Returns (Matrix Mult: [T x 1] broadcasted over [T x N])
            strat_returns = signals_shifted * returns
        
        # 4. Cost Adjustment
        # Cost is paid on turnover (change in signal).
        # Turnover Matrix: abs(Signal(t) - Signal(t-1))
        # Note: We use unshifted signals for turnover calc to capture trade moment
        trades = np.abs(np.diff(signals, axis=0, prepend=0))
        costs = trades * self.total_cost_pct
        
        net_returns = strat_returns - costs
        
        # 5. Metrics Calculation (Vectorized across columns)
        # Cumulative Sum (Equity Curve)
        equity_curves = np.cumsum(net_returns, axis=0)
        
        # Total Return
        total_ret = np.sum(net_returns, axis=0)
        
        # Volatility (Annualized? Assuming 250-tick bars ~ 2 mins -> huge scaling factor)
        # Let's just use per-bar std dev for Sharpe
        stdev = np.std(net_returns, axis=0) + 1e-9
        
        # Sharpe (Simple: Mean / Std)
        # Scaled by square root of annualization factor
        avg_ret = np.mean(net_returns, axis=0)
        sharpe = (avg_ret / stdev) * np.sqrt(self.annualization_factor)
        
        # Sortino (Mean / Downside Std)
        # Vectorized Downside Deviation
        downside_returns = np.minimum(net_returns, 0)
        downside_std = np.std(downside_returns, axis=0) + 1e-9
        sortino = (avg_ret / downside_std) * np.sqrt(self.annualization_factor)
        
        # --- NEW: Stability Metric (Prop Firm Integrity) ---
        # Penalize if > 50% of profit comes from a single trade.
        # How to calc per-trade PnL in vectorized way?
        # Difficult fully vectorized without loop, but we can approximate or do a quick pass.
        # Alternatively, use Max Win / Total Return.
        # We need "Max Single Bar Return" vs Total Return? No, trade can be multi-bar.
        # Approximation: Check max single-bar PnL. If a single bar is 50% of total PnL, that's bad too.
        max_bar_pnl = np.max(net_returns, axis=0)
        stability_ratio = max_bar_pnl / (total_ret + 1e-9)
        # NOTE: This is "Max Bar / Total". Real stability is "Max Trade / Total". 
        # But Max Bar is a decent proxy for "lucky spike".
        
        # 6. Assign Scores back to Strategy Objects
        results = []
        for i, strat in enumerate(population):
            # Apply Stability Penalty
            final_sortino = sortino[i]
            if stability_ratio[i] > 0.5 and total_ret[i] > 0:
                 final_sortino *= 0.5 # Severe penalty for "One Hit Wonders"
            
            strat.fitness = final_sortino
            metrics = {
                'id': strat.name,
                'sharpe': sharpe[i],
                'sortino': final_sortino,
                'total_return': total_ret[i],
                'trades': np.sum(trades[:, i]),
                'stability': stability_ratio[i]
            }
            results.append(metrics)
            
        results_df = pd.DataFrame(results)
        
        if return_series:
            return results_df, net_returns
        else:
            return results_df
            
        results_df = pd.DataFrame(results)
        
        if return_series:
            return results_df, net_returns
        else:
            return results_df

if __name__ == "__main__":
    # Test Driver
    import sys
    import config
    from feature_engine import FeatureEngine
    from strategy_genome import GenomeFactory
    import os
    
    # 1. Load Data
    print("Loading Data...")
    engine = FeatureEngine(config.DIRS['DATA_RAW_TICKS'])
    df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    engine.create_volume_bars(df, volume_threshold=250)
    
    # Generate some features for the strategies to use
    engine.add_features_to_bars(windows=[50])
    # engine.add_microstructure_features() # Add this if using those genes
    
    data = engine.bars
    
    # 2. Initialize Engine
    backtester = BacktestEngine(data, cost_bps=0.5)
    
    # 3. Generate Random Population
    print("\nSpawning 100 random strategies...")
    survivors_path = os.path.join(config.DIRS['DATA_DIR'], "survivors_60.json")
    
    # Mocking survivor list if file expects columns we haven't calc'd in this quick test
    # Just creating a factory that uses available columns for the test
    # Exclude timestamp columns to prevent type errors
    exclude_cols = ['time_start', 'time_end', 'close', 'ts_event']
    available_features = [c for c in data.columns if c not in exclude_cols and data[c].dtype.kind in 'bifc']
    
    # Manually patching factory for test
    factory = GenomeFactory(survivors_path)
    factory.features = available_features # Override with what we actually computed
    factory.set_stats(data)
    
    population = [factory.create_strategy(num_genes=2) for _ in range(100)]
    
    # 4. Run Evaluation
    print("\nRunning Vectorized Backtest (Train Set)...")
    import time
    start = time.time()
    results = backtester.evaluate_population(population, set_type='train')
    duration = time.time() - start
    
    print(f"\nCompleted in {duration:.4f}s")
    print("\nTop 5 Performers:")
    print(results.sort_values('sharpe', ascending=False).head())
