import pandas as pd
import numpy as np
from strategy_genome import Strategy, VALID_DELTA_LOOKBACKS, VALID_ZSCORE_WINDOWS

class BacktestEngine:
    """
    High-Performance Vectorized Backtester.
    Capable of evaluating thousands of strategies simultaneously via Matrix operations.
    """
    def __init__(self, data: pd.DataFrame, cost_bps=0.5, target_col='log_ret', annualization_factor=1.0):
        self.raw_data = data
        self.cost_pct = cost_bps / 10000.0
        self.target_col = target_col
        self.annualization_factor = annualization_factor
        self.context = {}
        
        # Pre-calculate log returns for the vectorizer if standard mode
        if target_col == 'log_ret' and 'log_ret' not in self.raw_data.columns:
            self.raw_data['log_ret'] = np.log(self.raw_data['close'] / self.raw_data['close'].shift(1))
            
        # Clean Data
        # Audit Fix: Drop NaNs instead of filling with 0 to prevent training on artifacts.
        if target_col in self.raw_data.columns:
            original_len = len(self.raw_data)
            self.raw_data = self.raw_data.dropna(subset=[target_col])
            # Check if we dropped too much
            if len(self.raw_data) < original_len * 0.9:
                print(f"⚠️ Warning: Dropped {original_len - len(self.raw_data)} rows due to NaNs in {target_col}")
            
        self.raw_data = self.raw_data.reset_index(drop=True)
        
        # Prepare Data Matrices (Numpy is faster than Pandas for Dot Products)
        self.returns_vec = self.raw_data[self.target_col].values.reshape(-1, 1).astype(np.float32)
        self.close_vec = self.raw_data['close'].values.astype(np.float32)
        
        # Split Indices (60% Train, 20% Val, 20% Test)
        n = len(self.raw_data)
        self.train_idx = int(n * 0.6)
        self.val_idx = int(n * 0.8)
        
        print(f"Engine Initialized. Data: {n} bars.")
        print(f"Train: 0-{self.train_idx} | Val: {self.train_idx}-{self.val_idx} | Test: {self.val_idx}-{n}")
        
        # Precompute Derived Features for Fast Evaluation
        self.precompute_context()

    def precompute_context(self):
        """
        Generates a dictionary of Numpy arrays for all raw and potential derived features.
        This enables O(1) feature lookup inside the tight strategy loop.
        Optimization: Uses float32 to reduce memory footprint.
        """
        print("⚡ Pre-computing Derived Features (Deltas/ZScores) [float32]...")
        # 1. Base Features
        # Filter for numeric columns only to avoid errors
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns
        
        # Store base arrays
        for col in numeric_cols:
            self.context[col] = self.raw_data[col].values.astype(np.float32)
            
        self.context['__len__'] = len(self.raw_data)
        
        # 2. Derived Features
        # We pre-calculate ALL valid mutations so Genes can just look them up.
        
        # A. Deltas
        for w in VALID_DELTA_LOOKBACKS:
            for col in numeric_cols:
                key = f"delta_{col}_{w}"
                arr = self.context[col]
                # Diff: arr[i] - arr[i-w]
                diff = np.zeros_like(arr)
                diff[w:] = arr[w:] - arr[:-w]
                self.context[key] = diff # Already float32 from base
                
        # B. ZScores
        for w in VALID_ZSCORE_WINDOWS:
            # Need pandas rolling for efficiency/correctness of std
            print(f"   > Computing ZScores (w={w})...")
            # We can't use the float32 array in pandas easily, so use raw_data
            for col in numeric_cols:
                key = f"zscore_{col}_{w}"
                series = self.raw_data[col] 
                
                r_mean = series.rolling(w).mean()
                r_std = series.rolling(w).std()
                
                z = (series - r_mean) / (r_std + 1e-9)
                self.context[key] = z.fillna(0).values.astype(np.float32)
        
        print(f"⚡ Context Ready. Total Feature Arrays: {len(self.context)}")

    def generate_signal_matrix(self, population: list[Strategy]) -> np.array:
        """
        Converts a population of Strategy objects into a Boolean Signal Matrix (Time x Strategies).
        Optimized with Gene Caching.
        """
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
        costs = trades * self.cost_pct
        
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
        
        # Drawdown
        # peaks = np.maximum.accumulate(equity_curves, axis=0)
        # drawdowns = peaks - equity_curves
        # max_dd = np.max(drawdowns, axis=0)
        
        # 6. Assign Scores back to Strategy Objects
        results = []
        for i, strat in enumerate(population):
            strat.fitness = sharpe[i]
            metrics = {
                'id': strat.name,
                'sharpe': sharpe[i],
                'total_return': total_ret[i],
                'trades': np.sum(trades[:, i]),
                # 'max_dd': max_dd[i]
            }
            results.append(metrics)
            
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
