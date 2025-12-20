import pandas as pd
import numpy as np
from strategy_genome import Strategy

class BacktestEngine:
    """
    High-Performance Vectorized Backtester.
    Capable of evaluating thousands of strategies simultaneously via Matrix operations.
    """
    def __init__(self, data: pd.DataFrame, cost_bps=0.5):
        self.raw_data = data
        self.cost_pct = cost_bps / 10000.0
        
        # Pre-calculate log returns for the vectorizer
        if 'log_ret' not in self.raw_data.columns:
            self.raw_data['log_ret'] = np.log(self.raw_data['close'] / self.raw_data['close'].shift(1))
            
        # Clean Nans
        self.raw_data = self.raw_data.dropna().reset_index(drop=True)
        
        # Prepare Data Matrices (Numpy is faster than Pandas for Dot Products)
        self.returns_vec = self.raw_data['log_ret'].values.reshape(-1, 1)
        self.close_vec = self.raw_data['close'].values
        
        # Split Indices (60% Train, 20% Val, 20% Test)
        n = len(self.raw_data)
        self.train_idx = int(n * 0.6)
        self.val_idx = int(n * 0.8)
        
        print(f"Engine Initialized. Data: {n} bars.")
        print(f"Train: 0-{self.train_idx} | Val: {self.train_idx}-{self.val_idx} | Test: {self.val_idx}-{n}")

    def generate_signal_matrix(self, population: list[Strategy]) -> np.array:
        """
        Converts a population of Strategy objects into a Boolean Signal Matrix (Time x Strategies).
        """
        # We need to evaluate each strategy. 
        # Ideally, this is the slowest part. 
        # Future Optimization: Compile genes to numba or eval string.
        # For now, standard python loop is fine for < 5000 strategies.
        
        num_strats = len(population)
        num_bars = len(self.raw_data)
        
        # Pre-allocate matrix (Bool)
        signal_matrix = np.zeros((num_bars, num_strats), dtype=int)
        
        for i, strat in enumerate(population):
            # This calls the strategy's internal vectorized check
            # Returns a 1D array of 0s and 1s
            signal_matrix[:, i] = strat.generate_signal(self.raw_data)
            
        return signal_matrix

    def evaluate_population(self, population: list[Strategy], set_type='train', return_series=False):
        """
        Runs the full backtest math on the entire population matrix.
        Returns a list of metrics dictionaries.
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
        # Strategy Returns = Signal(t) * Market_Return(t+1)
        # Shift signals forward by 1 to align with NEXT bar's return (No lookahead bias)
        signals_shifted = np.roll(signals, 1, axis=0)
        signals_shifted[0, :] = 0 # Zero out first row after shift
        
        # Gross Returns (Matrix Mult: [T x 1] broadcasted over [T x N])
        # Actually simple element-wise mult is easier here since dimensions align columns
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
        # Scaling factor: sqrt(bars_per_year). 
        # If 2 mins/bar -> 720 bars/day -> 180,000 bars/year? 
        # Let's stick to simple Sharpe per bar for relative fitness.
        avg_ret = np.mean(net_returns, axis=0)
        sharpe = avg_ret / stdev
        
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
