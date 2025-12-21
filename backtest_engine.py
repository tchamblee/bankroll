import pandas as pd
import numpy as np
from strategy_genome import Strategy
import config

class BacktestEngine:
    """
    High-Performance Vectorized Backtester.
    Capable of evaluating thousands of strategies simultaneously via Matrix operations.
    Refactored for Tiered Lot Sizing (1-3 Lots) and Account-Based PnL.
    """
    def __init__(self, data: pd.DataFrame, cost_bps=0.2, fixed_cost=2.0, spread_bps=1.0, account_size=None, target_col='log_ret', annualization_factor=1.0):
        self.raw_data = data
        self.target_col = target_col
        self.annualization_factor = annualization_factor
        self.context = {}
        
        # --- CONFIGURATION ---
        self.account_size = account_size if account_size else config.ACCOUNT_SIZE
        self.standard_lot = config.STANDARD_LOT_SIZE
        
        # --- COST MODELING ---
        # Variable Commission: 0.2 bps (0.00002)
        var_comm_pct = cost_bps / 10000.0
        
        # Fixed Commission: $2.00 min per trade
        # Convert to approximate bps based on 1 lot ($100k notional) for simplicity in vectorization, 
        # or treat as floor. For vectorization, we often use effective bps.
        # $2 per 100k = 0.2 bps.
        fixed_comm_pct = fixed_cost / 100000.0 # Assuming 1 lot min size for cost calc baseline
        
        # Effective Commission: Max of Variable vs Fixed
        effective_comm_pct = max(var_comm_pct, fixed_comm_pct)
        
        # Spread & Slippage: Always paid (e.g. 1.0 bps)
        spread_slippage_pct = spread_bps / 10000.0
        
        # Total Cost Per Side (Multiplier of Notional Traded)
        self.total_cost_pct = effective_comm_pct + spread_slippage_pct
        
        # print(f"💰 Cost Model Updated:")
        # print(f"   - Account Size: ${self.account_size:,.0f}")
        # print(f"   - Lot Size: {self.standard_lot:,.0f} Units")
        # print(f"   - Effective Cost: {self.total_cost_pct*10000:.2f} bps of Notional")
        
        # Pre-calculate log returns for the vectorizer if standard mode
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
        self.close_vec = self.raw_data['close'].values.reshape(-1, 1).astype(np.float32)
        
        # Split Indices
        n = len(self.raw_data)
        self.train_idx = int(n * 0.6)
        self.val_idx = int(n * 0.8)
        
        # print(f"Engine Initialized. Data: {n} bars.")
        
    def precompute_context(self):
        # print("⚡ Pre-computing Base Context (Time/Consecutive) [float32]...")
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
        
        # Track base keys to allow resetting JIT features
        self.base_keys = set(self.context.keys())
        # print(f"⚡ Base Context Ready. Total Arrays: {len(self.context)}")

    def reset_jit_context(self):
        """Removes any JIT-computed features to free memory."""
        keys_to_remove = [k for k in self.context.keys() if k not in self.base_keys]
        for k in keys_to_remove:
            del self.context[k]
        # print(f"🧹 JIT Context Cleared. Removed {len(keys_to_remove)} features.")

    def ensure_context(self, population: list[Strategy]):
        needed = set()
        for strat in population:
            all_genes = strat.long_genes + strat.short_genes
            for gene in all_genes:
                if gene.type == 'delta': needed.add(('delta', gene.feature, gene.lookback))
                elif gene.type == 'zscore': needed.add(('zscore', gene.feature, gene.window))
                    
        calc_count = 0
        for type_, feature, param in needed:
            key = f"{type_}_{feature}_{param}"
            if key in self.context: continue
            calc_count += 1
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
                
        # if calc_count > 0: print(f"   > JIT: Computed {calc_count} new features.")

    def generate_signal_matrix(self, population: list[Strategy]) -> np.array:
        self.ensure_context(population)
        num_strats = len(population)
        num_bars = len(self.raw_data)
        signal_matrix = np.zeros((num_bars, num_strats), dtype=np.int8)
        gene_cache = {}
        for i, strat in enumerate(population):
            signal_matrix[:, i] = strat.generate_signal(self.context, cache=gene_cache)
        return signal_matrix

    def evaluate_population(self, population: list[Strategy], set_type='train', return_series=False, prediction_mode=False):
        if not population: return []
        
        full_signal_matrix = self.generate_signal_matrix(population)
        
        if set_type == 'train':
            signals = full_signal_matrix[:self.train_idx]
            returns = self.returns_vec[:self.train_idx]
            prices = self.close_vec[:self.train_idx]
        elif set_type == 'validation':
            signals = full_signal_matrix[self.train_idx:self.val_idx]
            returns = self.returns_vec[self.train_idx:self.val_idx]
            prices = self.close_vec[self.train_idx:self.val_idx]
        elif set_type == 'test':
            signals = full_signal_matrix[self.val_idx:]
            returns = self.returns_vec[self.val_idx:]
            prices = self.close_vec[self.val_idx:]
        else: raise ValueError("set_type must be 'train', 'validation', or 'test'")
            
        # --- PnL CALCULATION (Account Based) ---
        if prediction_mode:
            # Prediction Mode: Signal * Target (No sizing/cost logic)
            strat_returns = signals * returns
            net_returns = strat_returns # Raw label match score
            trades = np.abs(signals) # Just count activations
        else:
            # Trading Mode: Sizing & Costs
            # 1. Align Signal with Price/Return
            # We enter at T (Signal T), PnL realized at T+1 (Close T+1)
            # Position Held (T) = Signal (T)
            # Market Return (T+1) = log(Close T+1 / Close T) -> returns_vec aligns with T+1 if standard log_ret
            # So: PnL(T+1) = Position(T) * MarketReturn(T+1)
            
            signals_shifted = np.roll(signals, 1, axis=0); signals_shifted[0, :] = 0
            
            # 2. Gross PnL ($)
            # Position Notional = Lots * StandardLotSize * Price
            # Approximation: Previous Position Notional * Log Return
            prev_position_lots = signals_shifted
            # Price at T (previous bar close) is needed for notional? 
            # log_ret is relative to prev close. So: Value(T) * log_ret approx PnL.
            # Value(T) = Lots * 100k * Price(T). 
            # But 'prices' vector corresponds to current bar T+1? 
            # We need Price(T) to match signals_shifted(T).
            prices_shifted = np.roll(prices, 1, axis=0); prices_shifted[0] = prices[0]
            
            position_notional = prev_position_lots * self.standard_lot * prices_shifted
            gross_pnl_dollar = position_notional * returns
            
            # 3. Transaction Costs ($)
            # Trade happens at T.
            # Change in lots = Signal(T) - Signal(T-1)
            # We pay cost on Notional Traded at Price(T).
            # Price(T) is 'prices' vector (unshifted) because signal change happens at close T.
            # Wait, if we use unshifted signals for turnover, we align with unshifted prices.
            
            lot_change = np.abs(np.diff(signals, axis=0, prepend=0))
            turnover_notional = lot_change * self.standard_lot * prices
            costs_dollar = turnover_notional * self.total_cost_pct
            
            # 4. Net PnL & Percentage Return
            net_pnl_dollar = gross_pnl_dollar - costs_dollar
            net_returns = net_pnl_dollar / self.account_size # % Return on Account
            trades = lot_change # Count lot changes as trade activity measure
        
        # --- METRICS ---
        total_ret = np.sum(net_returns, axis=0)
        stdev = np.std(net_returns, axis=0) + 1e-9
        avg_ret = np.mean(net_returns, axis=0)
        sharpe = (avg_ret / stdev) * np.sqrt(self.annualization_factor)
        
        downside = np.minimum(net_returns, 0)
        downside_std = np.std(downside, axis=0) + 1e-9
        sortino = (avg_ret / downside_std) * np.sqrt(self.annualization_factor)
        
        # Stability: Max Dollar Win / Total Dollar PnL (approx via % returns)
        max_win = np.max(net_returns, axis=0)
        stability_ratio = max_win / (total_ret + 1e-9)
        
        results = []
        for i, strat in enumerate(population):
            final_sortino = sortino[i]
            # Penalty for low activity or instability
            if np.sum(trades[:, i]) < 5: final_sortino = -1.0
            if stability_ratio[i] > 0.5 and total_ret[i] > 0: final_sortino *= 0.5
            
            strat.fitness = final_sortino
            results.append({
                'id': strat.name,
                'sharpe': sharpe[i],
                'sortino': final_sortino,
                'total_return': total_ret[i],
                'trades': np.sum(trades[:, i]),
                'stability': stability_ratio[i]
            })
            
        results_df = pd.DataFrame(results)
        return (results_df, net_returns) if return_series else results_df

    def evaluate_walk_forward(self, population: list[Strategy], folds=4):
        """
        Conducts a rigorous Rolling Walk-Forward Validation on the Development Set (0-80%).
        The last 20% is held out as the Final Exam.
        """
        if not population: return []

        full_signal_matrix = self.generate_signal_matrix(population)
        n_bars = len(self.raw_data)
        
        # Define Development Set (0% - 80%)
        dev_end_idx = int(n_bars * 0.8)
        
        # Fixed indices for 4 folds on ~22k bars (scaled to actual size)
        # We target ~18k dev set.
        # Window ~10k, Step ~2k.
        
        # Dynamic resizing based on actual dev_end_idx
        window_size = int(dev_end_idx * 0.55) # ~10k / 18k
        step_size = int(dev_end_idx * 0.11)   # ~2k / 18k
        
        fold_scores = np.zeros((len(population), folds))
        
        for f in range(folds):
            start = f * step_size
            train_end = start + window_size
            test_end = train_end + step_size
            
            # Clip to dev set end
            if test_end > dev_end_idx:
                test_end = dev_end_idx
                train_end = test_end - step_size # Maintain test size? or shrink?
            
            # Extract Test Slice for this fold
            signals = full_signal_matrix[train_end:test_end]
            returns = self.returns_vec[train_end:test_end]
            prices = self.close_vec[train_end:test_end]
            
            # --- Vectorized PnL (Same logic as evaluate_population) ---
            signals_shifted = np.roll(signals, 1, axis=0); signals_shifted[0, :] = 0
            prices_shifted = np.roll(prices, 1, axis=0); prices_shifted[0] = prices[0]
            
            position_notional = signals_shifted * self.standard_lot * prices_shifted
            gross_pnl = position_notional * returns
            
            lot_change = np.abs(np.diff(signals, axis=0, prepend=0))
            costs = lot_change * self.standard_lot * prices * self.total_cost_pct
            
            net_pnl = gross_pnl - costs
            net_ret = net_pnl / self.account_size
            
            # Metrics
            avg = np.mean(net_ret, axis=0)
            downside = np.std(np.minimum(net_ret, 0), axis=0) + 1e-9
            sortino = (avg / downside) * np.sqrt(self.annualization_factor)
            
            # Activity Filter (Per Fold)
            trades = np.sum(lot_change, axis=0)
            sortino[trades < 3] = -1.0 # Min 3 trades per 2k bars (~1 month)
            
            fold_scores[:, f] = sortino
            
        # Aggregated Score
        avg_sortino = np.mean(fold_scores, axis=1)
        min_sortino = np.min(fold_scores, axis=1) # Worst case scenario
        
        # Robust Score = Average - Penalty for Variance between folds
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
