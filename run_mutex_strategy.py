import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import itertools
import time
from numba import jit
import config
from genome import Strategy
from backtest import BacktestEngine
from trade_simulator import TradeSimulator

# ==============================================================================
#  HELPER FUNCTIONS
# ==============================================================================

def load_all_candidates():
    candidates = []
    print(f"Loading candidates from all horizons...")
    
    for h in config.PREDICTION_HORIZONS:
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}_top5_unique.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if data:
            for d in data:
                try:
                    s = Strategy.from_dict(d)
                    s.horizon = h
                    s.training_id = d.get('training_id', 'legacy')
                    # Use 'sortino_oos' if available, else fallback
                    s.sortino = d.get('metrics', {}).get('sortino_oos', d.get('test_sortino', 0))
                    candidates.append(s)
                except Exception as e:
                    pass
            
    # Initial Sort by Sortino (Best First)
    candidates.sort(key=lambda x: x.sortino, reverse=True)
    
    print(f"  Loaded {len(candidates)} total candidates.")
    return candidates

@jit(nopython=True)
def _jit_simulate_mutex_custom(sig_matrix, prices, highs, lows, atr, hours, weekdays, horizons, sl_mults, tp_mults, lot_size, spread_pct, comm_pct, account_size, end_hour, cooldown_bars):
    """
    JIT-compiled simulator for a Mutex Portfolio.
    Priority Logic: The first strategy (column 0) has highest priority. 
    If it signals, we take it. If not, check col 1, etc.
    """
    n_bars, n_strats = sig_matrix.shape
    net_returns = np.zeros(n_bars, dtype=np.float64)
    trades_count = 0
    
    # State
    position = 0.0
    entry_price = 0.0
    entry_idx = 0
    entry_atr = 0.0
    active_strat_idx = -1
    
    current_horizon = 0
    current_sl_mult = 0.0
    current_tp_mult = 0.0
    
    prev_pos = 0.0
    
    # Cooldown State
    strat_cooldowns = np.zeros(n_strats, dtype=np.int64)
    
    for i in range(n_bars):
        # Decrement Cooldowns
        for s in range(n_strats):
            if strat_cooldowns[s] > 0:
                strat_cooldowns[s] -= 1
                
        # 1. Check Barriers / Exits
        exit_trade = False
        barrier_price = 0.0
        exit_reason = 0 # 0: None, 1: Time/EOD, 2: SL, 3: TP
        
        # Force Close (Time/EOD)
        if hours[i] >= end_hour or weekdays[i] >= 5:
            exit_trade = True
            exit_reason = 1
            
        elif position != 0:
            # Time Limit
            if (i - entry_idx) >= current_horizon:
                exit_trade = True
                exit_reason = 1
            
            # SL/TP
            if not exit_trade:
                h_prev = highs[i-1] if i > 0 else prices[i]
                l_prev = lows[i-1] if i > 0 else prices[i]
                
                sl_dist = entry_atr * current_sl_mult
                tp_dist = entry_atr * current_tp_mult
                
                if position > 0:
                    if current_sl_mult > 0 and l_prev <= (entry_price - sl_dist):
                        exit_trade = True
                        barrier_price = entry_price - sl_dist
                        exit_reason = 2 # SL
                    elif current_tp_mult > 0 and h_prev >= (entry_price + tp_dist):
                        exit_trade = True
                        barrier_price = entry_price + tp_dist
                        exit_reason = 3 # TP
                else:
                    if current_sl_mult > 0 and h_prev >= (entry_price + sl_dist):
                        exit_trade = True
                        barrier_price = entry_price + sl_dist
                        exit_reason = 2 # SL
                    elif current_tp_mult > 0 and l_prev <= (entry_price - tp_dist):
                        exit_trade = True
                        barrier_price = entry_price - tp_dist
                        exit_reason = 3 # TP
            
            # Reversal Check (Active Strategy Only)
            if not exit_trade and active_strat_idx >= 0:
                sig = sig_matrix[i, active_strat_idx]
                # Check for Reversal Signal
                if sig != 0 and np.sign(sig) != np.sign(position):
                     # Reversal!
                     # Effectively close current and open new
                     # No cooldown on reversal usually, unless we consider it an exit?
                     # Let's treat reversal as atomic: Exit + Entry. 
                     # If the exit would have been a loss, should we cooldown?
                     # Usually reversals are intentional logic, so NO cooldown.
                     position = float(sig)
                     entry_price = prices[i]
                     entry_idx = i
                     entry_atr = atr[i]
                     # Params remain same since it's the same strategy
                        
        if exit_trade:
            # Apply Cooldown if SL
            if exit_reason == 2 and active_strat_idx >= 0:
                strat_cooldowns[active_strat_idx] = cooldown_bars
                
            position = 0.0
            active_strat_idx = -1
            
        # 2. Check Entries (Priority Mux)
        if position == 0.0 and not (hours[i] >= end_hour or weekdays[i] >= 5):
            for s_idx in range(n_strats):
                # Check Cooldown
                if strat_cooldowns[s_idx] > 0:
                    continue
                    
                sig = sig_matrix[i, s_idx]
                if sig != 0:
                    position = float(sig)
                    entry_price = prices[i]
                    entry_idx = i
                    entry_atr = atr[i]
                    active_strat_idx = s_idx
                    
                    # Capture Params from active strategy
                    current_horizon = horizons[s_idx]
                    current_sl_mult = sl_mults[s_idx]
                    current_tp_mult = tp_mults[s_idx]
                    break 
                    
        # 3. PnL Calculation
        curr_price = prices[i]
        if barrier_price != 0.0:
            curr_price = barrier_price
            
        if i > 0:
            price_change = curr_price - prices[i-1]
            gross_pnl = prev_pos * lot_size * price_change
            
            pos_change = abs(position - prev_pos)
            cost = pos_change * lot_size * curr_price * (0.5 * spread_pct + comm_pct)
            
            net_rets = (gross_pnl - cost) / account_size
            net_returns[i] = net_rets
            
            if pos_change > 0:
                trades_count += 1
        else:
            # First bar
            pos_change = abs(position - 0.0)
            cost = pos_change * lot_size * curr_price * (0.5 * spread_pct + comm_pct)
            net_returns[i] = -cost / account_size
            if pos_change > 0:
                trades_count += 1
                
        prev_pos = position
        
    return net_returns, trades_count

def simulate_mutex_portfolio(subset_indices, sig_matrix, horizons, sl_mults, tp_mults, prices, highs, lows, atr, hours, weekdays, account_size):
    """
    Wrapper for JIT simulator. 
    subset_indices: List of column indices in sig_matrix to use for this portfolio.
    """
    # Slice parameters
    sub_sig = sig_matrix[:, subset_indices].astype(np.float64)
    sub_horizons = horizons[subset_indices]
    sub_sl = sl_mults[subset_indices]
    sub_tp = tp_mults[subset_indices]
    
    return _jit_simulate_mutex_custom(
        sub_sig,
        prices, highs, lows, atr, hours, weekdays,
        sub_horizons, sub_sl, sub_tp,
        config.STANDARD_LOT_SIZE,
        config.SPREAD_BPS / 10000.0,
        config.COST_BPS / 10000.0,
        account_size,
        config.TRADING_END_HOUR,
        config.STOP_LOSS_COOLDOWN_BARS
    )

# ==============================================================================
#  OPTIMIZATION LOGIC
# ==============================================================================

def optimize_mutex_portfolio(candidates, backtester):
    print("\n" + "="*80)
    print("🧠 MUTEX PORTFOLIO COMBINATORIAL OPTIMIZATION")
    print("="*80)
    
    # 1. Prepare Data (OOS Slice)
    oos_start = backtester.val_idx
    print(f"Optimization Range: OOS Index {oos_start} to {len(backtester.open_vec)} ({len(backtester.open_vec) - oos_start} bars)")
    
    prices = backtester.open_vec[oos_start:].astype(np.float64)
    highs = backtester.high_vec[oos_start:].astype(np.float64)
    lows = backtester.low_vec[oos_start:].astype(np.float64)
    atr = backtester.atr_vec[oos_start:].astype(np.float64)
    
    # Time
    if hasattr(backtester.times_vec, 'iloc'):
        times = backtester.times_vec.iloc[oos_start:]
    else:
        times = backtester.times_vec[oos_start:]
        
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values.astype(np.int8)
        weekdays = times.dt.dayofweek.values.astype(np.int8)
    else:
        dt_idx = pd.to_datetime(times)
        hours = dt_idx.hour.values.astype(np.int8)
        weekdays = dt_idx.dayofweek.values.astype(np.int8)

    # 2. Limit Candidates
    MAX_CANDIDATES = 14 # 2^14 = 16,384 combinations (Fast)
    if len(candidates) > MAX_CANDIDATES:
        print(f"⚠️  Too many candidates ({len(candidates)}). Truncating to top {MAX_CANDIDATES} by standalone Sortino.")
        candidates = candidates[:MAX_CANDIDATES]
    
    n_c = len(candidates)
    print(f"Evaluating {2**n_c - 1} combinations for {n_c} candidates...")
    
    # 3. Generate Master Signal Matrix
    # We must ensure Next-Open execution logic here
    print("  Generating signal matrix...")
    backtester.ensure_context(candidates)
    raw_sig_matrix = backtester.generate_signal_matrix(candidates)
    
    # Shift for Next-Open: Signal at T affects Position at T+1
    # We insert a row of zeros at the top and remove the last row
    shifted_sig_matrix = np.vstack([np.zeros((1, n_c), dtype=raw_sig_matrix.dtype), raw_sig_matrix[:-1]])
    
    # Slice OOS
    oos_sig_matrix = shifted_sig_matrix[oos_start:]
    
    # 4. Pre-extract Strategy Params (Arrays)
    horizons = np.array([s.horizon for s in candidates], dtype=np.int64)
    sl_mults = np.array([getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for s in candidates], dtype=np.float64)
    tp_mults = np.array([getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for s in candidates], dtype=np.float64)
    
    # 5. Combinatorial Search
    best_combo = None
    best_stats = {}
    
    # Trackers for Fallback (Best Sortino)
    fallback_best_sortino = -999.0
    fallback_combo = None
    fallback_stats = {}
    
    # Trackers for Primary Objective (Max Profit subject to Constraints)
    best_profit = -999.0
    
    # Constraints
    MIN_SORTINO = 2.0
    MAX_DD_LIMIT = -0.10 # -10%
    
    start_time = time.time()
    tested_count = 0
    valid_portfolios_found = 0
    
    # Iterate all subset sizes (from 1 to N)
    for r in range(1, n_c + 1):
        for combo_indices in itertools.combinations(range(n_c), r):
            # Convert tuple to list for indexing
            idx_list = list(combo_indices)
            
            # Run Simulation
            rets, trades = simulate_mutex_portfolio(
                idx_list, 
                oos_sig_matrix, 
                horizons, sl_mults, tp_mults, 
                prices, highs, lows, atr, hours, weekdays, 
                backtester.account_size
            )
            
            # Metrics
            total_ret = np.sum(rets)
            avg_ret = np.mean(rets)
            
            if avg_ret <= 0 or trades < 10:
                sortino = -1.0
                max_dd = -1.0
            else:
                downside = np.sqrt(np.mean(np.minimum(rets, 0)**2)) + 1e-9
                sortino = (avg_ret / downside) * np.sqrt(config.ANNUALIZATION_FACTOR)
                
                # Max DD
                cum_ret = np.cumsum(rets)
                peak = np.maximum.accumulate(cum_ret)
                dd_series = cum_ret - peak
                max_dd = np.min(dd_series)
            
            tested_count += 1
            
            profit = total_ret * backtester.account_size
            stats = {
                'Sortino': sortino,
                'Profit': profit,
                'Trades': trades,
                'MaxDD': max_dd
            }
            
            # 1. Update Fallback (Best Sortino)
            if sortino > fallback_best_sortino:
                fallback_best_sortino = sortino
                fallback_combo = [candidates[i] for i in idx_list]
                fallback_stats = stats
                
            # 2. Check Primary Constraints
            if sortino >= MIN_SORTINO and max_dd >= MAX_DD_LIMIT:
                valid_portfolios_found += 1
                # Maximize Profit
                if profit > best_profit:
                    best_profit = profit
                    best_combo = [candidates[i] for i in idx_list]
                    best_stats = stats
                
    elapsed = time.time() - start_time
    print(f"  Optimization complete in {elapsed:.2f}s ({tested_count} combos).")
    print(f"  Valid Portfolios (Sortino > {MIN_SORTINO}, MaxDD > {MAX_DD_LIMIT*100:.0f}%): {valid_portfolios_found}")
    
    if best_combo:
        print(f"\n🏆 WINNING COMBINATION FOUND (Max Profit Objective)!")
        print(f"  Strategies: {len(best_combo)}")
        print(f"  Profit:     ${best_stats['Profit']:,.2f}")
        print(f"  Sortino:    {best_stats['Sortino']:.2f}")
        print(f"  MaxDD:      {best_stats['MaxDD']*100:.2f}%")
        print(f"  Trades:     {int(best_stats['Trades'])}")
        
        print("\nComposition:")
        for s in best_combo:
            print(f"  - {s.name} (H{s.horizon})")
            
        return best_combo, best_stats
        
    elif fallback_combo:
        print(f"\n⚠️  No portfolio met strict criteria. Falling back to Best Sortino.")
        print(f"🏆 WINNING COMBINATION FOUND (Fallback Objective)!")
        print(f"  Strategies: {len(fallback_combo)}")
        print(f"  Sortino:    {fallback_stats['Sortino']:.2f}")
        print(f"  Profit:     ${fallback_stats['Profit']:,.2f}")
        print(f"  MaxDD:      {fallback_stats['MaxDD']*100:.2f}%")
        
        print("\nComposition:")
        for s in fallback_combo:
            print(f"  - {s.name} (H{s.horizon})")
            
        return fallback_combo, fallback_stats
        
    else:
        print("❌ No profitable combination found.")
        return [], {}

def run_mutex_backtest():
    # 1. Setup
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # 2. Load Candidates
    candidates = load_all_candidates()
    if not candidates:
        print("❌ No candidates found.")
        return

    # 3. Optimize Portfolio
    best_portfolio, stats = optimize_mutex_portfolio(candidates, backtester)
    
    if not best_portfolio:
        return

    # 4. Save Result
    mutex_data = []
    for s in best_portfolio:
        s_dict = s.to_dict()
        s_dict['horizon'] = s.horizon
        s_dict['training_id'] = getattr(s, 'training_id', 'legacy')
        mutex_data.append(s_dict)

    mutex_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
    with open(mutex_path, 'w') as f:
        json.dump(mutex_data, f, indent=4)
    print(f"\n💾 Saved Optimized Mutex Portfolio to {mutex_path}")
    
    # 5. Visual Validation (Re-run best for plotting)
    # We do this to get the equity curve vector
    oos_start = backtester.val_idx
    prices = backtester.open_vec[oos_start:].astype(np.float64)
    highs = backtester.high_vec[oos_start:].astype(np.float64)
    lows = backtester.low_vec[oos_start:].astype(np.float64)
    atr = backtester.atr_vec[oos_start:].astype(np.float64)
    times = backtester.times_vec.iloc[oos_start:] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[oos_start:]
    
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values.astype(np.int8)
        weekdays = times.dt.dayofweek.values.astype(np.int8)
    else:
        dt_idx = pd.to_datetime(times)
        hours = dt_idx.hour.values.astype(np.int8)
        weekdays = dt_idx.dayofweek.values.astype(np.int8)
        
    backtester.ensure_context(best_portfolio)
    raw_sig = backtester.generate_signal_matrix(best_portfolio)
    shifted_sig = np.vstack([np.zeros((1, len(best_portfolio)), dtype=raw_sig.dtype), raw_sig[:-1]])
    oos_sig = shifted_sig[oos_start:]
    
    horizons = np.array([s.horizon for s in best_portfolio], dtype=np.int64)
    sl_mults = np.array([getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for s in best_portfolio], dtype=np.float64)
    tp_mults = np.array([getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for s in best_portfolio], dtype=np.float64)
    
    # Use internal jit func for simplicity
    rets, _ = _jit_simulate_mutex_custom(
        oos_sig.astype(np.float64), 
        prices, highs, lows, atr, hours, weekdays, 
        horizons, sl_mults, tp_mults, 
        config.STANDARD_LOT_SIZE, 
        config.SPREAD_BPS / 10000.0, 
        config.COST_BPS / 10000.0, 
        config.ACCOUNT_SIZE, 
        config.TRADING_END_HOUR,
        config.STOP_LOSS_COOLDOWN_BARS
    )
    
    cum_profit = np.cumsum(rets) * config.ACCOUNT_SIZE
    
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(cum_profit)), cum_profit, label='Optimized Mutex', color='green', linewidth=2)
    plt.title(f"Optimized Mutex Portfolio (OOS)\nSortino: {stats['Sortino']:.2f} | Profit: ${stats['Profit']:,.0f} | Trades: {stats['Trades']}")
    plt.ylabel("Cumulative Profit ($)")
    plt.xlabel("Bars (OOS)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = os.path.join(config.DIRS['PLOTS_DIR'], "mutex_optimized.png")
    plt.savefig(out_path)
    print(f"📸 Saved Performance Chart to {out_path}")

if __name__ == "__main__":
    run_mutex_backtest()