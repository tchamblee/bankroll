import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import itertools
import time
from numba import jit
import scipy.stats
from scipy.optimize import minimize
import config
from genome import Strategy
from backtest import BacktestEngine
from backtest.statistics import deflated_sharpe_ratio, estimated_sharpe_ratio
from trade_simulator import TradeSimulator

# ... (Previous Helper Functions) ...


# --- JIT SIMULATOR FOR MUTEX PORTFOLIO ---
@jit(nopython=True, nogil=True, cache=True)
def _jit_simulate_mutex_custom(signals: np.ndarray, prices: np.ndarray, 
                               highs: np.ndarray, lows: np.ndarray, atr_vec: np.ndarray,
                               hours: np.ndarray, weekdays: np.ndarray,
                               horizons: np.ndarray, sl_mults: np.ndarray, tp_mults: np.ndarray,
                               lot_size: float, spread_pct: float, comm_pct: float, 
                               account_size: float, end_hour: int, cooldown_bars: int):
    """
    Simulates a portfolio of strategies running concurrently on one account.
    Each strategy manages its own position logic (Horizon, SL, TP).
    Returns net returns vector for the aggregated account.
    """
    n_bars, n_strats = signals.shape
    
    # State tracking per strategy
    # 0: Flat, 1: Long, -1: Short
    positions = np.zeros(n_strats, dtype=np.float64) 
    entry_prices = np.zeros(n_strats, dtype=np.float64)
    entry_indices = np.zeros(n_strats, dtype=np.int64)
    entry_atrs = np.zeros(n_strats, dtype=np.float64)
    
    # Cooldown tracking
    cooldowns = np.zeros(n_strats, dtype=np.int64)
    
    net_returns = np.zeros(n_bars, dtype=np.float64)
    
    # Loop over bars
    for i in range(1, n_bars):
        bar_pnl = 0.0
        
        # Check Force Close (EOD or Weekend)
        force_close = (hours[i] >= end_hour) or (weekdays[i] >= 5)
        
        # Loop over strategies
        for s in range(n_strats):
            # Manage Cooldown
            if cooldowns[s] > 0:
                cooldowns[s] -= 1
                
            curr_pos = positions[s]
            
            # --- 1. EXIT CHECKS (If in position) ---
            exit_signal = False
            is_sl_hit = False
            
            # Check Barriers from PREVIOUS bar (i-1) to avoid look-ahead
            # We entered at or before Open[i-1]. We need to see if i-1 killed us.
            if curr_pos != 0 and i > 0:
                # Check Time Limit (Current time i vs Entry)
                if (i - entry_indices[s]) >= horizons[s]:
                    exit_signal = True
                
                # Check SL/TP against PREVIOUS bar volatility
                if not exit_signal:
                    e_price = entry_prices[s]
                    sl_dist = entry_atrs[s] * sl_mults[s]
                    tp_dist = entry_atrs[s] * tp_mults[s]
                    
                    # Use i-1 for High/Low checks
                    if curr_pos > 0:
                        if lows[i-1] <= (e_price - sl_dist):
                            exit_signal = True
                            is_sl_hit = True
                        elif tp_mults[s] > 0 and highs[i-1] >= (e_price + tp_dist):
                            exit_signal = True
                    else:
                        if highs[i-1] >= (e_price + sl_dist):
                            exit_signal = True
                            is_sl_hit = True
                        elif tp_mults[s] > 0 and lows[i-1] <= (e_price - tp_dist):
                            exit_signal = True

                if force_close:
                    exit_signal = True
                    # Don't trigger cooldown on EOD
                    is_sl_hit = False 
            
            # --- 2. EXECUTION LOGIC ---
            target_pos = curr_pos
            
            if exit_signal:
                target_pos = 0.0
                if is_sl_hit:
                    cooldowns[s] = cooldown_bars
            elif curr_pos == 0:
                # Check Entry (Signal at i is for Open[i])
                sig = signals[i, s]
                if sig != 0 and cooldowns[s] == 0 and not force_close:
                    target_pos = float(sig)
            
            # --- 3. STATE UPDATE & COST ---
            if target_pos != curr_pos:
                price = prices[i]
                
                # Determine Execution Price
                exec_price = price # Default to Open[i]
                
                if exit_signal and curr_pos != 0:
                    # If barrier hit in i-1, we assume we exited AT the barrier (plus slippage later)
                    # or at Open[i] if it was a Time Exit.
                    
                    if is_sl_hit or (target_pos == 0 and not force_close and (i - entry_indices[s]) < horizons[s]):
                        # Recalculate barrier levels
                        e_price = entry_prices[s]
                        sl_dist = entry_atrs[s] * sl_mults[s]
                        tp_dist = entry_atrs[s] * tp_mults[s]
                        
                        if is_sl_hit:
                             if curr_pos > 0: exec_price = e_price - sl_dist
                             else: exec_price = e_price + sl_dist
                        else:
                            # TP Hit (Check i-1 again to determine price)
                            # Note: This logic duplicates the check, but necessary to set price
                            # Simplified: We trust the exit_signal flag.
                            if curr_pos > 0: exec_price = e_price + tp_dist
                            else: exec_price = e_price - tp_dist
                            
                    # If Time Exit or Force Close, use Open[i] (price)
                
                change = abs(target_pos - curr_pos)
                
                # Costs
                # Spread
                cost_spread = change * lot_size * exec_price * (0.5 * spread_pct)
                # Comm
                raw_comm = change * lot_size * exec_price * comm_pct
                comm = max(2.0, raw_comm) if change > 0.5 else 0.0 # Min comm per order
                # Slippage (10% ATR)
                slip = 0.1 * atr_vec[i] * lot_size * change
                
                total_cost = cost_spread + comm + slip
                
                # PnL from Trade Close
                if curr_pos != 0 and target_pos == 0:
                    trade_pnl = (exec_price - entry_prices[s]) * curr_pos * lot_size
                    bar_pnl += (trade_pnl - total_cost)
                elif curr_pos == 0 and target_pos != 0:
                    # Entry
                    bar_pnl -= total_cost
                    entry_prices[s] = exec_price
                    entry_indices[s] = i
                    entry_atrs[s] = atr_vec[i]
                
                positions[s] = target_pos
        
        net_returns[i] = bar_pnl / account_size

    return net_returns, positions.astype(np.int64), 0

def optimize_mutex_portfolio(candidates, backtester):
    print("\n" + "="*80)
    print("⚔️ MUTEX COMBINATORIAL OPTIMIZATION")
    print("="*80)
    
    if not candidates: return [], {}

    # 1. Filter Candidates (Top 14 by Robust Score)
    # We rely on 'robust_score' which comes from WFV
    candidates.sort(key=lambda x: getattr(x, 'fitness', -999), reverse=True)
    top_candidates = candidates[:14]
    print(f"  Selected top {len(top_candidates)} candidates for combinatorial search.")
    
    # 2. Prepare Data (Validation Set)
    val_start = backtester.train_idx
    val_end = backtester.val_idx
    
    backtester.ensure_context(top_candidates)
    raw_sig = backtester.generate_signal_matrix(top_candidates)
    # Shift signals for Next-Open
    shifted_sig = np.vstack([np.zeros((1, len(top_candidates)), dtype=raw_sig.dtype), raw_sig[:-1]])
    
    val_sig = shifted_sig[val_start:val_end]
    prices = backtester.open_vec[val_start:val_end].astype(np.float64)
    highs = backtester.high_vec[val_start:val_end].astype(np.float64)
    lows = backtester.low_vec[val_start:val_end].astype(np.float64)
    atr = backtester.atr_vec[val_start:val_end].astype(np.float64)
    times = backtester.times_vec.iloc[val_start:val_end] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[val_start:val_end]
    
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values.astype(np.int8)
        weekdays = times.dt.dayofweek.values.astype(np.int8)
    else:
        dt_idx = pd.to_datetime(times)
        hours = dt_idx.hour.values.astype(np.int8)
        weekdays = dt_idx.dayofweek.values.astype(np.int8)
        
    # Pre-extract params
    horizons = np.array([c.horizon for c in top_candidates], dtype=np.int64)
    sl_mults = np.array([getattr(c, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for c in top_candidates], dtype=np.float64)
    tp_mults = np.array([getattr(c, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for c in top_candidates], dtype=np.float64)
    
    best_combo = []
    best_profit = -99999.0
    best_sortino = 0.0
    
    # 3. Combinatorial Loop
    # We test combinations of size 1 to 5
    for r in range(1, 6):
        print(f"  Testing combinations of size {r}...")
        for indices in itertools.combinations(range(len(top_candidates)), r):
            idxs = np.array(indices)
            
            # Slice inputs
            sub_sig = val_sig[:, idxs]
            sub_horizons = horizons[idxs]
            sub_sl = sl_mults[idxs]
            sub_tp = tp_mults[idxs]
            
            # Run JIT Simulation
            rets, _, _ = _jit_simulate_mutex_custom(
                sub_sig.astype(np.float64), 
                prices, highs, lows, atr, 
                hours, weekdays, 
                sub_horizons, sub_sl, sub_tp,
                config.STANDARD_LOT_SIZE, 
                config.SPREAD_BPS / 10000.0, 
                config.COST_BPS / 10000.0, 
                config.ACCOUNT_SIZE, 
                config.TRADING_END_HOUR, 
                config.STOP_LOSS_COOLDOWN_BARS
            )
            
            # Metrics
            total_ret = np.sum(rets)
            profit = total_ret * config.ACCOUNT_SIZE
            avg_ret = np.mean(rets)
            downside = np.std(np.minimum(rets, 0)) + 1e-9
            sortino = (avg_ret / downside) * np.sqrt(config.ANNUALIZATION_FACTOR)
            
            # Constraints
            if sortino > 3.0 and profit > best_profit:
                best_profit = profit
                best_sortino = sortino
                best_combo = [top_candidates[i] for i in idxs]
                # print(f"    New Best: ${profit:.0f} (Sortino: {sortino:.2f}) -> {[c.name for c in best_combo]}")

    print(f"✅ Mutex Optimization Complete.")
    print(f"  Best Portfolio: {len(best_combo)} Strategies")
    print(f"  Val Profit: ${best_profit:,.2f}")
    print(f"  Val Sortino: {best_sortino:.2f}")
    
    return best_combo, {'profit': best_profit, 'sortino': best_sortino}

import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def get_ivp(cov):
    # Inverse Variance Portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov, c_items):
    # Calculate cluster variance
    cov_slice = cov.loc[c_items, c_items]
    w = get_ivp(cov_slice).reshape(-1, 1)
    c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
    return c_var

def get_quasi_diag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]) #.append(df0)
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_rec_bisection(cov, sort_ix):
    # Recursive Bisection Allocation
    w = pd.Series(1.0, index=sort_ix)
    c_items = [sort_ix]
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        for i in range(0, len(c_items), 2):
            c_items0 = c_items[i]
            c_items1 = c_items[i + 1]
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha
            w[c_items1] *= 1 - alpha
    return w

def optimize_hrp_portfolio(candidates, backtester):
    print("\n" + "="*80)
    print("🌳 HIERARCHICAL RISK PARITY (HRP) OPTIMIZATION")
    print("="*80)
    
    if len(candidates) < 2:
        print("Not enough candidates for HRP.")
        return

    # 1. Generate Returns Matrix (Train + Val)
    opt_start = 0
    opt_end = backtester.val_idx
    
    # Generate full signals first
    backtester.ensure_context(candidates)
    raw_sig = backtester.generate_signal_matrix(candidates)
    shifted_sig = np.vstack([np.zeros((1, len(candidates)), dtype=raw_sig.dtype), raw_sig[:-1]])
    opt_sig = shifted_sig[opt_start:opt_end]
    
    prices = backtester.open_vec[opt_start:opt_end].astype(np.float64)
    times = backtester.times_vec.iloc[opt_start:opt_end] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[opt_start:opt_end]
    highs = backtester.high_vec[opt_start:opt_end].astype(np.float64)
    lows = backtester.low_vec[opt_start:opt_end].astype(np.float64)
    atr = backtester.atr_vec[opt_start:opt_end].astype(np.float64)

    print("  Simulating individual streams for Correlation...")
    rets_matrix, _ = backtester.run_simulation_batch(
        opt_sig, candidates, prices, times, 
        time_limit=config.DEFAULT_TIME_LIMIT, 
        highs=highs, lows=lows, atr=atr
    )
    
    # Convert to DataFrame
    rets_df = pd.DataFrame(rets_matrix, columns=[c.name for c in candidates])
    
    # 2. HRP Steps
    # a. Correlation & Distance
    corr = rets_df.corr().fillna(0)
    dist = np.sqrt((1 - corr) / 2)
    
    # b. Linkage (Clustering)
    dist_condensed = squareform(dist.values)
    link = sch.linkage(dist_condensed, 'single')
    
    # c. Quasi-Diagonalization
    sort_ix = get_quasi_diag(link)
    sort_ix = list(map(int, sort_ix)) # Ensure ints
    
    # d. Recursive Bisection
    # Need Covariance for allocation
    cov = rets_df.cov()
    # Map sort_ix (indices) to column names for the function if needed, 
    # but our helper uses integer indices if cov indices are integers?
    # Our helper expects cov to be a DataFrame with indices matching sort_ix values?
    # Actually, get_cluster_var does `cov.loc[c_items, c_items]`.
    # So if sort_ix contains integers [0, 5, 2...], cov must have integer index/columns.
    # rets_df.cov() produces string index if rets_df has string columns.
    # Let's reset cov to integer index for simplicity
    cov_int = cov.reset_index(drop=True)
    cov_int.columns = range(len(cov_int.columns))
    
    hrp_weights = get_rec_bisection(cov_int, sort_ix)
    
    # Re-map weights to candidates
    # hrp_weights is a Series with index matching sort_ix (integers)
    # We just need to align them.
    weights = hrp_weights.sort_index().values
    
    # Filter low weights
    final_portfolio = []
    for i, w in enumerate(weights):
        if w > 0.005: # 0.5% min weight
            s_dict = candidates[i].to_dict()
            s_dict['weight'] = float(w)
            s_dict['horizon'] = candidates[i].horizon
            final_portfolio.append(s_dict)
    
    final_portfolio.sort(key=lambda x: x['weight'], reverse=True)
    
    # OOS Performance check
    oos_start = backtester.val_idx
    oos_sig = shifted_sig[oos_start:]
    oos_prices = backtester.open_vec[oos_start:].astype(np.float64)
    oos_times = backtester.times_vec.iloc[oos_start:] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[oos_start:]
    oos_highs = backtester.high_vec[oos_start:].astype(np.float64)
    oos_lows = backtester.low_vec[oos_start:].astype(np.float64)
    oos_atr = backtester.atr_vec[oos_start:].astype(np.float64)

    oos_rets_matrix, _ = backtester.run_simulation_batch(
        oos_sig, candidates, oos_prices, oos_times, 
        highs=oos_highs, lows=oos_lows, atr=oos_atr
    )
    
    oos_port_rets = np.dot(oos_rets_matrix, weights)
    oos_profit = np.sum(oos_port_rets) * config.ACCOUNT_SIZE
    oos_sortino = (np.mean(oos_port_rets) / (np.std(np.minimum(oos_port_rets, 0)) + 1e-9)) * np.sqrt(config.ANNUALIZATION_FACTOR)
    
    print(f"\n🏆 HRP PORTFOLIO RESULT")
    print(f"  Strategies: {len(final_portfolio)}")
    print(f"  OOS Profit: ${oos_profit:,.2f}")
    print(f"  OOS Sortino: {oos_sortino:.2f}")
    
    out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "hrp_portfolio.json")
    with open(out_path, 'w') as f:
        json.dump(final_portfolio, f, indent=4)
    print(f"💾 Saved to {out_path}")
    
    return final_portfolio

def load_all_candidates():
    candidates_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "candidates.json")
    if not os.path.exists(candidates_path):
        print(f"⚠️ Candidates file not found: {candidates_path}")
        return []

    with open(candidates_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Error decoding candidates.json")
            return []

    strategies = []
    for s_dict in data:
        try:
            strat = Strategy.from_dict(s_dict)
            # Hydrate Params
            strat.horizon = s_dict.get('horizon', config.DEFAULT_TIME_LIMIT)
            strat.stop_loss_pct = s_dict.get('stop_loss_pct', config.DEFAULT_STOP_LOSS)
            strat.take_profit_pct = s_dict.get('take_profit_pct', config.DEFAULT_TAKE_PROFIT)
            
            # Store metrics for sorting/ranking
            if 'test_stats' in s_dict:
                 strat.fitness = s_dict['test_stats'].get('sortino', 0)
            elif 'metrics' in s_dict:
                 strat.fitness = s_dict['metrics'].get('sortino_oos', 0)
            else:
                 strat.fitness = 0.0
            
            strategies.append(strat)
        except Exception as e:
            print(f"⚠️ Error loading strategy {s_dict.get('name', 'Unknown')}: {e}")
            continue
            
    print(f"✅ Loaded {len(strategies)} candidates from {candidates_path}")
    return strategies

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

    # 3. Optimize Portfolio (Mutex)
    best_portfolio, stats = optimize_mutex_portfolio(candidates, backtester)
    
    if best_portfolio:
        # Save Mutex Result
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
        
        # Visual Validation (Re-run best for plotting)
        # ... (Existing Plotting Logic) ...
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
        
        rets, _, _ = _jit_simulate_mutex_custom(
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
        total_oos_profit = cum_profit[-1] if len(cum_profit) > 0 else 0.0
        
        avg_oos = np.mean(rets)
        downside_oos = np.sqrt(np.mean(np.minimum(rets, 0)**2)) + 1e-9
        sortino_oos = (avg_oos / downside_oos) * np.sqrt(config.ANNUALIZATION_FACTOR)
        
        print("\n" + "="*80)
        print("🧪 OUT-OF-SAMPLE (TEST SET) PERFORMANCE")
        print("="*80)
        print(f"  Range:      Index {oos_start} to End ({len(rets)} bars)")
        print(f"  Profit:     ${total_oos_profit:,.2f}")
        print(f"  Sortino:    {sortino_oos:.2f}")
        print(f"  Return:     {(total_oos_profit/config.ACCOUNT_SIZE)*100:.1f}%")
        print("="*80 + "\n")

        plt.figure(figsize=(15, 8))
        plt.plot(range(len(cum_profit)), cum_profit, label='Optimized Mutex', color='green', linewidth=2)
        plt.title(f"Optimized Mutex Portfolio (OOS)\nSortino: {sortino_oos:.2f} | Profit: ${total_oos_profit:,.0f}")
        plt.ylabel("Cumulative Profit ($)")
        plt.xlabel("Bars (OOS)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        out_path = os.path.join(config.DIRS['PLOTS_DIR'], "mutex_optimized.png")
        plt.savefig(out_path)
        print(f"📸 Saved Performance Chart to {out_path}")
    
    # 4. Optimize HRP Portfolio (Robust)
    optimize_hrp_portfolio(candidates[:100], backtester)

if __name__ == "__main__":
    run_mutex_backtest()