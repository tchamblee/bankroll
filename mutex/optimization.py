import numpy as np
import pandas as pd
import itertools
import os
import json
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import config
from .simulator import _jit_simulate_mutex_custom

def optimize_mutex_portfolio(candidates, backtester):
    print("\n" + "="*80)
    print("⚔️ MUTEX COMBINATORIAL OPTIMIZATION")
    print("="*80)
    
    if not candidates: return [], {}

    # 1. Filter Candidates (Top 14 by Robust Score)
    candidates.sort(key=lambda x: getattr(x, 'fitness', -999), reverse=True)
    top_candidates = candidates[:14]
    print(f"  Selected top {len(top_candidates)} candidates for combinatorial search.")
    
    # 2. Prepare Data (Validation Set)
    val_start = backtester.train_idx
    val_end = backtester.val_idx
    
    backtester.ensure_context(top_candidates)
    raw_sig = backtester.generate_signal_matrix(top_candidates)
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
        
    horizons = np.array([c.horizon for c in top_candidates], dtype=np.int64)
    sl_mults = np.array([getattr(c, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for c in top_candidates], dtype=np.float64)
    tp_mults = np.array([getattr(c, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for c in top_candidates], dtype=np.float64)
    
    best_combo = []
    best_profit = -99999.0
    best_sortino = 0.0
    
    # 3. Combinatorial Loop
    for r in range(1, 6):
        print(f"  Testing combinations of size {r}...")
        for indices in itertools.combinations(range(len(top_candidates)), r):
            idxs = np.array(indices)
            
            sub_sig = val_sig[:, idxs]
            sub_horizons = horizons[idxs]
            sub_sl = sl_mults[idxs]
            sub_tp = tp_mults[idxs]
            
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
            
            total_ret = np.sum(rets)
            profit = total_ret * config.ACCOUNT_SIZE
            avg_ret = np.mean(rets)
            downside = np.std(np.minimum(rets, 0)) + 1e-9
            sortino = (avg_ret / downside) * np.sqrt(config.ANNUALIZATION_FACTOR)
            
            if sortino > 1.0 and profit > best_profit:
                best_profit = profit
                best_sortino = sortino
                best_combo = [top_candidates[i] for i in idxs]

    print(f"✅ Mutex Optimization Complete.")
    print(f"  Best Portfolio: {len(best_combo)} Strategies")
    print(f"  Val Profit: ${best_profit:,.2f}")
    print(f"  Val Sortino: {best_sortino:.2f}")
    
    return best_combo, {'profit': best_profit, 'sortino': best_sortino}

# --- HRP HELPERS ---

def get_ivp(cov):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov, c_items):
    cov_slice = cov.loc[c_items, c_items]
    w = get_ivp(cov_slice).reshape(-1, 1)
    c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
    return c_var

def get_quasi_diag(link):
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
        sort_ix = pd.concat([sort_ix, df0]) 
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()

def get_rec_bisection(cov, sort_ix):
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
    
    rets_df = pd.DataFrame(rets_matrix, columns=[c.name for c in candidates])
    
    # 2. HRP Steps
    corr = rets_df.corr().fillna(0)
    dist = np.sqrt((1 - corr) / 2)
    dist_condensed = squareform(dist.values)
    link = sch.linkage(dist_condensed, 'single')
    
    sort_ix = get_quasi_diag(link)
    sort_ix = list(map(int, sort_ix)) 
    
    cov = rets_df.cov()
    cov_int = cov.reset_index(drop=True)
    cov_int.columns = range(len(cov_int.columns))
    
    hrp_weights = get_rec_bisection(cov_int, sort_ix)
    weights = hrp_weights.sort_index().values
    
    final_portfolio = []
    for i, w in enumerate(weights):
        if w > 0.005: 
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
