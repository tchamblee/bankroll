import numpy as np
import pandas as pd
import itertools
import os
import json
import config
from backtest.statistics import calculate_sortino_ratio
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
    # CRITICAL: Shift ATR by 1 to prevent Look-Ahead Bias
    if len(atr) > 1:
        atr = np.roll(atr, 1)
        atr[0] = atr[1]

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
            
            strat_rets, _, _, _, _, _ = _jit_simulate_mutex_custom(
                sub_sig.astype(np.float64),
                prices, highs, lows, atr,
                hours, weekdays,
                sub_horizons, sub_sl, sub_tp,
                config.STANDARD_LOT_SIZE,
                config.SPREAD_BPS / 10000.0,
                config.COST_BPS / 10000.0,
                config.ACCOUNT_SIZE,
                config.TRADING_END_HOUR,
                config.STOP_LOSS_COOLDOWN_BARS,
                config.MIN_COMMISSION,
                config.SLIPPAGE_ATR_FACTOR,
                config.COMMISSION_THRESHOLD
            )            
            rets = np.sum(strat_rets, axis=1)
            total_ret = np.sum(rets)
            profit = total_ret * config.ACCOUNT_SIZE
            sortino = calculate_sortino_ratio(rets, config.ANNUALIZATION_FACTOR)
            
            # Constraint: No individual losers in the portfolio (Validation Set)
            strat_profits = np.sum(strat_rets, axis=0)
            no_losers = np.all(strat_profits > 0)
            
            if sortino > 1.0 and profit > best_profit and no_losers:
                best_profit = profit
                best_sortino = sortino
                best_combo = [top_candidates[i] for i in idxs]

    print(f"✅ Mutex Optimization Complete.")
    print(f"  Best Portfolio: {len(best_combo)} Strategies")
    print(f"  Val Profit: ${best_profit:,.2f}")
    print(f"  Val Sortino: {best_sortino:.2f}")
    
    return best_combo, {'profit': best_profit, 'sortino': best_sortino}

