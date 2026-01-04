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
    print("⚔️ MUTEX COMBINATORIAL OPTIMIZATION (FULL PIPELINE)")
    print("="*80)
    
    if not candidates: return [], {}

    # 1. Filter Candidates
    # We still use the robust score to pick the top pool, as checking 2^N on 100+ strats is impossible.
    def get_score(s):
        if hasattr(s, 'robust_score'): return s.robust_score
        return 0

    candidates.sort(key=get_score, reverse=True)
    top_candidates = candidates[:25]
    print(f"  Selected top {len(top_candidates)} candidates for combinatorial search.")
    
    # 2. Prepare Data for ALL Sets
    # Indices
    train_end = backtester.train_idx
    val_end = backtester.val_idx
    
    # Pre-generate Matrix for all candidates
    backtester.ensure_context(top_candidates)
    raw_sig = backtester.generate_signal_matrix(top_candidates)
    
    # Common Vectors
    full_prices = backtester.open_vec.astype(np.float64)
    full_highs = backtester.high_vec.astype(np.float64)
    full_lows = backtester.low_vec.astype(np.float64)
    full_atr = backtester.atr_vec.astype(np.float64)
    
    # Shift ATR
    if len(full_atr) > 1:
        full_atr = np.roll(full_atr, 1)
        full_atr[0] = full_atr[1]
        
    # Time Features
    times = backtester.times_vec
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values.astype(np.int8)
        weekdays = times.dt.dayofweek.values.astype(np.int8)
    else:
        dt_idx = pd.to_datetime(times)
        hours = dt_idx.hour.values.astype(np.int8)
        weekdays = dt_idx.dayofweek.values.astype(np.int8)
        
    # Helper to slice
    def get_slice(start, end):
        # Signal Shift logic: 
        # raw_sig[i] is signal generated at Close[i]. Execution is Open[i+1].
        # So Signal[i] aligns with Open[i+1].
        # The simulator expects `signals` aligned with `prices` (Open[t]).
        # So we need shifted_sig[t] = raw_sig[t-1].
        
        # Construct full shifted matrix first?
        # shift_sig = np.vstack([zeros, raw_sig[:-1]])
        # slice = shift_sig[start:end]
        
        # Optimization: Slice raw first then shift? No, boundary conditions.
        # Let's shift the whole thing once.
        return start, end

    # Full Shifted Signal Matrix
    shifted_sig = np.vstack([np.zeros((1, len(top_candidates)), dtype=raw_sig.dtype), raw_sig[:-1]])

    # Define Sets
    sets = {
        'Train': (0, train_end),
        'Val':   (train_end, val_end),
        'Test':  (val_end, len(full_prices))
    }
    
    # Extract Params
    horizons = np.array([c.horizon for c in top_candidates], dtype=np.int64)
    sl_mults = np.array([getattr(c, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for c in top_candidates], dtype=np.float64)
    tp_mults = np.array([getattr(c, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for c in top_candidates], dtype=np.float64)

    best_combo = []
    best_total_profit = -99999.0
    best_stats = {}
    
    # 3. Combinatorial Loop
    # We limit combination size to 5 for speed/complexity
    for r in range(1, 6):
        print(f"  Testing combinations of size {r}...")
        for indices in itertools.combinations(range(len(top_candidates)), r):
            idxs = np.array(indices)
            
            # Subset Params
            sub_horizons = horizons[idxs]
            sub_sl = sl_mults[idxs]
            sub_tp = tp_mults[idxs]
            
            # Evaluate on ALL Sets
            set_results = {}
            valid_combo = True
            
            for set_name, (s_start, s_end) in sets.items():
                sub_sig = shifted_sig[s_start:s_end][:, idxs]
                
                # Check for empty set (e.g. if backtest was short)
                if len(sub_sig) == 0: continue

                strat_rets, _, _, _, _, _ = _jit_simulate_mutex_custom(
                    sub_sig.astype(np.float64),
                    full_prices[s_start:s_end], full_highs[s_start:s_end], full_lows[s_start:s_end], full_atr[s_start:s_end],
                    hours[s_start:s_end], weekdays[s_start:s_end],
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
                
                # 1. Check Constraint: No Individual Losers
                strat_profits = np.sum(strat_rets, axis=0) * config.ACCOUNT_SIZE
                if np.any(strat_profits < 0):
                    valid_combo = False
                    break # Fail fast
                
                # Aggregate
                rets = np.sum(strat_rets, axis=1)
                total_profit = np.sum(rets) * config.ACCOUNT_SIZE
                sortino = calculate_sortino_ratio(rets, config.ANNUALIZATION_FACTOR)
                
                set_results[set_name] = {
                    'profit': total_profit,
                    'sortino': sortino
                }

            if valid_combo:
                # Calculate Global Metrics
                total_profit = sum(r['profit'] for r in set_results.values())
                avg_sortino = sum(r['sortino'] for r in set_results.values()) / 3.0
                
                # Filter: Must have decent avg Sortino
                if avg_sortino > 1.0:
                    if total_profit > best_total_profit:
                        best_total_profit = total_profit
                        best_combo = [top_candidates[i] for i in idxs]
                        best_stats = set_results
                        best_stats['avg_sortino'] = avg_sortino
                        best_stats['total_profit'] = total_profit

    print(f"✅ Mutex Optimization Complete.")
    
    if best_combo:
        print(f"  Best Portfolio: {len(best_combo)} Strategies")
        print(f"  Total Profit:   ${best_stats['total_profit']:,.2f}")
        print(f"  Avg Sortino:    {best_stats['avg_sortino']:.2f}")
        print(f"  Breakdown:")
        print(f"    Train: ${best_stats['Train']['profit']:,.0f} (Sort: {best_stats['Train']['sortino']:.2f})")
        print(f"    Val:   ${best_stats['Val']['profit']:,.0f} (Sort: {best_stats['Val']['sortino']:.2f})")
        print(f"    Test:  ${best_stats['Test']['profit']:,.0f} (Sort: {best_stats['Test']['sortino']:.2f})")
    else:
        print("❌ No valid portfolio found that satisfies all constraints (No Losers in Train/Val/Test).")

    return best_combo, best_stats

