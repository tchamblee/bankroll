import os
import json
import pandas as pd
import numpy as np
import config
from genome import Strategy


def get_barrier_params(strategy):
    """
    Extract direction-specific SL/TP parameters from a single strategy.

    Returns dict with keys: sl_long, sl_short, tp_long, tp_short

    Falls back to symmetric base values (stop_loss_pct, take_profit_pct)
    if direction-specific values are not available.
    """
    base_sl = getattr(strategy, 'stop_loss_pct', config.DEFAULT_STOP_LOSS)
    base_tp = getattr(strategy, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT)

    if hasattr(strategy, 'get_effective_sl'):
        sl_long = strategy.get_effective_sl('long')
        sl_short = strategy.get_effective_sl('short')
    else:
        sl_long = getattr(strategy, 'sl_long', base_sl)
        sl_short = getattr(strategy, 'sl_short', base_sl)

    if hasattr(strategy, 'get_effective_tp'):
        tp_long = strategy.get_effective_tp('long')
        tp_short = strategy.get_effective_tp('short')
    else:
        tp_long = getattr(strategy, 'tp_long', base_tp)
        tp_short = getattr(strategy, 'tp_short', base_tp)

    return {
        'sl_long': sl_long,
        'sl_short': sl_short,
        'tp_long': tp_long,
        'tp_short': tp_short,
    }


def extract_barrier_params(strategies):
    """
    Extract direction-specific SL/TP parameters from a list of strategies.

    Returns numpy arrays for use in simulation:
        sl_longs, sl_shorts, tp_longs, tp_shorts
    """
    params = [get_barrier_params(s) for s in strategies]
    sl_longs = np.array([p['sl_long'] for p in params], dtype=np.float64)
    sl_shorts = np.array([p['sl_short'] for p in params], dtype=np.float64)
    tp_longs = np.array([p['tp_long'] for p in params], dtype=np.float64)
    tp_shorts = np.array([p['tp_short'] for p in params], dtype=np.float64)
    return sl_longs, sl_shorts, tp_longs, tp_shorts

def find_strategy_in_files(strategy_name):
    """Searches all strategy output files for a strategy with the given name."""
    search_patterns = [
        "found_strategies.json",
        "candidates.json",
        "apex_strategies_*_top5_unique.json",
        "apex_strategies_*_top10.json",
        "apex_strategies_*.json",
        "optimized_*.json"
    ]
    
    import glob
    for pattern in search_patterns:
        files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], pattern))
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for s_dict in data:
                        if s_dict.get('name') == strategy_name:
                            # Found it!
                            # Ensure horizon is set if missing (infer from filename)
                            if 'horizon' not in s_dict:
                                # extracting horizon from filename e.g., apex_strategies_90_...
                                parts = os.path.basename(file_path).split('_')
                                for p in parts:
                                    if p.isdigit():
                                        s_dict['horizon'] = int(p)
                                        break
                            
                            # Ensure metrics are present
                            if 'metrics' not in s_dict:
                                s_dict['metrics'] = {
                                    'sortino_oos': s_dict.get('test_sortino', 0),
                                    'robust_return': s_dict.get('test_return', s_dict.get('robust_return', 0))
                                }
                                
                            return s_dict
            except Exception:
                continue
    return None

def refresh_strategies(strategies_data, purge_failing=False):
    """
    Re-simulates strategies to ensure fresh performance metrics.
    Updates the dictionaries in-place and returns the list.

    If purge_failing=True, removes strategies that no longer pass quality thresholds.
    """
    from .engine import BacktestEngine
    if not strategies_data:
        return []

    print(f"🔄 Refreshing metrics for {len(strategies_data)} strategies...")
    
    # Load Data
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("⚠️  Feature Matrix not found. Cannot refresh metrics.")
        return strategies_data
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])

    # Apply time filter (same as evolution engine)
    if hasattr(config, 'TRAIN_START_DATE') and config.TRAIN_START_DATE:
        if 'time_start' in df.columns:
            # Ensure datetime compatibility
            if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
                df['time_start'] = pd.to_datetime(df['time_start'], utc=True)

            # Handle Timezone
            ts_col = df['time_start']
            if ts_col.dt.tz is None:
                ts_col = ts_col.dt.tz_localize('UTC')
            else:
                ts_col = ts_col.dt.tz_convert('UTC')

            start_ts = pd.Timestamp(config.TRAIN_START_DATE).tz_localize('UTC')

            if ts_col.min() < start_ts:
                df = df[ts_col >= start_ts].reset_index(drop=True)

    engine = BacktestEngine(df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # Convert dicts to Strategy objects
    strat_objects = []
    
    # Pre-clean strategies_data to ensure valid dicts
    # We will map by name
    
    for s_data in strategies_data:
        try:
            strat = Strategy.from_dict(s_data)
            # Ensure horizon is set
            strat.horizon = s_data.get('horizon', config.DEFAULT_TIME_LIMIT)
            strat_objects.append(strat)
        except Exception as e:
            print(f"⚠️  Skipping invalid strategy {s_data.get('name', '?')}: {e}")

    if not strat_objects:
        return strategies_data

    # Group by horizon for batch processing
    from collections import defaultdict
    by_horizon = defaultdict(list)
    for s in strat_objects:
        h = getattr(s, 'horizon', config.DEFAULT_TIME_LIMIT)
        by_horizon[h].append(s)
        
    for h, group in by_horizon.items():
        # Test (OOS)
        stats_test, net_returns_matrix = engine.evaluate_population(
            group, set_type='test', return_series=True, time_limit=h
        )
        
        # Train
        stats_train = engine.evaluate_population(group, set_type='train', time_limit=h)
        
        # Validation
        stats_val = engine.evaluate_population(group, set_type='validation', time_limit=h)

        # CPCV (Combinatorial Purged Cross-Validation)
        stats_cpcv = engine.evaluate_combinatorial_purged_cv(group, time_limit=h)

        # Create lookup maps
        test_map = {row['id']: row for _, row in stats_test.iterrows()}
        train_map = {row['id']: row for _, row in stats_train.iterrows()}
        val_map = {row['id']: row for _, row in stats_val.iterrows()}
        cpcv_map = {row['id']: row for _, row in stats_cpcv.iterrows()}
        
        # Update original data dicts
        for s in group:
            # Find the original dict object
            target_dict = next((d for d in strategies_data if d.get('name') == s.name), None)
            if not target_dict: continue
            
            if s.name in test_map and s.name in train_map and s.name in val_map:
                t_stats = train_map[s.name]
                v_stats = val_map[s.name]
                te_stats = test_map[s.name]
                
                # --- Update Flat Keys (Inbox Style) ---
                target_dict['train_return'] = float(t_stats['total_return'])
                target_dict['val_return'] = float(v_stats['total_return'])
                target_dict['test_return'] = float(te_stats['total_return'])
                
                target_dict['train_sortino'] = float(t_stats['sortino'])
                target_dict['val_sortino'] = float(v_stats['sortino'])
                target_dict['test_sortino'] = float(te_stats['sortino'])
                
                target_dict['test_trades'] = int(te_stats['trades'])
                target_dict['train_trades'] = int(t_stats['trades'])
                target_dict['val_trades'] = int(v_stats['trades'])
                
                # Additional Helper for Reporting
                returns = net_returns_matrix[:, group.index(s)] # Need correct index in group
                # Wait, 'net_returns_matrix' columns correspond to 'group' list order.
                # 'group' list order is preserved.
                
                # Calculate Sharpe/Drawdown if possible (fast approx)
                # But evaluate_population doesn't return full series by default unless return_series=True
                # We have return_series=True for Test.
                
                # Calculate Sharpe/DD for Test
                test_ret_vec = returns # already extracted column? No.
                # net_returns_matrix is (bars x strategies)
                test_ret_vec = net_returns_matrix[:, group.index(s)]
                
                ann_factor = config.ANNUALIZATION_FACTOR
                avg_ret = np.mean(test_ret_vec)
                std_ret = np.std(test_ret_vec)
                sharpe = (avg_ret * ann_factor) / (std_ret * np.sqrt(ann_factor) + 1e-9)
                
                cum_pnl = np.cumsum(test_ret_vec)
                peak = np.maximum.accumulate(cum_pnl)
                dd = cum_pnl - peak
                max_dd = np.min(dd) # Simple PnL drawdown (approx)
                
                target_dict['test_sharpe'] = float(sharpe)
                target_dict['max_drawdown'] = float(max_dd)
                target_dict['test_ann_return'] = float(avg_ret * ann_factor)

                # --- Update Structured Stats (Candidate/Optimizer Style) ---
                target_dict['train_stats'] = {'ret': float(t_stats['total_return']), 'sortino': float(t_stats['sortino']), 'trades': int(t_stats['trades'])}
                target_dict['val_stats'] = {'ret': float(v_stats['total_return']), 'sortino': float(v_stats['sortino']), 'trades': int(v_stats['trades'])}
                target_dict['test_stats'] = {'ret': float(te_stats['total_return']), 'sortino': float(te_stats['sortino']), 'trades': int(te_stats['trades'])}
                
                # --- Legacy/Report fields ---
                target_dict['metrics'] = {
                    'sortino_oos': float(te_stats['sortino']),
                    'robust_return': float(te_stats['total_return'])
                }

                # --- CPCV Stats ---
                if s.name in cpcv_map:
                    cpcv_stats = cpcv_map[s.name]
                    target_dict['cpcv_p5'] = float(cpcv_stats['cpcv_p5_sortino'])
                    target_dict['cpcv_min'] = float(cpcv_stats['cpcv_min'])

    engine.shutdown()
    print("✅ Metrics refreshed.")

    # Optionally purge strategies that no longer pass quality thresholds
    if purge_failing:
        min_test_sortino = config.MIN_TEST_SORTINO
        min_val_sortino = getattr(config, 'MIN_VAL_SORTINO', config.MIN_SORTINO_THRESHOLD)
        min_cpcv = config.MIN_CPCV_THRESHOLD
        max_decay = config.MAX_TRAIN_TEST_DECAY

        passing = []
        purged = []

        for s in strategies_data:
            test_sort = s.get('test_sortino', 0)
            val_sort = s.get('val_sortino', 0)
            cpcv = s.get('cpcv_min', 0)
            train_ret = s.get('train_return', 0)
            test_ret = s.get('test_return', 0)
            decay = 1 - (test_ret / train_ret) if train_ret > 0 else 1.0

            fails = []
            if test_sort < min_test_sortino: fails.append(f"test_sort={test_sort:.2f}")
            if val_sort < min_val_sortino: fails.append(f"val_sort={val_sort:.2f}")
            if cpcv < min_cpcv: fails.append(f"cpcv={cpcv:.2f}")
            if decay > max_decay: fails.append(f"decay={decay:.0%}")

            if fails:
                purged.append((s.get('name', '?'), fails))
            else:
                passing.append(s)

        if purged:
            print(f"🗑️  Purged {len(purged)} strategies that no longer pass thresholds:")
            for name, reasons in purged[:5]:
                print(f"   - {name}: {', '.join(reasons)}")
            if len(purged) > 5:
                print(f"   ... and {len(purged) - 5} more")

        return passing

    return strategies_data

def check_direction_consistency(engine, strategy, horizon=None):
    """
    Checks if a strategy's long and short sides are consistent between train and test.

    A direction is considered "flipped" if:
    - It was profitable (>0) in train but unprofitable (<0) in test, or vice versa
    - Has at least MIN_TRADES_FOR_CHECK trades in both periods

    Returns:
        dict with keys:
            - 'consistent': bool - True if both directions are consistent
            - 'long_flipped': bool - True if long direction flipped
            - 'short_flipped': bool - True if short direction flipped
            - 'long_train_ret': float - Long return in train
            - 'long_test_ret': float - Long return in test
            - 'short_train_ret': float - Short return in train
            - 'short_test_ret': float - Short return in test
            - 'warning': str or None - Description of inconsistency
    """
    MIN_TRADES_FOR_CHECK = 5  # Need at least 5 trades to consider a direction active

    h = horizon if horizon else getattr(strategy, 'horizon', config.DEFAULT_TIME_LIMIT)

    # Generate signals for the full dataset
    signals = engine.generate_signal_matrix([strategy], horizon=h)

    # Shift signals by 1 to enforce next-open execution
    signals = np.roll(signals, 1, axis=0)
    signals[0] = 0

    # Split signals into long-only and short-only
    signals_long = signals.copy()
    signals_long[signals_long < 0] = 0

    signals_short = signals.copy()
    signals_short[signals_short > 0] = 0

    results = {
        'consistent': True,
        'long_flipped': False,
        'short_flipped': False,
        'long_train_ret': 0.0,
        'long_test_ret': 0.0,
        'short_train_ret': 0.0,
        'short_test_ret': 0.0,
        'long_train_trades': 0,
        'long_test_trades': 0,
        'short_train_trades': 0,
        'short_test_trades': 0,
        'warning': None
    }

    # Helper to run simulation on a segment
    def run_segment(sig_matrix, start_idx, end_idx):
        seg_slice = slice(start_idx, end_idx)
        signals_seg = sig_matrix[seg_slice]
        prices_seg = engine.open_vec[seg_slice]
        times_seg = engine.times_vec[seg_slice]
        highs_seg = engine.high_vec[seg_slice]
        lows_seg = engine.low_vec[seg_slice]
        atr_seg = engine.atr_vec[seg_slice] if engine.atr_vec is not None else None

        ret, trades = engine.run_simulation_batch(
            signals_seg, [strategy], prices_seg, times_seg,
            time_limit=h, highs=highs_seg, lows=lows_seg, atr=atr_seg
        )
        return np.sum(ret[:, 0]), int(trades[0])

    # Run on train
    long_train_ret, long_train_trades = run_segment(signals_long, 0, engine.train_idx)
    short_train_ret, short_train_trades = run_segment(signals_short, 0, engine.train_idx)

    # Run on test
    long_test_ret, long_test_trades = run_segment(signals_long, engine.val_idx, len(engine.close_vec))
    short_test_ret, short_test_trades = run_segment(signals_short, engine.val_idx, len(engine.close_vec))

    results['long_train_ret'] = long_train_ret
    results['long_test_ret'] = long_test_ret
    results['short_train_ret'] = short_train_ret
    results['short_test_ret'] = short_test_ret
    results['long_train_trades'] = long_train_trades
    results['long_test_trades'] = long_test_trades
    results['short_train_trades'] = short_train_trades
    results['short_test_trades'] = short_test_trades

    warnings = []

    # Check long direction consistency (only if enough trades in both periods)
    if long_train_trades >= MIN_TRADES_FOR_CHECK and long_test_trades >= MIN_TRADES_FOR_CHECK:
        # Check for sign flip
        if (long_train_ret > 0 and long_test_ret < 0) or (long_train_ret < 0 and long_test_ret > 0):
            results['long_flipped'] = True
            results['consistent'] = False
            warnings.append(f"LONG flipped: train={long_train_ret:.2%} -> test={long_test_ret:.2%}")

    # Check short direction consistency (only if enough trades in both periods)
    if short_train_trades >= MIN_TRADES_FOR_CHECK and short_test_trades >= MIN_TRADES_FOR_CHECK:
        # Check for sign flip
        if (short_train_ret > 0 and short_test_ret < 0) or (short_train_ret < 0 and short_test_ret > 0):
            results['short_flipped'] = True
            results['consistent'] = False
            warnings.append(f"SHORT flipped: train={short_train_ret:.2%} -> test={short_test_ret:.2%}")

    if warnings:
        results['warning'] = "; ".join(warnings)

    return results


def prepare_simulation_data(prices, highs=None, lows=None, atr=None):
    """
    Prepares data vectors for simulation, handling ATR fallback and Look-Ahead prevention.
    Returns: atr_vec
    """
    # ATR Fallback
    if atr is None:
        print(f"⚠️  WARNING: ATR data missing. Using fallback {config.ATR_FALLBACK_BPS} bps.")
        if highs is not None and lows is not None:
            # Simple Range if ATR missing
            min_atr = prices * (config.MIN_ATR_BPS / 10000.0)
            atr_vec = np.maximum(highs - lows, min_atr)
        else:
            atr_vec = prices * (config.ATR_FALLBACK_BPS / 10000.0) 
    else:
        atr_vec = atr.astype(np.float64)
    
    # Enforce Floor (MIN_ATR_BPS)
    min_atr = prices * (config.MIN_ATR_BPS / 10000.0)
    atr_vec = np.maximum(atr_vec, min_atr)

    # CRITICAL: Shift ATR by 1 to prevent Look-Ahead Bias
    # Simulator executes at Open[t], so it must use Volatility[t-1]
    if len(atr_vec) > 1:
        atr_vec = np.roll(atr_vec, 1)
        atr_vec[0] = atr_vec[1] # Backfill first element
        
    return atr_vec