import sys
import os
import json
import argparse
import config
import pandas as pd
import numpy as np
from genome import Strategy
from backtest.engine import BacktestEngine
from backtest.utils import refresh_strategies, find_strategy_in_files, check_direction_consistency, prepare_simulation_data, extract_barrier_params
from backtest.reporting import print_candidate_table, get_min_sortino, get_cpcv_p5, get_cpcv_min
from backtest.statistics import calculate_sortino_ratio
from mutex.simulator import _jit_simulate_mutex_custom

import glob

CANDIDATES_FILE = config.CANDIDATES_FILE

# Cache for backtester to avoid reloading data
_cached_engine = None

def _get_engine():
    """Get or create a cached BacktestEngine instance."""
    global _cached_engine
    if _cached_engine is None:
        if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
            raise RuntimeError("Feature Matrix not found")
        df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
        _cached_engine = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
    return _cached_engine

def check_standalone_profitability(strategy_data, verbose=True):
    """
    Check if a strategy is profitable STANDALONE using the actual mutex simulator.
    This is the same simulator used for live trading, with stops, costs, and position sizing.

    Returns: (passes, results_dict) where results_dict contains Train/Val/Test profits and sortinos
    """
    engine = _get_engine()

    strat = Strategy.from_dict(strategy_data)
    strat.horizon = strategy_data.get('horizon', 60)

    # Data slices
    train_end = engine.train_idx
    val_end = engine.val_idx
    n = len(engine.open_vec)

    sets = {
        'Train': (0, train_end),
        'Val': (train_end, val_end),
        'Test': (val_end, n)
    }

    # Prepare data
    full_prices = engine.open_vec.astype(np.float64)
    full_highs = engine.high_vec.astype(np.float64)
    full_lows = engine.low_vec.astype(np.float64)
    full_atr = engine.atr_vec.astype(np.float64)

    if len(full_atr) > 1:
        full_atr = np.roll(full_atr, 1)
        full_atr[0] = full_atr[1]

    times = engine.times_vec
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values.astype(np.int8)
        weekdays = times.dt.dayofweek.values.astype(np.int8)
    else:
        dt_idx = pd.to_datetime(times)
        hours = dt_idx.hour.values.astype(np.int8)
        weekdays = dt_idx.dayofweek.values.astype(np.int8)

    # Generate signals
    engine.ensure_context([strat])
    raw_sig = engine.generate_signal_matrix([strat])
    shifted_sig = np.vstack([np.zeros((1, 1), dtype=raw_sig.dtype), raw_sig[:-1]])

    horizons = np.array([strat.horizon], dtype=np.int64)
    sl_longs, sl_shorts, tp_longs, tp_shorts = extract_barrier_params([strat])
    target_risk_dollars = config.ACCOUNT_SIZE * config.RISK_PER_TRADE_PERCENT

    results = {}

    for set_name, (s_start, s_end) in sets.items():
        sub_sig = shifted_sig[s_start:s_end]

        strat_rets, strat_trades, _, _, _, _ = _jit_simulate_mutex_custom(
            sub_sig.astype(np.float64),
            full_prices[s_start:s_end], full_highs[s_start:s_end],
            full_lows[s_start:s_end], full_atr[s_start:s_end],
            hours[s_start:s_end], weekdays[s_start:s_end],
            horizons,
            sl_longs, sl_shorts, tp_longs, tp_shorts,
            config.STANDARD_LOT_SIZE,
            config.SPREAD_BPS / 10000.0,
            config.COST_BPS / 10000.0,
            config.ACCOUNT_SIZE,
            config.TRADING_END_HOUR,
            config.STOP_LOSS_COOLDOWN_BARS,
            config.MIN_COMMISSION,
            config.SLIPPAGE_ATR_FACTOR,
            config.COMMISSION_THRESHOLD,
            True,  # Vol Targeting
            target_risk_dollars,
            float(config.MIN_LOTS),
            float(config.MAX_LOTS),
            config.VOL_SCALE_THRESHOLD,
            config.VOL_SCALE_TIGHTEN
        )

        rets = np.sum(strat_rets, axis=1)
        profit = np.sum(rets) * config.ACCOUNT_SIZE
        sortino = calculate_sortino_ratio(rets, config.ANNUALIZATION_FACTOR)
        trades = strat_trades[0]

        results[set_name] = {'profit': profit, 'sortino': sortino, 'trades': trades}

    # Check criteria: must be profitable on Train AND Test, with decent Test Sortino
    train_profit = results['Train']['profit']
    test_profit = results['Test']['profit']
    test_sortino = results['Test']['sortino']

    passes = True
    failures = []

    if train_profit <= 0:
        passes = False
        failures.append(f"Train PnL ${train_profit:,.0f} <= $0")
    if test_profit <= 0:
        passes = False
        failures.append(f"Test PnL ${test_profit:,.0f} <= $0")
    if test_sortino < 0.5:
        passes = False
        failures.append(f"Test Sortino {test_sortino:.2f} < 0.5")

    if verbose:
        if passes:
            print(f"    ✅ Standalone: Train ${train_profit:,.0f}, Test ${test_profit:,.0f}, Sortino {test_sortino:.2f}")
        else:
            print(f"    ❌ Standalone FAILED: {', '.join(failures)}")

    return passes, results, failures

def load_candidates():
    if not os.path.exists(CANDIDATES_FILE):
        return []
    with open(CANDIDATES_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_candidates(candidates):
    with open(CANDIDATES_FILE, 'w') as f:
        json.dump(candidates, f, indent=4)
    print(f"💾 Saved {len(candidates)} candidates to {CANDIDATES_FILE}")

def list_candidates():
    candidates = load_candidates()
    if not candidates:
        print("Empty candidate list.")
        return

    # REFRESH METRICS
    candidates = refresh_strategies(candidates)
    save_candidates(candidates)
    
    # Sort by CPCV P5 (most robust first)
    candidates.sort(key=get_cpcv_p5, reverse=True)
    
    # Print standardized table
    print_candidate_table(candidates)

def add_strategy(name):
    candidates = load_candidates()
    # Check if already exists
    if any(c['name'] == name for c in candidates):
        print(f"⚠️  Strategy '{name}' is already in the candidate list.")
        return

    print(f"🔍 Searching for '{name}'...")
    strategy_data = find_strategy_in_files(name)

    if not strategy_data:
        print(f"❌ Could not find strategy '{name}' in any output files.")
        return

    # === QUALITY GATES (Stored Metrics) ===
    test_sortino = strategy_data.get('test_sortino', 0)
    val_sortino = strategy_data.get('val_sortino', 0)
    cpcv_min = strategy_data.get('cpcv_min', 0)
    train_ret = strategy_data.get('train_return', 0)
    test_ret = strategy_data.get('test_return', 0)

    # Calculate decay
    decay = 1 - (test_ret / train_ret) if train_ret > 0 else 1.0

    # Check thresholds
    failures = []
    if test_sortino < config.MIN_TEST_SORTINO:
        failures.append(f"test_sortino={test_sortino:.2f} < {config.MIN_TEST_SORTINO}")
    if hasattr(config, 'MIN_VAL_SORTINO') and val_sortino < config.MIN_VAL_SORTINO:
        failures.append(f"val_sortino={val_sortino:.2f} < {config.MIN_VAL_SORTINO}")
    if cpcv_min < config.MIN_CPCV_THRESHOLD:
        failures.append(f"cpcv_min={cpcv_min:.2f} < {config.MIN_CPCV_THRESHOLD}")
    if decay > config.MAX_TRAIN_TEST_DECAY:
        failures.append(f"decay={decay:.0%} > {config.MAX_TRAIN_TEST_DECAY:.0%}")

    if failures:
        print(f"❌ Strategy '{name}' REJECTED - fails quality gates:", file=sys.stderr)
        for f in failures:
            print(f"   • {f}", file=sys.stderr)
        sys.exit(1)  # Non-zero exit so promote_strategy.py sees the failure

    # === STANDALONE PROFITABILITY CHECK (Actual Simulator) ===
    print(f"🔬 Running standalone simulation check...")
    passes, sim_results, sim_failures = check_standalone_profitability(strategy_data, verbose=True)

    if not passes:
        print(f"❌ Strategy '{name}' REJECTED - fails standalone profitability:", file=sys.stderr)
        for f in sim_failures:
            print(f"   • {f}", file=sys.stderr)
        sys.exit(1)

    # Passed all gates
    candidates.append(strategy_data)
    print(f"✅ Added '{name}' (Horizon: {strategy_data.get('horizon')}, CPCV: {cpcv_min:.2f}, Decay: {decay:.0%})")
    save_candidates(candidates)

def remove_strategy(name):
    candidates = load_candidates()
    initial_len = len(candidates)
    candidates = [c for c in candidates if c['name'] != name]
    
    if len(candidates) < initial_len:
        print(f"🗑️  Removed '{name}'.")
        save_candidates(candidates)
    else:
        print(f"⚠️  Strategy '{name}' not found in candidate list.")

def clear_list():
    save_candidates([])
    print("🧹 Candidate list cleared.")

def clear_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if os.path.exists(inbox_path):
        with open(inbox_path, 'w') as f:
            json.dump([], f, indent=4)
        print("🧹 Strategy Inbox cleared.")
    else:
        print("⚠️  Inbox file not found.")


def prune_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    candidates = load_candidates()
    candidate_names = {c['name'] for c in candidates}
    
    with open(inbox_path, 'r') as f:
        inbox_strategies = json.load(f)
        
    initial_count = len(inbox_strategies)
    pruned_strategies = [s for s in inbox_strategies if s['name'] not in candidate_names]
    removed_count = initial_count - len(pruned_strategies)
    
    if removed_count > 0:
        with open(inbox_path, 'w') as f:
            json.dump(pruned_strategies, f, indent=4)
        print(f"✂️  Pruned {removed_count} duplicates from Inbox. {len(pruned_strategies)} strategies remain.")
    else:
        print("✅ No duplicates found in Inbox.")

def cleanup_inbox(min_sortino=1.0):
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    with open(inbox_path, 'r') as f:
        try:
            strategies = json.load(f)
        except:
            print("Inbox is corrupted.")
            return

    if not strategies:
        print("Inbox is empty.")
        return

    min_test_trades = config.MIN_TRADES_TEST
    max_decay = config.MAX_TRAIN_TEST_DECAY
    print(f"🧹 Cleaning Inbox (Ret<=0, Sortino<{min_sortino}, Trades<{min_test_trades}, Decay>{max_decay*100:.0f}%)...")

    kept_strategies = []
    removed_count = 0

    for s in strategies:
        # Check Returns (Must be positive across board if data exists)
        # Using flat keys which are now reliable after fix
        t_r = s.get('train_return', 0)
        v_r = s.get('val_return', 0)
        te_r = s.get('test_return', 0)

        t_s = s.get('train_sortino', 0)
        v_s = s.get('val_sortino', 0)
        te_s = s.get('test_sortino', 0)

        te_trades = s.get('test_trades', 0)

        # If stats dicts exist, they might be more accurate if refresh just happened
        if 'train_stats' in s: t_r = s['train_stats'].get('ret', t_r)
        if 'val_stats' in s: v_r = s['val_stats'].get('ret', v_r)
        if 'test_stats' in s: te_r = s['test_stats'].get('ret', te_r)

        # Logic: If any dataset has Negative Return, it's a loser.
        # Logic: If any dataset has Sortino < min_sortino, it's weak.
        # Exception: Test set might be 0 if not evaluated? No, refresh ensures eval.

        is_failing = False
        reasons = []

        if t_r <= 0: is_failing = True; reasons.append(f"Train Ret {t_r*100:.1f}%")
        if v_r <= 0: is_failing = True; reasons.append(f"Val Ret {v_r*100:.1f}%")
        if te_r <= 0: is_failing = True; reasons.append(f"Test Ret {te_r*100:.1f}%")

        if t_s < min_sortino: is_failing = True; reasons.append(f"Train Sort {t_s:.2f}")
        if v_s < min_sortino: is_failing = True; reasons.append(f"Val Sort {v_s:.2f}")

        # Trade count filter
        if te_trades < min_test_trades: is_failing = True; reasons.append(f"Test Trades {te_trades}")

        # Decay filter: reject if test < (1 - max_decay) * train
        if t_r > 0:
            decay = 1.0 - (te_r / t_r)
            if decay > max_decay: is_failing = True; reasons.append(f"Decay {decay*100:.0f}%")
        
        if is_failing:
            removed_count += 1
            # print(f"   ❌ Removing {s.get('name')}: {', '.join(reasons)}")
        else:
            kept_strategies.append(s)

    if removed_count > 0:
        with open(inbox_path, 'w') as f:
            json.dump(kept_strategies, f, indent=4)
        print(f"✅ Removed {removed_count} failing strategies. {len(kept_strategies)} remain.")
    else:
        print("✅ Inbox is clean. No failing strategies found.")

def list_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox is empty (no file found).")
        return

    with open(inbox_path, 'r') as f:
        try:
            strategies = json.load(f)
        except:
            print("Inbox is corrupted.")
            return

    if not strategies:
        print("Inbox is empty.")
        return

    # REFRESH METRICS and PURGE strategies that no longer pass thresholds
    strategies = refresh_strategies(strategies, purge_failing=True)

    # Save back to inbox (with purged strategies removed)
    with open(inbox_path, 'w') as f:
        json.dump(strategies, f, indent=4)

    # Sort by CPCV P5 (most robust first)
    strategies.sort(key=get_cpcv_p5, reverse=True)

    # Print standardized table
    print_candidate_table(strategies, title="INBOX STRATEGIES")

def remove_from_inbox(name):
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    with open(inbox_path, 'r') as f:
        try:
            strategies = json.load(f)
        except:
            print("Inbox is corrupted.")
            return

    initial_len = len(strategies)
    strategies = [s for s in strategies if s.get('name') != name]
    
    if len(strategies) < initial_len:
        with open(inbox_path, 'w') as f:
            json.dump(strategies, f, indent=4)
        print(f"🗑️  Removed '{name}' from Inbox.")
    else:
        print(f"⚠️  Strategy '{name}' not found in Inbox.")

def clear_negative_cmin_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    with open(inbox_path, 'r') as f:
        try:
            strategies = json.load(f)
        except:
            print("Inbox is corrupted.")
            return

    if not strategies:
        print("Inbox is empty.")
        return

    print("🧹 Clearing strategies with negative CMin from Inbox...")

    kept_strategies = []
    removed_count = 0

    for s in strategies:
        cmin = get_cpcv_min(s)
        if cmin < 0:
            removed_count += 1
        else:
            kept_strategies.append(s)

    if removed_count > 0:
        with open(inbox_path, 'w') as f:
            json.dump(kept_strategies, f, indent=4)
        print(f"✅ Removed {removed_count} strategies with negative CMin. {len(kept_strategies)} remain.")
    else:
        print("✅ No strategies with negative CMin found.")

def add_all_from_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    with open(inbox_path, 'r') as f:
        inbox_strategies = json.load(f)

    candidates = load_candidates()
    candidate_names = {c['name'] for c in candidates}

    print(f"🔬 Checking {len(inbox_strategies)} inbox strategies for standalone profitability...")

    added_count = 0
    rejected_count = 0

    for s in inbox_strategies:
        if s['name'] in candidate_names:
            continue  # Already in candidates

        print(f"\n  Checking {s['name']}...")
        passes, sim_results, failures = check_standalone_profitability(s, verbose=True)

        if passes:
            candidates.append(s)
            candidate_names.add(s['name'])
            added_count += 1
        else:
            rejected_count += 1

    if added_count > 0:
        save_candidates(candidates)
        print(f"\n✅ Added {added_count} strategies from Inbox to Candidates.")
    if rejected_count > 0:
        print(f"❌ Rejected {rejected_count} strategies (failed standalone profitability).")
    if added_count == 0 and rejected_count == 0:
        print("⚠️  No new strategies to add (all already in candidates).")


def verify_candidates():
    """
    Re-verify all candidates against standalone profitability using the actual simulator.
    Removes any strategies that no longer pass.
    """
    candidates = load_candidates()
    if not candidates:
        print("No candidates to verify.")
        return

    print(f"🔬 Verifying {len(candidates)} candidates for standalone profitability...")

    kept = []
    removed = []

    for s in candidates:
        print(f"\n  Checking {s['name']}...")
        passes, sim_results, failures = check_standalone_profitability(s, verbose=True)

        if passes:
            kept.append(s)
        else:
            removed.append((s['name'], failures))

    if removed:
        print(f"\n❌ Removing {len(removed)} strategies that failed standalone check:")
        for name, failures in removed:
            print(f"   - {name}: {', '.join(failures)}")
        save_candidates(kept)
    else:
        print(f"\n✅ All {len(candidates)} candidates passed standalone verification.")

def audit_direction_consistency(source='candidates'):
    """
    Checks direction consistency for all strategies in candidates or inbox.
    Flags strategies where long/short performance flipped between train and test.
    """
    if source == 'candidates':
        strategies = load_candidates()
        source_name = "Candidates"
    else:
        inbox_path = config.DIRS['STRATEGY_INBOX']
        if not os.path.exists(inbox_path):
            print("Inbox file not found.")
            return
        with open(inbox_path, 'r') as f:
            strategies = json.load(f)
        source_name = "Inbox"

    if not strategies:
        print(f"{source_name} is empty.")
        return

    print(f"🔍 Auditing direction consistency for {len(strategies)} strategies in {source_name}...")
    print(f"   Loading market data...")

    # Load data
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])

    # Apply time filter
    if hasattr(config, 'TRAIN_START_DATE') and config.TRAIN_START_DATE:
        if 'time_start' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
                df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
            ts_col = df['time_start']
            if ts_col.dt.tz is None:
                ts_col = ts_col.dt.tz_localize('UTC')
            else:
                ts_col = ts_col.dt.tz_convert('UTC')
            start_ts = pd.Timestamp(config.TRAIN_START_DATE).tz_localize('UTC')
            if ts_col.min() < start_ts:
                df = df[ts_col >= start_ts].reset_index(drop=True)

    engine = BacktestEngine(df, cost_bps=config.COST_BPS)

    consistent_count = 0
    inconsistent_strategies = []

    print(f"\n{'Strategy':<50} | {'Long':<20} | {'Short':<20} | Status")
    print("-" * 100)

    for s_data in strategies:
        try:
            strat = Strategy.from_dict(s_data)
            strat.horizon = s_data.get('horizon', config.DEFAULT_TIME_LIMIT)

            result = check_direction_consistency(engine, strat, strat.horizon)

            # Format output
            long_str = f"T:{result['long_train_ret']:+.1%} -> Te:{result['long_test_ret']:+.1%}"
            short_str = f"T:{result['short_train_ret']:+.1%} -> Te:{result['short_test_ret']:+.1%}"

            if result['consistent']:
                status = "✅ OK"
                consistent_count += 1
            else:
                status = f"⚠️  FLIPPED"
                inconsistent_strategies.append((s_data['name'], result['warning']))
                # Store warning in strategy data
                s_data['direction_warning'] = result['warning']

            print(f"{s_data['name'][:50]:<50} | {long_str:<20} | {short_str:<20} | {status}")

        except Exception as e:
            print(f"{s_data.get('name', '?')[:50]:<50} | {'ERROR':<20} | {'ERROR':<20} | ❌ {str(e)[:30]}")

    engine.shutdown()

    print("-" * 100)
    print(f"\n📊 Summary: {consistent_count}/{len(strategies)} strategies are direction-consistent")

    if inconsistent_strategies:
        print(f"\n⚠️  Strategies with direction flips:")
        for name, warning in inconsistent_strategies:
            print(f"   - {name}: {warning}")


def clear_direction_flipped_inbox():
    """
    Removes strategies from inbox that have direction flips (long/short performance
    changed sign between train and test).
    """
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    with open(inbox_path, 'r') as f:
        strategies = json.load(f)

    if not strategies:
        print("Inbox is empty.")
        return

    print(f"🔍 Checking direction consistency for {len(strategies)} inbox strategies...")

    # Load data
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])

    # Apply time filter
    if hasattr(config, 'TRAIN_START_DATE') and config.TRAIN_START_DATE:
        if 'time_start' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
                df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
            ts_col = df['time_start']
            if ts_col.dt.tz is None:
                ts_col = ts_col.dt.tz_localize('UTC')
            else:
                ts_col = ts_col.dt.tz_convert('UTC')
            start_ts = pd.Timestamp(config.TRAIN_START_DATE).tz_localize('UTC')
            if ts_col.min() < start_ts:
                df = df[ts_col >= start_ts].reset_index(drop=True)

    engine = BacktestEngine(df, cost_bps=config.COST_BPS)

    kept_strategies = []
    removed_count = 0

    for s_data in strategies:
        try:
            strat = Strategy.from_dict(s_data)
            strat.horizon = s_data.get('horizon', config.DEFAULT_TIME_LIMIT)

            result = check_direction_consistency(engine, strat, strat.horizon)

            if result['consistent']:
                kept_strategies.append(s_data)
            else:
                removed_count += 1
                print(f"   ❌ Removing {s_data['name']}: {result['warning']}")

        except Exception as e:
            # Keep strategies we can't check
            kept_strategies.append(s_data)
            print(f"   ⚠️  Could not check {s_data.get('name', '?')}: {e}")

    engine.shutdown()

    if removed_count > 0:
        with open(inbox_path, 'w') as f:
            json.dump(kept_strategies, f, indent=4)
        print(f"\n✅ Removed {removed_count} strategies with direction flips. {len(kept_strategies)} remain.")
    else:
        print("\n✅ No direction-flipped strategies found.")

def main():
    parser = argparse.ArgumentParser(description="Manage Strategy Candidates for Mutex Portfolio")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    subparsers.add_parser('list', help='List current candidates')
    subparsers.add_parser('inbox', help='List found strategies in inbox')

    add_parser = subparsers.add_parser('add', help='Add a strategy by name')
    add_parser.add_argument('name', type=str, help='Name of the strategy (e.g., Child_2080)')

    subparsers.add_parser('add-all', help='Add ALL strategies from inbox to candidates')

    rm_parser = subparsers.add_parser('remove', help='Remove a strategy from CANDIDATES list')
    rm_parser.add_argument('name', type=str, help='Name of the strategy')

    rm_inbox_parser = subparsers.add_parser('remove-inbox', help='Remove a strategy from INBOX')
    rm_inbox_parser.add_argument('name', type=str, help='Name of the strategy')

    subparsers.add_parser('clear', help='Clear the candidate list')
    subparsers.add_parser('clear-inbox', help='Clear the strategy inbox')
    subparsers.add_parser('clear-negative-cmin', help='Clear inbox strategies with negative CMin')
    subparsers.add_parser('clear-direction-flipped', help='Clear inbox strategies with direction flips')
    subparsers.add_parser('prune', help='Remove candidates from inbox')
    subparsers.add_parser('cleanup', help='Remove losing strategies from inbox (Ret <= 0)')
    subparsers.add_parser('verify', help='Re-verify candidates against standalone profitability (actual simulator)')

    audit_dir_parser = subparsers.add_parser('audit-directions', help='Audit direction consistency')
    audit_dir_parser.add_argument('--inbox', action='store_true', help='Audit inbox instead of candidates')

    subparsers.add_parser('help', help='Show this help message')

    args = parser.parse_args()

    if args.command == 'list':
        list_candidates()
    elif args.command == 'inbox':
        list_inbox()
    elif args.command == 'add':
        add_strategy(args.name)
    elif args.command == 'add-all':
        add_all_from_inbox()
    elif args.command == 'remove':
        remove_strategy(args.name)
    elif args.command == 'remove-inbox':
        remove_from_inbox(args.name)
    elif args.command == 'clear':
        clear_list()
    elif args.command == 'clear-inbox':
        clear_inbox()
    elif args.command == 'clear-negative-cmin':
        clear_negative_cmin_inbox()
    elif args.command == 'clear-direction-flipped':
        clear_direction_flipped_inbox()
    elif args.command == 'prune':
        prune_inbox()
    elif args.command == 'cleanup':
        cleanup_inbox()
    elif args.command == 'verify':
        verify_candidates()
    elif args.command == 'audit-directions':
        source = 'inbox' if args.inbox else 'candidates'
        audit_direction_consistency(source)
    elif args.command == 'help':
        parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
