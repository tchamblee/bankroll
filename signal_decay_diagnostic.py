#!/usr/bin/env python3
"""
Signal Decay Diagnostic
-----------------------
Analyzes how quickly a strategy's signal loses predictive power over time.
This helps identify if strategies are holding positions beyond their signal's useful life.
"""

import pandas as pd
import numpy as np
import json
import sys
import os
sys.path.insert(0, '/home/tony/bankroll')

import config
from genome.strategy import Strategy
from backtest.core import BacktestEngineBase
from backtest.feature_computation import ensure_feature_context

def load_data():
    """Load the feature matrix."""
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    print(f"Loaded {len(df):,} bars from {df['time_start'].min()} to {df['time_start'].max()}")
    return df

def load_strategy(filepath, idx=0):
    """Load a strategy from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    strat_dict = data[idx]
    strat = Strategy.from_dict(strat_dict)
    print(f"Loaded strategy: {strat.name}")
    print(f"  Horizon: {strat.horizon} bars")
    print(f"  Long genes: {len(strat.long_genes)}, Short genes: {len(strat.short_genes)}")
    return strat, strat_dict

def setup_backtester_context(df, strategies):
    """
    Set up the backtester which computes derived features (correlations, zscores, etc.)
    """
    print("\nInitializing backtester and computing derived features...")
    bt = BacktestEngineBase(df)
    # Ensure all features needed by the strategies are computed
    ensure_feature_context(strategies, bt.temp_dir, bt.existing_keys)
    return bt

def build_context_from_backtester(bt):
    """Load all features from backtester's temp directory into a context dict."""
    context = {}
    for key in bt.existing_keys:
        path = os.path.join(bt.temp_dir, f"{key}.npy")
        if os.path.exists(path):
            context[key] = np.load(path)
    context['__len__'] = bt.data_len
    return context

def get_required_features(strategy):
    """Extract all features required by a strategy's genes."""
    features = set()

    for gene in strategy.long_genes + strategy.short_genes:
        gene_dict = gene.to_dict()
        for key, val in gene_dict.items():
            if 'feature' in key.lower() and isinstance(val, str):
                features.add(val)

    return features

def compute_raw_votes(strategy, context):
    """
    Compute raw vote counts (continuous signal) instead of discrete +1/-1/0.
    Returns long_votes, short_votes as continuous arrays.
    """
    # Check for missing features
    required = get_required_features(strategy)
    available = set(context.keys())
    missing = required - available

    if missing:
        print(f"\n  [WARNING] Missing {len(missing)} features:")
        for f in sorted(missing)[:10]:
            print(f"    - {f}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    n_rows = context.get('__len__', 0)
    if n_rows == 0:
        # Try to infer from close array
        n_rows = len(context.get('close', []))
        context['__len__'] = n_rows

    cache = {}

    def get_votes(genes):
        if not genes:
            return np.zeros(n_rows, dtype=np.float32)
        votes = np.zeros(n_rows, dtype=np.float32)
        for i, gene in enumerate(genes):
            try:
                result = gene.evaluate(context, cache)
                if result.shape[0] == n_rows:
                    active = result.sum()
                    pct = 100 * active / n_rows
                    print(f"    Gene {i}: {gene.type} -> {active:.0f} active bars ({pct:.2f}%)")
                    votes += result
                else:
                    print(f"    Gene {gene} returned wrong shape: {result.shape}")
            except Exception as e:
                print(f"    Gene {i} ({gene.type}) failed: {e}")
        return votes

    l_votes = get_votes(strategy.long_genes)
    s_votes = get_votes(strategy.short_genes)

    # Net signal strength (positive = bullish, negative = bearish)
    # Normalize by gene count to get -1 to +1 range
    max_genes = max(len(strategy.long_genes), len(strategy.short_genes), 1)
    net_signal = (l_votes - s_votes) / max_genes

    return l_votes, s_votes, net_signal

def compute_forward_returns(context, lags=[1, 2, 5, 10, 20, 30, 50, 75, 100, 120]):
    """Compute forward returns at various lags."""
    fwd_rets = {}
    close = context['close']
    n = len(close)

    for lag in lags:
        # Forward return from bar t to bar t+lag
        fwd_ret = np.zeros(n)
        fwd_ret[:-lag] = (close[lag:] - close[:-lag]) / close[:-lag]
        fwd_ret[-lag:] = np.nan  # Can't compute for last `lag` bars
        fwd_rets[lag] = fwd_ret

    return fwd_rets

def compute_signal_decay(signal, fwd_rets, direction='long'):
    """
    Compute correlation between signal and forward returns at various lags.

    Args:
        signal: Raw signal array (positive = bullish)
        fwd_rets: Dict of {lag: forward_returns_array}
        direction: 'long' (positive signal should predict positive returns)
                  or 'short' (positive signal should predict negative returns)

    Returns:
        Dict of {lag: correlation}
    """
    decay_curve = {}

    # For short direction, flip the expected correlation
    sign = 1 if direction == 'long' else -1

    for lag, fwd_ret in fwd_rets.items():
        # Only use rows where both signal and forward return are valid
        mask = ~np.isnan(fwd_ret) & ~np.isnan(signal) & (signal != 0)

        if mask.sum() < 100:
            decay_curve[lag] = np.nan
            continue

        # Compute correlation
        corr = np.corrcoef(signal[mask] * sign, fwd_ret[mask])[0, 1]
        decay_curve[lag] = corr

    return decay_curve

def estimate_half_life(decay_curve):
    """Find lag where |correlation| drops to half of peak."""
    valid_corrs = {k: v for k, v in decay_curve.items() if not np.isnan(v)}
    if not valid_corrs:
        return 0

    peak = max(abs(v) for v in valid_corrs.values())
    if peak < 0.01:  # No meaningful signal
        return 0

    half_peak = peak / 2

    for lag in sorted(valid_corrs.keys()):
        if abs(valid_corrs[lag]) <= half_peak:
            return lag

    return max(valid_corrs.keys())  # Doesn't decay within range

def analyze_trade_durations(context, strategy, signal):
    """
    Simulate trades and compute actual holding durations.
    Simplified version - just estimates based on signal changes.
    """
    # Binary signal (when we'd be in a position)
    in_position = signal != 0

    # Find position changes
    position_changes = np.diff(in_position.astype(int), prepend=0)
    entries = np.where(position_changes == 1)[0]
    exits = np.where(position_changes == -1)[0]

    # Match entries to exits
    durations = []
    for entry in entries:
        # Find next exit after this entry
        later_exits = exits[exits > entry]
        if len(later_exits) > 0:
            exit_idx = later_exits[0]
            duration = exit_idx - entry
            # Cap at horizon (strategy would force-exit)
            duration = min(duration, strategy.horizon)
            durations.append(duration)

    if not durations:
        return {'avg': 0, 'median': 0, 'max': 0, 'count': 0}

    return {
        'avg': np.mean(durations),
        'median': np.median(durations),
        'max': np.max(durations),
        'count': len(durations)
    }

def analyze_single_strategy(df, bt, context, strategy, strat_dict, verbose=True):
    """Analyze a single strategy and return results."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy.name}")
        print(f"  Horizon: {strategy.horizon} bars")
        print(f"  Long genes: {len(strategy.long_genes)}, Short genes: {len(strategy.short_genes)}")
        print(f"{'='*60}")

    # Ensure features are computed for this strategy
    ensure_feature_context([strategy], bt.temp_dir, bt.existing_keys)
    context = build_context_from_backtester(bt)

    # Compute raw votes
    l_votes, s_votes, net_signal = compute_raw_votes(strategy, context)

    signal_active = np.abs(net_signal) > 0.01
    if verbose:
        print(f"  Signal active on {signal_active.sum():,} bars ({100*signal_active.mean():.1f}%)")

    # Compute forward returns
    lags = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 100, 120]
    fwd_rets = compute_forward_returns(context, lags)

    # Determine direction
    has_long = len(strategy.long_genes) > 0
    has_short = len(strategy.short_genes) > 0

    if has_long and not has_short:
        direction = 'long'
        signal_for_decay = l_votes / max(len(strategy.long_genes), 1)
    elif has_short and not has_long:
        direction = 'short'
        signal_for_decay = s_votes / max(len(strategy.short_genes), 1)
    else:
        direction = 'both'
        signal_for_decay = net_signal

    decay_curve = compute_signal_decay(signal_for_decay, fwd_rets,
                                        direction='long' if direction != 'short' else 'short')

    half_life = estimate_half_life(decay_curve)

    # Trade durations
    try:
        discrete_signal = strategy.generate_signal(context)
    except:
        discrete_signal = np.sign(net_signal) * (np.abs(net_signal) >= 0.5)

    duration_stats = analyze_trade_durations(context, strategy, discrete_signal)

    # Find peak correlation
    valid_corrs = [abs(v) for v in decay_curve.values() if not np.isnan(v)]
    peak_corr = max(valid_corrs) if valid_corrs else 0

    if verbose:
        print(f"\n  Peak correlation: {peak_corr:.4f}")
        print(f"  Half-life: {half_life} bars")
        print(f"  Avg trade duration: {duration_stats['avg']:.1f} bars")
        print(f"  Trade count: {duration_stats['count']}")

    return {
        'name': strategy.name,
        'horizon': strategy.horizon,
        'peak_corr': peak_corr,
        'half_life': half_life,
        'avg_duration': duration_stats['avg'],
        'trade_count': duration_stats['count'],
        'decay_curve': decay_curve,
        'test_sortino': strat_dict.get('test_sortino', 0),
        'test_return': strat_dict.get('test_return', 0)
    }

def main():
    # 1. Load data
    print("=" * 60)
    print("SIGNAL DECAY DIAGNOSTIC - MULTI-STRATEGY ANALYSIS")
    print("=" * 60)

    df = load_data()

    # 2. Load multiple strategies from apex_120
    apex_file = config.APEX_FILE_TEMPLATE.format(120)
    with open(apex_file) as f:
        all_strats = json.load(f)

    print(f"\nLoaded {len(all_strats)} strategies from {apex_file}")

    # 3. Set up backtester context
    print("\nInitializing backtester...")
    bt = BacktestEngineBase(df)
    context = build_context_from_backtester(bt)
    print(f"  Base context has {len(context)} keys")

    # 4. Analyze first 5 strategies
    results = []
    for idx in range(min(5, len(all_strats))):
        strat_dict = all_strats[idx]
        strategy = Strategy.from_dict(strat_dict)
        result = analyze_single_strategy(df, bt, context, strategy, strat_dict, verbose=True)
        results.append(result)

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Strategy':<35} {'Peak Corr':>10} {'Half-Life':>10} {'Avg Dur':>10} {'Trades':>8} {'Test SR':>10}")
    print("-" * 85)
    for r in results:
        name = r['name'][:34]
        print(f"{name:<35} {r['peak_corr']:>10.4f} {r['half_life']:>10} {r['avg_duration']:>10.1f} {r['trade_count']:>8} {r['test_sortino']:>10.2f}")

    # Diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    low_corr = [r for r in results if r['peak_corr'] < 0.01]
    if low_corr:
        print(f"\n[WARNING] {len(low_corr)}/{len(results)} strategies have peak correlation < 0.01")
        print("These strategies likely have no real predictive power.")

    misaligned = [r for r in results if r['half_life'] > 0 and r['half_life'] < r['avg_duration'] * 0.5]
    if misaligned:
        print(f"\n[WARNING] {len(misaligned)}/{len(results)} strategies have signal decay faster than holding period")

    # Cleanup backtester
    bt.shutdown()

    return results

if __name__ == "__main__":
    main()
