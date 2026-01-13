#!/usr/bin/env python3
"""
Feature Predictive Power Analysis
---------------------------------
Scans all features in the feature matrix to find which ones
actually predict future returns at various horizons.

This answers the fundamental question: Is there alpha in this feature space?
"""

import pandas as pd
import numpy as np
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

sys.path.insert(0, '/home/tony/bankroll')
import config

warnings.filterwarnings('ignore')

def load_data():
    """Load the feature matrix."""
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    print(f"Loaded {len(df):,} bars, {len(df.columns)} columns")
    print(f"Date range: {df['time_start'].min()} to {df['time_start'].max()}")
    return df

def compute_forward_returns(close, horizons=[10, 20, 30, 60, 90, 120]):
    """Compute forward returns at various horizons."""
    fwd_rets = {}
    for h in horizons:
        fwd = np.zeros(len(close))
        fwd[:-h] = (close[h:] - close[:-h]) / close[:-h]
        fwd[-h:] = np.nan
        fwd_rets[h] = fwd
    return fwd_rets

def analyze_feature(feature_name, feature_vals, fwd_rets, horizons):
    """
    Analyze a single feature's predictive power.
    Returns dict with correlation at each horizon.
    """
    results = {'feature': feature_name}

    # Skip if too many NaNs
    valid_pct = (~np.isnan(feature_vals)).mean()
    if valid_pct < 0.5:
        for h in horizons:
            results[f'corr_{h}'] = np.nan
        results['best_horizon'] = 0
        results['best_corr'] = 0
        results['valid_pct'] = valid_pct
        return results

    results['valid_pct'] = valid_pct

    best_corr = 0
    best_horizon = 0

    for h in horizons:
        fwd = fwd_rets[h]

        # Mask for valid values
        mask = ~np.isnan(feature_vals) & ~np.isnan(fwd)

        if mask.sum() < 1000:
            results[f'corr_{h}'] = np.nan
            continue

        # Compute correlation
        try:
            corr = np.corrcoef(feature_vals[mask], fwd[mask])[0, 1]
            if np.isnan(corr):
                corr = 0
        except:
            corr = 0

        results[f'corr_{h}'] = corr

        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_horizon = h

    results['best_horizon'] = best_horizon
    results['best_corr'] = best_corr

    return results

def analyze_feature_wrapper(args):
    """Wrapper for parallel processing."""
    feature_name, feature_vals, fwd_rets, horizons = args
    return analyze_feature(feature_name, feature_vals, fwd_rets, horizons)

def compute_information_coefficient(feature_vals, fwd_ret, n_quantiles=10):
    """
    Compute Information Coefficient (IC) using quantile analysis.
    More robust than simple correlation for non-linear relationships.
    """
    mask = ~np.isnan(feature_vals) & ~np.isnan(fwd_ret)
    if mask.sum() < 1000:
        return 0, {}

    f = feature_vals[mask]
    r = fwd_ret[mask]

    # Compute quantile bins
    try:
        quantiles = pd.qcut(f, n_quantiles, labels=False, duplicates='drop')
    except:
        return 0, {}

    # Mean return per quantile
    quantile_returns = {}
    for q in range(n_quantiles):
        qmask = quantiles == q
        if qmask.sum() > 0:
            quantile_returns[q] = r[qmask].mean()

    if len(quantile_returns) < 3:
        return 0, quantile_returns

    # IC = correlation between quantile rank and mean return
    ranks = list(quantile_returns.keys())
    rets = [quantile_returns[q] for q in ranks]

    try:
        ic = np.corrcoef(ranks, rets)[0, 1]
        if np.isnan(ic):
            ic = 0
    except:
        ic = 0

    return ic, quantile_returns

def analyze_feature_deep(feature_name, feature_vals, fwd_rets, horizons):
    """
    Deep analysis of a single feature including:
    - Linear correlation
    - Information Coefficient (quantile-based)
    - Monotonicity score
    """
    results = {'feature': feature_name}

    valid_pct = (~np.isnan(feature_vals)).mean()
    if valid_pct < 0.5:
        return None

    results['valid_pct'] = valid_pct

    best_corr = 0
    best_ic = 0
    best_horizon = 0

    for h in horizons:
        fwd = fwd_rets[h]

        mask = ~np.isnan(feature_vals) & ~np.isnan(fwd)
        if mask.sum() < 1000:
            continue

        # Linear correlation
        try:
            corr = np.corrcoef(feature_vals[mask], fwd[mask])[0, 1]
            if np.isnan(corr):
                corr = 0
        except:
            corr = 0

        results[f'corr_{h}'] = corr

        # Information Coefficient
        ic, _ = compute_information_coefficient(feature_vals, fwd)
        results[f'ic_{h}'] = ic

        # Track best
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_horizon = h

        if abs(ic) > abs(best_ic):
            best_ic = ic

    results['best_corr'] = best_corr
    results['best_ic'] = best_ic
    results['best_horizon'] = best_horizon

    return results

def main():
    print("=" * 70)
    print("FEATURE PREDICTIVE POWER ANALYSIS")
    print("=" * 70)

    # 1. Load data
    df = load_data()

    # 2. Prepare data
    close = df['close'].values
    horizons = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240]

    print(f"\nAnalyzing {len(df.columns)} features across {len(horizons)} horizons...")
    print(f"Horizons: {horizons}")

    # 3. Compute forward returns
    print("\nComputing forward returns...")
    fwd_rets = compute_forward_returns(close, horizons)

    # 4. Get numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude metadata columns
    exclude = ['time_start', 'open', 'high', 'low', 'close', 'volume', 'log_ret',
               'time_hour', 'time_weekday', 'bar_idx']
    numeric_cols = [c for c in numeric_cols if c not in exclude and not c.startswith('fwd_')]

    print(f"Analyzing {len(numeric_cols)} numeric features...")

    # 5. Analyze all features (parallel)
    print("\nRunning analysis (this may take a few minutes)...")

    results = []

    # Prepare args for parallel processing
    args_list = [
        (col, df[col].values, fwd_rets, horizons)
        for col in numeric_cols
    ]

    # Use multiprocessing for speed
    from multiprocessing import Pool, cpu_count

    with Pool(processes=min(cpu_count(), 8)) as pool:
        for i, result in enumerate(pool.imap_unordered(analyze_feature_wrapper, args_list, chunksize=50)):
            results.append(result)
            if (i + 1) % 200 == 0:
                print(f"  Processed {i + 1}/{len(numeric_cols)} features...")

    print(f"  Processed {len(results)} features total.")

    # 6. Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df['abs_best_corr'] = results_df['best_corr'].abs()
    results_df = results_df.sort_values('abs_best_corr', ascending=False)

    # 7. Report top features
    print("\n" + "=" * 70)
    print("TOP 30 FEATURES BY PREDICTIVE POWER")
    print("=" * 70)

    top_n = 30
    print(f"\n{'Feature':<50} {'Best Corr':>10} {'Horizon':>8} {'Valid%':>8}")
    print("-" * 78)

    for _, row in results_df.head(top_n).iterrows():
        name = row['feature'][:49]
        corr = row['best_corr']
        horizon = int(row['best_horizon']) if not np.isnan(row['best_horizon']) else 0
        valid = row['valid_pct'] * 100

        # Color coding (terminal)
        if abs(corr) >= 0.05:
            marker = "***"
        elif abs(corr) >= 0.03:
            marker = "**"
        elif abs(corr) >= 0.02:
            marker = "*"
        else:
            marker = ""

        print(f"{name:<50} {corr:>+10.4f} {horizon:>8} {valid:>7.1f}% {marker}")

    # 8. Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    abs_corrs = results_df['abs_best_corr'].dropna()

    print(f"\nTotal features analyzed: {len(results_df)}")
    print(f"Features with |corr| >= 0.05: {(abs_corrs >= 0.05).sum()}")
    print(f"Features with |corr| >= 0.03: {(abs_corrs >= 0.03).sum()}")
    print(f"Features with |corr| >= 0.02: {(abs_corrs >= 0.02).sum()}")
    print(f"Features with |corr| >= 0.01: {(abs_corrs >= 0.01).sum()}")

    print(f"\nMax |correlation|: {abs_corrs.max():.4f}")
    print(f"Mean |correlation|: {abs_corrs.mean():.4f}")
    print(f"Median |correlation|: {abs_corrs.median():.4f}")

    # 9. Best horizon analysis
    print("\n" + "=" * 70)
    print("BEST HORIZON DISTRIBUTION")
    print("=" * 70)

    # Only for features with meaningful correlation
    meaningful = results_df[results_df['abs_best_corr'] >= 0.02]
    if len(meaningful) > 0:
        horizon_counts = meaningful['best_horizon'].value_counts().sort_index()
        print(f"\nFor {len(meaningful)} features with |corr| >= 0.02:")
        for h, count in horizon_counts.items():
            if count > 0:
                print(f"  Horizon {int(h):>3} bars: {count:>3} features")
    else:
        print("\nNo features with |corr| >= 0.02 found.")

    # 10. Feature category analysis
    print("\n" + "=" * 70)
    print("FEATURE CATEGORY ANALYSIS")
    print("=" * 70)

    # Group features by prefix
    categories = {}
    for _, row in results_df.iterrows():
        feature = row['feature']
        # Extract category (first part before underscore or number)
        parts = feature.split('_')
        if len(parts) > 1:
            cat = parts[0]
        else:
            cat = feature

        if cat not in categories:
            categories[cat] = []
        categories[cat].append(row['abs_best_corr'])

    # Compute mean correlation by category
    cat_stats = []
    for cat, corrs in categories.items():
        if len(corrs) >= 3:  # Only categories with 3+ features
            cat_stats.append({
                'category': cat,
                'count': len(corrs),
                'mean_corr': np.mean(corrs),
                'max_corr': np.max(corrs)
            })

    cat_df = pd.DataFrame(cat_stats).sort_values('max_corr', ascending=False)

    print(f"\n{'Category':<20} {'Count':>8} {'Mean |Corr|':>12} {'Max |Corr|':>12}")
    print("-" * 54)
    for _, row in cat_df.head(20).iterrows():
        print(f"{row['category']:<20} {row['count']:>8} {row['mean_corr']:>12.4f} {row['max_corr']:>12.4f}")

    # 11. Deep dive on top features
    print("\n" + "=" * 70)
    print("DEEP DIVE: TOP 10 FEATURES")
    print("=" * 70)

    for _, row in results_df.head(10).iterrows():
        feature = row['feature']
        print(f"\n{feature}")
        print("-" * len(feature))

        # Show correlation at each horizon
        horizon_corrs = []
        for h in horizons:
            col = f'corr_{h}'
            if col in row and not np.isnan(row[col]):
                horizon_corrs.append((h, row[col]))

        if horizon_corrs:
            print("  Horizon ->", end="")
            for h, _ in horizon_corrs[:8]:
                print(f" {h:>6}", end="")
            print()
            print("  Corr    ->", end="")
            for _, c in horizon_corrs[:8]:
                print(f" {c:>+6.3f}", end="")
            print()

    # 12. Save results
    output_file = '/home/tony/bankroll/output/feature_predictive_power.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")

    # 13. Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    strong_features = (abs_corrs >= 0.05).sum()
    moderate_features = (abs_corrs >= 0.03).sum()
    weak_features = (abs_corrs >= 0.02).sum()

    if strong_features >= 10:
        print("\n[GOOD] Found multiple features with strong predictive power (|corr| >= 0.05)")
        print("       Alpha exists in this feature space.")
    elif moderate_features >= 10:
        print("\n[MODERATE] Found features with moderate predictive power (|corr| >= 0.03)")
        print("           Alpha may exist but signals are weak.")
    elif weak_features >= 10:
        print("\n[WEAK] Only weak predictive signals found (|corr| >= 0.02)")
        print("       Consider different features or shorter horizons.")
    else:
        print("\n[POOR] Very few features show any predictive power.")
        print("       This feature space may not contain exploitable alpha.")
        print("       Consider:")
        print("       - Different feature engineering approaches")
        print("       - Higher frequency data / shorter horizons")
        print("       - Alternative data sources")

    return results_df

if __name__ == "__main__":
    results = main()
