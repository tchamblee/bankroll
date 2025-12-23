import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os
import config
from feature_engine import FeatureEngine
from validate_features import triple_barrier_labels

import warnings

import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os
import config
from feature_engine import FeatureEngine
from validate_features import triple_barrier_labels

import warnings

def get_feature_correlation_matrix(df, features):
    # Optimized: Rank then Pearson (Equivalent to Spearman but much faster)
    return df[features].rank(method='average').corr(method='pearson')

def purge_features(df, horizon, target_col='target_return', ic_threshold=0.005, p_threshold=0.05, corr_threshold=0.92, stability_threshold=0.25, silent=False):
    """
    Identifies features to purge based on Variance, IC, and Collinearity.
    Saves survivors to data/survivors_{horizon}.json.
    """
    if not silent:
        print(f"\n--- 💀 THE PURGE [Horizon: {horizon}] (IC > {ic_threshold}, Stability > {stability_threshold}) ---")
    
    # 1. Identify Candidate Columns
    exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 
               'cum_vol', 'vol_proxy', 'bar_id', 'log_ret', target_col,
               'avg_bid_price', 'avg_ask_price', 'avg_bid_size', 'avg_ask_size', 'avg_spread',
               'ticket_imbalance', 'residual_bund']
    candidates = [c for c in df.columns if c not in exclude]
    
    kill_list = []
    
    # 2. Variance Check (Dead Features) - Vectorized
    # if not silent: print("\n[Step 1] Checking for Dead Features (Zero Variance)...")
    
    # Calculatenunique and std only once
    stats_df = df[candidates].agg(['nunique', 'std']).T
    dead_features = stats_df[(stats_df['nunique'] <= 1) | (stats_df['std'] == 0) | (stats_df['nunique'].isna())].index.tolist()
    
    for f in dead_features:
        kill_list.append({'Feature': f, 'Reason': 'Zero Variance / NaN'})
        
    survivors = [c for c in candidates if c not in dead_features]
    
    if not survivors:
        if not silent: print("No survivors after Variance Check.")
        return [], kill_list

    # --- OPTIMIZATION: Rank Once Strategy ---
    # Rank all survivor features globally once. 
    # This avoids re-ranking in folds and correlation matrix.
    # We use pct=False (ranks) to be compatible with Pearson-on-Ranks = Spearman.
    # if not silent: print(f"    ... Pre-ranking {len(survivors)} features for speed ...")
    ranked_features = df[survivors].rank(method='average')
    
    # Also rank the target (we need this for correlation)
    # Note: Target might have NaNs (triple barrier edges). 
    # We mask NaNs later or let corr handle it (Pearson ignores NaNs).
    ranked_target = df[[target_col]].rank(method='average')
    
    # Combine for easier slicing
    ranked_df = pd.concat([ranked_features, ranked_target], axis=1)
    
    # 3. Stability & Weakness Check (Walk-Forward Analysis)
    # if not silent: print("\n[Step 2] Checking for Weak & Unstable Features (Walk-Forward IC)...")
    
    # Create 5 chronological fold INDICES (not copies of data)
    n = len(df)
    fold_size = n // 5
    fold_slices = []
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else n
        fold_slices.append((start, end))
    
    # 1. Pre-calculate all Fold ICs using Ranked Data (Pearson on Ranks)
    fold_ic_matrix = pd.DataFrame(index=survivors, columns=range(5))
    
    for i, (start, end) in enumerate(fold_slices):
        # Slice the PRE-RANKED data
        fold_data = ranked_df.iloc[start:end]
        
        # Calculate Pearson correlation on ranks (Approximation of Spearman on Fold)
        # Note: True Spearman would re-rank within the fold. 
        # But Global Rank is a very strong proxy and 10x faster.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # method='pearson' is default for corrwith
            c = fold_data[survivors].corrwith(fold_data[target_col])
        
        # Enforce min_periods check using raw data counts (not ranks)
        # We need to look at original NaNs
        raw_fold_target = df[target_col].iloc[start:end]
        raw_fold_feats = df[survivors].iloc[start:end]
        
        target_valid = raw_fold_target.notna()
        # Count valid overlaps
        # This can be slow if we do it for every feature. 
        # Optimization: Most features are dense. Only target has NaNs usually.
        # If features are dense, count is just sum(target_valid).
        # We'll assume features are mostly dense due to generation logic, but let's be safe.
        # Fast check: valid_count = (~raw_fold_feats.isna() & target_valid.values[:, None]).sum()
        
        # Slower but accurate path if needed, or just assume valid if not dropped by variance
        # Let's trust the correlation to return NaN if not enough data, 
        # but we need to zero it out if low data.
        # For speed, we skip the rigorous "count < 20" check per feature unless crucial.
        # We'll assume if correlation is computed, it's valid enough, 
        # or rely on the global P-value later.
        
        fold_ic_matrix[i] = c
    
    fold_ic_matrix = fold_ic_matrix.fillna(0)
    
    # 2. Calculate Stability
    mean_ic = fold_ic_matrix.mean(axis=1)
    std_ic = fold_ic_matrix.std(axis=1)
    stability_scores = mean_ic.abs() / (std_ic + 1e-6)
    
    # 3. Calculate Full ICs (Pearson on Global Ranks)
    # efficient calculation for all survivors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        full_corrs = ranked_df[survivors].corrwith(ranked_df[target_col])
    full_corrs = full_corrs.fillna(0)
    
    # 4. Vectorized P-Value Calculation
    # We avoid the loop with stats.spearmanr
    n_samples = len(df.dropna(subset=[target_col])) # Approximation of N
    
    # t-statistic: t = r * sqrt((n-2) / (1-r^2))
    # We clip r to +/- 0.999 to avoid division by zero
    r_clipped = full_corrs.clip(-0.9999, 0.9999)
    t_stats = r_clipped * np.sqrt((n_samples - 2) / (1 - r_clipped**2))
    
    # Survival function for 2-tailed p-value
    p_values = 2 * stats.t.sf(np.abs(t_stats), n_samples - 2)
    p_values_series = pd.Series(p_values, index=full_corrs.index)

    # 5. Filter Candidates
    # Vectorized Filtering
    pass_ic_mask = full_corrs.abs() >= ic_threshold
    pass_stab_mask = stability_scores >= stability_threshold
    
    # Identify failures
    # Fix: Convert survivors list to Index for boolean masking
    survivors_idx = pd.Index(survivors)
    
    fail_ic = survivors_idx[~pass_ic_mask].tolist()
    fail_stab = survivors_idx[pass_ic_mask & ~pass_stab_mask].tolist() # Only check stab if passed IC
    
    # Record Kill Reasons (Optional, for logging)
    # For speed, we might skip detailed logging per feature if N is huge,
    # but constructing the list is fast enough.
    
    for f in fail_ic: # These are indices now
        kill_list.append({'Feature': f, 'Reason': f'Weak Signal (IC={full_corrs[f]:.4f})'})
        
    for f in fail_stab:
        kill_list.append({'Feature': f, 'Reason': f'Unstable (Stability={stability_scores[f]:.2f})'})
        
    # Valid survivors of Step 2
    ic_survivors = survivors_idx[pass_ic_mask & pass_stab_mask].tolist()
    
    if len(ic_survivors) == 0:
        if not silent: print("No survivors after Weakness Check.")
        return [], kill_list

    # Construct Stats DataFrame for sorting
    stats_df = pd.DataFrame({
        'IC': full_corrs[ic_survivors],
        'Abs_IC': full_corrs[ic_survivors].abs(),
        'P-Value': p_values_series[ic_survivors],
        'Stability': stability_scores[ic_survivors]
    })
    
    # Add Fold ICs (optional, maybe skip if not needed for logic)
    # stats_df['Fold_ICs'] = fold_ic_matrix.loc[ic_survivors].values.tolist()
    
    # Sort by Signal Strength
    sorted_survivors = stats_df.sort_values('Abs_IC', ascending=False).index.tolist()
    
    # 4. Redundancy Check (Collinearity)
    # if not silent: print("\n[Step 3] Checking for Redundant Features (Collinearity)...")
    final_survivors = []
    dropped_redundant = set()
    
    # --- OPTIMIZATION: Subsampled Correlation Matrix ---
    # If N is huge, subsample for the N^2 matrix calculation.
    # 50k rows is enough to establish 0.75 correlation.
    if len(ranked_df) > 50000:
        corr_sample = ranked_df[sorted_survivors].sample(n=50000, random_state=42)
    else:
        corr_sample = ranked_df[sorted_survivors]
        
    # Calculate Correlation Matrix of Survivors (Pearson on Ranks)
    # This is O(M^2 * N_sample)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corr_matrix = corr_sample.corr(method='pearson')
    
    # Iterate High Signal -> Low Signal
    for i, f1 in enumerate(sorted_survivors):
        if f1 in dropped_redundant: continue
        
        final_survivors.append(f1)
        
        # Check against weaker features
        # Vectorized check? 
        # We can find all f2 where corr > threshold
        
        # Get correlations of f1 with all remaining candidates
        remaining = sorted_survivors[i+1:]
        if not remaining: continue
        
        # Extract correlations for this row, slicing only relevant columns
        f1_corrs = corr_matrix.loc[f1, remaining]
        
        # Identify redundant ones
        redundant_mask = f1_corrs.abs() > corr_threshold
        redundant_feats = f1_corrs[redundant_mask].index.tolist()
        
        for f2 in redundant_feats:
            if f2 not in dropped_redundant:
                kill_list.append({'Feature': f2, 'Reason': f'Redundant with {f1} (Corr={f1_corrs[f2]:.2f})'})
                dropped_redundant.add(f2)

    # Output Results
    if not silent:
        print(f"\n💀 THE PURGE SUMMARY [Horizon: {horizon}]")
        print(f"-------------------------------------------")
        
        # Helper to extract base name (remove trailing numbers/windows)
        # e.g., 'rsi_14' -> 'rsi', 'volatility_roc_3200' -> 'volatility_roc'
        import re
        def get_base_name(s):
            return re.sub(r'_\d+$', '', s)

        # Categorize kills for reporting
        dead_kills = [k['Feature'] for k in kill_list if 'Zero Variance' in k['Reason']]
        weak_ic_kills = [k['Feature'] for k in kill_list if 'Weak Signal' in k['Reason']]
        unstable_kills = [k['Feature'] for k in kill_list if 'Unstable' in k['Reason']]
        
        # Filter Redundant Kills: Show only if Base Name is DIFFERENT
        # We need to look up the "Killer" (f1) for each "Victim" (f2)
        # The 'Reason' string contains "Redundant with {f1}..."
        redundant_kills = []
        for k in kill_list:
            if 'Redundant' in k['Reason']:
                victim = k['Feature']
                # Extract killer from reason string
                # Reason format: "Redundant with {f1} (Corr=...)"
                match = re.search(r'Redundant with (.+) \(', k['Reason'])
                if match:
                    killer = match.group(1)
                    # Only report if they are fundamentally different features
                    if get_base_name(victim) != get_base_name(killer):
                        redundant_kills.append(victim)
                else:
                    # Fallback if parsing fails
                    redundant_kills.append(victim)

        if dead_kills:
            print(f"❌ DEAD (Zero Var): {len(dead_kills)} features.")
            print(f"   Examples: {dead_kills[:10]}")
            
        if weak_ic_kills:
            print(f"📉 WEAK (Low IC): {len(weak_ic_kills)} features.")
            print(f"   Examples: {weak_ic_kills[:10]}")
            
        if unstable_kills:
            print(f"🎢 UNSTABLE: {len(unstable_kills)} features.")
            print(f"   Examples: {unstable_kills[:10]}")
            
        if redundant_kills:
            print(f"👯 REDUNDANT (Cross-Feature): {len(redundant_kills)} features.")
            print(f"   Examples: {redundant_kills[:10]}")

        print(f"\n🛡️  SURVIVORS: {len(final_survivors)}")
        print(f"   Top 10: {final_survivors[:10]}")
    
    # Save Survivors List
    os.makedirs(config.DIRS['FEATURES_DIR'], exist_ok=True)
    survivors_path = os.path.join(config.DIRS['FEATURES_DIR'], f"survivors_{horizon}.json")
    with open(survivors_path, "w") as f:
        json.dump(final_survivors, f, indent=4)
    
    return final_survivors, kill_list

if __name__ == "__main__":
    marker_path = os.path.join(config.DIRS['FEATURES_DIR'], "PURGE_COMPLETE")
    if os.path.exists(marker_path):
        os.remove(marker_path)

    # Check if all survivor lists already exist
    all_exist = True
    for h in config.PREDICTION_HORIZONS:
        p = os.path.join(config.DIRS['FEATURES_DIR'], f"survivors_{h}.json")
        if not os.path.exists(p):
            all_exist = False
            break
    
    if all_exist:
        print(f"⏩ Survivor lists for all horizons ({config.PREDICTION_HORIZONS}) already exist. Skipping.")
        exit(0)

    print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found. Run generate_features.py first.")
        exit(1)
        
    base_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # --- FIX: DATA LEAKAGE PREVENTION ---
    # Only use the Training Set (First 60%) for Feature Selection.
    # The remaining 40% (Val/Test) must remain unseen by the "Hunger Games".
    train_size = int(len(base_df) * 0.6)
    train_df = base_df.iloc[:train_size].copy()
    
    print(f"\n🔒 LOCKDOWN: Feature Selection restricted to TRAINING SET only.")
    print(f"   Range: {train_df['time_start'].min()} to {train_df['time_start'].max()}")
    print(f"   Size: {len(train_df)} bars (Total: {len(base_df)})")
    # ------------------------------------
    
    # Collect all candidates (all numeric features except metadata)
    exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 
               'cum_vol', 'vol_proxy', 'bar_id', 'log_ret', 
               'avg_bid_price', 'avg_ask_price', 'avg_bid_size', 'avg_ask_size', 'avg_spread',
               'ticket_imbalance', 'residual_bund']
    all_candidates = set([c for c in train_df.columns if c not in exclude])
    
    useful_features = set()
    global_kill_reasons = {} # feature -> list of reasons
    
    print(f"\n⏳ Analyzing {len(config.PREDICTION_HORIZONS)} horizons...")
    
    # Loop through Horizons from Config
    for horizon in config.PREDICTION_HORIZONS:
        print(f"   - Processing Horizon {horizon}...")
        
        # Optimization: In-place update (avoid full copy)
        # We reuse train_df and just overwrite the target column
        # Using default TP=1.5% SL=0.5% (Config alignment)
        train_df['target_return'] = triple_barrier_labels(train_df, lookahead=horizon, tp_pct=config.DEFAULT_TAKE_PROFIT, sl_pct=config.DEFAULT_STOP_LOSS)
        
        # Run the Purge (Silent Mode)
        survivors, kill_list = purge_features(train_df, horizon, silent=True)
        
        useful_features.update(survivors)
        
        # Record reasons for this horizon
        for k in kill_list:
            feat = k['Feature']
            if feat not in global_kill_reasons:
                global_kill_reasons[feat] = []
            global_kill_reasons[feat].append(f"H{horizon}:{k['Reason']}")

    # Calculate Global Rejects (Features that failed in ALL horizons)
    useless_features = all_candidates - useful_features
    
    print(f"\n🌍 GLOBAL PURGE REPORT")
    print(f"======================")
    print(f"Total Candidates: {len(all_candidates)}")
    print(f"Useful Features (Survived >= 1 Horizon): {len(useful_features)}")
    print(f"Useless Features (Failed ALL Horizons): {len(useless_features)}")
    
    if useless_features:
        print(f"\n🗑️  Feature Drop Summary (Failed in ALL Horizons):")
        
        # Categorize
        dead_list = []
        redundant_list = []
        weak_list = []
        
        for f in useless_features:
            reasons = global_kill_reasons.get(f, [])
            is_dead = any("Zero Variance" in r for r in reasons)
            is_redundant = any("Redundant" in r for r in reasons)
            
            if is_dead:
                dead_list.append(f)
            elif is_redundant:
                redundant_list.append(f)
            else:
                weak_list.append(f)
                
        # Helper to print truncated list
        def print_truncated(label, items, icon=""):
            items = sorted(items)
            limit = 10
            print(f"\n{icon} {label} [{len(items)}]:")
            if len(items) > limit:
                shown = items[:limit]
                print(f"   {', '.join(shown)} ... and {len(items)-limit} more")
            else:
                print(f"   {', '.join(items)}")

        # Print Groups
        if dead_list:
            print_truncated("DEAD (Zero Variance)", dead_list, "❌")
            
        if redundant_list:
            print_truncated("REDUNDANT (Collinear)", redundant_list, "👯")
            
        if weak_list:
            print_truncated("LOW SIGNAL (Weak/Unstable)", weak_list, "📉")
            
    else:
        print("\n✅ Clean Feature Set! No completely useless features found.")

    # Signal completion
    with open(marker_path, 'w') as f:
        f.write("Updated")
    print(f"✅ Purge Complete. Marker saved to {marker_path}")
