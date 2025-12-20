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

def purge_features(df, horizon, target_col='target_return', ic_threshold=0.005, p_threshold=0.05, corr_threshold=0.75, stability_threshold=0.25):
    """
    Identifies features to purge based on Variance, IC, and Collinearity.
    Saves survivors to data/survivors_{horizon}.json.
    """
    print(f"\n--- 💀 THE PURGE [Horizon: {horizon}] (IC > {ic_threshold}, Stability > {stability_threshold}) ---")
    
    # 1. Identify Candidate Columns
    exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 
               'cum_vol', 'vol_proxy', 'bar_id', 'log_ret', target_col,
               'avg_bid_price', 'avg_ask_price', 'avg_bid_size', 'avg_ask_size', 'avg_spread',
               'ticket_imbalance', 'residual_bund']
    candidates = [c for c in df.columns if c not in exclude]
    
    kill_list = []
    survivors = []
    feature_stats = []

    # 2. Variance Check (Dead Features)
    print("\n[Step 1] Checking for Dead Features (Zero Variance)...")
    for f in candidates:
        if df[f].std() == 0 or df[f].isna().all():
            kill_list.append({'Feature': f, 'Reason': 'Zero Variance / NaN'})
            # print(f"  ❌ {f}: Dead")
        else:
            survivors.append(f)
            
    # 3. Stability & Weakness Check (Walk-Forward Analysis)
    print("\n[Step 2] Checking for Weak & Unstable Features (Walk-Forward IC)...")
    ic_survivors = []
    
    # Create 5 chronological folds
    n = len(df)
    fold_size = n // 5
    folds = []
    for i in range(5):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else n
        folds.append(df.iloc[start:end])
    
    # --- OPTIMIZED: Vectorized Fold Analysis ---
    # Calculates ICs for all features at once, avoiding the slow loop overhead.
    
    # 1. Pre-calculate all Fold ICs
    fold_ic_matrix = pd.DataFrame(index=survivors, columns=range(5))
    
    print("    ... Vectorizing fold calculations (Speedup) ...")
    
    for i, fold_data in enumerate(folds):
        # Calculate Spearman correlation
        # corrwith matches pairwise deletion of NaNs (like the loop's dropna)
        
        # Suppress ConstantInputWarning (occurs when a feature is constant within a single fold)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = fold_data[survivors].corrwith(fold_data[target_col], method='spearman')
        
        # Enforce min_periods=20 rule
        # Count non-NaNs in features where target is also non-NaN
        target_valid = fold_data[target_col].notna()
        counts = fold_data.loc[target_valid, survivors].count()
        c[counts < 20] = 0
        
        fold_ic_matrix[i] = c
    
    fold_ic_matrix = fold_ic_matrix.fillna(0)
    
    # 2. Calculate Stability
    mean_ic = fold_ic_matrix.mean(axis=1)
    std_ic = fold_ic_matrix.std(axis=1)
    stability_scores = mean_ic.abs() / (std_ic + 1e-6)
    
    # 3. Calculate Full ICs
    full_corr_series = df[survivors].corrwith(df[target_col], method='spearman').fillna(0)
    
    # 4. Filter and Collect Stats
    for f in survivors:
        full_corr = full_corr_series.loc[f]
        stability = stability_scores.loc[f]
        fold_ics = fold_ic_matrix.loc[f].tolist()
        
        pass_ic = abs(full_corr) >= ic_threshold
        pass_stab = stability >= stability_threshold
        
        full_p = 1.0 # Default/Dummy
        
        if not pass_ic:
             kill_list.append({'Feature': f, 'Reason': f'Weak Signal (IC={full_corr:.4f})'})
        elif not pass_stab:
             kill_list.append({'Feature': f, 'Reason': f'Unstable (Stability={stability:.2f})'})
        else:
             # Only calculate P-value for survivors (Expensive op, done only on survivors)
             full_valid = df[[f, target_col]].dropna()
             if len(full_valid) > 2:
                 _, full_p = stats.spearmanr(full_valid[f], full_valid[target_col])
             
             ic_survivors.append(f)

        feature_stats.append({
            'Feature': f, 
            'IC': full_corr, 
            'Abs_IC': abs(full_corr), 
            'P-Value': full_p,
            'Stability': stability,
            'Fold_ICs': [round(x, 3) for x in fold_ics]
        })
            
    survivors = ic_survivors
    
    # Sort stats for redundancy check (Keep best performers)
    if not survivors:
        print("No survivors after Weakness Check.")
        return []

    stats_df = pd.DataFrame(feature_stats).set_index('Feature')
    
    # --- OPTIMIZATION: Sort by Signal Strength (Abs_IC) ---
    # We must ensure we iterate from Strongest -> Weakest so that
    # when we find a correlation, we drop the WEAKER one.
    survivors = stats_df.sort_values('Abs_IC', ascending=False).index.tolist()
    
    # ----------------------------------------------------
    # [REMOVED] Step 2.5: Conceptual Redundancy (Highlander Rule)
    # We now allow multiple windows of the same concept (e.g. vol_20 and vol_200) to survive
    # provided they are not strictly collinear (handled in Step 3).
    # ----------------------------------------------------
    
    # 4. Redundancy Check (Collinearity)
    print("\n[Step 3] Checking for Redundant Features (Collinearity)...")
    final_survivors = []
    dropped_redundant = set()
    
    # --- OPTIMIZATION: Fast Rank-Based Correlation ---
    # Pandas .corr(method='spearman') is very slow (O(N^2) sorting).
    # Instead, we rank once, then use Pearson (optimized matrix multiplication).
    print(f"    ... Computing Correlation Matrix for {len(survivors)} survivors ...")
    ranked_df = df[survivors].rank(method='average')
    corr_matrix = ranked_df.corr(method='pearson')
    
    for i, f1 in enumerate(survivors):
        if f1 in dropped_redundant: continue
        
        final_survivors.append(f1)
        
        for f2 in survivors[i+1:]:
            if f2 in dropped_redundant: continue
            
            # Check correlation
            rho = corr_matrix.loc[f1, f2]
            if abs(rho) > corr_threshold:
                kill_list.append({'Feature': f2, 'Reason': f'Redundant with {f1} (Corr={rho:.2f})'})
                dropped_redundant.add(f2)
                # print(f"  ❌ {f2}: Redundant with {f1} ({rho:.2f})")

    # Output Results
    print(f"\n💀 Purged {len(kill_list)} features.")
    print(f"🛡️  {len(final_survivors)} Survivors remaining.")
    
    print("\n--- 🛡️ THE SURVIVORS (Elite Gene Pool) ---")
    print(stats_df.loc[final_survivors][['IC', 'Stability', 'P-Value']])
    
    # Save Survivors List
    survivors_path = os.path.join(config.DIRS['FEATURES_DIR'], f"survivors_{horizon}.json")
    with open(survivors_path, "w") as f:
        json.dump(final_survivors, f, indent=4)
    
    return final_survivors

if __name__ == "__main__":
    
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
    
    # Loop through Horizons from Config
    for horizon in config.PREDICTION_HORIZONS:
        print(f"\n\n==============================================")
        print(f"Running Feature Hunger Games for Horizon: {horizon}")
        print(f"==============================================")
        
        # Use only Training Data
        df = train_df.copy()
        df['target_return'] = triple_barrier_labels(df, lookahead=horizon, pt_sl_multiple=2.0)
        
        # Run the Purge for this horizon
        purge_features(df, horizon)
