import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os
import config
from feature_engine import FeatureEngine
from validate_features import triple_barrier_labels

def get_feature_correlation_matrix(df, features):
    return df[features].corr(method='spearman')

def purge_features(df, horizon, target_col='target_return', ic_threshold=0.01, p_threshold=0.05, corr_threshold=0.75, stability_threshold=0.5):
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
    
    # --- NEW: CONCEPT DEDUPLICATION (Highlander Rule) ---
    print("\n[Step 2.5] Checking for Conceptual Redundancy (Best Window per Feature)...")
    
    # Helper to get base name (e.g., 'volatility_200' -> 'volatility')
    def get_base_concept(name):
        # 1. Strip 'delta_' prefix
        clean_name = name.replace('delta_', '')
        
        # 2. Strip window suffixes (_10, _50, _200, etc)
        parts = clean_name.split('_')
        while parts and parts[-1].isdigit():
            parts.pop()
        base = "_".join(parts)
        
        # 3. Custom Grouping for GDELT Volume Proxies
        # These are highly correlated in small samples; force them to compete.
        gdelt_vol_proxies = ['news_vol_usd', 'news_vol_eur', 'news_vol_diff', 
                             'news_crisis_vol', 'epu_total', 'epu_usd', 'epu_eur', 
                             'inflation_vol', 'conflict_intensity', 'news_instability',
                             'news_velocity_usd', 'news_velocity_eur']
        
        if base in gdelt_vol_proxies:
            return 'GDELT_Attention_Metrics'
            
        return base

    concept_map = {}
    for f in survivors:
        concept = get_base_concept(f)
        if concept not in concept_map:
            concept_map[concept] = []
        concept_map[concept].append(f)
        
    concept_survivors = []
    for concept, group in concept_map.items():
        if len(group) == 1:
            concept_survivors.append(group[0])
        else:
            # Pick winner based on Abs_IC
            # For GDELT features, we might want to keep the top 2 if they are distinct enough?
            # But the collinearity check will catch them anyway. Let's keep top 2 for GDELT group.
            
            sorted_group = stats_df.loc[group].sort_values('Abs_IC', ascending=False).index.tolist()
            
            if concept == 'GDELT_Attention_Metrics':
                # Keep top 3 for GDELT to allow for EUR vs USD vs Diff nuances
                winners = sorted_group[:3]
            else:
                winners = sorted_group[:1]
                
            concept_survivors.extend(winners)
            
            # Log the kills
            losers = [g for g in group if g not in winners]
            for l in losers:
                kill_list.append({'Feature': l, 'Reason': f'Concept Redundant with {winners[0]}'})
                # print(f"  ❌ {l}: Concept Redundant with {winners[0]} (Weaker IC)")
                
    survivors = stats_df.loc[concept_survivors].sort_values('Abs_IC', ascending=False).index.tolist()
    # ----------------------------------------------------
    
    # 4. Redundancy Check (Collinearity)
    print("\n[Step 3] Checking for Redundant Features (Collinearity)...")
    final_survivors = []
    dropped_redundant = set()
    
    corr_matrix = get_feature_correlation_matrix(df, survivors)
    
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
                print(f"  ❌ {f2}: Redundant with {f1} ({rho:.2f})")

    # Output Results
    print(f"\n💀 Purged {len(kill_list)} features.")
    print(f"🛡️  {len(final_survivors)} Survivors remaining.")
    
    print("\n--- 🛡️ THE SURVIVORS (Elite Gene Pool) ---")
    print(stats_df.loc[final_survivors][['IC', 'Stability', 'P-Value']])
    
    # SAVE SURVIVORS TO JSON
    filename = f"survivors_{horizon}.json"
    output_path = os.path.join(config.DIRS['FEATURES_DIR'], filename)
    try:
        os.makedirs(config.DIRS['FEATURES_DIR'], exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(final_survivors, f, indent=4)
        print(f"\n💾 Saved {len(final_survivors)} survivors to {output_path}")
    except Exception as e:
        print(f"Error saving survivors: {e}")
    
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
