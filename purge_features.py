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
    
    for f in survivors:
        fold_ics = []
        valid_folds = 0
        
        for fold_data in folds:
            valid = fold_data[[f, target_col]].dropna()
            if len(valid) < 20: 
                fold_ics.append(0)
                continue
                
            corr, p_val = stats.spearmanr(valid[f], valid[target_col])
            # Handle potential nan if constant in fold
            if np.isnan(corr): corr = 0
            fold_ics.append(corr)
            valid_folds += 1
            
        mean_ic = np.mean(fold_ics)
        std_ic = np.std(fold_ics)
        # Avoid div by zero
        stability = abs(mean_ic) / (std_ic + 1e-6)
        
        # Overall stats (full dataset) for redundancy check
        full_valid = df[[f, target_col]].dropna()
        full_corr, full_p = stats.spearmanr(full_valid[f], full_valid[target_col])
        
        feature_stats.append({
            'Feature': f, 
            'IC': full_corr, 
            'Abs_IC': abs(full_corr), 
            'P-Value': full_p,
            'Stability': stability,
            'Fold_ICs': [round(x, 3) for x in fold_ics]
        })
        
        # Filter Logic
        if abs(full_corr) < ic_threshold:
            kill_list.append({'Feature': f, 'Reason': f'Weak Signal (IC={full_corr:.4f})'})
        elif stability < stability_threshold:
            kill_list.append({'Feature': f, 'Reason': f'Unstable (Stability={stability:.2f})'})
            # print(f"  ❌ {f}: Unstable (Stab={stability:.2f})")
        else:
            ic_survivors.append(f)
            
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
        # specific handling for delta features to keep them separate from base features
        if name.startswith('delta_'):
            # delta_volatility_50_50 -> delta_volatility
            parts = name.split('_')
            # Check if last parts are numbers
            while parts and parts[-1].isdigit():
                parts.pop()
            return "_".join(parts)
        else:
            # volatility_200 -> volatility
            parts = name.split('_')
            while parts and parts[-1].isdigit():
                parts.pop()
            return "_".join(parts)

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
            winner = stats_df.loc[group].sort_values('Abs_IC', ascending=False).index[0]
            concept_survivors.append(winner)
            # Log the kills
            losers = [g for g in group if g != winner]
            for l in losers:
                kill_list.append({'Feature': l, 'Reason': f'Concept Redundant with {winner}'})
                print(f"  ❌ {l}: Concept Redundant with {winner} (Weaker IC)")
                
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
    output_path = os.path.join(config.DIRS['DATA_DIR'], filename)
    try:
        os.makedirs(config.DIRS['DATA_DIR'], exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(final_survivors, f, indent=4)
        print(f"\n💾 Saved {len(final_survivors)} survivors to {output_path}")
    except Exception as e:
        print(f"Error saving survivors: {e}")
    
    return final_survivors

if __name__ == "__main__":
    DATA_PATH = config.DIRS['DATA_RAW_TICKS']
    engine = FeatureEngine(DATA_PATH)
    
    # Load Data (Same setup as validation)
    primary_df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    engine.create_volume_bars(primary_df, volume_threshold=250)
    
    for ticker, suffix in [("RAW_TICKS_TNX*.parquet", "_tnx"), 
                           ("RAW_TICKS_DXY*.parquet", "_dxy"), 
                           ("RAW_TICKS_BUND*.parquet", "_bund"),
                           ("RAW_TICKS_SPY*.parquet", "_spy")]:
        corr_df = engine.load_ticker_data(ticker)
        if corr_df is not None:
            engine.add_correlator_residual(corr_df, suffix=suffix)
            
    engine.add_features_to_bars(windows=[50, 100, 200, 400]) 
    
    # --- GDELT INTEGRATION ---
    gdelt_df = engine.load_gdelt_data()
    if gdelt_df is not None:
        engine.add_gdelt_features(gdelt_df)
        
    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_monster_features()
    engine.add_delta_features(lookback=10) 
    engine.add_delta_features(lookback=50) 
    
    base_df = engine.bars.copy()
    
    # Loop through Horizons from Config
    for horizon in config.PREDICTION_HORIZONS:
        print(f"\n\n==============================================")
        print(f"Running Feature Hunger Games for Horizon: {horizon}")
        print(f"==============================================")
        
        df = base_df.copy()
        df['target_return'] = triple_barrier_labels(df, lookahead=horizon, pt_sl_multiple=2.0)
        
        # Run the Purge for this horizon
        purge_features(df, horizon)
