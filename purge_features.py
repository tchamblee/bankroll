import pandas as pd
import numpy as np
import scipy.stats as stats
import json
import os
from feature_engine import FeatureEngine
from validate_features import triple_barrier_labels

def get_feature_correlation_matrix(df, features):
    return df[features].corr(method='spearman')

def purge_features(df, target_col='target_return', ic_threshold=0.01, p_threshold=0.05, corr_threshold=0.95):
    """
    Identifies features to purge based on Variance, IC, and Collinearity.
    Saves survivors to data/survivors.json.
    """
    print(f"\n--- 💀 THE PURGE (IC > {ic_threshold}, P < {p_threshold}, Corr < {corr_threshold}) ---")
    
    # 1. Identify Candidate Columns
    exclude = ['time_start', 'time_end', 'open', 'high', 'low', 'close', 'volume', 'net_aggressor_vol', 
               'cum_vol', 'vol_proxy', 'bar_id', 'log_ret', target_col]
    candidates = [c for c in df.columns if c not in exclude]
    
    kill_list = []
    survivors = []
    feature_stats = []

    # 2. Variance Check (Dead Features)
    print("\n[Step 1] Checking for Dead Features (Zero Variance)...")
    for f in candidates:
        if df[f].std() == 0 or df[f].isna().all():
            kill_list.append({'Feature': f, 'Reason': 'Zero Variance / NaN'})
            print(f"  ❌ {f}: Dead")
        else:
            survivors.append(f)
            
    # 3. Weakness Check (Low IC)
    print("\n[Step 2] Checking for Weak Features (Low Predictive Power)...")
    ic_survivors = []
    
    for f in survivors:
        valid = df[[f, target_col]].dropna()
        if len(valid) < 50:
            kill_list.append({'Feature': f, 'Reason': 'Insufficient Data'})
            continue
            
        corr, p_val = stats.spearmanr(valid[f], valid[target_col])
        feature_stats.append({'Feature': f, 'IC': corr, 'Abs_IC': abs(corr), 'P-Value': p_val})
        
        if abs(corr) < ic_threshold or p_val > p_threshold:
            kill_list.append({'Feature': f, 'Reason': f'Weak Signal (IC={corr:.4f}, P={p_val:.4f})'})
            # print(f"  ❌ {f}: Weak")
        else:
            ic_survivors.append(f)
            
    survivors = ic_survivors
    
    # Sort stats for redundancy check (Keep best performers)
    if not survivors:
        print("No survivors after Weakness Check.")
        return []

    stats_df = pd.DataFrame(feature_stats).set_index('Feature')
    survivors = stats_df.loc[survivors].sort_values('Abs_IC', ascending=False).index.tolist()
    
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
    print(stats_df.loc[final_survivors][['IC', 'P-Value']])
    
    # SAVE SURVIVORS TO JSON
    output_path = os.path.join("data", "survivors.json")
    try:
        os.makedirs("data", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(final_survivors, f, indent=4)
        print(f"\n💾 Saved {len(final_survivors)} survivors to {output_path}")
    except Exception as e:
        print(f"Error saving survivors: {e}")
    
    return final_survivors

if __name__ == "__main__":
    DATA_PATH = "/home/tony/bankroll/data/raw_ticks"
    engine = FeatureEngine(DATA_PATH)
    
    # Load Data (Same setup as validation)
    primary_df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    engine.create_volume_bars(primary_df, volume_threshold=500)
    
    for ticker, suffix in [("RAW_TICKS_TNX*.parquet", "_tnx"), 
                           ("RAW_TICKS_DXY*.parquet", "_dxy"), 
                           ("RAW_TICKS_BUND*.parquet", "_bund"),
                           ("RAW_TICKS_SPY*.parquet", "_spy")]:
        corr_df = engine.load_ticker_data(ticker)
        if corr_df is not None:
            engine.add_correlator_residual(corr_df, suffix=suffix)
            
    engine.add_features_to_bars(windows=[50, 100, 200, 400]) 
    engine.add_physics_features()
    engine.add_delta_features(lookback=10) 
    engine.add_delta_features(lookback=50) 
    
    df = engine.bars.copy()
    df['target_return'] = triple_barrier_labels(df, lookahead=120, pt_sl_multiple=2.0)
    
    # Run the Purge
    purge_features(df)