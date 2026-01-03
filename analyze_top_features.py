import pandas as pd
import numpy as np
import scipy.stats as stats
import os
import config
from validate_features import triple_barrier_labels
import warnings
import argparse

# Suppress warnings
warnings.filterwarnings("ignore")

def get_feature_correlation_matrix(df, features):
    """
    Optimized: Rank then Pearson (Equivalent to Spearman but much faster)
    """
    return df[features].rank(method='average').corr(method='pearson')

def analyze_horizon(df, horizon, top_n=20, use_survivors=False):
    print(f"\n==============================================")
    print(f"📊 ANALYZING HORIZON: {horizon} bars")
    print(f"==============================================")
    
    # 1. Generate Labels (Triple Barrier)
    # Using default TP/SL from config to match training conditions
    print(f"   Generating Target Labels (TP={config.DEFAULT_TAKE_PROFIT}xATR, SL={config.DEFAULT_STOP_LOSS}xATR)...")
    df = df.copy()
    df['target_return'] = triple_barrier_labels(df, lookahead=horizon, tp_pct=config.DEFAULT_TAKE_PROFIT, sl_pct=config.DEFAULT_STOP_LOSS)
    
    # 2. Identify Candidates
    exclude_cols = ['time_start', 'time_end', 'ts_event', 'open', 'high', 'low', 'close', 
                    'volume', 'net_aggressor_vol', 'cum_vol', 'vol_proxy', 'bar_id', 
                    'target_return', 'log_ret',
                    'avg_ask_price', 'avg_bid_price', 'avg_ask_size', 'avg_bid_size',
                    'avg_spread', 'ticket_imbalance', 'residual_bund', 'residual_tnx', 'residual_usdchf',
                     # BLACKLIST (Suspicious Leakage known from purge_features.py)
                    'rel_strength_z_6e', 
                    'delta_rel_strength_z_6e_25', 'delta_rel_strength_z_6e_50', 'delta_rel_strength_z_6e_100',
                    'divergence_50_6e',
                    'delta_divergence_50_6e_25', 'delta_divergence_50_6e_50', 'delta_divergence_50_6e_100']
    
    noise_patterns = ['kyle_lambda', 'tick_spread', 'tick_volatility', 'tick_ofi', 'pres_imbalance', 'avg_spread']
    
    if use_survivors:
    survivors_path = config.SURVIVORS_FILE_TEMPLATE.format(horizon)
    if not os.path.exists(survivors_path):
            import json
            with open(survivors_path, 'r') as f:
                feature_cols = json.load(f)
            print(f"   Loaded {len(feature_cols)} survivor features from {survivors_path}")
        else:
            print(f"   ⚠️ Survivors file not found: {survivors_path}. Falling back to all features.")
            feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype.kind in 'bifc' and not any(p in c for p in noise_patterns)]
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype.kind in 'bifc' and not any(p in c for p in noise_patterns)]

    # Explicitly remove raw price columns to avoid leakage/spurious regression
    feature_cols = [c for c in feature_cols if not c.startswith('price_')]

    # Remove constant features
    valid_features = []
    for c in feature_cols:
        if c in df.columns and df[c].nunique() > 1:
            valid_features.append(c)
    
    print(f"   Evaluating {len(valid_features)} features...")
    
    # 3. Calculate IC (Spearman Correlation)
    target_col = 'target_return'
    
    # Drop rows where target is NaN (edges)
    valid_df = df[valid_features + [target_col]].dropna(subset=[target_col])
    
    if len(valid_df) < 50:
        print("   ❌ Not enough valid data points.")
        return

    # Rank Data (Vectorized Spearman)
    ranked_df = valid_df.rank(method='average')
    
    # Calculate Correlations
    corrs = ranked_df[valid_features].corrwith(ranked_df[target_col])
    
    # Construct Results
    results = pd.DataFrame({
        'Feature': valid_features,
        'IC': corrs.values,
        'Abs_IC': corrs.abs()
    })
    
    # Sort by Absolute IC (Strength)
    top_results = results.sort_values('Abs_IC', ascending=False).head(top_n)
    
    print(f"\n   🏆 TOP {top_n} FEATURES (by Predictive Strength):")
    print(f"   {'Rank':<5} {'Feature':<40} {'IC':<10} {'Direction'}")
    print(f"   {'-'*70}")
    
    for rank, (idx, row) in enumerate(top_results.iterrows(), 1):
        direction = "Positive" if row['IC'] > 0 else "Negative"
        print(f"   {rank:<5} {row['Feature']:<40} {row['IC']:<10.4f} {direction}")

    # Optional: Save to CSV
    out_path = os.path.join(config.DIRS['OUTPUT_DIR'], f"top_features_horizon_{horizon}.csv")
    top_results.to_csv(out_path, index=False)
    print(f"\n   💾 Saved top list to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Top Features for Prediction")
    parser.add_argument("--survivors", action="store_true", help="Only analyze features that survived the purge")
    parser.add_argument("--top", type=int, default=20, help="Number of top features to display")
    parser.add_argument("--horizon", type=int, help="Specific horizon to analyze (default: all in config)")
    args = parser.parse_args()

    print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        exit(1)
        
    base_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])

    # --- DATE FILTERING ---
    if hasattr(config, 'TRAIN_START_DATE') and config.TRAIN_START_DATE:
        if 'time_start' in base_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(base_df['time_start']):
                 base_df['time_start'] = pd.to_datetime(base_df['time_start'], utc=True)
            
            ts_col = base_df['time_start']
            if ts_col.dt.tz is None: ts_col = ts_col.dt.tz_localize('UTC')
            else: ts_col = ts_col.dt.tz_convert('UTC')
                 
            start_ts = pd.Timestamp(config.TRAIN_START_DATE).tz_localize('UTC')
            
            if ts_col.min() < start_ts:
                original_len = len(base_df)
                base_df = base_df[ts_col >= start_ts].reset_index(drop=True)
                print(f"📅 Training Start Date Applied: {config.TRAIN_START_DATE} (Dropped {original_len - len(base_df)} rows)")

    # Use only Training Data (Standard Practice)
    train_size = int(len(base_df) * config.TRAIN_SPLIT_RATIO)
    train_df = base_df.iloc[:train_size].copy()
    print(f"🔒 Using Training Set ({len(train_df)} bars)")

    horizons = [args.horizon] if args.horizon else config.PREDICTION_HORIZONS

    for h in horizons:
        analyze_horizon(train_df, h, top_n=args.top, use_survivors=args.survivors)

if __name__ == "__main__":
    main()
