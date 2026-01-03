import pandas as pd
import numpy as np
import sys
from feature_engine import FeatureEngine
import scipy.stats as stats
import os
import config

from numba import jit

@jit(nopython=True)
def _jit_triple_barrier(closes, highs, lows, lookahead, tp_pct, sl_pct):
    """
    Numba-optimized core logic for Triple Barrier labeling.
    Uses fixed percentages to align with TradeSimulator logic.
    """
    n = len(closes)
    outcomes = np.empty(n)
    outcomes[:] = np.nan
    
    for i in range(n - lookahead):
        current_price = closes[i]
        
        # Fixed Percentage Barriers
        upper_barrier = current_price * (1.0 + tp_pct)
        lower_barrier = current_price * (1.0 - sl_pct)
        
        # Path analysis
        first_upper = -1
        first_lower = -1
        
        for k in range(1, lookahead + 1):
            idx = i + k
            # Upper barrier check
            if highs[idx] >= upper_barrier:
                first_upper = k
                break
        
        for k in range(1, lookahead + 1):
            idx = i + k
            # Lower barrier check
            if lows[idx] <= lower_barrier:
                first_lower = k
                break
        
        # Outcome Logic
        if first_upper == -1 and first_lower == -1:
            # Vertical barrier (time limit) - Return at Horizon
            outcomes[i] = (closes[i + lookahead] - current_price) / current_price
        elif first_upper != -1 and (first_lower == -1 or first_upper < first_lower):
            # Hit Upper Barrier (TP)
            outcomes[i] = tp_pct
        else:
            # Hit Lower Barrier (SL)
            outcomes[i] = -sl_pct
            
    return outcomes

def triple_barrier_labels(df, lookahead=config.DEFAULT_TIME_LIMIT, tp_pct=config.DEFAULT_TAKE_PROFIT, sl_pct=config.DEFAULT_STOP_LOSS):
    """
    Implements Triple-Barrier Method for labeling.
    Uses Numba for high performance.
    """
    if df is None or len(df) == 0: return pd.Series(dtype=np.float64)
    
    closes = df['close'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    
    res = _jit_triple_barrier(closes, highs, lows, lookahead, tp_pct, sl_pct)
    
    return pd.Series(res, index=df.index)

if __name__ == "__main__":
    # Check if Purge is complete
    marker_path = config.PURGE_MARKER_FILE
    if not os.path.exists(marker_path):
        print(f"❌ Purge marker not found at {marker_path}. Run purge_features.py first.")
        sys.exit(1)
    
    # Consume marker
    os.remove(marker_path)

    print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found. Run generate_features.py first.")
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

    for horizon in config.PREDICTION_HORIZONS:
        # print(f"\n\n==============================================")
        # print(f"⚔️  FEATURE VALIDATION: Horizon {horizon} ⚔️")
        # print(f"==============================================")
        
        df = base_df.copy()
        
        # 2. Generate Labels (Triple Barrier)
        # Using default TP=1.5% SL=0.5% as a standard baseline
        df['target_return'] = triple_barrier_labels(df, lookahead=horizon, tp_pct=config.DEFAULT_TAKE_PROFIT, sl_pct=config.DEFAULT_STOP_LOSS)
        
        # 3. Analyze (Hunger Games)
        # Broader inclusion logic: Exclude metadata, keep everything else
        exclude_cols = ['time_start', 'time_end', 'ts_event', 'open', 'high', 'low', 'close', 
                        'volume', 'net_aggressor_vol', 'cum_vol', 'vol_proxy', 'bar_id', 
                        'target_return', 'log_ret',
                        'avg_ask_price', 'avg_bid_price', 'avg_ask_size', 'avg_bid_size']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype.kind in 'bifc']
        
        # Remove constant features to avoid warnings/errors
        constant_features = [c for c in feature_cols if df[c].nunique() <= 1]
        if constant_features:
            # print(f"Dropping {len(constant_features)} constant features: {constant_features}")
            feature_cols = [c for c in feature_cols if c not in constant_features]

        target_col = 'target_return'
        
        results = []
        # print("\n--- Feature Validation Results (Information Coefficient) ---")
        
        # Optimized Vectorized Spearman Correlation
        # 1. Drop rows where target is NaN
        valid_df = df[feature_cols + [target_col]].dropna(subset=[target_col])
        
        if len(valid_df) > 50:
            # 2. Rank the features and target
            # method='average' is standard for Spearman
            ranked_df = valid_df.rank(method='average')
            
            # 3. Calculate Pearson correlation on ranks (which is Spearman)
            # This is much faster than iterative spearmanr
            corrs = ranked_df[feature_cols].corrwith(ranked_df[target_col])
            
            # 4. Calculate P-values using t-distribution approximation
            # n = number of non-NaN pairs for each feature
            # We use a slightly conservative approach with global n for speed, 
            # or we can compute it per feature if NaNs vary significantly.
            n = valid_df[feature_cols].notna().sum()
            
            # Avoid division by zero for features with all NaNs or constant values
            mask = (n > 2) & (corrs.abs() < 1.0)
            t_stats = np.zeros(len(corrs))
            t_stats[mask] = corrs[mask] * np.sqrt((n[mask] - 2) / (1 - corrs[mask]**2))
            
            p_vals = np.ones(len(corrs))
            # Use survival function (sf) for 2-tailed p-value: 2 * (1 - cdf(|t|))
            from scipy.stats import t
            p_vals[mask] = 2 * t.sf(np.abs(t_stats[mask]), df=n[mask]-2)
            
            results_df = pd.DataFrame({
                'Feature': feature_cols,
                'IC': corrs.values,
                'P-Value': p_vals
            }).sort_values('IC', ascending=False)
        else:
            results_df = pd.DataFrame(columns=['Feature', 'IC', 'P-Value'])

        # print(results_df)
        
        # 4. Bonus: Random Forest Importance
        try:
            from sklearn.ensemble import ExtraTreesRegressor
            # print("\n🌲 Random Forest Importance Check (Optimized)...")
            
            valid_data = df[feature_cols + [target_col]].dropna()
            if len(valid_data) > 100:
                # Performance optimization: Subsample if too large
                if len(valid_data) > 10000:
                    valid_data = valid_data.sample(n=10000, random_state=42)

                X = valid_data[feature_cols]
                y = valid_data[target_col]
                
                model = ExtraTreesRegressor(n_estimators=30, max_depth=5, n_jobs=-1, random_state=42)
                model.fit(X, y)
                
                importances = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # print(importances.head(20))
                
                # Merge and Save Metrics
                # Ensure index is reset for merge
                if 'Feature' not in results_df.columns: results_df = results_df.reset_index()
                
                # Merge on Feature (IC + Importance)
                merged_metrics = pd.merge(importances, results_df, on='Feature', how='outer')
                metrics_path = os.path.join(config.DIRS['FEATURES_DIR'], f"feature_metrics_{horizon}.csv")
                merged_metrics.to_csv(metrics_path, index=False)
                print(f"\n💾 Saved Feature Metrics to {metrics_path}")
            else:
                print("Not enough valid data for Random Forest check.")
        except Exception as e:
            print(f"RF Check failed: {e}")
