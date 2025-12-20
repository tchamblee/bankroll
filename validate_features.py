import pandas as pd
import numpy as np
from feature_engine import FeatureEngine
import scipy.stats as stats
import os
import config

def triple_barrier_labels(df, lookahead=120, pt_sl_multiple=2.0, vol_window=100):
    """
    Implements Triple-Barrier Method for labeling.
    """
    if df is None or len(df) == 0: return pd.Series()
    
    print(f"Calculating Triple-Barrier Labels (Lookahead: {lookahead}, Multiplier: {pt_sl_multiple})...")
    
    # Estimate daily volatility (simple close-to-close std dev)
    daily_vol = df['close'].pct_change().rolling(vol_window).std()
    
    # Store results
    outcomes = []
    
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    vols = daily_vol.values
    n = len(df)
    
    for i in range(n):
        if i + lookahead >= n:
            outcomes.append(np.nan)
            continue
            
        current_price = closes[i]
        vol = vols[i]
        
        if np.isnan(vol) or vol == 0:
            outcomes.append(np.nan)
            continue
            
        # Dynamic Barriers
        upper_barrier = current_price * (1 + vol * pt_sl_multiple)
        lower_barrier = current_price * (1 - vol * pt_sl_multiple)
        
        # Path analysis
        future_highs = highs[i+1 : i+1+lookahead]
        future_lows = lows[i+1 : i+1+lookahead]
        future_closes = closes[i+1 : i+1+lookahead]
        
        upper_breaches = future_highs >= upper_barrier
        lower_breaches = future_lows <= lower_barrier
        
        first_upper = np.argmax(upper_breaches) if upper_breaches.any() else lookahead + 1
        first_lower = np.argmax(lower_breaches) if lower_breaches.any() else lookahead + 1
        
        if first_upper == lookahead + 1 and first_lower == lookahead + 1:
            ret = (future_closes[-1] - current_price) / current_price
        elif first_upper < first_lower:
            ret = (upper_barrier - current_price) / current_price
        else:
            ret = (lower_barrier - current_price) / current_price
            
        outcomes.append(ret)
        
    return pd.Series(outcomes, index=df.index)

if __name__ == "__main__":
    
    print(f"Loading Feature Matrix from {config.DIRS['FEATURE_MATRIX']}...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found. Run generate_features.py first.")
        exit(1)
        
    base_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    for horizon in config.PREDICTION_HORIZONS:
        print(f"\n\n==============================================")
        print(f"⚔️  FEATURE VALIDATION: Horizon {horizon} ⚔️")
        print(f"==============================================")
        
        df = base_df.copy()
        
        # 2. Generate Labels (Triple Barrier)
        df['target_return'] = triple_barrier_labels(df, lookahead=horizon, pt_sl_multiple=2.0)
        
        # 3. Analyze (Hunger Games)
        # Broader inclusion logic: Exclude metadata, keep everything else
        exclude_cols = ['time_start', 'time_end', 'ts_event', 'open', 'high', 'low', 'close', 
                        'volume', 'net_aggressor_vol', 'cum_vol', 'vol_proxy', 'bar_id', 
                        'target_return', 'log_ret']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype.kind in 'bifc']
        target_col = 'target_return'
        
        results = []
        print("\n--- Feature Validation Results (Information Coefficient) ---")
        for f in feature_cols:
            valid = df[[f, target_col]].dropna()
            if len(valid) < 50: continue
            
            # Check for zero variance to avoid warnings
            if valid[f].std() == 0 or valid[f].nunique() <= 1:
                results.append({'Feature': f, 'IC': np.nan, 'P-Value': np.nan})
                continue
                
            corr, p_val = stats.spearmanr(valid[f], valid[target_col])
            results.append({'Feature': f, 'IC': corr, 'P-Value': p_val})
            
        results_df = pd.DataFrame(results).sort_values('IC', ascending=False)
        print(results_df)
        
        # 4. Bonus: Random Forest Importance
        try:
            from sklearn.ensemble import RandomForestRegressor
            print("\n🌲 Random Forest Importance Check...")
            
            valid_data = df[feature_cols + [target_col]].dropna()
            if len(valid_data) > 100:
                X = valid_data[feature_cols]
                y = valid_data[target_col]
                
                model = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
                model.fit(X, y)
                
                importances = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                print(importances.head(20))
                
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
