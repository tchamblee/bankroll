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
    DATA_PATH = config.DIRS['DATA_RAW_TICKS']
    
    # 1. Generate Features
    engine = FeatureEngine(DATA_PATH)
    
    # Load Primary (EUR/USD) - The True Target
    primary_df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    
    if primary_df is None:
        print("Failed to load primary data.")
        exit()

    print(f"Loaded {len(primary_df)} primary ticks.")
    # For FX, Tick Bars (fixed number of quotes) are often more robust than estimated volume
    # Let's use 250 ticks per bar. 700k ticks -> 2800 bars.
    engine.create_volume_bars(primary_df, volume_threshold=250)
    
    if engine.bars is None or len(engine.bars) < 10:
        print(f"Not enough bars generated ({len(engine.bars) if engine.bars is not None else 0}).")
        exit()

    # Load Correlators
    for ticker, suffix in [("RAW_TICKS_TNX*.parquet", "_tnx"), 
                           ("RAW_TICKS_DXY*.parquet", "_dxy"), 
                           ("RAW_TICKS_BUND*.parquet", "_bund"),
                           ("RAW_TICKS_SPY*.parquet", "_spy")]:
        corr_df = engine.load_ticker_data(ticker)
        if corr_df is not None:
            engine.add_correlator_residual(corr_df, suffix=suffix)
    
    # Add Standard Features
    engine.add_features_to_bars(windows=[50, 100, 200, 400]) 

    # --- MACRO VOLTAGE ---
    engine.add_macro_voltage_features()
    
    # Add Physics Features
    engine.add_physics_features()
    
    # Add Microstructure Features
    engine.add_microstructure_features()

    # Add Advanced Physics Features
    engine.add_advanced_physics_features()
    
    # Add Delta Features (Flow) - Multiple Horizons
    engine.add_delta_features(lookback=10) 
    engine.add_delta_features(lookback=50) 
    
    base_df = engine.bars.copy()
    
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
                metrics_path = os.path.join(config.DIRS['DATA_DIR'], f"feature_metrics_{horizon}.csv")
                merged_metrics.to_csv(metrics_path, index=False)
                print(f"\n💾 Saved Feature Metrics to {metrics_path}")
            else:
                print("Not enough valid data for Random Forest check.")
        except Exception as e:
            print(f"RF Check failed: {e}")
