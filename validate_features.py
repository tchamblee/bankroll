import pandas as pd
import numpy as np
from feature_engine import FeatureEngine
import scipy.stats as stats

def triple_barrier_labels(df, lookahead=120, pt_sl_multiple=2.0, vol_window=100):
    """
    Implements Triple-Barrier Method for labeling.
    1. Horizontal Barriers: defined by dynamic volatility * pt_sl_multiple.
    2. Vertical Barrier: defined by 'lookahead' bars.
    
    Returns a Series of 'realized_return' based on the first barrier touched.
    """
    print(f"Calculating Triple-Barrier Labels (Lookahead: {lookahead}, Multiplier: {pt_sl_multiple})...")
    
    # Estimate daily volatility (simple close-to-close std dev)
    # In a real system, we'd use the feature engine's volatility, but recalculating here for independence.
    daily_vol = df['close'].pct_change().rolling(vol_window).std()
    
    # Store results
    outcomes = []
    
    # This loop is slow in Python, but necessary for path-dependent logic.
    # We can optimize with numba later if needed. For validation, it's fine.
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
        
        if np.isnan(vol):
            outcomes.append(np.nan)
            continue
            
        # Dynamic Barriers
        upper_barrier = current_price * (1 + vol * pt_sl_multiple)
        lower_barrier = current_price * (1 - vol * pt_sl_multiple)
        
        # Path analysis
        # Slice the future window
        future_highs = highs[i+1 : i+1+lookahead]
        future_lows = lows[i+1 : i+1+lookahead]
        future_closes = closes[i+1 : i+1+lookahead]
        
        # Check touches
        # np.argmax returns index of first True. If no True, returns 0 (which is risky, so we check max)
        # We need first occurrence of High > Upper OR Low < Lower
        
        upper_breaches = future_highs >= upper_barrier
        lower_breaches = future_lows <= lower_barrier
        
        first_upper = np.argmax(upper_breaches) if upper_breaches.any() else lookahead + 1
        first_lower = np.argmax(lower_breaches) if lower_breaches.any() else lookahead + 1
        
        if first_upper == lookahead + 1 and first_lower == lookahead + 1:
            # Vertical Barrier Hit (Time Out)
            # Return is simply the drift at the end
            ret = (future_closes[-1] - current_price) / current_price
        elif first_upper < first_lower:
            # Hit Profit Target first
            ret = (upper_barrier - current_price) / current_price
        else:
            # Hit Stop Loss first (or both same tick, we assume worst case usually, but let's say SL)
            ret = (lower_barrier - current_price) / current_price
            
        outcomes.append(ret)
        
    return pd.Series(outcomes, index=df.index)

def analyze_features(df, target_col='target_return'):
    """
    Calculates Information Coefficient (Spearman Correlation) 
    between features and the target.
    """
    feature_cols = [c for c in df.columns if any(x in c for x in ['velocity', 'efficiency', 'volatility', 'autocorr', 'trend_strength', 'imbalance', 'frac_diff', 'hurst'])]
    
    results = []
    print("\n--- Feature Validation Results (Information Coefficient) ---")
    for f in feature_cols:
        # Drop NaNs for correlation
        valid = df[[f, target_col]].dropna()
        if len(valid) < 100: continue
        
        # Spearman Rank Correlation (non-linear relationships)
        corr, p_val = stats.spearmanr(valid[f], valid[target_col])
        results.append({'Feature': f, 'IC': corr, 'P-Value': p_val})
        
    results_df = pd.DataFrame(results).sort_values('IC', ascending=False)
    print(results_df)
    return results_df

if __name__ == "__main__":
    DATA_PATH = "/home/tony/bankroll/data/raw_ticks"
    
    # 1. Generate Features
    engine = FeatureEngine(DATA_PATH)
    
    # Load Primary (SPY)
    spy_df = engine.load_ticker_data("RAW_TICKS_SPY*.parquet")
    
    if spy_df is None:
        print("Failed to load primary data.")
        exit()

    engine.create_volume_bars(spy_df, volume_threshold=50000)
    
    # Load Correlator (TNX) - Optional
    tnx_df = engine.load_ticker_data("RAW_TICKS_TNX*.parquet")
    if tnx_df is not None:
        engine.merge_correlator(tnx_df, suffix="_tnx")
        engine.add_divergence_features(suffix="_tnx")
    
    # Add Standard Features
    engine.add_features_to_bars(windows=[50, 100, 200, 400]) 
    
    # Add Physics Features
    engine.add_physics_features()
    
    # Add Delta Features (Flow)
    engine.add_delta_features(lookback=10)
    
    df = engine.bars.copy()
    
    # 2. Generate Labels (Triple Barrier)
    df['target_return'] = triple_barrier_labels(df, lookahead=120, pt_sl_multiple=2.0)
    
    # 3. Analyze (Hunger Games)
    print("\n⚔️  FEATURE HUNGER GAMES ⚔️")
    
    # Update filter to include delta
    feature_cols = [c for c in df.columns if any(x in c for x in ['velocity', 'efficiency', 'volatility', 'autocorr', 'trend_strength', 'imbalance', 'frac_diff', 'hurst', 'divergence', 'delta'])]
    target_col = 'target_return'
    
    results = []
    print("\n--- Feature Validation Results (Information Coefficient) ---")
    for f in feature_cols:
        # Drop NaNs for correlation
        valid = df[[f, target_col]].dropna()
        if len(valid) < 100: continue
        
        # Spearman Rank Correlation (non-linear relationships)
        corr, p_val = stats.spearmanr(valid[f], valid[target_col])
        results.append({'Feature': f, 'IC': corr, 'P-Value': p_val})
        
    results_df = pd.DataFrame(results).sort_values('IC', ascending=False)
    print(results_df)
    
    # 4. Bonus: Random Forest Importance (Non-Linear Check)
    try:
        from sklearn.ensemble import RandomForestRegressor
        print("\n🌲 Random Forest Importance Check...")
        
        # Prepare Data
        # Re-select cols in case list changed
        clean_df = df[feature_cols + ['target_return']].dropna()
        
        X = clean_df[feature_cols]
        y = clean_df['target_return']
        
        model = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
        model.fit(X, y)
        
        importances = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(importances)
    except ImportError:
        print("sklearn not installed. Skipping RF check.")
