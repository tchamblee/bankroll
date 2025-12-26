import pandas as pd
import numpy as np
import config
import scipy.stats

def get_regime_stats(prices, returns, label):
    if len(prices) < 100: return
    
    # 1. Returns
    total_ret = (prices[-1] - prices[0]) / prices[0]
    
    # 2. Volatility (Annualized)
    std = np.std(returns)
    ann_vol = std * np.sqrt(config.ANNUALIZATION_FACTOR)
    
    # 3. Hurst (Trendiness)
    # Using a simplified hurst or just rough approx if compute_hurst is heavy
    try:
        # Calculate Hurst on log prices
        log_prices = np.log(prices)
        # Simple R/S analysis is complex to implement from scratch reliably in one go, 
        # so we'll infer trendiness from Annualized Return / Annualized Volatility (Sharpe-like)
        trend_score = abs(total_ret) # Raw magnitude of move
    except:
        trend_score = 0
        
    print(f"\n--- {label} SET ---")
    print(f"  Bars:       {len(prices)}")
    print(f"  Return:     {total_ret*100:.2f}%")
    print(f"  Ann. Vol:   {ann_vol*100:.2f}%")
    
    # Heuristic: Is it efficient?
    # If return is small but vol is high -> Choppy/Mean Reverting
    # If return is high -> Trending
    
    if abs(total_ret) < 0.02:
        regime = "Choppy / Flat"
    elif total_ret > 0.02:
        regime = "Bullish Trend"
    else:
        regime = "Bearish Trend"
        
    print(f"  Regime:     {regime}")
    
    # Check Difficulty (Return vs 1.25% Threshold)
    threshold = config.MIN_RETURN_THRESHOLD
    if total_ret < threshold:
        print(f"  ⚠️  DIFFICULTY WARNING: Market Return ({total_ret*100:.2f}%) is below your strategy threshold ({threshold*100:.2f}%).")
        print(f"      Long-only strategies will naturally fail here. Strategies must be active & precise.")

def diagnose():
    print("🏥 DIAGNOSING PIPELINE HEALTH...")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # Reconstruct Splits based on config
    n = len(df)
    train_idx = int(n * config.TRAIN_SPLIT_RATIO)
    val_idx = int(n * config.VAL_SPLIT_RATIO)
    
    prices = df['open'].values
    # Calculate log returns for vol
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    returns = df['log_ret'].values
    
    # 1. Train
    get_regime_stats(prices[:train_idx], returns[:train_idx], "TRAINING")
    
    # 2. Validation
    get_regime_stats(prices[train_idx:val_idx], returns[train_idx:val_idx], "VALIDATION")
    
    # 3. Test
    get_regime_stats(prices[val_idx:], returns[val_idx:], "TEST (OOS)")

import os
if __name__ == "__main__":
    diagnose()
