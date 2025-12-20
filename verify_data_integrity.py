import pandas as pd
import numpy as np
import config
import os

def check_leaks():
    print("🕵️  Starting Data Leakage Investigation...")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        return

    # Load Data
    print(f"Loading {config.DIRS['FEATURE_MATRIX']}...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # Calculate Target (Next Day Return)
    # log_ret is return from T-1 to T.
    # We want to predict log_ret at T+1 (Target).
    # So Target[i] = log_ret[i+1].
    # Which corresponds to df['log_ret'].shift(-1).
    
    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
    df['target_next_ret'] = df['log_ret'].shift(-1)
    
    # Drop NaNs created by shift
    valid_df = df.dropna()
    
    print(f"Analyzing {len(valid_df)} rows and {len(valid_df.columns)} features...")
    
    # 1. Check Correlation with Future Return (The Holy Grail Leak)
    print("\n🔍 Checking for Direct Future Leaks (Correlation with T+1 Return)...")
    
    # Suppress divide by zero warnings for constant features
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs = valid_df.corrwith(valid_df['target_next_ret']).abs().sort_values(ascending=False)
    
    # Exclude the target itself from the report
    corrs = corrs.drop('target_next_ret', errors='ignore')
    
    print(corrs.head(20))
    
    potential_leaks = corrs[corrs > 0.2] # 0.2 is essentially impossible for daily data
    if not potential_leaks.empty:
        print(f"\n⚠️  FOUND {len(potential_leaks)} POTENTIAL LEAKS (Corr > 0.2 with Future):")
        print(potential_leaks)
    else:
        print("\n✅ No obvious direct linear leaks found (Max Corr < 0.2).")

    # 2. Check for "Perfect Predictors" of TODAY's return (which might be confused by the engine)
    print("\n🔍 Checking for Features Identical to Current Return (T)...")
    with np.errstate(divide='ignore', invalid='ignore'):
        corrs_current = valid_df.corrwith(valid_df['log_ret']).abs().sort_values(ascending=False)
    
    # Exclude the feature itself (log_ret) from the proxy check
    corrs_current = corrs_current.drop('log_ret', errors='ignore')
    
    print(corrs_current.head(10))

    # 3. Deep Dive into Top Suspects
    # If any feature has corr > 0.99 with current return, and the engine has an off-by-one error, that's the smoking gun.
    suspects = corrs_current[corrs_current > 0.95].index.tolist()
    if suspects:
        print(f"\n⚠️  {len(suspects)} features are effectively proxies for Current Return:")
        for s in suspects:
            print(f"   - {s}")

if __name__ == "__main__":
    check_leaks()