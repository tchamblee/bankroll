import pandas as pd
import numpy as np
import config
import os

def check_data_quality(df):
    print("\n🏥 Starting General Data Health Check...")
    
    # Identify non-numeric columns to skip (like timestamp)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Exclude metadata
    metadata_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    feature_cols = [c for c in numeric_cols if c not in metadata_cols]
    
    print(f"Checking {len(feature_cols)} features for anomalies...")
    
    issues_found = False
    
    # 1. NaN Check
    nan_counts = df[feature_cols].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    # Threshold for expected NaNs due to rolling windows (max window ~3200)
    NAN_THRESHOLD = 3500 
    
    unexpected_nans = nan_cols[nan_cols > NAN_THRESHOLD]
    expected_nans = nan_cols[nan_cols <= NAN_THRESHOLD]
    
    if not unexpected_nans.empty:
        issues_found = True
        print(f"\n⚠️  {len(unexpected_nans)} features contain EXCESSIVE NaNs (> {NAN_THRESHOLD}):")
        print(unexpected_nans.sort_values(ascending=False).head(20))
    elif not expected_nans.empty:
        print(f"✅ NaNs detected but within expected warmup range (<= {NAN_THRESHOLD} rows).")
    else:
        print("✅ No NaNs found in features.")

    # 2. Infinite Check
    # Check for both positive and negative infinity
    inf_counts = np.isinf(df[feature_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    
    if not inf_cols.empty:
        issues_found = True
        print(f"\n❌ {len(inf_cols)} features contain Infinite values (CRITICAL):")
        print(inf_cols.sort_values(ascending=False).head(20))
    else:
        print("✅ No Infinite values found.")
        
    # 3. Constant Check (Zero Variance)
    # Use std() == 0
    stds = df[feature_cols].std()
    const_cols = stds[stds == 0].index.tolist()
    
    if const_cols:
        issues_found = True
        print(f"\n⚠️  {len(const_cols)} features are CONSTANT (Zero Variance):")
        print(const_cols[:20])
        if len(const_cols) > 20:
            print(f"...and {len(const_cols)-20} more.")
    else:
        print("✅ No constant features found.")
        
    return issues_found

def check_leaks():
    print("🕵️  Starting Data Leakage Investigation...")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        return

    # Load Data
    print(f"Loading {config.DIRS['FEATURE_MATRIX']}...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # --- RUN HEALTH CHECK FIRST ---
    check_data_quality(df)
    
    # Calculate Target (Next Day Return)
    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
    df['target_next_ret'] = df['log_ret'].shift(-1)
    
    # Drop NaNs created by shift
    valid_df = df.dropna()
    
    print(f"\nAnalyzing {len(valid_df)} rows and {len(valid_df.columns)} features for LEAKAGE...")
    
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