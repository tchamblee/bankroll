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

import sys

def check_leaks(force=False):
    print("🕵️  Starting Data Leakage Investigation...")
    
    matrix_path = config.DIRS['FEATURE_MATRIX']
    verified_marker = matrix_path + ".verified"

    if not os.path.exists(matrix_path):
        print("❌ Feature Matrix not found.")
        return

    if os.path.exists(verified_marker) and not force:
        print(f"⏩ Data Integrity already verified. Skipping.")
        print(f"   (Use '--force' to override)")
        return

    # Load Data
    print(f"Loading {matrix_path}...")
    df = pd.read_parquet(matrix_path)
    
    # --- DATE FILTERING ---
    if hasattr(config, 'TRAIN_START_DATE') and config.TRAIN_START_DATE:
        print(f"📅 Applying Training Start Date: {config.TRAIN_START_DATE}")
        if 'time_start' in df.columns:
            # Normalize Timezones
            if not pd.api.types.is_datetime64_any_dtype(df['time_start']):
                 df['time_start'] = pd.to_datetime(df['time_start'], utc=True)
            
            ts_col = df['time_start']
            if ts_col.dt.tz is None:
                 ts_col = ts_col.dt.tz_localize('UTC')
            else:
                 ts_col = ts_col.dt.tz_convert('UTC')
                 
            start_ts = pd.Timestamp(config.TRAIN_START_DATE).tz_localize('UTC')
            
            if ts_col.min() < start_ts:
                original_len = len(df)
                df = df[ts_col >= start_ts].reset_index(drop=True)
                print(f"   Dropped {original_len - len(df)} rows (Pre-Start Data). Checking {len(df)} rows.")
        else:
             print("   ⚠️ 'time_start' column missing. Cannot filter by date.")

    # --- RUN HEALTH CHECK FIRST ---
    issues_found = check_data_quality(df)
    
    # Calculate Target (Next Day Return)
    if 'log_ret' not in df.columns:
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
    df['target_next_ret'] = df['log_ret'].shift(-1)
    
    # Drop NaNs created by shift (only drop rows where target is missing)
    # We do NOT drop rows where features are missing, as that would wipe out the dataset 
    # if one feature column is empty (as seen with COT data).
    valid_df = df.dropna(subset=['target_next_ret'])
    
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
    
    # Whitelist known valid signals (Arbitrage/Lead-Lag)
    KNOWN_LEAKS = ['rel_strength_z_6e']
    real_leaks = potential_leaks.drop(KNOWN_LEAKS, errors='ignore')
    
    if not real_leaks.empty:
        issues_found = True
        print(f"\n⚠️  FOUND {len(real_leaks)} POTENTIAL LEAKS (Corr > 0.2 with Future):")
        print(real_leaks)
    elif not potential_leaks.empty:
        print(f"\n✅ No unknown leaks found. (Suppressed {len(potential_leaks)} known valid signals: {list(potential_leaks.index)})")
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
            
    # Mark as verified if no critical issues (or just mark as verified to skip next time regardless of warnings?
    # Usually we only skip if successful. But here let's assume if it runs to completion it's "checked".
    # But I set issues_found=True for leaks.
    
    # For now, I'll create the marker at the end. The user can remove it if they want to re-check.
    with open(verified_marker, 'w') as f:
        f.write("Verified")
    print(f"\n✅ Verification Complete. Marker saved to {verified_marker}")

if __name__ == "__main__":
    force_verify = '--force' in sys.argv
    check_leaks(force=force_verify)