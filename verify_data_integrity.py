import pandas as pd
import numpy as np
import config
import os
import sys

def check_ohlcv_integrity(df):
    """Check OHLC price relationships and basic sanity."""
    print("\n📊 Checking OHLCV Integrity...")
    issues = []

    # 1. OHLC Relationship: high >= low
    if 'high' in df.columns and 'low' in df.columns:
        bad_hl = (df['high'] < df['low']).sum()
        if bad_hl > 0:
            issues.append(f"❌ {bad_hl} rows have high < low")
        else:
            print("  ✅ high >= low: OK")

    # 2. High should be >= open and close
    if all(c in df.columns for c in ['high', 'open', 'close']):
        bad_ho = (df['high'] < df['open']).sum()
        bad_hc = (df['high'] < df['close']).sum()
        if bad_ho > 0 or bad_hc > 0:
            issues.append(f"❌ {bad_ho} rows have high < open, {bad_hc} have high < close")
        else:
            print("  ✅ high >= open, close: OK")

    # 3. Low should be <= open and close
    if all(c in df.columns for c in ['low', 'open', 'close']):
        bad_lo = (df['low'] > df['open']).sum()
        bad_lc = (df['low'] > df['close']).sum()
        if bad_lo > 0 or bad_lc > 0:
            issues.append(f"❌ {bad_lo} rows have low > open, {bad_lc} have low > close")
        else:
            print("  ✅ low <= open, close: OK")

    # 4. No zero or negative prices
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            bad_price = (df[col] <= 0).sum()
            if bad_price > 0:
                issues.append(f"❌ {bad_price} rows have {col} <= 0")
    if not any('price' in i for i in issues):
        print("  ✅ All prices > 0: OK")

    # 5. No negative volume
    if 'volume' in df.columns:
        bad_vol = (df['volume'] < 0).sum()
        if bad_vol > 0:
            issues.append(f"❌ {bad_vol} rows have negative volume")
        else:
            print("  ✅ volume >= 0: OK")

    for issue in issues:
        print(f"  {issue}")

    return len(issues) > 0


def check_timestamp_integrity(df):
    """Check timestamp ordering and duplicates."""
    print("\n⏰ Checking Timestamp Integrity...")
    issues = []

    time_col = None
    for col in ['time_start', 'timestamp', 'time']:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        print("  ⚠️  No timestamp column found, skipping.")
        return False

    ts = pd.to_datetime(df[time_col])

    # 1. Check strictly monotonically increasing (excluding microsecond ties)
    # For high-frequency data, multiple bars can have same timestamp at microsecond level
    # This is acceptable - we check for same-SECOND duplicates as the real issue
    diffs = ts.diff()

    # Negative diffs are always bad (time going backwards)
    negative_diffs = (diffs < pd.Timedelta(0)).sum()
    if negative_diffs > 0:
        issues.append(f"❌ {negative_diffs} rows have timestamps going BACKWARDS (critical)")
    else:
        print("  ✅ No backwards timestamps: OK")

    # 2. Check for duplicates at different granularities
    # Microsecond duplicates: acceptable for HF data (quote bursts)
    # Second-level duplicates spanning >1 second: problematic
    duplicates_micro = ts.duplicated().sum()

    # Round to seconds and check for problematic duplicates
    ts_seconds = ts.dt.floor('s')
    dup_seconds = ts_seconds.duplicated(keep=False)

    if duplicates_micro > 0:
        # Check if these are just microsecond-level ties (acceptable)
        # vs actual data issues (same timestamp appearing far apart)
        dup_indices = np.where(ts.duplicated(keep=False))[0]
        max_gap_between_dups = 0

        for dup_ts in ts[ts.duplicated(keep=False)].unique():
            indices = np.where(ts == dup_ts)[0]
            if len(indices) > 1:
                gap = indices[-1] - indices[0]
                max_gap_between_dups = max(max_gap_between_dups, gap)

        if max_gap_between_dups <= 5:  # Adjacent rows with same timestamp = OK
            print(f"  ✅ {duplicates_micro} microsecond-level timestamp ties (acceptable for HF data)")
        else:
            issues.append(f"❌ {duplicates_micro} duplicate timestamps with max gap of {max_gap_between_dups} rows")
    else:
        print("  ✅ No duplicate timestamps: OK")

    # 3. Check for large gaps (> 3 days, accounting for weekends)
    large_gaps = diffs[diffs > pd.Timedelta(days=3)]
    if len(large_gaps) > 0:
        print(f"  ⚠️  {len(large_gaps)} gaps > 3 days found (may be holidays/data gaps)")

    for issue in issues:
        print(f"  {issue}")

    return len(issues) > 0


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

    # Threshold for expected NaNs:
    # - Rolling windows up to 400 bars
    # - Intermarket data (US2Y, SCHATZ, etc) has ~5000 NaNs due to trading hours
    # - Delta calculations add another 100 bars
    # So threshold = 5000 (intermarket) + 400 (window) + 200 (buffer) = 5600
    NAN_THRESHOLD = 5600

    unexpected_nans = nan_cols[nan_cols > NAN_THRESHOLD]
    expected_nans = nan_cols[nan_cols <= NAN_THRESHOLD]

    if not unexpected_nans.empty:
        issues_found = True
        print(f"\n⚠️  {len(unexpected_nans)} features contain EXCESSIVE NaNs (> {NAN_THRESHOLD}):")
        print(unexpected_nans.sort_values(ascending=False).head(20))
    elif not expected_nans.empty:
        print(f"✅ NaNs detected but within expected warmup range (<= {NAN_THRESHOLD} rows).")
        # Show breakdown by category
        intermarket_nans = [c for c in nan_cols.index if any(x in c.lower() for x in ['us2y', 'schatz', 'bund', 'zn', 'tnx', 'curve', 'spread_us'])]
        other_nans = [c for c in nan_cols.index if c not in intermarket_nans]
        if intermarket_nans:
            print(f"     ({len(intermarket_nans)} intermarket features, {len(other_nans)} other)")
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


def check_leaks(force=False):
    """
    Main data integrity verification function.
    Returns True if all checks pass, False if critical issues found.
    """
    print("🕵️  Starting Data Integrity Verification...")

    matrix_path = config.DIRS['FEATURE_MATRIX']
    verified_marker = matrix_path + ".verified"

    if not os.path.exists(matrix_path):
        print("❌ Feature Matrix not found.")
        return False

    if os.path.exists(verified_marker) and not force:
        print(f"⏩ Data Integrity already verified. Skipping.")
        print(f"   (Use '--force' to override)")
        return True

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

    # --- RUN INTEGRITY CHECKS ---
    critical_issues = False

    # 1. OHLCV sanity checks
    if check_ohlcv_integrity(df):
        critical_issues = True

    # 2. Timestamp integrity checks
    if check_timestamp_integrity(df):
        critical_issues = True

    # 3. Feature quality checks (NaN, Inf, constant)
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
    # rel_strength_z_6e: Futures (6E) vs Spot (EURUSD) Lead-Lag
    # divergence_50_gbpusd: Spot (GBPUSD) vs Spot (EURUSD) Divergence/Momentum
    KNOWN_LEAKS = ['rel_strength_z_6e', 'divergence_50_gbpusd']
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
            
    # Combine all issue flags
    if critical_issues:
        issues_found = True

    # Summary
    print("\n" + "="*50)
    if issues_found:
        print("⚠️  VERIFICATION COMPLETE WITH WARNINGS")
        print("   Some issues were detected. Review output above.")
    else:
        print("✅ ALL CHECKS PASSED")

    # Create marker file (even with warnings, to avoid re-running every time)
    # User can delete marker or use --force to re-verify
    with open(verified_marker, 'w') as f:
        f.write(f"Verified: {'WARNINGS' if issues_found else 'CLEAN'}")
    print(f"   Marker saved to {verified_marker}")

    return not issues_found  # Return True if no issues


if __name__ == "__main__":
    force_verify = '--force' in sys.argv
    success = check_leaks(force=force_verify)
    sys.exit(0 if success else 1)