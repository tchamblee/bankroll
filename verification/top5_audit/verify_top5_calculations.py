import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from feature_engine import FeatureEngine

def audit_calculation(df, feature_name, description):
    print(f"\n--- Auditing: {feature_name} ({description}) ---")
    
    if feature_name not in df.columns:
        print(f"  ❌ Missing column: {feature_name}")
        return

    series = df[feature_name]
    print(f"  ✅ Data Found. Length: {len(series)}")
    print(f"  📊 Stats: Mean={series.mean():.4f}, Std={series.std():.4f}, Min={series.min():.4f}, Max={series.max():.4f}")
    
    # Check for frozen values
    if series.std() == 0:
        print("  ❌ FATAL: Feature is frozen (Zero Variance).")
    
    # Check Delta Logic if it's a Delta feature
    if 'delta_' in feature_name:
        # Reconstruct manually
        parts = feature_name.split('_')
        # Format: delta_base_feature_window
        # e.g. delta_beta_dxy_50 -> base=beta_dxy, window=50
        # This parsing is tricky, let's assume we know the base for the Top 5
        pass

if __name__ == "__main__":
    from feature_engine import create_full_feature_engine
    
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw_ticks'))
    
    print("Loading Data for Top 5 Audit...")
    engine = create_full_feature_engine(DATA_PATH)
    
    df = engine.bars
    
    # Top 5 Champions (Horizon 60)
    champions = [
        ("frac_diff_02", "Memory: Fractional Differentiation (0.2)"),
        ("beta_tnx", "Macro: Sensitivity to US 10Y Rates"),
        ("residual_dxy", "Macro: Idiosyncratic moves vs Dollar"),
        ("hurst_200", "Regime: Long-term Hurst Exponent"),
        ("delta_beta_spy_50", "Macro Flow: Change in SPY Sensitivity")
    ]
    
    for feat, desc in champions:
        audit_calculation(df, feat, desc)
        
    # Manual Logic Check for #1: delta_beta_dxy_50
    print("\n--- 🔬 Deep Dive: delta_beta_dxy_50 ---")
    if 'beta_dxy' in df.columns:
        manual_delta = df['beta_dxy'].diff(50)
        check = np.allclose(df['delta_beta_dxy_50'].fillna(0), manual_delta.fillna(0), atol=1e-5)
        print(f"  ✅ Manual Recalculation Match: {check}")
        print(f"  Sample Values:\n{df[['beta_dxy', 'delta_beta_dxy_50']].tail()}")
    else:
        print("  ❌ Base feature 'beta_dxy' missing.")
