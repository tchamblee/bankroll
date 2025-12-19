import pandas as pd
import numpy as np
import physics_features as phys

# Generate dummy price data (random walk)
np.random.seed(42)
prices = np.cumprod(1 + np.random.normal(0, 0.01, 1000)) * 100
series = pd.Series(prices)

print("--- Debugging FracDiff ---")
print(f"Input Length: {len(series)}")

# Test d=0.4 with stricter threshold
print("\nTesting d=0.4 (1e-5)...")
try:
    out = phys.frac_diff_ffd(series, d=0.4, thres=1e-5)
    print(f"NaN Count: {out.isna().sum()}")
except Exception as e:
    print(f"Error: {e}")

# Test d=0.2 with stricter threshold
print("\nTesting d=0.2 (1e-5)...")
try:
    out = phys.frac_diff_ffd(series, d=0.2, thres=1e-5)
    print(f"NaN Count: {out.isna().sum()}")
except Exception as e:
    print(f"Error: {e}")
