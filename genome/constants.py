# Strict grid to prevent overfitting
VALID_DELTA_LOOKBACKS = [5, 10, 20, 50, 100, 200]
VALID_ZSCORE_WINDOWS = [20, 50, 100, 200, 400]
VALID_CORR_WINDOWS = [20, 50, 100, 200]
VALID_FLUX_LAGS = [5, 10, 20, 50]
VALID_EFF_WINDOWS = [20, 50, 100, 200]
VALID_SLOPE_WINDOWS = [20, 50, 100] # Re-added for Divergence

# Threshold grids to prevent overfitting (no magic numbers)
VALID_SIGMA_THRESHOLDS = [-2.5, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 2.5]
VALID_SIGMA_THRESHOLDS_POSITIVE = [1.5, 2.0, 2.5, 3.0]  # For mean-reversion (always positive magnitude)
VALID_CORR_THRESHOLDS = [-0.6, -0.4, -0.2, 0.2, 0.4, 0.6]
VALID_EFFICIENCY_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
VALID_PERCENTAGE_THRESHOLDS = [0.7, 0.8, 0.9, 0.95]
