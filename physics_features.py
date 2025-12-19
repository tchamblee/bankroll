import numpy as np
import pandas as pd
from scipy.stats import linregress

def get_weights(d, size):
    """
    Calculates weights for fractional differentiation.
    w_k = -w_{k-1} * (d - k + 1) / k
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    return np.array(w[::-1]) # Reverse for convolution

def frac_diff_ffd(series, d=0.4, thres=1e-4):
    """
    Fractional Differentiation with Fixed Window (FFD).
    Preserves memory while making series stationary.
    """
    # 1. Determine window size where weights drop below threshold
    w_test = [1.0]
    k = 1
    while abs(w_test[-1]) > thres:
        w_k = -w_test[-1] * (d - k + 1) / k
        w_test.append(w_k)
        k += 1
    
    width = len(w_test)
    weights = np.array(w_test[::-1]) # Weights for dot product
    
    # 2. Apply weights via rolling window
    # Valid only after 'width' data points
    df = pd.Series(series).dropna()
    res = df.rolling(window=width).apply(lambda x: np.dot(x, weights), raw=True)
    
    return res

def get_hurst_exponent(series, window=100, lags=[2, 4, 8, 16, 32, 64]):
    """
    Estimates Hurst Exponent using variance of differences at multiple lags.
    H = 0.5 (Random), >0.5 (Trend), <0.5 (Mean Revert).
    
    This is a rolling simplified version suitable for feature generation.
    """
    # We calculate the std dev of returns at different lags
    # log(std) ~ H * log(lag)
    
    def _calc_hurst(x):
        if len(x) < max(lags) + 1: return np.nan
        
        y_vals = []
        x_vals = np.log(lags)
        
        for lag in lags:
            # Std dev of price differences (returns) at this lag
            res = x - np.roll(x, lag)
            res = res[lag:] # remove nan
            if len(res) == 0: return np.nan
            y_vals.append(np.std(res))
        
        y_vals = np.log(y_vals)
        
        # Slope of log-log plot is H
        try:
            slope, _, _, _, _ = linregress(x_vals, y_vals)
            return slope
        except:
            return np.nan

    # Rolling application
    return series.rolling(window=window).apply(_calc_hurst, raw=True)

def get_efficiency_ratio(series, window=100):
    """
    Kaufman Efficiency Ratio (Fractal Efficiency).
    ER = Displacement / Path Length
    """
    # Displacement: Change from t-n to t
    change = series.diff(window).abs()
    
    # Path Length: Sum of absolute tick-to-tick changes
    # volatility = sum(|price[i] - price[i-1]|)
    path = series.diff().abs().rolling(window).sum()
    
    return change / path

def get_shannon_entropy(series, window=100, bins=20):
    """
    Calculates Rolling Shannon Entropy (Information Content).
    Higher Entropy = More Disorder/Noise. Lower Entropy = More Structure/Trend.
    
    H(X) = -sum(p(x) * log2(p(x)))
    """
    def _calc_entropy(x):
        # Discretize data into bins to estimate probability distribution
        counts, _ = np.histogram(x, bins=bins, density=False)
        # Probabilities
        p = counts / counts.sum()
        # Remove zeros to avoid log error
        p = p[p > 0]
        # Entropy
        return -np.sum(p * np.log2(p))

    # Apply on rolling log-returns (or stationarized series)
    return series.rolling(window=window).apply(_calc_entropy, raw=True)
