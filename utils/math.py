import numpy as np
from numba import jit

@jit(nopython=True, nogil=True, cache=True)
def calc_slope(arr: np.ndarray, w: int) -> np.ndarray:
    """
    Calculates rolling linear regression slope over window w.
    """
    n = len(arr)
    slope = np.zeros(n, dtype=np.float32)
    if n < w:
        return slope
        
    # Precompute X sums
    sum_x = (w * (w - 1)) / 2.0
    sum_x2 = (w * (w - 1) * (2 * w - 1)) / 6.0
    divisor = w * sum_x2 - sum_x ** 2
    
    # Kernel for sum_xy (decreasing weights because convolution flips)
    # y[i] is current. x is 0..w-1. 
    # We want slope of y vs x. 
    # x = [0, 1, ..., w-1]
    # y = [y[i-w+1], ..., y[i]]
    # Convolution correlates reversed kernel.
    # So kernel should be x[::-1] = [w-1, w-2, ..., 0]
    
    kernel_xy = np.arange(w, dtype=np.float32)[::-1]
    kernel_y = np.ones(w, dtype=np.float32)
    
    # Use convolution for speed
    sum_xy = np.convolve(arr, kernel_xy, mode='full')[:n]
    sum_y = np.convolve(arr, kernel_y, mode='full')[:n]
    
    # Formula: (N * Sum(XY) - Sum(X)*Sum(Y)) / (N * Sum(X^2) - Sum(X)^2)
    numerator = w * sum_xy - sum_x * sum_y
    
    # Handle division by zero (unlikely with fixed x unless w=1)
    if divisor == 0:
        return slope
        
    slope[:] = numerator / divisor
    slope[:w-1] = 0.0 # Warmup
    return slope

@jit(nopython=True, nogil=True, cache=True)
def calc_zscore(arr: np.ndarray, w: int) -> np.ndarray:
    """
    Calculates rolling Z-Score over window w.
    """
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    if n < w:
        return z
        
    kernel = np.ones(w, dtype=np.float32)
    
    sum_x = np.convolve(arr, kernel, mode='full')[:n]
    sum_x2 = np.convolve(arr * arr, kernel, mode='full')[:n]
    
    mean = sum_x / w
    mean_x2 = sum_x2 / w
    var = np.maximum(mean_x2 - mean**2, 0.0)
    std = np.sqrt(var)
    
    # Avoid div/0
    mask = std > 1e-9
    z[mask] = (arr[mask] - mean[mask]) / std[mask]
    
    z[:w-1] = 0.0
    return z

@jit(nopython=True, nogil=True, cache=True)
def calc_correlation(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    """
    Calculates rolling Pearson correlation between a and b over window w.
    """
    n = len(a)
    corr = np.zeros(n, dtype=np.float32)
    if n < w:
        return corr
        
    kernel = np.ones(w, dtype=np.float32)
    
    sum_a = np.convolve(a, kernel, mode='full')[:n]
    sum_b = np.convolve(b, kernel, mode='full')[:n]
    sum_ab = np.convolve(a * b, kernel, mode='full')[:n]
    sum_aa = np.convolve(a * a, kernel, mode='full')[:n]
    sum_bb = np.convolve(b * b, kernel, mode='full')[:n]
    
    var_a_term = np.maximum(w * sum_aa - sum_a**2, 0.0)
    var_b_term = np.maximum(w * sum_bb - sum_b**2, 0.0)
    
    numerator = w * sum_ab - sum_a * sum_b
    denominator = np.sqrt(var_a_term * var_b_term)
    
    mask = denominator > 1e-9
    corr[mask] = numerator[mask] / denominator[mask]
    
    corr[:w-1] = 0.0
    return np.clip(corr, -1.0, 1.0)

@jit(nopython=True, nogil=True, cache=True)
def calc_efficiency(arr: np.ndarray, w: int) -> np.ndarray:
    """
    Calculates Kaufman Efficiency Ratio: |Change| / Sum(|Diffs|)
    """
    n = len(arr)
    er = np.zeros(n, dtype=np.float32)
    if n <= w:
        return er
        
    # Net Change over w
    change = np.zeros(n, dtype=np.float32)
    change[w:] = np.abs(arr[w:] - arr[:-w])
    
    # Sum of absolute diffs (Path Length)
    diffs = np.zeros(n, dtype=np.float32)
    diffs[1:] = np.abs(arr[1:] - arr[:-1])
    
    kernel = np.ones(w, dtype=np.float32)
    path = np.convolve(diffs, kernel, mode='full')[:n]
    
    mask = path > 1e-9
    er[mask] = change[mask] / path[mask]
    
    er[:w] = 0.0
    return er

@jit(nopython=True, nogil=True, cache=True)
def calc_delta(arr: np.ndarray, w: int) -> np.ndarray:
    """
    Calculates Delta (Change) over window w.
    """
    n = len(arr)
    delta = np.zeros(n, dtype=np.float32)
    if n <= w:
        return delta
    
    delta[w:] = arr[w:] - arr[:-w]
    return delta

@jit(nopython=True, nogil=True, cache=True)
def calc_flux(arr: np.ndarray, lag: int) -> np.ndarray:
    """
    Calculates Flux (Second Derivative / Acceleration approximation).
    Flux = (x[t] - x[t-lag]) - (x[t-lag] - x[t-2*lag])
    """
    n = len(arr)
    flux = np.zeros(n, dtype=np.float32)
    if n <= 2 * lag:
        return flux
        
    # First derivative
    d1 = np.zeros(n, dtype=np.float32)
    d1[lag:] = arr[lag:] - arr[:-lag]
    
    # Second derivative (change of d1)
    flux[lag:] = d1[lag:] - d1[:-lag]
    
    return flux
