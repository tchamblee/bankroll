import numpy as np
import pandas as pd
from numba import jit
from scipy.signal import lfilter

@jit(nopython=True)
def _jit_fisher_transform(prices, window):
    """
    Calculates Ehlers Fisher Transform.
    """
    n = len(prices)
    fisher = np.full(n, np.nan)
    signal = np.full(n, np.nan)
    value1 = np.zeros(n)
    
    if n < window:
        return fisher, signal
        
    for i in range(1, n):
        # 1. Normalize price to -1 to 1 range over window
        start_idx = max(0, i - window + 1)
        
        mx = prices[i]
        mn = prices[i]
        
        # Find Min/Max in window
        for j in range(start_idx, i + 1):
             val = prices[j]
             if val > mx: mx = val
             if val < mn: mn = val
             
        if mx == mn:
            norm = 0.0
        else:
            norm = 2.0 * ((prices[i] - mn) / (mx - mn) - 0.5)
            
        # 2. Smooth (Value1)
        # Value1 = 0.33 * 2 * ((Price - Min) / (Max - Min) - 0.5) + 0.67 * Value1[1]
        prev_val1 = value1[i-1]
        val = 0.33 * norm + 0.67 * prev_val1
        
        # Soft clip to avoid singularities
        if val > 0.99: val = 0.999
        if val < -0.99: val = -0.999
        
        value1[i] = val
        
        # 3. Fisher Transform
        # Fish = 0.5 * log((1 + Value1) / (1 - Value1)) + 0.5 * Fish[1]
        prev_fish = 0.0 if np.isnan(fisher[i-1]) else fisher[i-1]
        
        current_fish = 0.5 * np.log((1.0 + val) / (1.0 - val)) + 0.5 * prev_fish
        
        fisher[i] = current_fish
        signal[i] = prev_fish # Shifted by 1
        
    return fisher, signal

def _get_hilbert_fir_coeffs(num_taps=31):
    """
    Generates coefficients for a causal Type III FIR Hilbert Transformer.
    Delay = (num_taps - 1) / 2
    """
    center = (num_taps - 1) / 2
    h = np.zeros(num_taps)
    for i in range(num_taps):
        k = i - center
        if k == 0:
            h[i] = 0
        elif k % 2 != 0: # Odd
            h[i] = 2 / (np.pi * k)
        else: # Even
            h[i] = 0
            
    # Apply Hamming Window to reduce ripple
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(num_taps) / (num_taps - 1))
    return h * w

def add_dsp_features(df, windows=[20, 40]):
    """
    Adds Digital Signal Processing features:
    1. Ehlers Fisher Transform (Gaussian Normalization)
    2. Hilbert Sine Wave (Cycle Phase via Causal FIR Hilbert Transform)
    """
    if df is None: return None
    print("Calculating DSP Features (Fisher, Hilbert)...")
    df = df.copy()
    
    # Use Median Price for DSP (High+Low)/2 or Close
    # Ehlers prefers (H+L)/2
    src = ((df['high'] + df['low']) / 2).values
    
    # 1. Fisher Transform
    for w in windows:
        f, s = _jit_fisher_transform(src, w)
        df[f'fisher_{w}'] = f
        df[f'fisher_signal_{w}'] = s
        
    # 2. Hilbert Sine Wave (Causal)
    # We use a fixed tap size for the Hilbert filter.
    # N=61 provides ~30 bar delay but good accuracy.
    # N=31 provides ~15 bar delay.
    taps = 61
    delay = (taps - 1) // 2
    coeffs = _get_hilbert_fir_coeffs(taps)
    
    # Pre-calculate Detrended Series for each window (High Pass)
    # Sine Wave needs to operate on a detrended series to see cycles.
    for w in windows:
        # Simple robust detrender: Price - SMA
        # Or Price - LinearRegression? SMA is faster/stable.
        # Use a longer SMA for detrending to preserve the cycle of interest (w).
        # E.g. if looking for cycle w, detrend with 2*w or 4*w? 
        # Actually, Ehlers Sine Wave is usually adaptive. 
        # Here we make a "Spectral Sine Wave" at specific frequencies.
        
        sma = df['close'].rolling(w).mean() # High pass cutoff ~ w
        detrended = (df['close'] - sma).fillna(0).values
        
        # Apply Hilbert Filter -> Quadrature
        # lfilter(b, a, x)
        quadrature = lfilter(coeffs, 1.0, detrended)
        
        # InPhase is simply the Detrended signal delayed
        # in_phase = detrended.shift(delay)
        in_phase = np.full_like(detrended, np.nan)
        in_phase[delay:] = detrended[:-delay]
        
        # Compute Phase
        # We need to handle the startup NaNs
        valid_idx = delay
        
        # Avoid division by zero in arctan2 (handled by numpy)
        # Phase = atan(Q / I)
        # Using negative Q to match Ehlers convention? 
        # Analytic signal is I + jQ. 
        phase = np.arctan2(quadrature, in_phase)
        
        # Sine Indicators
        # Shift back to align with price? 
        # No, the filter delay implies the feature at time t corresponds to event at t-delay.
        # This is strictly causal. The "Phase" tells us the phase of the cycle *delayed by 15 bars*.
        # For prediction, we want to know the phase *now*.
        # We cannot know the phase *now* perfectly without lag (uncertainty principle).
        # However, ML can learn "Phase at t-15 was X, so Price at t+1 is Y".
        
        df[f'sine_{w}'] = np.sin(phase)
        df[f'leadsine_{w}'] = np.sin(phase + np.pi/4)
        
    return df
