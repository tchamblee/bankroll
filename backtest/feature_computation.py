import os
import numpy as np
import pandas as pd

def _save_feature(temp_dir, existing_keys, name, arr):
    path = os.path.join(temp_dir, f"{name}.npy")
    np.save(path, arr)
    existing_keys.add(name)

def precompute_base_features(raw_data, temp_dir, existing_keys):
    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        _save_feature(temp_dir, existing_keys, col, raw_data[col].values.astype(np.float32))
    
    if 'time_start' in raw_data.columns:
        dt = raw_data['time_start'].dt
        _save_feature(temp_dir, existing_keys, 'time_hour', dt.hour.values.astype(np.float32))
        _save_feature(temp_dir, existing_keys, 'time_weekday', dt.dayofweek.values.astype(np.float32))
        
    close = raw_data['close'].values
    up_mask = (close > np.roll(close, 1)); up_mask[0] = False
    
    def get_streak(mask):
        streaks = np.zeros(len(mask), dtype=np.float32)
        current = 0
        for i in range(len(mask)):
            current = (current + 1) if mask[i] else 0
            streaks[i] = current
        return streaks

    # Precompute Base ATR (for Dynamic Barriers)
    # TR = Max(H-L, |H-C_prev|, |L-C_prev|)
    h, l, c = raw_data['high'].values, raw_data['low'].values, raw_data['close'].values
    c_prev = np.roll(c, 1); c_prev[0] = c[0]
    
    tr1 = h - l
    tr2 = np.abs(h - c_prev)
    tr3 = np.abs(l - c_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Simple Moving Average for ATR (Window 50 for stability approx 2 hours)
    # Using convolution for speed
    w_atr = 50
    if len(tr) >= w_atr:
        kernel = np.ones(w_atr) / w_atr
        atr = np.convolve(tr, kernel, mode='full')[:len(tr)]
        atr[:w_atr] = tr[:w_atr] # Fill warmup with raw TR or mean of available
    else:
        atr = tr
        
    _save_feature(temp_dir, existing_keys, 'atr_base', atr.astype(np.float32))

def ensure_feature_context(population, temp_dir, existing_keys):
    needed = set()
    for strat in population:
        all_genes = strat.long_genes + strat.short_genes + strat.regime_genes
        for gene in all_genes:
            if gene.type == 'delta': needed.add(('delta', gene.feature, gene.lookback))
            elif gene.type == 'zscore': needed.add(('zscore', gene.feature, gene.window))
            
            # Implicit Z-Score dependency for EventGene
            elif gene.type == 'event': 
                if gene.feature.startswith('zscore_'):
                    # format: zscore_{feature}_{window}
                    parts = gene.feature.split('_')
                    # Last part is window, rest is feature name (rejoined)
                    # zscore_close_800 -> feature='close', window=800
                    # zscore_delta_close_50_800 -> feature='delta_close_50', window=800
                    w = int(parts[-1])
                    feat = "_".join(parts[1:-1])
                    needed.add(('zscore', feat, w))
            
            elif gene.type == 'correlation': needed.add(('correlation', gene.feature_left, gene.feature_right, gene.window))
            elif gene.type == 'flux': needed.add(('flux', gene.feature, gene.lag))
            elif gene.type == 'divergence': 
                needed.add(('slope', gene.feature_a, gene.window))
                needed.add(('slope', gene.feature_b, gene.window))
            elif gene.type == 'efficiency': needed.add(('efficiency', gene.feature, gene.window))
    
    # Helper to load feature from disk for calculation
    def get_data(key):
        if key not in existing_keys: return None
        return np.load(os.path.join(temp_dir, f"{key}.npy"))

    for item in needed:
        type_ = item[0]
        
        if type_ == 'delta':
            feature, param = item[1], item[2]
            key = f"delta_{feature}_{param}"
            if key in existing_keys: continue
            
            arr = get_data(feature)
            if arr is not None:
                w = param
                diff = np.zeros_like(arr)
                if w < len(arr): diff[w:] = arr[w:] - arr[:-w]
                _save_feature(temp_dir, existing_keys, key, diff)

        elif type_ == 'flux':
            feature, lag = item[1], item[2]
            key = f"flux_{feature}_{lag}"
            if key in existing_keys: continue
            
            arr = get_data(feature)
            if arr is not None:
                # Flux = Delta(Delta(X, lag), lag)
                d1 = np.zeros_like(arr)
                if len(arr) > lag: d1[lag:] = arr[lag:] - arr[:-lag]
                
                flux = np.zeros_like(arr)
                if len(arr) > lag: flux[lag:] = d1[lag:] - d1[:-lag]
                _save_feature(temp_dir, existing_keys, key, flux)

        elif type_ == 'slope':
            feature, w = item[1], item[2]
            key = f"slope_{feature}_{w}"
            if key in existing_keys: continue
            
            arr = get_data(feature)
            if arr is not None:
                y = arr
                n = w
                if len(y) > w:
                    sum_x = (n * (n - 1)) / 2
                    sum_x2 = (n * (n - 1) * (2 * n - 1)) / 6
                    divisor = n * sum_x2 - sum_x ** 2
                    
                    kernel_xy = np.arange(n)[::-1]
                    sum_xy = np.convolve(y, kernel_xy, mode='full')[:len(y)]
                    sum_y = np.convolve(y, np.ones(n), mode='full')[:len(y)]
                    
                    slope = (n * sum_xy - sum_x * sum_y) / (divisor + 1e-9)
                    slope[:n] = 0.0
                    _save_feature(temp_dir, existing_keys, key, slope.astype(np.float32))

        elif type_ == 'efficiency':
            feature, w = item[1], item[2]
            key = f"eff_{feature}_{w}"
            if key in existing_keys: continue
            
            arr = get_data(feature)
            if arr is not None:
                # ER = Change / Path
                change = np.zeros_like(arr)
                if len(arr) > w: change[w:] = np.abs(arr[w:] - arr[:-w])
                
                diff1 = np.zeros_like(arr)
                diff1[1:] = np.abs(arr[1:] - arr[:-1])
                
                kernel = np.ones(w)
                path = np.convolve(diff1, kernel, mode='full')[:len(arr)]
                
                er = np.divide(change, path, out=np.zeros_like(change), where=path!=0)
                er[:w] = 0.0
                _save_feature(temp_dir, existing_keys, key, er.astype(np.float32))

        elif type_ == 'zscore':
            feature, param = item[1], item[2]
            key = f"zscore_{feature}_{param}"
            if key in existing_keys: continue

            # Vectorized Numpy Rolling Z-Score
            arr = get_data(feature)
            if arr is not None:
                w = param
                n_len = len(arr)
                if n_len >= w:
                    kernel = np.ones(w)
                    
                    # Rolling Sums
                    sum_x = np.convolve(arr, kernel, 'full')[:n_len]
                    sum_x2 = np.convolve(arr * arr, kernel, 'full')[:n_len]
                    
                    # Rolling Mean and Variance
                    # Mean = Sum / w
                    # Var = E[x^2] - (E[x])^2
                    
                    mean = sum_x / w
                    # To avoid precision issues with Mean(x^2) - Mean(x)^2, use robust variance if needed
                    # But standard fast method:
                    mean_x2 = sum_x2 / w
                    var = mean_x2 - mean**2
                    var = np.maximum(var, 0) # Clip negative 0
                    
                    std = np.sqrt(var)
                    
                    z = np.divide(arr - mean, std, out=np.zeros_like(arr), where=std!=0)
                    
                    # Fix startup (first w-1 are invalid due to growing window in 'full' convolution)
                    z[:w-1] = 0.0
                    
                    _save_feature(temp_dir, existing_keys, key, z.astype(np.float32))
        
        elif type_ == 'correlation':
            f1, f2, w = item[1], item[2], item[3]
            f1, f2 = sorted([f1, f2])
            key = f"corr_{f1}_{f2}_{w}"
            if key in existing_keys: continue
            
            a = get_data(f1)
            b = get_data(f2)
            
            if a is not None and b is not None:
                n = w
                kernel = np.ones(n)
                
                aa = a * a
                bb = b * b
                ab = a * b
                
                sum_a = np.convolve(a, kernel, 'full')[:len(a)]
                sum_b = np.convolve(b, kernel, 'full')[:len(b)]
                sum_ab = np.convolve(ab, kernel, 'full')[:len(ab)]
                sum_aa = np.convolve(aa, kernel, 'full')[:len(aa)]
                sum_bb = np.convolve(bb, kernel, 'full')[:len(bb)]
                
                var_a_term = np.maximum(n * sum_aa - sum_a**2, 0)
                var_b_term = np.maximum(n * sum_bb - sum_b**2, 0)
                
                numerator = n * sum_ab - sum_a * sum_b
                denominator = np.sqrt(var_a_term * var_b_term)
                
                corr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
                corr[:n-1] = 0.0
                corr = np.clip(corr, -1.0, 1.0)
                
                _save_feature(temp_dir, existing_keys, key, corr.astype(np.float32))
