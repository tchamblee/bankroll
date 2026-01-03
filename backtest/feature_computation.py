import os
import numpy as np
import pandas as pd
from utils import math as umath
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

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
    # Streak: consecutive ups/downs? The original code had a simple streak logic
    # "up_mask = ... streaks = ..."
    # We'll preserve it or use a math util if complex. It was simple.
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
    
    # Simple Moving Average for ATR (Window 50)
    w_atr = 50
    if len(tr) >= w_atr:
        kernel = np.ones(w_atr) / w_atr
        atr = np.convolve(tr, kernel, mode='full')[:len(tr)]
        atr[:w_atr] = tr[:w_atr] 
    else:
        atr = tr
        
    _save_feature(temp_dir, existing_keys, 'atr_base', atr.astype(np.float32))

def ensure_feature_context(population, temp_dir, existing_keys):
    needed = set()
    
    def parse_feature_dependencies(feature_name):
        parts = feature_name.split('_')
        
        if feature_name.startswith('delta_') and parts[-1].isdigit():
             lookback = int(parts[-1])
             inner_feature = "_".join(parts[1:-1])
             needed.add(('delta', inner_feature, lookback))
             parse_feature_dependencies(inner_feature)
             
        elif feature_name.startswith('zscore_') and parts[-1].isdigit():
             window = int(parts[-1])
             inner_feature = "_".join(parts[1:-1])
             needed.add(('zscore', inner_feature, window))
             parse_feature_dependencies(inner_feature)
        
        elif feature_name.startswith('slope_') and parts[-1].isdigit():
             window = int(parts[-1])
             inner_feature = "_".join(parts[1:-1])
             needed.add(('slope', inner_feature, window))
             parse_feature_dependencies(inner_feature)

        elif feature_name.startswith('flux_') and parts[-1].isdigit():
             lag = int(parts[-1])
             inner_feature = "_".join(parts[1:-1])
             needed.add(('flux', inner_feature, lag))
             parse_feature_dependencies(inner_feature)
             
        elif feature_name.startswith('eff_') and parts[-1].isdigit():
             # 'eff' usually efficiency
             window = int(parts[-1])
             inner_feature = "_".join(parts[1:-1])
             needed.add(('efficiency', inner_feature, window))
             parse_feature_dependencies(inner_feature)

    # print(f"    [FeatureCtx] Parsing dependencies for {len(population)} strategies...")
    for i, strat in enumerate(population):
        # if (i+1) % 2000 == 0: print(f"      ... Parsed {i+1} strategies", end='\r')
        all_genes = strat.long_genes + strat.short_genes
        for gene in all_genes:
            if gene.type == 'delta': 
                needed.add(('delta', gene.feature, gene.lookback))
                parse_feature_dependencies(gene.feature)
            elif gene.type == 'zscore': 
                needed.add(('zscore', gene.feature, gene.window))
                parse_feature_dependencies(gene.feature)
            elif gene.type == 'persistence':
                parse_feature_dependencies(gene.feature)
            elif gene.type == 'cross' or gene.type == 'relational':
                if hasattr(gene, 'feature_left'): parse_feature_dependencies(gene.feature_left)
                if hasattr(gene, 'feature_right'): parse_feature_dependencies(gene.feature_right)
            elif gene.type == 'correlation': 
                needed.add(('correlation', gene.feature_left, gene.feature_right, gene.window))
                parse_feature_dependencies(gene.feature_left)
                parse_feature_dependencies(gene.feature_right)
            elif gene.type == 'flux': 
                needed.add(('flux', gene.feature, gene.lag))
                parse_feature_dependencies(gene.feature)
            elif gene.type == 'divergence': 
                needed.add(('slope', gene.feature_a, gene.window))
                needed.add(('slope', gene.feature_b, gene.window))
                parse_feature_dependencies(gene.feature_a)
                parse_feature_dependencies(gene.feature_b)
            elif gene.type == 'efficiency': 
                needed.add(('efficiency', gene.feature, gene.window))
                parse_feature_dependencies(gene.feature)
            elif gene.type == 'event':
                 if hasattr(gene, 'feature'): parse_feature_dependencies(gene.feature)
    # print("") # Newline
    
    # LRU-style Cache for Input Data (OrderedDict for O(1) access)
    _cache = OrderedDict()
    MAX_CACHE_ITEMS = 6000 
    cache_lock = Lock()

    def get_data_safe(key):
        """Thread-safe retrieval from cache or disk."""
        # 1. Try Cache (Fast, O(1))
        with cache_lock:
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]

        if key not in existing_keys: return None

        # 2. Load from Disk (Slow)
        try:
            arr = np.load(os.path.join(temp_dir, f"{key}.npy"))
        except:
            return None
            
        # 3. Update Cache
        with cache_lock:
            _cache[key] = arr
            _cache.move_to_end(key)
            if len(_cache) > MAX_CACHE_ITEMS:
                _cache.popitem(last=False) # Pop oldest
        return arr

    def compute_feature_task(item):
        """Worker function for a single feature."""
        type_ = item[0]
        result = None
        key = None
        
        if type_ == 'delta':
            feature, param = item[1], item[2]
            key = f"delta_{feature}_{param}"
            if key in existing_keys: return None # Skip if done
            arr = get_data_safe(feature)
            if arr is not None: result = umath.calc_delta(arr, param)

        elif type_ == 'flux':
            feature, lag = item[1], item[2]
            key = f"flux_{feature}_{lag}"
            if key in existing_keys: return None
            arr = get_data_safe(feature)
            if arr is not None: result = umath.calc_flux(arr, lag)

        elif type_ == 'slope':
            feature, w = item[1], item[2]
            key = f"slope_{feature}_{w}"
            if key in existing_keys: return None
            arr = get_data_safe(feature)
            if arr is not None: result = umath.calc_slope(arr, w)

        elif type_ == 'efficiency':
            feature, w = item[1], item[2]
            key = f"eff_{feature}_{w}"
            if key in existing_keys: return None
            arr = get_data_safe(feature)
            if arr is not None: result = umath.calc_efficiency(arr, w)

        elif type_ == 'zscore':
            feature, param = item[1], item[2]
            key = f"zscore_{feature}_{param}"
            if key in existing_keys: return None
            arr = get_data_safe(feature)
            if arr is not None: result = umath.calc_zscore(arr, param)

        elif type_ == 'correlation':
            f1, f2, w = item[1], item[2], item[3]
            f1, f2 = sorted([f1, f2])
            key = f"corr_{f1}_{f2}_{w}"
            if key in existing_keys: return None
            a = get_data_safe(f1)
            b = get_data_safe(f2)
            if a is not None and b is not None: result = umath.calc_correlation(a, b, w)
            
        if result is not None and key is not None:
            return key, result
        return None

    max_passes = 3
    # print(f"    [FeatureCtx] Identified {len(needed)} potential derived features.")
    
    # Parallel Execution Loop
    for pass_num in range(max_passes):
        start_count = len(existing_keys)
        computed_count = 0
        
        # Sort to prioritize cache locality (though parallel execution mixes it up)
        sorted_needed = sorted(list(needed), key=lambda x: (x[1], x[0], str(x[2])))
        
        # Use ThreadPool to release GIL during Numba calculations
        # Max workers = 2x CPU count usually good for mixed I/O-CPU
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Submit all candidates
            # Note: Candidates depending on OTHER candidates computed in THIS pass will fail 
            # and be picked up in the NEXT pass. This is expected.
            futures = [executor.submit(compute_feature_task, item) for item in sorted_needed]
            
            for i, future in enumerate(as_completed(futures)):
                # Progress logging
                # if i % 1000 == 0 and i > 0:
                #      print(f"      ... Processed {i}/{len(sorted_needed)} tasks in pass {pass_num+1}", end='\r')
                
                res = future.result()
                if res:
                    key, arr = res
                    # Save to Disk (thread-safe by file system mostly, but we are unique keys)
                    _save_feature(temp_dir, existing_keys, key, arr)
                    # Add to Cache
                    with cache_lock:
                        _cache[key] = arr
                        _cache.move_to_end(key)
                        if len(_cache) > MAX_CACHE_ITEMS:
                            _cache.popitem(last=False)
                    computed_count += 1

        # print(f"    [FeatureCtx] Pass {pass_num+1}/{max_passes}: Computed {computed_count} new features.")
        if len(existing_keys) == start_count:
            break
            
    # Clear cache
    _cache.clear()