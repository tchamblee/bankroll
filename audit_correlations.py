import sys
import os
import json
import numpy as np
import pandas as pd
import argparse
import config
from genome import Strategy
from backtest.engine import BacktestEngine
from backtest.statistics import calculate_sortino_ratio
from backtest.reporting import get_avg_sortino, get_min_sortino

def audit_correlations(threshold=0.8):
    # Load Candidates
    candidates_path = config.CANDIDATES_FILE
    if not os.path.exists(candidates_path):
        print("Candidates file not found.")
        return

    with open(candidates_path, 'r') as f:
        cand_dicts = json.load(f)
    
    strategies = []
    meta_map = {}
    for d in cand_dicts:
        try:
            s = Strategy.from_dict(d)
            strategies.append(s)
            meta_map[s.name] = {
                'avg_s': get_avg_sortino(d),
                'min_s': get_min_sortino(d)
            }
        except Exception as e:
            print(f"Skipping strategy {d.get('name', 'unknown')} due to load error: {e}")
            
    print(f"Loaded {len(strategies)} candidates.")
    
    if len(strategies) < 2:
        print("Need at least 2 strategies to check correlation.")
        return

    # 2. Initialize Backtester
    print("Initializing Backtester...")
    # This automatically loads data
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
    print(f"Backtester initialized with {backtester.data_len} bars.")
    
    # 3. Generate Signals & Returns
    print("Generating Signals & Simulating Returns...")
    
    # Ensure features exist
    backtester.ensure_context(strategies)
    
    # Generate signals
    raw_sig = backtester.generate_signal_matrix(strategies)
    
    # Shift signals for simulation (Open execution)
    shifted_sig = np.vstack([np.zeros((1, len(strategies)), dtype=raw_sig.dtype), raw_sig[:-1]])
    
    prices = backtester.open_vec
    times = backtester.times_vec
    highs = backtester.high_vec
    lows = backtester.low_vec
    atr = backtester.atr_vec
    
    # Run Simulation
    rets_matrix, _ = backtester.run_simulation_batch(
        shifted_sig, strategies, prices, times, 
        highs=highs, lows=lows, atr=atr
    )
    
    # 4. Compute Correlation & Sortinos
    df = pd.DataFrame(rets_matrix, columns=[s.name for s in strategies])
    
    # Drop columns with zero variance (no trades) to avoid NaNs
    df = df.loc[:, df.std() > 0]
    
    if df.empty:
        print("No strategies with non-zero returns found.")
        return

    corr_matrix = df.corr()

    # 5. Find High Correlations
    print(f"\n🔍 Auditing for Correlations > {threshold}...")
    
    high_corr_pairs = []
    
    # Iterate over upper triangle
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            s1 = cols[i]
            s2 = cols[j]
            val = corr_matrix.iloc[i, j]
            
            if val > threshold:
                m1 = meta_map.get(s1, {'avg_s': 0, 'min_s': 0})
                m2 = meta_map.get(s2, {'avg_s': 0, 'min_s': 0})
                
                high_corr_pairs.append((s1, s2, val, m1, m2))
    
    if not high_corr_pairs:
        print("✅ No high correlations found.")
    else:
        # Sort by correlation desc
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"⚠️  Found {len(high_corr_pairs)} highly correlated pairs:")
        header = f"{'Strategy A':<35} | {'A Avg':<5} | {'A Min':<5} | {'Strategy B':<35} | {'B Avg':<5} | {'B Min':<5} | {'Corr':<5}"
        print(header)
        print("-" * len(header))
        for s1, s2, c, m1, m2 in high_corr_pairs:
            print(f"{s1[:35]:<35} | {m1['avg_s']:5.2f} | {m1['min_s']:5.2f} | {s2[:35]:<35} | {m2['avg_s']:5.2f} | {m2['min_s']:5.2f} | {c:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Strategy Correlations")
    parser.add_argument("--threshold", type=float, default=0.8, help="Correlation threshold (default: 0.8)")
    args = parser.parse_args()
    
    audit_correlations(args.threshold)
