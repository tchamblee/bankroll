#!/usr/bin/env python3
"""
Entry point for Evolutionary Alpha Discovery.
Logic has been refactored into the `evolution` package.
"""
import argparse
import os
import pandas as pd
import config
from evolution import EvolutionaryAlphaFactory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survivors", type=str, required=True)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--pop_size", type=int, default=5000)
    parser.add_argument("--gens", type=int, default=50)
    args = parser.parse_args()

    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        exit(1)
        
    bars_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    factory = EvolutionaryAlphaFactory(
        bars_df, 
        args.survivors, 
        population_size=args.pop_size, 
        generations=args.gens,
        target_col='log_ret',
        prediction_mode=False
    )
    
    try:
        factory.evolve(horizon=args.horizon)
    finally:
        factory.cleanup()