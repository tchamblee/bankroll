#!/usr/bin/env python3
"""
Entry point for Evolutionary Alpha Discovery.
Logic has been refactored into the `evolution` package.
"""
import argparse
import os
import sys
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
        sys.exit(1)

    if not os.path.exists(args.survivors):
        print(f"❌ Survivors file not found: {args.survivors}")
        sys.exit(1)

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
        print("\n✅ Evolution complete.")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n⚠️ Evolution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Evolution failed: {e}")
        sys.exit(1)
    finally:
        factory.cleanup()