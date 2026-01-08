#!/usr/bin/env python3
import sys
import os
import json
import argparse
import config
from backtest.utils import find_strategy_in_files
from genome import Strategy

def load_candidates():
    if not os.path.exists(config.CANDIDATES_FILE):
        return []
    with open(config.CANDIDATES_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_candidates(candidates):
    with open(config.CANDIDATES_FILE, 'w') as f:
        json.dump(candidates, f, indent=4)
    print(f"💾 Saved {len(candidates)} candidates to {config.CANDIDATES_FILE}")

def main():
    parser = argparse.ArgumentParser(description="Strip LONG genes from a strategy and save as a new candidate.")
    parser.add_argument("name", type=str, help="Name of the source strategy")
    args = parser.parse_args()

    print(f"🔍 Searching for '{args.name}'...")
    strategy_data = find_strategy_in_files(args.name)

    if not strategy_data:
        print(f"❌ Strategy '{args.name}' not found.")
        sys.exit(1)

    # Create new strategy dict
    new_strat = strategy_data.copy()
    new_name = f"{args.name}_ShortOnly"
    new_strat['name'] = new_name
    new_strat['long_genes'] = [] # STRIP LONG GENES
    
    # Reset stats because logic changed
    keys_to_reset = ['train_return', 'val_return', 'test_return', 
                     'train_sortino', 'val_sortino', 'test_sortino',
                     'train_stats', 'val_stats', 'test_stats',
                     'metrics', 'robust_score', 'dsr_val', 'psr_test']
    
    for k in keys_to_reset:
        if k in new_strat:
            del new_strat[k]
            
    # Add parent pointer
    new_strat['parent_name'] = args.name
    new_strat['modification'] = 'strip_long_genes'

    # Load existing candidates
    candidates = load_candidates()
    
    # Check for duplicates
    for c in candidates:
        if c['name'] == new_name:
            print(f"⚠️  Strategy '{new_name}' already exists in candidates. Updating it.")
            candidates.remove(c)
            break
    
    candidates.append(new_strat)
    save_candidates(candidates)
    print(f"✅ Created '{new_name}' (Short Only) and added to candidates.")

if __name__ == "__main__":
    main()
