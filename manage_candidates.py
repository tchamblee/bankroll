import sys
import os
import json
import argparse
import config
import pandas as pd
import numpy as np
from genome import Strategy
from backtest.utils import refresh_strategies, find_strategy_in_files

CANDIDATES_FILE = os.path.join(config.DIRS['STRATEGIES_DIR'], "candidates.json")

def load_candidates():
    if not os.path.exists(CANDIDATES_FILE):
        return []
    with open(CANDIDATES_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_candidates(candidates):
    with open(CANDIDATES_FILE, 'w') as f:
        json.dump(candidates, f, indent=4)
    print(f"💾 Saved {len(candidates)} candidates to {CANDIDATES_FILE}")

def list_candidates():
    candidates = load_candidates()
    if not candidates:
        print("Empty candidate list.")
        return

    # REFRESH METRICS
    candidates = refresh_strategies(candidates)
    save_candidates(candidates)

    print(f"\n📋 CURRENT CANDIDATE LIST ({len(candidates)} strategies)")
    # Headers
    header = f"{'Name':<50} | {'H':<4} | {'Trds':<6} | {'Train (R%/S)':<14} | {'Val (R%/S)':<14} | {'Test (R%/S)':<14}"
    print(header)
    print("-" * len(header))
    
    for c in candidates:
        name = c.get('name', 'Unknown')
        horizon = str(c.get('horizon', '?'))
        
        # Helper to extract metrics from various possible schemas
        def get_m(prefix):
            # Try new stats dict first (from optimizer)
            stats = c.get(f'{prefix}_stats', {})
            if stats:
                return stats.get('ret', 0) * 100, stats.get('sortino', 0), int(stats.get('trades', 0))
            
            # Fallback to flat keys
            ret = c.get(f'{prefix}_return', 0) * 100
            sort = c.get(f'{prefix}_sortino', 0)
            trades = c.get(f'{prefix}_trades', 0)
            
            if prefix == 'test':
                 if sort == 0: sort = c.get('test_sortino', 0)
                 if trades == 0: trades = c.get('test_trades', 0)
            
            return ret, sort, trades

        t_r, t_s, t_tr = get_m('train')
        v_r, v_s, v_tr = get_m('val')
        te_r, te_s, te_tr = get_m('test')
        
        # Handle cases where robust_return/sortino_oos were used in metrics dict
        if te_r == 0 and te_s == 0:
            m = c.get('metrics', {})
            te_r = m.get('robust_return', 0) * 100
            te_s = m.get('sortino_oos', 0)

        row = f"{name[:50]:<50} | {horizon:<4} | {te_tr:<6} | {t_r:5.1f}%/{t_s:4.2f} | {v_r:5.1f}%/{v_s:4.2f} | {te_r:5.1f}%/{te_s:4.2f}"
        print(row)
    print("-" * len(header))

def add_strategy(name):
    candidates = load_candidates()
    # Check if already exists
    if any(c['name'] == name for c in candidates):
        print(f"⚠️  Strategy '{name}' is already in the candidate list.")
        return

    print(f"🔍 Searching for '{name}'...")
    strategy_data = find_strategy_in_files(name)
    
    if strategy_data:
        candidates.append(strategy_data)
        print(f"✅ Added '{name}' (Horizon: {strategy_data.get('horizon')})")
        save_candidates(candidates)
    else:
        print(f"❌ Could not find strategy '{name}' in any output files.")

def remove_strategy(name):
    candidates = load_candidates()
    initial_len = len(candidates)
    candidates = [c for c in candidates if c['name'] != name]
    
    if len(candidates) < initial_len:
        print(f"🗑️  Removed '{name}'.")
        save_candidates(candidates)
    else:
        print(f"⚠️  Strategy '{name}' not found in candidate list.")

def clear_list():
    save_candidates([])
    print("🧹 Candidate list cleared.")

def clear_inbox():
    inbox_path = config.DIRS.get('STRATEGY_INBOX', os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json"))
    if os.path.exists(inbox_path):
        with open(inbox_path, 'w') as f:
            json.dump([], f, indent=4)
        print("🧹 Strategy Inbox cleared.")
    else:
        print("⚠️  Inbox file not found.")

def prune_inbox():
    inbox_path = config.DIRS.get('STRATEGY_INBOX', os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json"))
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    candidates = load_candidates()
    candidate_names = {c['name'] for c in candidates}
    
    with open(inbox_path, 'r') as f:
        inbox_strategies = json.load(f)
        
    initial_count = len(inbox_strategies)
    pruned_strategies = [s for s in inbox_strategies if s['name'] not in candidate_names]
    removed_count = initial_count - len(pruned_strategies)
    
    if removed_count > 0:
        with open(inbox_path, 'w') as f:
            json.dump(pruned_strategies, f, indent=4)
        print(f"✂️  Pruned {removed_count} duplicates from Inbox. {len(pruned_strategies)} strategies remain.")
    else:
        print("✅ No duplicates found in Inbox.")

def list_inbox():
    inbox_path = config.DIRS.get('STRATEGY_INBOX', os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json"))
    if not os.path.exists(inbox_path):
        print("Inbox is empty (no file found).")
        return

    with open(inbox_path, 'r') as f:
        try:
            strategies = json.load(f)
        except:
            print("Inbox is corrupted.")
            return

    if not strategies:
        print("Inbox is empty.")
        return

    # REFRESH METRICS
    strategies = refresh_strategies(strategies)
    
    # Save back to inbox
    with open(inbox_path, 'w') as f:
        json.dump(strategies, f, indent=4)

    # Sort by Sortino
    strategies.sort(key=lambda x: x.get('test_sortino', 0), reverse=True)

    print(f"\n📥 INBOX STRATEGIES ({len(strategies)} found)")
    print(f"{'Name':<50} | {'Horizon':<8} | {'Sortino':<8} | {'Train %':<10} | {'Val %':<10} | {'Test %':<10} | {'Gen':<5}")
    print("-" * 130)
    
    for s in strategies:
        name = s.get('name', 'Unknown')
        horizon = s.get('horizon', '?')
        sortino = s.get('test_sortino', 0)
        
        r_train = s.get('train_return', 0) * 100
        r_val = s.get('val_return', 0) * 100
        r_test = s.get('test_return', 0) * 100
        gen = s.get('generation', '?')
        
        print(f"{name[:50]:<50} | {horizon:<8} | {sortino:<8.2f} | {r_train:<10.2f} | {r_val:<10.2f} | {r_test:<10.2f} | {gen}")
    print("-" * 130)

def main():
    parser = argparse.ArgumentParser(description="Manage Strategy Candidates for Mutex Portfolio")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    subparsers.add_parser('list', help='List current candidates')
    subparsers.add_parser('inbox', help='List found strategies in inbox')
    
    add_parser = subparsers.add_parser('add', help='Add a strategy by name')
    add_parser.add_argument('name', type=str, help='Name of the strategy (e.g., Child_2080)')
    
    rm_parser = subparsers.add_parser('remove', help='Remove a strategy by name')
    rm_parser.add_argument('name', type=str, help='Name of the strategy')
    
    subparsers.add_parser('clear', help='Clear the candidate list')
    subparsers.add_parser('clear-inbox', help='Clear the strategy inbox')
    subparsers.add_parser('prune', help='Remove candidates from inbox')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_candidates()
    elif args.command == 'inbox':
        list_inbox()
    elif args.command == 'add':
        add_strategy(args.name)
    elif args.command == 'remove':
        remove_strategy(args.name)
    elif args.command == 'clear':
        clear_list()
    elif args.command == 'clear-inbox':
        clear_inbox()
    elif args.command == 'prune':
        prune_inbox()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
