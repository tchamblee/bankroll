import sys
import os
import json
import argparse
import config
import pandas as pd
import numpy as np
from genome import Strategy
from backtest.engine import BacktestEngine
from backtest.utils import refresh_strategies, find_strategy_in_files
from backtest.reporting import print_candidate_table, get_min_sortino, get_cpcv_p5, get_cpcv_min

import glob

CANDIDATES_FILE = config.CANDIDATES_FILE

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
    
    # Sort by CPCV P5 (most robust first)
    candidates.sort(key=get_cpcv_p5, reverse=True)
    
    # Print standardized table
    print_candidate_table(candidates)

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
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if os.path.exists(inbox_path):
        with open(inbox_path, 'w') as f:
            json.dump([], f, indent=4)
        print("🧹 Strategy Inbox cleared.")
    else:
        print("⚠️  Inbox file not found.")


def prune_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
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

def cleanup_inbox(min_sortino=1.0):
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
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

    print(f"🧹 Cleaning Inbox (Removing strategies with Return <= 0 or Sortino < {min_sortino})...")
    
    kept_strategies = []
    removed_count = 0
    
    for s in strategies:
        # Check Returns (Must be positive across board if data exists)
        # Using flat keys which are now reliable after fix
        t_r = s.get('train_return', 0)
        v_r = s.get('val_return', 0)
        te_r = s.get('test_return', 0)
        
        t_s = s.get('train_sortino', 0)
        v_s = s.get('val_sortino', 0)
        te_s = s.get('test_sortino', 0)
        
        # If stats dicts exist, they might be more accurate if refresh just happened
        if 'train_stats' in s: t_r = s['train_stats'].get('ret', t_r)
        if 'val_stats' in s: v_r = s['val_stats'].get('ret', v_r)
        if 'test_stats' in s: te_r = s['test_stats'].get('ret', te_r)

        # Logic: If any dataset has Negative Return, it's a loser.
        # Logic: If any dataset has Sortino < min_sortino, it's weak.
        # Exception: Test set might be 0 if not evaluated? No, refresh ensures eval.
        
        is_failing = False
        reasons = []
        
        if t_r <= 0: is_failing = True; reasons.append(f"Train Ret {t_r*100:.1f}%")
        if v_r <= 0: is_failing = True; reasons.append(f"Val Ret {v_r*100:.1f}%")
        if te_r <= 0: is_failing = True; reasons.append(f"Test Ret {te_r*100:.1f}%")
        
        if t_s < min_sortino: is_failing = True; reasons.append(f"Train Sort {t_s:.2f}")
        if v_s < min_sortino: is_failing = True; reasons.append(f"Val Sort {v_s:.2f}")
        
        if is_failing:
            removed_count += 1
            # print(f"   ❌ Removing {s.get('name')}: {', '.join(reasons)}")
        else:
            kept_strategies.append(s)

    if removed_count > 0:
        with open(inbox_path, 'w') as f:
            json.dump(kept_strategies, f, indent=4)
        print(f"✅ Removed {removed_count} failing strategies. {len(kept_strategies)} remain.")
    else:
        print("✅ Inbox is clean. No failing strategies found.")

def list_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
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

    # Sort by CPCV P5 (most robust first)
    strategies.sort(key=get_cpcv_p5, reverse=True)

    # Print standardized table
    print_candidate_table(strategies, title="INBOX STRATEGIES")

def remove_from_inbox(name):
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    with open(inbox_path, 'r') as f:
        try:
            strategies = json.load(f)
        except:
            print("Inbox is corrupted.")
            return

    initial_len = len(strategies)
    strategies = [s for s in strategies if s.get('name') != name]
    
    if len(strategies) < initial_len:
        with open(inbox_path, 'w') as f:
            json.dump(strategies, f, indent=4)
        print(f"🗑️  Removed '{name}' from Inbox.")
    else:
        print(f"⚠️  Strategy '{name}' not found in Inbox.")

def clear_negative_cmin_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
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

    print("🧹 Clearing strategies with negative CMin from Inbox...")

    kept_strategies = []
    removed_count = 0

    for s in strategies:
        cmin = get_cpcv_min(s)
        if cmin < 0:
            removed_count += 1
        else:
            kept_strategies.append(s)

    if removed_count > 0:
        with open(inbox_path, 'w') as f:
            json.dump(kept_strategies, f, indent=4)
        print(f"✅ Removed {removed_count} strategies with negative CMin. {len(kept_strategies)} remain.")
    else:
        print("✅ No strategies with negative CMin found.")

def add_all_from_inbox():
    inbox_path = config.DIRS['STRATEGY_INBOX']
    if not os.path.exists(inbox_path):
        print("Inbox file not found.")
        return

    with open(inbox_path, 'r') as f:
        inbox_strategies = json.load(f)
        
    candidates = load_candidates()
    candidate_names = {c['name'] for c in candidates}
    
    added_count = 0
    for s in inbox_strategies:
        if s['name'] not in candidate_names:
            candidates.append(s)
            candidate_names.add(s['name'])
            added_count += 1
            
    if added_count > 0:
        save_candidates(candidates)
        print(f"✅ Added {added_count} strategies from Inbox to Candidates.")
    else:
        print("⚠️  No new strategies to add (all already in candidates).")

def main():
    parser = argparse.ArgumentParser(description="Manage Strategy Candidates for Mutex Portfolio")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    subparsers.add_parser('list', help='List current candidates')
    subparsers.add_parser('inbox', help='List found strategies in inbox')
    
    add_parser = subparsers.add_parser('add', help='Add a strategy by name')
    add_parser.add_argument('name', type=str, help='Name of the strategy (e.g., Child_2080)')
    
    subparsers.add_parser('add-all', help='Add ALL strategies from inbox to candidates')
    
    rm_parser = subparsers.add_parser('remove', help='Remove a strategy from CANDIDATES list')
    rm_parser.add_argument('name', type=str, help='Name of the strategy')

    rm_inbox_parser = subparsers.add_parser('remove-inbox', help='Remove a strategy from INBOX')
    rm_inbox_parser.add_argument('name', type=str, help='Name of the strategy')
    
    subparsers.add_parser('clear', help='Clear the candidate list')
    subparsers.add_parser('clear-inbox', help='Clear the strategy inbox')
    subparsers.add_parser('clear-negative-cmin', help='Clear inbox strategies with negative CMin')
    subparsers.add_parser('prune', help='Remove candidates from inbox')
    subparsers.add_parser('cleanup', help='Remove losing strategies from inbox (Ret <= 0)')
    subparsers.add_parser('help', help='Show this help message')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_candidates()
    elif args.command == 'inbox':
        list_inbox()
    elif args.command == 'add':
        add_strategy(args.name)
    elif args.command == 'add-all':
        add_all_from_inbox()
    elif args.command == 'remove':
        remove_strategy(args.name)
    elif args.command == 'remove-inbox':
        remove_from_inbox(args.name)
    elif args.command == 'clear':
        clear_list()
    elif args.command == 'clear-inbox':
        clear_inbox()
    elif args.command == 'clear-negative-cmin':
        clear_negative_cmin_inbox()
    elif args.command == 'prune':
        prune_inbox()
    elif args.command == 'cleanup':
        cleanup_inbox()
    elif args.command == 'help':
        parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
