import sys
import os
import json
import argparse
import config
from genome import Strategy

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

def find_strategy_in_files(strategy_name):
    """Searches all strategy output files for a strategy with the given name."""
    search_patterns = [
        "found_strategies.json",
        "apex_strategies_*_top5_unique.json",
        "apex_strategies_*_top10.json",
        "apex_strategies_*.json"
    ]
    
    import glob
    for pattern in search_patterns:
        files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], pattern))
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for s_dict in data:
                        if s_dict.get('name') == strategy_name:
                            # Found it!
                            # Ensure horizon is set if missing (infer from filename)
                            if 'horizon' not in s_dict:
                                # extracting horizon from filename e.g., apex_strategies_90_...
                                parts = os.path.basename(file_path).split('_')
                                for p in parts:
                                    if p.isdigit():
                                        s_dict['horizon'] = int(p)
                                        break
                            
                            # Ensure metrics are present
                            if 'metrics' not in s_dict:
                                s_dict['metrics'] = {
                                    'sortino_oos': s_dict.get('test_sortino', 0),
                                    'robust_return': s_dict.get('test_return', s_dict.get('robust_return', 0))
                                }
                                
                            return s_dict
            except Exception:
                continue
    return None

def list_candidates():
    candidates = load_candidates()
    if not candidates:
        print("Empty candidate list.")
        return

    print(f"\n📋 CURRENT CANDIDATE LIST ({len(candidates)} strategies)")
    print(f"{'Name':<20} | {'Horizon':<8} | {'Sortino':<8} | {'Return':<8}")
    print("-" * 50)
    
    for c in candidates:
        name = c.get('name', 'Unknown')
        horizon = c.get('horizon', '?')
        metrics = c.get('metrics', {})
        sortino = metrics.get('sortino_oos', c.get('test_sortino', 0))
        ret = metrics.get('robust_return', 0)
        
        if ret == 0:
            ret = c.get('test_return', 0)
        
        print(f"{name:<20} | {horizon:<8} | {sortino:<8.2f} | {ret*100:<7.2f}%")
    print("-" * 50)

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

    # Sort by Sortino
    strategies.sort(key=lambda x: x.get('test_sortino', 0), reverse=True)

    print(f"\n📥 INBOX STRATEGIES ({len(strategies)} found)")
    print(f"{'Name':<20} | {'Horizon':<8} | {'Sortino':<8} | {'Train %':<10} | {'Val %':<10} | {'Test %':<10} | {'Gen':<5}")
    print("-" * 100)
    
    for s in strategies:
        name = s.get('name', 'Unknown')
        horizon = s.get('horizon', '?')
        sortino = s.get('test_sortino', 0)
        
        r_train = s.get('train_return', 0) * 100
        r_val = s.get('val_return', 0) * 100
        r_test = s.get('test_return', 0) * 100
        gen = s.get('generation', '?')
        
        print(f"{name:<20} | {horizon:<8} | {sortino:<8.2f} | {r_train:<10.2f} | {r_val:<10.2f} | {r_test:<10.2f} | {gen}")
    print("-" * 100)

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
