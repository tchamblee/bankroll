import os
import json
import argparse
import re
import config
from backtest.utils import find_strategy_in_files
from genome import Strategy

# --- CONFIGURATION ---
import glob

# --- CONFIGURATION ---
def get_strategy_files():
    """
    Dynamically finds all strategy files to scan/update.
    """
    patterns = [
        "mutex_portfolio.json",
        "candidates.json",
        "found_strategies.json",
        "apex_strategies_*.json",
        "optimized_*.json"
    ]
    
    files = set()
    for pattern in patterns:
        matched = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], pattern))
        files.update(matched)
        
    return list(files)

FILES_TO_UPDATE = get_strategy_files()

# Semantic Mapping for Technical Features
FEATURE_TAGS = {
    'slope': 'Trend',
    'trend': 'Trend',
    'ema': 'Trend',
    'sma': 'Trend',
    'rsi': 'Osc',      # Oscillator
    'stoch': 'Osc',
    'zscore': 'Dev',   # Deviation
    'delta': 'Mom',    # Momentum
    'roc': 'Mom',
    'volatility': 'Vol',
    'atr': 'Vol',
    'volume': 'Vol',
    'ofi': 'Flow',     # Order Flow
    'book': 'Flow',
    'skew': 'Stat',
    'kurtosis': 'Stat',
    'autocorr': 'Persist', # Persistence
    'hurst': 'Regime',
    'entropy': 'Chaos',
    'correlation': 'Corr',
    'beta': 'Corr',
    'gdelt': 'News',
    'sentiment': 'News',
    'panic': 'Panic',
    'yield': 'Macro',
    'spread': 'Macro'
}

def load_json(path):
    if not os.path.exists(path): return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except: return []

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 Updated {path}")

def get_feature_tag(feature_name):
    """Extracts a short semantic tag from a feature name."""
    for key, tag in FEATURE_TAGS.items():
        if key in feature_name.lower():
            return tag
    return "Tech" # Generic Technical

def analyze_dna(strategy_data):
    """
    Analyzes strategy dictionary to produce a semantic description.
    """
    strat = Strategy.from_dict(strategy_data)
    horizon = strategy_data.get('horizon', 120)
    sl = strategy_data.get('stop_loss_pct', 2.0)
    tp = strategy_data.get('take_profit_pct', 4.0)
    
    # 1. Analyze Genes
    all_genes = strat.long_genes + strat.short_genes
    features = set()
    conditions = []
    
    for g in all_genes:
        # Extract feature name from string repr if possible, or object attributes
        # Assuming Gene.__repr__ or to_dict holds the key
        d = g.to_dict()
        # Look for feature keys
        if 'feature_name' in d:
            features.add(d['feature_name'])
        elif 'feature' in d:
            features.add(d['feature'])
        elif 'params' in d and 'feature_name' in d['params']:
             features.add(d['params']['feature_name'])
        
    # 2. Determine Style
    tags = [get_feature_tag(f) for f in features]
    
    # Count tags
    from collections import Counter
    counts = Counter(tags)
    primary_style = counts.most_common(1)[0][0] if counts else "Hybrid"
    
    # 3. Risk Profile
    risk_profile = "Neutral"
    if sl < 1.0: risk_profile = "Scalp"
    elif sl > 3.0: risk_profile = "Swing"
    
    # 4. Construct Description
    desc = f"Style: {primary_style} ({risk_profile})\n"
    desc += f"   Horizon: {horizon} bars\n"
    desc += f"   Risk: SL {sl}xATR | TP {tp}xATR\n"
    desc += f"   Key Features:\n"
    for f in features:
        desc += f"     - {f} ({get_feature_tag(f)})\n"
        
    return desc, primary_style, list(features), horizon

def generate_name(current_name, primary_style, features, horizon):
    """
    Generates a name: [Style]_[DominantFeature]_[Horizon]
    e.g. Trend_Slope_H120
    """
    # Find most specific feature name (shortest meaningful part)
    # e.g. "slope_linear_decay_100" -> "Slope"
    best_feat = "Mix"
    if features:
        # Heuristic: Pick the one that matches the primary style
        style_feats = [f for f in features if get_feature_tag(f) == primary_style]
        target_list = style_feats if style_feats else features
        
        # Pick the shortest name to keep it clean
        shortest = min(target_list, key=len)
        
        # Clean it up: remove numbers, 'delta', etc if redundant
        clean = re.sub(r'_\d+$', '', shortest) # remove suffix numbers
        clean = re.sub(r'^delta_', '', clean)
        best_feat = clean.title().replace("_", "")
        
    # Shorten Style
    style_map = {
        'Osc': 'MeanRev',
        'Vol': 'Vol',
        'Flow': 'Flow',
        'Trend': 'Trend',
        'News': 'News',
        'Macro': 'Macro'
    }
    style_str = style_map.get(primary_style, primary_style)
    
    # Construct Name
    proposal = f"{style_str}_{best_feat}_H{horizon}"
    
    # Ensure uniqueness
    base_proposal = proposal
    counter = 1
    while check_name_exists(proposal) and proposal != current_name:
        counter += 1
        proposal = f"{base_proposal}_v{counter}"
        
    return proposal

def check_name_exists(name):
    """Checks if a strategy name exists in any of the tracked files."""
    for path in FILES_TO_UPDATE:
        data = load_json(path)
        if not data: continue
        for s in data:
            if s.get('name') == name:
                return True
    return False

def rename_strategy_in_files(old_name, new_name):
    if check_name_exists(new_name):
        print(f"❌ Error: Strategy '{new_name}' already exists! Aborting rename.")
        return 0

    count = 0
    for path in FILES_TO_UPDATE:
        data = load_json(path)
        if not data: continue
        
        changed = False
        for s in data:
            if s.get('name') == old_name:
                s['name'] = new_name
                changed = True
                count += 1
        
        if changed:
            save_json(path, data)
            
    return count

def process_strategy(name, auto_accept=False):
    # 1. Find Strategy
    strat_data = find_strategy_in_files(name)
    if not strat_data:
        print(f"❌ Strategy '{name}' not found.")
        return

    # 2. Analyze
    desc, style, features, horizon = analyze_dna(strat_data)
    
    # 3. Propose Name
    proposal = generate_name(name, style, features, horizon)
    
    final_name = None
    if auto_accept:
        final_name = proposal
        print(f"   Renaming {name} -> {final_name}")
    else:
        print(f"\n🧬 Analysis for {name}: {style} | {features}")
        print(f"💡 Proposed Name: {proposal}")
        
        while True:
            choice = input("Accept? [Y/n/custom]: ").strip()
            if choice.lower() in ['', 'y', 'yes']:
                final_name = proposal
                break
            elif choice.lower() in ['n', 'no']:
                return
            else:
                if check_name_exists(choice):
                    print(f"❌ Name '{choice}' already exists! Please choose another.")
                else:
                    final_name = choice
                    break

    # 4. Execute
    if final_name and final_name != name:
        count = rename_strategy_in_files(name, final_name)
        if count > 0:
            print(f"✅ Renamed to {final_name}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and Rename Strategies")
    parser.add_argument("name", type=str, nargs='?', help="Current Strategy Name")
    parser.add_argument("--new", type=str, help="Skip interactive mode and use this new name")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-accept generated name")
    parser.add_argument("--all", action="store_true", help="Process all strategies in inbox")
    
    args = parser.parse_args()

    if args.all:
        inbox_path = config.DIRS['STRATEGY_INBOX']
        data = load_json(inbox_path)
        print(f"Processing {len(data)} strategies...")
        for s in data:
            process_strategy(s['name'], auto_accept=True)
    elif args.name:
        if args.new:
            rename_strategy_in_files(args.name, args.new)
        else:
            process_strategy(args.name, auto_accept=args.yes)
    else:
        print("Please provide a name or use --all")

if __name__ == "__main__":
    main()
