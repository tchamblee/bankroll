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
    # Ensure uniqueness? The calling function handles checking, this just proposes.
    proposal = f"{style_str}_{best_feat}_H{horizon}"
    return proposal

def rename_strategy_in_files(old_name, new_name):
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

def main():
    parser = argparse.ArgumentParser(description="Analyze and Rename Strategies")
    parser.add_argument("name", type=str, help="Current Strategy Name (e.g. Mutant_1234)")
    parser.add_argument("--new", type=str, help="Skip interactive mode and use this new name")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-accept generated name")
    
    args = parser.parse_args()
    
    # 1. Find Strategy
    print(f"🔍 Searching for '{args.name}'...")
    strat_data = find_strategy_in_files(args.name)
    
    if not strat_data:
        print(f"❌ Strategy '{args.name}' not found in any output files.")
        return

    # 2. Analyze
    desc, style, features, horizon = analyze_dna(strat_data)
    
    print("\n🧬 STRATEGY DNA ANALYSIS")
    print("="*40)
    print(desc)
    print("="*40)
    
    # 3. Propose Name
    proposal = generate_name(args.name, style, features, horizon)
    
    print(f"\n💡 Proposed Name:  \033[1m{proposal}\033[0m")
    
    final_name = None
    
    if args.new:
        final_name = args.new
    elif args.yes:
        final_name = proposal
    else:
        print(f"Current Name:   {args.name}")
        choice = input("\nAccept proposal? [Y/n/custom]: ").strip().lower()
        
        if choice in ['', 'y', 'yes']:
            final_name = proposal
        elif choice in ['n', 'no']:
            print("❌ Rename cancelled.")
            return
        else:
            final_name = choice # Custom name
            
    # 4. Execute
    if final_name:
        # Safety Check: Does new name exist?
        exists = find_strategy_in_files(final_name)
        if exists:
            print(f"⚠️  Warning: A strategy named '{final_name}' already exists.")
            confirm = input("Overwrite/Merge? [y/N]: ").strip().lower()
            if confirm not in ['y', 'yes']:
                print("❌ Cancelled.")
                return

        print(f"\n🚀 Renaming '{args.name}' -> '{final_name}'...")
        count = rename_strategy_in_files(args.name, final_name)
        
        if count > 0:
            print(f"✅ Success! Updated {count} occurrences.")
            print("   The Dashboard and Reports will reflect this change immediately.")
        else:
            print("⚠️  Odd... No files were updated. Maybe the file changed externally?")

if __name__ == "__main__":
    main()
