import json
import os
import config
from collections import Counter

def classify_strategy(strat):
    """Classifies a strategy based on its genes."""
    genes = strat.get('long_genes', []) + strat.get('short_genes', [])
    
    scores = {
        'Trend Following': 0,
        'Mean Reversion': 0,
        'Breakout': 0,
        'Microstructure': 0,
        'Regime Aware': 0
    }
    
    for gene in genes:
        g_type = gene.get('type')
        features = []
        if 'feature' in gene: features.append(gene['feature'])
        if 'feature_left' in gene: features.append(gene['feature_left'])
        if 'feature_right' in gene: features.append(gene['feature_right'])
        
        full_text = " ".join(features).lower()
        
        # Type based scoring
        if g_type == 'zscore':
            scores['Mean Reversion'] += 2
        elif g_type == 'persistence':
            scores['Trend Following'] += 1
            scores['Regime Aware'] += 1
        elif g_type == 'flux' or g_type == 'delta':
            scores['Trend Following'] += 1
        elif g_type == 'cross':
            scores['Trend Following'] += 2
        elif g_type == 'squeeze':
            scores['Breakout'] += 3
        elif g_type == 'divergence':
            scores['Mean Reversion'] += 2
        elif g_type == 'relational':
            # Relational can be anything, check context
            pass
            
        # Feature based scoring
        if 'trend' in full_text or 'velocity' in full_text:
            scores['Trend Following'] += 1
        if 'fdi' in full_text or 'hurst' in full_text or 'efficiency' in full_text or 'volatility' in full_text:
            scores['Regime Aware'] += 2
        if 'ofi' in full_text or 'kyle' in full_text or 'imbalance' in full_text or 'spread' in full_text:
            scores['Microstructure'] += 2
        if 'decay' in full_text or 'shock' in full_text or 'bars_since' in full_text:
            scores['Breakout'] += 2
        if 'skew' in full_text or 'kurt' in full_text:
            scores['Regime Aware'] += 1
            
    # Normalize
    max_score = max(scores.values()) if scores.values() else 0
    if max_score == 0:
        return "Hybrid/Unknown"
        
    primary = [k for k, v in scores.items() if v == max_score]
    return primary[0] if len(primary) == 1 else f"Hybrid ({'/'.join(primary)})"

def analyze_inbox():
    inbox_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json")
    if not os.path.exists(inbox_path):
        print("Inbox not found.")
        return

    with open(inbox_path, 'r') as f:
        strategies = json.load(f)

    if not strategies:
        print("Inbox is empty.")
        return
        
    print("\n🧬 GENETIC ANALYSIS OF INBOX")
    print("=" * 60)
    
    # 1. Feature Usage
    feature_counts = Counter()
    gene_types = Counter()
    
    for s in strategies:
        genes = s.get('long_genes', []) + s.get('short_genes', [])
        for g in genes:
            gene_types[g.get('type')] += 1
            if 'feature' in g: feature_counts[g['feature']] += 1
            if 'feature_left' in g: feature_counts[g['feature_left']] += 1
            if 'feature_right' in g: feature_counts[g['feature_right']] += 1

    print(f"Total Strategies: {len(strategies)}")
    
    print("\n🔹 Common Gene Types:")
    for g, c in gene_types.most_common(5):
        print(f"  - {g}: {c}")
        
    print("\n🔹 Common Features:")
    for f, c in feature_counts.most_common(5):
        print(f"  - {f}: {c}")
        
    print("\n🏷️  Strategy Classification:")
    print(f"{ 'Name':<20} | { 'Type':<20} | { 'Key Traits'}")
    print("-" * 75)
    
    for s in strategies:
        cls = classify_strategy(s)
        traits = []
        genes = s.get('long_genes', []) + s.get('short_genes', [])
        # Identify "exotic" traits
        all_feats = " ".join([str(g) for g in genes]).lower()
        if 'skew' in all_feats: traits.append("Skew")
        if 'fdi' in all_feats: traits.append("Fractal")
        if 'kyle' in all_feats: traits.append("Liq")
        if 'decay' in all_feats: traits.append("Event")
        if 'volatility' in all_feats: traits.append("Vol")
        
        trait_str = ", ".join(traits) if traits else "Standard"
        print(f"{s['name']:<20} | {cls:<20} | {trait_str}")

if __name__ == "__main__":
    analyze_inbox()
