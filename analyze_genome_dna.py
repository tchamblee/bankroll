import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from collections import Counter
import config

def analyze_dna():
    print("Analyzing Genome DNA...")
    
    # 1. Load DNA from all apex files
    dna_files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], "apex_strategies_*.json"))
    if not dna_files:
        print(f"❌ Error: No apex strategies found in {config.DIRS['STRATEGIES_DIR']}. Run evolution first.")
        return
        
    all_strategies = []
    for fpath in dna_files:
        with open(fpath, "r") as f:
            all_strategies.extend(json.load(f))
            
    # Extract Feature Counts from long_genes and short_genes
    feature_counts = Counter()
    for strat in all_strategies:
        genes = strat.get('long_genes', []) + strat.get('short_genes', [])
        for gene in genes:
            # Handle different gene types
            if 'feature' in gene:
                feature_counts[gene['feature']] += 1
            elif 'feature_left' in gene:
                feature_counts[gene['feature_left']] += 1
                feature_counts[gene['feature_right']] += 1
            
    if not feature_counts:
        print("❌ No features found in DNA.")
        return
            
    dna_features = list(feature_counts.keys())
    
    # 2. Load Metrics (Try 60, then 30, then 90)
    metrics_file = None
    for h in [60, 30, 90]:
        fpath = os.path.join(config.DIRS['FEATURES_DIR'], f"feature_metrics_{h}.csv")
        if os.path.exists(fpath):
            metrics_file = fpath
            # print(f"Loading metrics from {fpath}...")
            break
            
    if metrics_file:
        metrics_df = pd.read_csv(metrics_file)
        # Filter to only features present in DNA
        # Merge Counts
        dna_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['Count']).reset_index()
        dna_df.columns = ['Feature', 'Count']
        
        # Merge with Metrics
        merged = pd.merge(dna_df, metrics_df, on='Feature', how='left')
        
        # Sort by Importance (descending)
        merged = merged.sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # We plot Top 20 by Importance
        top_20 = merged.head(20).sort_values('Importance', ascending=True) # Ascending for barh
        
        # Create Bar Chart
        bars = plt.barh(top_20['Feature'], top_20['Importance'], color='teal')
        
        plt.title(f"Feature Power Ranking (RF Importance) in Elite DNA", fontsize=16)
        plt.xlabel("Random Forest Importance")
        plt.tight_layout()
        out_path = os.path.join(config.DIRS['PLOTS_DIR'], "genome_dna_analysis.png")
        plt.savefig(out_path)
        print(f"📸 Saved DNA Analysis to {out_path}")
        
        print("\n--- Top 20 Dominant Features (Ranked by Importance) ---")
        print(merged[['Feature', 'Importance', 'IC', 'Count']].head(20))
        
    else:
        print("⚠️ No feature metrics found. Falling back to count-based analysis.")
        # Fallback to old logic (Counts only)
        features_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['count'])
        features_df = features_df.sort_values('count', ascending=True).tail(20)
        
        plt.figure(figsize=(12, 10))
        features_df['count'].plot(kind='barh', color='skyblue')
        plt.title(f"Dominant Genes (Count)", fontsize=16)
        out_path = os.path.join(config.DIRS['PLOTS_DIR'], "genome_dna_analysis.png")
        plt.savefig(out_path)
        print(features_df.sort_values('count', ascending=False).head(20))

if __name__ == "__main__":
    analyze_dna()
