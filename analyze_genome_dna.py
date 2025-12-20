import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from collections import Counter

def analyze_dna():
    print("Analyzing Genome DNA...")
    
    # 1. Load DNA
    path = "data/final_population.json"
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found. Run evolution first.")
        return
        
    with open(path, "r") as f:
        population = json.load(f)
        
    # Extract Feature Counts
    feature_counts = Counter()
    for strat in population:
        for gene in strat['genes']:
            feature_counts[gene['feature']] += 1
            
    dna_features = list(feature_counts.keys())
    
    # 2. Load Metrics (Try 60, then 30, then 90)
    metrics_file = None
    for h in [60, 30, 90]:
        fpath = f"data/feature_metrics_{h}.csv"
        if os.path.exists(fpath):
            metrics_file = fpath
            print(f"Loading metrics from {fpath}...")
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
        plt.savefig("data/genome_dna_analysis.png")
        print(f"📸 Saved DNA Analysis to data/genome_dna_analysis.png")
        
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
        plt.savefig("data/genome_dna_analysis.png")
        print(features_df.sort_values('count', ascending=False).head(20))

if __name__ == "__main__":
    analyze_dna()
