import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from collections import Counter

def analyze_dna():
    print("Analyzing Genome DNA...")
    
    # Load Final Population
    path = "data/final_population.json"
    if not os.path.exists(path):
        print(f"❌ Error: {path} not found. Run evolution first.")
        return
        
    with open(path, "r") as f:
        population = json.load(f)
        
    # Extract Feature Usage
    feature_counts = Counter()
    operator_counts = Counter()
    
    winners = [s for s in population if s['sharpe'] > 0]
    # If no winners, analyze the whole elite pool (Top 100)
    data_to_analyze = winners if winners else population
    
    print(f"Analyzing DNA of {len(data_to_analyze)} elite strategies...")
    
    for strat in data_to_analyze:
        for gene in strat['genes']:
            feature_counts[gene['feature']] += 1
            operator_counts[gene['op']] += 1
                
    if not feature_counts:
        print("No DNA found to analyze.")
        return

    # Plot Top 20 Features
    features_df = pd.DataFrame.from_dict(feature_counts, orient='index', columns=['count'])
    features_df = features_df.sort_values('count', ascending=True).tail(20)
    
    plt.figure(figsize=(12, 10))
    features_df['count'].plot(kind='barh', color='skyblue')
    plt.title(f"Dominant Genes ({'Winners' if winners else 'Top 100'})", fontsize=16)
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig("data/genome_dna_analysis.png")
    print(f"📸 Saved DNA Analysis to data/genome_dna_analysis.png")
    
    print("\n--- Top 20 Dominant Features ---")
    print(features_df.sort_values('count', ascending=False).head(20))

if __name__ == "__main__":
    analyze_dna()
