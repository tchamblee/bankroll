import json
import os
import re
import pandas as pd
import config
from collections import Counter

def parse_genes_from_logic(logic_str):
    """
    Parses the string representation of a strategy to extract gene definitions.
    Example Logic: [Strat_X] LONG:(volatility_25 > 0.001) | SHORT:(hurst_roc_100 < 0.0)
    """
    # Simple regex to capture text inside parentheses
    # This might need refinement depending on exactly how genes are printed
    # Current repr: "Feature > Threshold" or "Feature < Threshold" or "F1 > F2"
    # The str(gene) in strategy_genome.py:
    # Static: f"{self.feature} {self.operator} {self.threshold:.4f}"
    # Relational: f"{self.feature_left} {self.operator} {self.feature_right}"
    # Delta: f"Delta({self.feature}, {self.lookback}) {self.operator} {self.threshold:.4f}"
    # ZScore: f"Z({self.feature}, {self.window}) {self.operator} {self.threshold:.2f}σ"
    
    # We'll just look for the gene components inside the LONG:(...) and SHORT:(...) blocks.
    # Actually, the 'logic' field in json is str(strategy), which is "[Name] LONG:(...) | SHORT:(...)"
    
    genes = []
    
    # Extract LONG and SHORT parts
    long_part_match = re.search(r"LONG:\((.*?)\)", logic_str)
    short_part_match = re.search(r"SHORT:\((.*?)\)", logic_str)
    
    parts = []
    if long_part_match: parts.append(long_part_match.group(1))
    if short_part_match: parts.append(short_part_match.group(1))
    
    for part in parts:
        # Genes are joined by " AND "
        raw_genes = part.split(" AND ")
        for g in raw_genes:
            if g == "None": continue
            # Attempt to identify the feature name
            # Heuristic: The first word is usually the feature, unless it's Delta(...) or Z(...)
            
            # Handle Delta(...)
            delta_match = re.match(r"Delta\((.*?),", g)
            if delta_match:
                genes.append(delta_match.group(1))
                continue
                
            # Handle Z(...)
            z_match = re.match(r"Z\((.*?),", g)
            if z_match:
                genes.append(z_match.group(1))
                continue
                
            # Handle Standard/Relational
            # Split by space and take the first token
            tokens = g.split(' ')
            if tokens:
                genes.append(tokens[0])
                # For relational (F1 > F2), the third token is also a feature
                # We can try to detect if the 3rd token is a known feature, but let's stick to dominant "primary" feature for now
                
    return genes

def main():
    print("\n" + "="*80)
    print("🏆 APEX STRATEGY REPORT 🏆")
    print("="*80 + "\n")
    
    horizons = config.PREDICTION_HORIZONS
    
    # Aggregate gene counts across all horizons
    global_gene_counts = Counter()
    
    for h in horizons:
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}.json")
        
        if not os.path.exists(file_path):
            continue
            
        print(f"--- Horizon: {h} Bars ---")
        try:
            with open(file_path, 'r') as f:
                strategies = json.load(f)
                
            if not strategies:
                print("  No strategies found.")
                continue
                
            # Sort by Test Sharpe
            strategies.sort(key=lambda x: x.get('test_sharpe', -999), reverse=True)
            
            # Create DataFrame for display
            df_data = []
            horizon_genes = []
            
            for s in strategies:
                name = s.get('name', 'Unknown')
                sharpe = s.get('test_sharpe', 0.0)
                logic = s.get('logic', '')
                
                genes = parse_genes_from_logic(logic)
                horizon_genes.extend(genes)
                global_gene_counts.update(genes)
                
                # Format logic for display (truncate if too long)
                display_logic = logic
                if len(display_logic) > 100:
                    display_logic = display_logic[:97] + "..."
                
                df_data.append({
                    'Name': name,
                    'Sharpe': f"{sharpe:.4f}",
                    'Dominant Genes': ", ".join(genes[:3]) + ("..." if len(genes)>3 else "") 
                })
            
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False))
            print("-" * 80)
            
            # Horizon specific dominant genes
            print(f"  Dominant Genes (Horizon {h}):")
            common = Counter(horizon_genes).most_common(5)
            for gene, count in common:
                print(f"    - {gene}: {count}")
            print("\n")
            
        except Exception as e:
            print(f"  Error reading file: {e}")

    print("="*80)
    print("🧬 GLOBAL DOMINANT GENES (All Horizons) 🧬")
    print("="*80)
    
    if global_gene_counts:
        common = global_gene_counts.most_common(20)
        for i, (gene, count) in enumerate(common, 1):
            print(f"{i:2d}. {gene:<40} (Count: {count})")
    else:
        print("No genes found across strategies.")

if __name__ == "__main__":
    main()
