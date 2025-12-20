import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from feature_engine import FeatureEngine

def plot_top5(df):
    print("Generating Top 5 Visualization...")
    
    # Champions
    features = [
        "frac_diff_02",
        "beta_tnx",
        "residual_dxy",
        "hurst_200",
        "delta_beta_spy_50"
    ]
    
    # Create Layout: Price + 5 Features
    fig, axes = plt.subplots(6, 1, figsize=(18, 20), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1, 1]})
    
    # 1. Price Panel
    ax_price = axes[0]
    ax_price.plot(df['time_end'], df['close'], color='black', linewidth=1.5, label='EUR/USD')
    ax_price.set_title("EUR/USD Price vs Top 5 Features", fontsize=14, fontweight='bold')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    
    # 2. Feature Panels
    colors = ['tab:red', 'tab:orange', 'tab:purple', 'tab:green', 'tab:blue']
    
    for i, feature in enumerate(features):
        ax = axes[i+1]
        
        if feature in df.columns:
            data = df[feature]
            
            # Plot Logic
            ax.plot(df['time_end'], data, color=colors[i], linewidth=1.2, label=feature)
            
            # Zero Line for Deltas
            if 'delta_' in feature or 'beta_' in feature:
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Value")
        else:
            ax.text(0.5, 0.5, f"Missing: {feature}", ha='center', va='center')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '../../data/top5_audit_chart.png')
    plt.savefig(output_path)
    print(f"📸 Saved Top 5 Chart to {output_path}")

if __name__ == "__main__":
    from feature_engine import create_full_feature_engine
    
    DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw_ticks'))
    engine = create_full_feature_engine(DATA_PATH)
    
    plot_top5(engine.bars)
