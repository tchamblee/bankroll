import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import config

def visualize_top5():
    print("\n🎨 VISUALIZING TOP 5 SIGNALS 🎨")
    
    # Check if Feature Matrix exists
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print(f"❌ Feature Matrix not found at {config.DIRS['FEATURE_MATRIX']}")
        sys.exit(1)
        
    try:
        df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    except Exception as e:
        print(f"❌ Failed to load Feature Matrix: {e}")
        sys.exit(1)
    
    os.makedirs(config.DIRS['PLOTS_DIR'], exist_ok=True)
    
    for horizon in config.PREDICTION_HORIZONS:
        print(f"\n--- Visualizing Horizon: {horizon} ---")
        
        survivors_file = os.path.join(config.DIRS['FEATURES_DIR'], f"survivors_{horizon}.json")
        if not os.path.exists(survivors_file):
            print(f"Skipping {horizon}, no survivors file.")
            continue
            
        try:
            with open(survivors_file, 'r') as f:
                survivors = json.load(f)
        except Exception as e:
            print(f"Error reading {survivors_file}: {e}")
            continue
            
        if not survivors:
            continue
            
        top5 = survivors[:5]
        
        # Select a sample (last 1000 bars)
        sample = df.tail(1000).copy()
        
        # Plot
        fig, axes = plt.subplots(len(top5) + 1, 1, figsize=(15, 3 * (len(top5) + 1)), sharex=True)
        
        # Price
        if 'time_start' in sample.columns and 'close' in sample.columns:
            axes[0].plot(sample['time_start'], sample['close'], color='black', label='Close Price')
            axes[0].set_title(f"EUR/USD Price (Last 1000 Bars) - Horizon {horizon}")
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "Missing 'time_start' or 'close' columns", ha='center')
        
        # Features
        for i, feature in enumerate(top5):
            ax = axes[i+1]
            if feature in sample.columns:
                ax.plot(sample['time_start'], sample[feature], label=feature, color='tab:blue')
                
                # Add horizontal line at 0 if relevant
                if sample[feature].min() < 0 < sample[feature].max():
                    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                    
                ax.set_title(f"Feature: {feature}")
                ax.legend(loc='upper left')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f"Feature {feature} missing", ha='center')
        
        plt.tight_layout()
        output_path = os.path.join(config.DIRS['PLOTS_DIR'], f"top5_audit_horizon_{horizon}.png")
        try:
            plt.savefig(output_path)
            print(f"✅ Saved plot to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save plot: {e}")
            
        plt.close(fig)

if __name__ == "__main__":
    visualize_top5()
