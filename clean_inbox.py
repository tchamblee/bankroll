import json
import os
import config

def clean_inbox():
    path = os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json")
    if not os.path.exists(path):
        return

    with open(path, 'r') as f:
        strategies = json.load(f)
        
    cleaned = []
    removed_count = 0
    
    for s in strategies:
        train_ret = s.get('train_return', 0.0)
        val_ret = s.get('val_return', 0.0)
        test_ret = s.get('test_return', 0.0)
        
        # Strict check: Train/Val must be >= Threshold (In-Sample Quality)
        # Test must be > 0 (OOS Profitability - just don't lose money)
        threshold = config.MIN_RETURN_THRESHOLD
        
        if train_ret < threshold or val_ret < threshold:
             removed_count += 1
             print(f"Removing {s['name']} (In-Sample): Train {train_ret*100:.2f}%, Val {val_ret*100:.2f}%")
        elif test_ret <= 0:
             removed_count += 1
             print(f"Removing {s['name']} (OOS Failure): Test {test_ret*100:.2f}%")
        else:
            cleaned.append(s)
            
    with open(path, 'w') as f:
        json.dump(cleaned, f, indent=4)
        
    print(f"✅ Removed {removed_count} strategies. {len(cleaned)} remain.")

if __name__ == "__main__":
    clean_inbox()
