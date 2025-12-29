import json
import os
import glob
import config

def clean_apex_files():
    pattern = os.path.join(config.DIRS['STRATEGIES_DIR'], "apex_strategies_*.json")
    files = glob.glob(pattern)
    
    total_removed = 0
    
    for file_path in files:
        print(f"Cleaning {os.path.basename(file_path)}...")
        try:
            with open(file_path, 'r') as f:
                strategies = json.load(f)
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
            
        cleaned = []
        removed_in_file = 0
        
        for s in strategies:
            # Check Test Return (Handle various key names just in case, but standard is 'test_return')
            test_ret = s.get('test_return', 0.0)
            
            # Also check 'metrics' dict if it exists (older format?)
            if 'metrics' in s and 'test_return' in s['metrics']:
                test_ret = s['metrics']['test_return']
            
            if test_ret <= 0:
                removed_in_file += 1
            else:
                cleaned.append(s)
        
        if removed_in_file > 0:
            with open(file_path, 'w') as f:
                json.dump(cleaned, f, indent=4)
            print(f"  Removed {removed_in_file} strategies.")
            total_removed += removed_in_file
        else:
            print("  No bad strategies found.")
            
    print(f"\n✅ Total strategies removed across all files: {total_removed}")

if __name__ == "__main__":
    clean_apex_files()
