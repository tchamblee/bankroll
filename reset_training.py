import os
import glob
import config
import argparse

def reset_training(hard=False):
    print("🧹 Cleaning up training artifacts to force a fresh run...")

    # 1. Delete Feature Matrix and Verification Marker
    feature_matrix = config.DIRS['FEATURE_MATRIX']
    verified_marker = feature_matrix + ".verified"
    
    if os.path.exists(feature_matrix):
        os.remove(feature_matrix)
        print(f"✅ Deleted Feature Matrix: {feature_matrix}")
    else:
        print(f"   Feature Matrix not found (skipping): {feature_matrix}")
        
    if os.path.exists(verified_marker):
        os.remove(verified_marker)
        print(f"✅ Deleted Verified Marker: {verified_marker}")

    # 2. Delete Purge Markers and Survivors
    features_dir = config.DIRS['FEATURES_DIR']
    purge_marker = os.path.join(features_dir, "PURGE_COMPLETE")
    
    if os.path.exists(purge_marker):
        os.remove(purge_marker)
        print(f"✅ Deleted Purge Marker: {purge_marker}")
        
    # Delete survivor files
    survivor_files = glob.glob(os.path.join(features_dir, "survivors_*.json"))
    for f in survivor_files:
        os.remove(f)
        print(f"✅ Deleted Survivor File: {f}")

    # 3. Optional: Delete Cleaned Ticks (Hard Reset)
    if hard:
        clean_ticks_dir = config.DIRS['DATA_CLEAN_TICKS']
        if os.path.exists(clean_ticks_dir):
            import shutil
            shutil.rmtree(clean_ticks_dir)
            print(f"✅ [HARD] Deleted Cleaned Ticks Directory: {clean_ticks_dir}")
        else:
            print(f"   [HARD] Cleaned Ticks Directory not found: {clean_ticks_dir}")

    print("\n✨ Reset Complete! The next run of 'train_forever.py' will rebuild features and re-run selection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset training artifacts to force fresh feature generation.")
    parser.add_argument("--hard", action="store_true", help="Also delete cleaned tick data (Data Lake clean reset)")
    args = parser.parse_args()
    
    reset_training(hard=args.hard)
