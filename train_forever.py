import subprocess
import time
import sys
import os
import glob
import config
from datetime import datetime

def train_forever():
    run_count = 0
    print("♾️  Starting Alpha Factory Training Loop (train_forever.py)")
    print("Press Ctrl+C to stop.\n")

    # --- INITIAL CLEANUP: Delete artifacts ONCE to force fresh Feature Generation & Selection on first run ---
    try:
        # 1. Delete Feature Matrix
        matrix_path = config.DIRS['FEATURE_MATRIX']
        if os.path.exists(matrix_path):
            print(f"🧹 Deleting {matrix_path} to force regeneration...")
            os.remove(matrix_path)
        
        # 2. Delete Verified Marker
        marker_path = matrix_path + ".verified"
        if os.path.exists(marker_path):
            os.remove(marker_path)

        # 3. Delete Survivor/Purge Files
        survivor_pattern = os.path.join(config.DIRS['FEATURES_DIR'], "survivors_*.json")
        for f in glob.glob(survivor_pattern):
            print(f"🧹 Deleting {f}...")
            os.remove(f)
            
        purge_marker = config.PURGE_MARKER_FILE
        if os.path.exists(purge_marker):
            os.remove(purge_marker)
            
    except Exception as e:
        print(f"⚠️ Warning during cleanup: {e}")

    while True:
        run_count += 1
        start_time = datetime.now()
        print(f"{ '='*80}")
        print(f"🔄 STARTING RUN #{run_count}")
        print(f"⏰ Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{ '='*80}\n")

        try:
            # Execute run_pipeline.py
            # Using sys.executable to ensure we use the same python environment
            process = subprocess.run([sys.executable, "run_pipeline.py"], check=True) 
            
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"\n{ '='*80}")
            print(f"✅ RUN #{run_count} COMPLETED")
            print(f"⏱️  Duration: {duration}")
            print(f"⏰ Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{ '='*80}\n")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ RUN #{run_count} FAILED with exit code {e.returncode}")
            print("Restarting in 10 seconds...")
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Training loop stopped by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            print("Restarting in 10 seconds...")
            time.sleep(10)

        # Optional: Small delay between successful runs
        time.sleep(2)

if __name__ == "__main__":
    train_forever()
