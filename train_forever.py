import subprocess
import time
import sys
import os
from datetime import datetime

def train_forever():
    run_count = 0
    print("♾️  Starting Alpha Factory Training Loop (train_forever.py)")
    print("Press Ctrl+C to stop.\n")

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
