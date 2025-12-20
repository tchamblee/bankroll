import subprocess
import sys
import os
import time

def run_step(script_path, description):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {description}")
    print(f"📄 Script: {script_path}")
    print(f"{ '='*60}\n")
    
    start = time.time()
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        sys.exit(1)
        
    try:
        # Run script and stream output
        result = subprocess.run([sys.executable, script_path], check=True)
        duration = time.time() - start
        print(f"\n✅ {description} Completed in {duration:.2f}s")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline Failed at Step: {description}")
        print(f"Exit Code: {e.returncode}")
        sys.exit(e.returncode)

def main():
    print("\n🌍 STARTING ALPHA FACTORY PIPELINE 🌍\n")
    
    # 1. Data ETL
    run_step("clean_data_lake.py", "Data Cleaning & Normalization")
    run_step("verify_data_integrity.py", "Data Integrity Audit")
    
    # 2. Feature Engineering & Selection
    # Note: purge_features.py generates the survivors_*.json files needed later
    run_step("purge_features.py", "Feature Hunger Games (Selection)")
    run_step("validate_features.py", "Feature Validation (OOS Check)")
    
    # 3. Verification & Audit
    run_step("verification/top5_audit/verify_top5_calculations.py", "Top 5 Feature Audit")
    run_step("verification/top5_audit/visualize_top5_signals.py", "Top 5 Visualization")
    
    # 4. Strategy Evolution (The Brain)
    run_step("evolutionary_loop.py", "Evolutionary Strategy Discovery")
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
    print("Output Artifacts:")
    print("  - Data: data/clean_ticks/")
    print("  - Features: data/survivors_*.json")
    print("  - Strategies: data/apex_strategies.json")
    print("  - Plots: data/top5_audit_chart.png")

if __name__ == "__main__":
    main()
