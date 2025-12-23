import subprocess
import sys
import os
import time
import shlex
import config

def run_step(script_command, description):
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {description}")
    print(f"📄 Command: {script_command}")
    print(f"{ '='*60}\n")
    
    start = time.time()
    
    # Split command into script and args
    args = shlex.split(script_command)
    script_path = args[0]
    
    # Check if script exists
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found at {script_path}")
        sys.exit(1)
        
    try:
        # Run script and stream output
        # Prefix with python executable
        full_command = [sys.executable] + args
        result = subprocess.run(full_command, check=True)
        duration = time.time() - start
        print(f"\n✅ {description} Completed in {duration:.2f}s")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline Failed at Step: {description}")
        print(f"Exit Code: {e.returncode}")
        sys.exit(e.returncode)

def main():
    print("\n🌍 STARTING ALPHA FACTORY PIPELINE 🌍\n")

    # Ensure all directories exist
    for dir_path in config.DIRS.values():
        if isinstance(dir_path, str) and not dir_path.endswith('.parquet'):
            os.makedirs(dir_path, exist_ok=True)

    run_step("clean_data_lake.py", "Data Cleaning & Normalization")
    
    # 2. Feature Engineering
    run_step("generate_features.py", "Feature Matrix Generation")
    run_step("verify_data_integrity.py", "Data Integrity Audit")
    
    # Feature Selection & Validation
    # Note: purge_features.py generates the survivors_*.json files needed later
    run_step("purge_features.py", "Feature Hunger Games (Selection)")
    run_step("validate_features.py", "Feature Validation (OOS Check)")
        
    # 4. Strategy Evolution (The Brain)
    for horizon in config.PREDICTION_HORIZONS:
        survivors_file = os.path.join(config.DIRS['FEATURES_DIR'], f"survivors_{horizon}.json")
        run_step(f"evolutionary_loop.py --survivors {survivors_file} --horizon {horizon} --pop_size 20000 --gens 100", 
                 f"Evolutionary Strategy Discovery (Horizon: {horizon})")
    
    # 5. Analysis & Visualization
    run_step("report_top_strategies.py", "Strategy Selection & Reporting")
    run_step("run_mutex_strategy.py", "Mutex Portfolio Backtest")
    run_step("visualize_strategy_performance.py", "Strategy Account Performance Visualization")
    run_step("generate_trade_atlas.py", "Consolidated Trade Atlas Generation")
    run_step("analyze_genome_dna.py", "Genome DNA Analysis")
    run_step("generate_prop_report.py", "Prop Desk Strategy Analysis Reports")
    
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
    print("Output Artifacts:")
    print(f"  - Data: {config.DIRS['DATA_CLEAN_TICKS']}")
    print(f"  - Features: {config.DIRS['FEATURES_DIR']}/survivors_*.json")
    print(f"  - Strategies: {config.DIRS['STRATEGIES_DIR']}/apex_strategies.json")
    print(f"  - Plots: {config.DIRS['PLOTS_DIR']}/top5_audit_chart.png")
    print(f"  - Report: Check stdout for 'APEX STRATEGY REPORT'")

if __name__ == "__main__":
    main()
