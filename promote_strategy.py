import argparse
import subprocess
import os
import json
import re
import sys
import config

def run_command(command, description):
    print(f"\n🚀 {description}...")
    print(f"   Command: {command}")
    try:
        # Run command and capture output
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        print("   ✅ Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed (Exit Code: {e.returncode})")
        print(f"   Error Output:\n{e.stderr}")
        return None

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def main():
    parser = argparse.ArgumentParser(description="Pipeline: Optimize -> Tune Stops -> Rename -> Promote")
    parser.add_argument("name", type=str, help="Initial Strategy Name (from Inbox)")
    
    args = parser.parse_args()
    current_name = args.name
    
    print(f"🔥 Starting Promotion Pipeline for '{current_name}'")
    
    # ------------------------------------------------------------------
    # STEP 1: Optimize Candidate (Gene Mutation/Selection)
    # ------------------------------------------------------------------
    cmd_opt = f"python3 optimize_candidate.py {current_name}"
    output_opt = run_command(cmd_opt, "Step 1: Structural Optimization")
    
    if output_opt:
        # Check if an optimized file was created
        # The script saves to output/strategies/optimized_{name}.json
        opt_file = os.path.join(config.DIRS['STRATEGIES_DIR'], f"optimized_{current_name}.json")
        data = load_json(opt_file)
        
        if data and len(data) > 0:
            # The file is sorted by quality. First one is best.
            best_opt = data[0]
            new_name = best_opt['name']
            
            # Simple heuristic: Is it actually different?
            if new_name != current_name:
                print(f"   ✨ Found improved variant: {new_name}")
                current_name = new_name
            else:
                print("   ℹ️  No better structural variant found (or name unchanged). Keeping original.")
        else:
            print("   ⚠️  Optimization ran but no 'optimized_*.json' found or it was empty. Proceeding with original.")
    else:
        print("   ⚠️  Optimization step failed or produced no output. Proceeding with original.")

    # ------------------------------------------------------------------
    # STEP 2: Optimize Stops (SL/TP Grid Search)
    # ------------------------------------------------------------------
    # NOTE: optimize_stops.py saves to optimized_stops_{name}.json
    cmd_stops = f"python3 optimize_stops.py {current_name} --sl-start 1.0 --sl-end 4.0 --sl-step 0.5 --tp-start 2.0 --tp-end 8.0 --tp-step 1.0"
    output_stops = run_command(cmd_stops, f"Step 2: Stop Loss Optimization ({current_name})")
    
    if output_stops:
        stops_file = os.path.join(config.DIRS['STRATEGIES_DIR'], f"optimized_stops_{current_name}.json")
        data = load_json(stops_file)
        
        if data and len(data) > 0:
            best_stop = data[0]
            stop_name = best_stop['name']
            
            if stop_name != current_name:
                print(f"   ✨ Found better Stop/Limit configuration: {stop_name}")
                current_name = stop_name
            else:
                print("   ℹ️  Existing stops were optimal.")
        else:
             print("   ⚠️  Stop optimization produced no results file. Keeping current.")
    
    # ------------------------------------------------------------------
    # STEP 3: Rename Strategy
    # ------------------------------------------------------------------
    cmd_rename = f"python3 rename_strategy.py {current_name} --yes"
    output_rename = run_command(cmd_rename, f"Step 3: Renaming ({current_name})")
    
    if output_rename:
        # Parse output for "Renamed to X"
        match = re.search(r"Renamed to ([\w_]+)", output_rename)
        if match:
            new_name = match.group(1)
            print(f"   🏷️  Renamed: {current_name} -> {new_name}")
            current_name = new_name
        else:
            print("   ℹ️  Name unchanged (or parsing failed).")
            # If parsing failed, we might be in trouble if the name *did* change.
            # But duplicate check logic in rename_strategy might have prevented change if it was already good.
            # Or maybe it just said "Strategy X not found" if we messed up.
            if "not found" in output_rename:
                print("   ❌ Critical: Strategy lost during renaming step.")
                sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 4: Add to Candidates
    # ------------------------------------------------------------------
    cmd_add = f"python3 manage_candidates.py add {current_name}"
    run_command(cmd_add, f"Step 4: Promotion ({current_name})")
    
    print("\n✅ Pipeline Complete!")
    print(f"   Final Strategy: {current_name}")

if __name__ == "__main__":
    main()
