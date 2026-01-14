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
    parser.add_argument("--skip-optimization", action="store_true", help="Skip optimization, just rename and promote original")

    args = parser.parse_args()
    original_name = args.name
    optimized_name = None  # Will be set if optimization produces a different strategy

    print(f"🔥 Starting Promotion Pipeline for '{original_name}'")

    # ------------------------------------------------------------------
    # STEP 1: Optimize Candidate (Gene Mutation/Selection)
    # ------------------------------------------------------------------
    if not args.skip_optimization:
        cmd_opt = f"python3 optimize_candidate.py {original_name} --walk-forward"
        output_opt = run_command(cmd_opt, "Step 1: Structural Optimization")

        if output_opt:
            # Check if an optimized file was created
            # The script saves to output/strategies/optimized_{name}.json
            opt_file = os.path.join(config.DIRS['STRATEGIES_DIR'], f"optimized_{original_name}.json")
            data = load_json(opt_file)

            if data and len(data) > 0:
                # The file is sorted by quality. First one is best.
                best_opt = data[0]
                new_name = best_opt['name']

                # Simple heuristic: Is it actually different?
                if new_name != original_name:
                    print(f"   ✨ Found improved variant: {new_name}")
                    optimized_name = new_name
                else:
                    print("   ℹ️  No better structural variant found (or name unchanged).")
            else:
                print("   ⚠️  Optimization ran but no 'optimized_*.json' found or it was empty.")
        else:
            print("   ⚠️  Optimization step failed or produced no output.")

        # ------------------------------------------------------------------
        # STEP 2: Optimize Stops on Optimized Variant (SL/TP Grid Search)
        # ------------------------------------------------------------------
        if optimized_name:
            cmd_stops = f"python3 optimize_stops.py {optimized_name} --sl-start 1.0 --sl-end 4.0 --sl-step 0.5 --tp-start 2.0 --tp-end 8.0 --tp-step 1.0"
            output_stops = run_command(cmd_stops, f"Step 2: Stop Loss Optimization ({optimized_name})")

            if output_stops:
                stops_file = os.path.join(config.DIRS['STRATEGIES_DIR'], f"optimized_stops_{optimized_name}.json")
                data = load_json(stops_file)

                if data and len(data) > 0:
                    best_stop = data[0]
                    stop_name = best_stop['name']

                    if stop_name != optimized_name:
                        print(f"   ✨ Found better Stop/Limit configuration: {stop_name}")
                        optimized_name = stop_name
                    else:
                        print("   ℹ️  Existing stops were optimal.")
                else:
                    print("   ⚠️  Stop optimization produced no results file. Keeping current.")
    else:
        print("   ⏭️  Skipping optimization (--skip-optimization flag set)")

    # ------------------------------------------------------------------
    # STEP 3: Rename Original Strategy
    # ------------------------------------------------------------------
    final_original_name = original_name
    cmd_rename = f"python3 rename_strategy.py {original_name} --yes"
    output_rename = run_command(cmd_rename, f"Step 3a: Renaming Original ({original_name})")

    if output_rename:
        match = re.search(r"Renamed to ([\w_]+)", output_rename)
        if match:
            final_original_name = match.group(1)
            print(f"   🏷️  Renamed: {original_name} -> {final_original_name}")
        else:
            print("   ℹ️  Name unchanged (or parsing failed).")
            if "not found" in output_rename:
                print("   ❌ Critical: Original strategy lost during renaming step.")
                sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 3b: Rename Optimized Strategy (if different)
    # ------------------------------------------------------------------
    final_optimized_name = None
    if optimized_name:
        cmd_rename_opt = f"python3 rename_strategy.py {optimized_name} --yes"
        output_rename_opt = run_command(cmd_rename_opt, f"Step 3b: Renaming Optimized ({optimized_name})")

        if output_rename_opt:
            match = re.search(r"Renamed to ([\w_]+)", output_rename_opt)
            if match:
                final_optimized_name = match.group(1)
                print(f"   🏷️  Renamed: {optimized_name} -> {final_optimized_name}")
            else:
                final_optimized_name = optimized_name
                print("   ℹ️  Optimized name unchanged (or parsing failed).")
                if "not found" in output_rename_opt:
                    print("   ⚠️  Optimized strategy lost during renaming step. Continuing with original only.")
                    final_optimized_name = None

    # ------------------------------------------------------------------
    # STEP 4a: Add Original to Candidates
    # ------------------------------------------------------------------
    cmd_add = f"python3 manage_candidates.py add {final_original_name}"
    output_add = run_command(cmd_add, f"Step 4a: Promoting Original ({final_original_name})")

    if output_add is None:
        print(f"\n⚠️  Original strategy '{final_original_name}' failed quality gates.")
        print("   Check thresholds: MIN_TEST_SORTINO, MIN_VAL_SORTINO, MIN_CPCV_THRESHOLD, MAX_TRAIN_TEST_DECAY")
    else:
        print(f"   ✅ Original strategy '{final_original_name}' added to candidates")

    # ------------------------------------------------------------------
    # STEP 4b: Add Optimized to Candidates (if applicable)
    # ------------------------------------------------------------------
    output_add_opt = None
    if final_optimized_name:
        cmd_add_opt = f"python3 manage_candidates.py add {final_optimized_name}"
        output_add_opt = run_command(cmd_add_opt, f"Step 4b: Promoting Optimized ({final_optimized_name})")

        if output_add_opt is None:
            print(f"\n⚠️  Optimized strategy '{final_optimized_name}' failed quality gates.")
        else:
            print(f"   ✅ Optimized strategy '{final_optimized_name}' added to candidates")

    # Check if at least one was promoted
    if output_add is None and (final_optimized_name is None or output_add_opt is None):
        print(f"\n❌ Pipeline ABORTED: No strategies passed quality gates.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 5: Remove Original from Inbox
    # ------------------------------------------------------------------
    cmd_remove = f"python3 manage_candidates.py remove-inbox {args.name}"
    run_command(cmd_remove, f"Step 5: Cleanup Inbox ({args.name})")

    print("\n✅ Pipeline Complete!")
    promoted = [final_original_name] if output_add else []
    if final_optimized_name and output_add_opt:
        promoted.append(final_optimized_name)
    print(f"   Promoted Strategies: {', '.join(promoted)}")

if __name__ == "__main__":
    main()
