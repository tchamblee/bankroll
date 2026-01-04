import os
import json
import random
import subprocess
import numpy as np
import pandas as pd
import scipy.stats
import config
from genome import Strategy
from backtest.statistics import deflated_sharpe_ratio, estimated_sharpe_ratio

# Import Optimizer
try:
    from optimize_candidate import StrategyOptimizer
except ImportError:
    # Fallback if running from a different context where root is not in path
    import sys
    sys.path.append(os.getcwd())
    from optimize_candidate import StrategyOptimizer

def play_success_sound():
    """
    Attempts to play a custom alert sound.
    Falls back to system bell if file not found or player missing.
    """
    sound_path = os.path.join(os.getcwd(), 'resources', 'alert.mp3')
    
    if os.path.exists(sound_path):
        # List of candidate players to try
        # Note: 'aplay' is typically for wav, but some versions handle others. 
        # We assume 'alert.mp3' for now, so mpg123/ffplay are best bets.
        players = [
            ['mpg123', '-q', sound_path],
            ['ffplay', '-nodisp', '-autoexit', '-hide_banner', sound_path],
            ['paplay', sound_path],
            ['vlc', '--intf', 'dummy', '--play-and-exit', sound_path]
        ]
        
        for cmd in players:
            try:
                # Use Popen to play asynchronously (don't block the loop)
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except (FileNotFoundError, OSError):
                continue
    
    # Fallback
    print('\a' * 3) # 🔔 Audible Alert

def save_campaign_results(hall_of_fame, backtester, horizon, training_id, total_strategies_evaluated):
    print(f"\n--- 🛡️ FINAL SELECTION (Using Diverse HOF) ---")
    
    top_candidates = [entry['strat'] for entry in hall_of_fame[:100]]
    val_fitness_map = {entry['strat'].name: entry['fit'] for entry in hall_of_fame[:100]}
    
    if not top_candidates:
        print("No strategies survived.")
        return

    # 1. Evaluate on Development Sets (Train + Validation) ONLY
    val_res, val_returns = backtester.evaluate_population(top_candidates, set_type='validation', return_series=True, time_limit=horizon, min_trades=config.MIN_TRADES_FOR_METRICS)
    train_res = backtester.evaluate_population(top_candidates, set_type='train', time_limit=horizon, min_trades=config.MIN_TRADES_FOR_METRICS)
    
    # Create lookup maps
    val_map = {row['id']: row for _, row in val_res.iterrows()}
    train_map = {row['id']: row for _, row in train_res.iterrows()}

    best_rejected_name = None
    best_rejected_min_ret = -999.0
    best_rejected_details = ""

    # 2. Strict Filtering Loop (Train + Val ONLY)
    survivor_candidates = []
    survivor_val_indices = []

    for i, s in enumerate(top_candidates):
        if s.name not in val_map or s.name not in train_map:
            continue
        
        t_stats = train_map[s.name]
        v_stats = val_map[s.name]
        
        train_ret = t_stats['total_return']
        val_ret = v_stats['total_return']
        
        # --- GATEKEEPING ---
        min_ret_threshold = config.MIN_RETURN_THRESHOLD
        
        train_sortino = float(t_stats.get('sortino', 0))
        val_sortino = float(v_stats.get('sortino', 0))
        
        passed_gate = True
        rejection_reason = []

        # 1. Strict Sortino Filter (> 1.5 on Train/Val)
        if train_sortino < 1.5 or val_sortino < 1.5:
            passed_gate = False
            if train_sortino < 1.5: rejection_reason.append(f"Train_Sort({train_sortino:.2f})")
            if val_sortino < 1.5: rejection_reason.append(f"Val_Sort({val_sortino:.2f})")
        
        # 2. Positive Return Filter
        elif train_ret <= 0 or val_ret <= 0:
            passed_gate = False
            if train_ret <= 0: rejection_reason.append(f"Train_Ret({train_ret*100:.2f}%)")
            if val_ret <= 0: rejection_reason.append(f"Val_Ret({val_ret*100:.2f}%)")

        if not passed_gate:
            # Track closest call for debugging/info
            # Prioritize Sortino failures in logging if Return was okay
            min_sort_here = min(train_sortino, val_sortino)
            if min_sort_here > 1.0: # Only log if it was somewhat close
                 best_rejected_details = f"TrainSort:{train_sortino:.2f}, ValSort:{val_sortino:.2f}"
            continue
        
        # --- SENSITIVITY ANALYSIS (Jitter Test) ---
        if val_ret > 0:
            clones = []
            for _ in range(3):
                clone = Strategy(name=f"{s.name}_jit", long_genes=[g.copy() for g in s.long_genes], short_genes=[g.copy() for g in s.short_genes], min_concordance=s.min_concordance)
                for g in clone.long_genes + clone.short_genes:
                    if hasattr(g, 'threshold'): g.threshold *= random.uniform(0.95, 1.05)
                    if hasattr(g, 'window') and isinstance(g.window, int): 
                        g.window = max(2, int(g.window * random.uniform(0.95, 1.05)))
                clones.append(clone)
            
            try:
                clone_stats = backtester.evaluate_population(clones, set_type='validation', return_series=False, time_limit=horizon)
                if not clone_stats.empty:
                    min_clone_ret = clone_stats['total_return'].min()
                    # Relaxed Jitter: Allow drop to 25% of original (was 50%)
                    jitter_thresh = val_ret * 0.25
                    if min_clone_ret < jitter_thresh:
                        continue
            except Exception as e:
                print(f"Warning: Jitter test error for {s.name}: {e}")
                continue

        survivor_candidates.append(s)
        survivor_val_indices.append(i)

    # 3. Optimization & Final Evaluation of Survivors
    filtered_candidates = []
    filtered_data = []

    if survivor_candidates:
        print(f"    🔍 Optimizing & Evaluating {len(survivor_candidates)} survivors on Test Set...")
        
        # Initial Test Eval (to get baseline stats and return vectors for DSR)
        test_res, test_returns = backtester.evaluate_population(survivor_candidates, set_type='test', return_series=True, time_limit=horizon, min_trades=config.MIN_TRADES_FOR_METRICS)
        test_map = {row['id']: row for _, row in test_res.iterrows()}

        for i, s in enumerate(survivor_candidates):
            orig_idx = survivor_val_indices[i]
            val_r_vec = val_returns[:, orig_idx]
            
            # Retrieve Parent Stats
            t_stats = train_map[s.name]
            v_stats = val_map[s.name]
            te_stats = test_map.get(s.name, {'total_return': 0.0, 'sortino': 0.0, 'trades': 0})
            
            final_strat = s
            final_test_ret = te_stats['total_return']
            final_test_sortino = float(te_stats['sortino'])
            final_test_trades = int(te_stats['trades'])
            final_test_r_vec = test_returns[:, i]
            
            final_train_ret = t_stats['total_return']
            final_val_ret = v_stats['total_return']
            
            is_optimized = False

            # --- OPTIMIZATION STEP ---
            try:
                # print(f"       ✨ Optimizing {s.name}...") 
                optimizer = StrategyOptimizer(s.name, strategy_dict=s.to_dict(), backtester=backtester, verbose=False, horizon=horizon)
                best_var, best_stats = optimizer.get_best_variant()
                
                if best_var:
                    # Check if improved (get_best_variant already ensures strict improvement in Sortino across board + higher test sortino)
                    # We trust get_best_variant's rigorous checks.
                    
                    print(f"       🚀 Replaced {s.name} with {best_var.name} (Test Sort: {te_stats['sortino']:.2f} -> {best_stats['test']['sortino']:.2f})")
                    
                    final_strat = best_var
                    final_test_ret = best_stats['test']['ret']
                    final_test_sortino = best_stats['test']['sortino']
                    final_test_trades = best_stats['test']['trades']
                    final_train_ret = best_stats['train']['ret']
                    final_val_ret = best_stats['val']['ret']
                    
                    # Need fresh return vector for PSR/DSR
                    res, ret_matrix = backtester.evaluate_population([best_var], set_type='test', return_series=True, time_limit=horizon)
                    final_test_r_vec = ret_matrix[:, 0]
                    
                    # Also need fresh Val return vector for DSR_Val?
                    # Ideally yes. But Val improvement is guaranteed by optimizer.
                    # Let's quickly fetch Val vector too to be accurate.
                    _, val_ret_matrix = backtester.evaluate_population([best_var], set_type='validation', return_series=True, time_limit=horizon)
                    val_r_vec = val_ret_matrix[:, 0]
                    
                    is_optimized = True
            except Exception as e:
                print(f"       ⚠️ Optimization skipped for {s.name}: {e}")

            # --- FINAL GATE: Test Performance Check ---
            if final_test_ret <= 0 or final_test_sortino < 1.5 or final_test_trades < config.MIN_TRADES_FOR_METRICS:
                    continue

            # DSR calculation on Validation Set (Updated if optimized)
            val_sr = estimated_sharpe_ratio(val_r_vec, config.ANNUALIZATION_FACTOR)
            dsr_val = deflated_sharpe_ratio(
                observed_sr=val_sr, returns=val_r_vec, n_trials=total_strategies_evaluated,
                var_returns=np.var(val_r_vec), skew_returns=scipy.stats.skew(val_r_vec), kurt_returns=scipy.stats.kurtosis(val_r_vec),
                annualization_factor=config.ANNUALIZATION_FACTOR
            )

            test_sr = estimated_sharpe_ratio(final_test_r_vec, config.ANNUALIZATION_FACTOR)
            
            # PSR calculation on Test Set
            psr_test = deflated_sharpe_ratio(
                observed_sr=test_sr, returns=final_test_r_vec, n_trials=1,
                var_returns=np.var(final_test_r_vec), skew_returns=scipy.stats.skew(final_test_r_vec), kurt_returns=scipy.stats.kurtosis(final_test_r_vec),
                annualization_factor=config.ANNUALIZATION_FACTOR
            )
            
            # Determine final stats for saving
            final_train_sortino = float(best_stats['train']['sortino']) if is_optimized else train_sortino
            final_val_sortino = float(best_stats['val']['sortino']) if is_optimized else val_sortino

            strat_data = final_strat.to_dict()
            strat_data['test_sortino'] = float(final_test_sortino)
            strat_data['test_return'] = float(final_test_ret)
            strat_data['test_trades'] = int(final_test_trades)
            strat_data['train_return'] = float(final_train_ret)
            strat_data['val_return'] = float(final_val_ret)
            strat_data['train_sortino'] = float(final_train_sortino)
            strat_data['val_sortino'] = float(final_val_sortino)
            strat_data['robust_score'] = float(val_fitness_map.get(s.name, -999.0)) # Keep original robust score for sorting
            strat_data['dsr_val'] = float(dsr_val)
            strat_data['psr_test'] = float(psr_test)
            strat_data['training_id'] = training_id
            strat_data['generation'] = getattr(s, 'generation_found', -1)
            if is_optimized:
                strat_data['parent_name'] = s.name
                strat_data['optimized'] = True
            
            filtered_candidates.append(final_strat)
            filtered_data.append(strat_data)

    # --- PORTFOLIO & SAVING ---
    output = filtered_data
    
    if not output:
        print(f"❌ No strategies passed the strict Train/Val/Test filters (Threshold: {config.MIN_RETURN_THRESHOLD*100:.2f}%).")
        if best_rejected_name:
            print(f"   👀 Closest Candidate: {best_rejected_name}")
            print(f"      Stats: {best_rejected_details}")
        else:
            print("   (No strategy reached even the basic Train/Val sorting gate)")
        return

    print(f"✅ {len(output)} Strategies passed final filtering.")
    
    # Save Apex Strategies
    os.makedirs(config.DIRS['STRATEGIES_DIR'], exist_ok=True)
    out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")
    
    print("\n--- 🧩 CONSTRUCTING PORTFOLIO ---")
    
    # Merge with existing
    existing_data = []
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f: 
                raw_existing = json.load(f)
                existing_data = [x for x in raw_existing if x.get('test_return', 0) > 0]
        except: pass
    
    combined = existing_data + output
    combined.sort(key=lambda x: x.get('robust_score', -999), reverse=True)
    
    # Dedup
    seen = set()
    unique = []
    for s in combined:
        if s['name'] not in seen:
            unique.append(s)
            seen.add(s['name'])
    
    with open(out_path, "w") as f: json.dump(unique[:1000], f, indent=4)
    print(f"💾 Saved results to {out_path}")
    
    print("\n  Top Horizon Champions:")
    for i, s in enumerate(unique[:5]):
        print(f"    {i+1}. {s['name']} (Robust: {s['robust_score']:.2f} | Test: {s['test_return']*100:.2f}%)")

    # --- Persist to Global Inbox (Accumulate ALL finds) ---
    inbox_path = config.DIRS['STRATEGY_INBOX']
    inbox_data = []
    if os.path.exists(inbox_path):
        try:
            with open(inbox_path, "r") as f: inbox_data = json.load(f)
        except: pass
    
    # Filter Inbox for SAME HORIZON to check correlation
    # We only care about redundancy within the same timeframe
    existing_inbox_objs = []
    existing_inbox_dicts = []
    
    for d in inbox_data:
        h = d.get('horizon', None)
        if h == horizon:
            try:
                s = Strategy.from_dict(d)
                s.horizon = h
                existing_inbox_objs.append(s)
                existing_inbox_dicts.append(d)
            except: pass

    # Prepare New Candidates (filtered_candidates matches output list index-wise)
    # We need to simulate both Existing and New to get aligned return vectors
    
    candidates_to_add_map = {} # full_idx -> data
    strategies_to_remove = set() # names of existing strategies to remove
    
    if not output:
        print("No candidates to add.")
    else:
        print(f"    🔍 Checking correlations against {len(existing_inbox_objs)} existing H{horizon} inbox strategies...")
        
        # Combine: [Existing... , New...]
        combined_sim = existing_inbox_objs + filtered_candidates
        
        # Use Validation Set for Correlation (Fast & Representative)
        _, ret_matrix = backtester.evaluate_population(combined_sim, set_type='validation', return_series=True, time_limit=horizon)
        
        # Handle case where evaluation returns None or empty
        if ret_matrix is None or ret_matrix.shape[1] != len(combined_sim):
            print("    ⚠️  Correlation check failed (Simulation Error). Adding all unique names.")
            # Fallback to name check only
            for s_data in output:
                candidates_to_add_map[output.index(s_data) + len(existing_inbox_objs)] = s_data
        else:
            # Calculate Correlation Matrix
            df_rets = pd.DataFrame(ret_matrix)
            # fillna(0) to prevent errors with flatline strategies
            df_rets = df_rets.fillna(0)
            corr_matrix = df_rets.corr().values
            
            n_existing = len(existing_inbox_objs)
            n_new = len(filtered_candidates)
            
            # Indices of accepted strategies (starts with Existing)
            accepted_indices = set(range(n_existing))
            
            added_count = 0
            replaced_count = 0
            
            for i in range(n_new):
                full_idx = n_existing + i
                new_strat_data = output[i]
                name = new_strat_data['name']
                
                # 1. Name Duplication Check
                if any(x['name'] == name for x in inbox_data):
                    continue
                    
                # 2. Correlation Check
                max_corr = -1.0
                conflict_idx = -1
                
                if accepted_indices:
                    # Check against ALL currently accepted (Existing + Accepted New)
                    # We iterate to find max
                    for acc_idx in accepted_indices:
                        c = corr_matrix[full_idx, acc_idx]
                        if np.isnan(c): c = 0.0
                        if c > max_corr:
                            max_corr = c
                            conflict_idx = acc_idx
                
                if max_corr > 0.75:
                    # Identify Conflict Source
                    conflict_data = None
                    is_existing_conflict = False
                    
                    if conflict_idx < n_existing:
                        # Conflict with Existing Inbox Strategy
                        conflict_data = existing_inbox_dicts[conflict_idx]
                        is_existing_conflict = True
                    else:
                        # Conflict with Newly Accepted Strategy
                        conflict_data = candidates_to_add_map.get(conflict_idx)
                        is_existing_conflict = False

                    if conflict_data:
                        # STRICT COMPARISON: Better on EVERY dataset?
                        # Prefer Sortino, fallback to Return
                        
                        # New Stats
                        n_tr_sort = new_strat_data.get('train_sortino', 0)
                        n_val_sort = new_strat_data.get('val_sortino', 0)
                        n_te_sort = new_strat_data.get('test_sortino', 0)
                        
                        # Old Stats (Handle missing legacy keys)
                        o_tr_sort = conflict_data.get('train_sortino', 0)
                        o_val_sort = conflict_data.get('val_sortino', 0)
                        o_te_sort = conflict_data.get('test_sortino', 0)
                        
                        # Calculate Averages
                        n_avg = (n_tr_sort + n_val_sort + n_te_sort) / 3.0
                        o_avg = (o_tr_sort + o_val_sort + o_te_sort) / 3.0
                        
                        if n_avg > o_avg:
                            print(f"      🔄 Replacing Correlated Strategy ({max_corr:.2f}): {conflict_data['name']} -> {name}")
                            
                            # Remove Conflict
                            accepted_indices.remove(conflict_idx)
                            if is_existing_conflict:
                                strategies_to_remove.add(conflict_data['name'])
                            else:
                                if conflict_idx in candidates_to_add_map:
                                    del candidates_to_add_map[conflict_idx]
                            
                            # Accept New
                            new_strat_data['horizon'] = horizon
                            candidates_to_add_map[full_idx] = new_strat_data
                            accepted_indices.add(full_idx)
                            replaced_count += 1
                        else:
                            print(f"      🗑️ Rejected {name}: High Correlation ({max_corr:.2f}) with {conflict_data['name']} and not strictly better.")
                    else:
                        # Should not happen
                        print(f"      ⚠️ Conflict data missing for idx {conflict_idx}. Rejecting {name}.")
                        
                else:
                    # No Correlation -> Accept
                    new_strat_data['horizon'] = horizon 
                    candidates_to_add_map[full_idx] = new_strat_data
                    accepted_indices.add(full_idx)
                    added_count += 1
    
    # Merge and Save
    if candidates_to_add_map or strategies_to_remove:
        # 1. Remove
        if strategies_to_remove:
            inbox_data = [d for d in inbox_data if d['name'] not in strategies_to_remove]
            print(f"      ✂️ Removed {len(strategies_to_remove)} inferior existing strategies.")
            
        # 2. Add
        to_add_list = list(candidates_to_add_map.values())
        if to_add_list:
            inbox_data.extend(to_add_list)
            print(f"      📦 Added {len(to_add_list)} new strategies to Inbox.")

        with open(inbox_path, "w") as f: json.dump(inbox_data, f, indent=4)
        play_success_sound()
    else:
        print("📦 No new strategies added (all duplicates or correlated and inferior).")
