import os
import json
import random
import numpy as np
import scipy.stats
import config
from genome import Strategy
from backtest.statistics import deflated_sharpe_ratio, estimated_sharpe_ratio

def save_campaign_results(hall_of_fame, backtester, horizon, training_id, total_strategies_evaluated):
    print(f"\n--- 🛡️ FINAL SELECTION (Using Diverse HOF) ---")
    
    top_candidates = [entry['strat'] for entry in hall_of_fame[:100]]
    val_fitness_map = {entry['strat'].name: entry['fit'] for entry in hall_of_fame[:100]}
    
    if not top_candidates:
        print("No strategies survived.")
        return

    # 1. Evaluate on Development Sets (Train + Validation) ONLY
    val_res, val_returns = backtester.evaluate_population(top_candidates, set_type='validation', return_series=True, time_limit=horizon, min_trades=10)
    train_res = backtester.evaluate_population(top_candidates, set_type='train', time_limit=horizon, min_trades=10)
    
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
        
        passed_gate = False
        rejection_reason = []

        # Check 1: Raw Return Threshold
        if train_ret >= min_ret_threshold and val_ret >= min_ret_threshold:
            passed_gate = True
        
        # Check 2: Stability Rescue (Sniper Logic)
        if not passed_gate:
            train_sortino = t_stats.get('sortino', 0)
            val_sortino = v_stats.get('sortino', 0)
            if train_ret > 0 and val_ret > 0 and train_sortino > 1.25 and val_sortino > 1.25:
                passed_gate = True

        if not passed_gate:
            if train_ret < min_ret_threshold: rejection_reason.append(f"Train_Ret({train_ret*100:.2f}%)")
            if val_ret < min_ret_threshold: rejection_reason.append(f"Val_Ret({val_ret*100:.2f}%)")
            
            # Track closest call
            min_ret_here = min(train_ret, val_ret)
            if min_ret_here > best_rejected_min_ret:
                best_rejected_min_ret = min_ret_here
                best_rejected_name = s.name
                best_rejected_details = f"Train: {train_ret*100:.2f}%, Val: {val_ret*100:.2f}%"
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

    # 3. Final Evaluation of Survivors (Now we can look at Test for reporting)
    filtered_candidates = []
    filtered_data = []

    if survivor_candidates:
        print(f"    🔍 Evaluating {len(survivor_candidates)} survivors on Test Set...")
        test_res, test_returns = backtester.evaluate_population(survivor_candidates, set_type='test', return_series=True, time_limit=horizon, min_trades=10)
        test_map = {row['id']: row for _, row in test_res.iterrows()}

        for i, s in enumerate(survivor_candidates):
            orig_idx = survivor_val_indices[i]
            val_r_vec = val_returns[:, orig_idx]
            
            # DSR calculation on Validation Set
            val_sr = estimated_sharpe_ratio(val_r_vec, config.ANNUALIZATION_FACTOR)
            dsr_val = deflated_sharpe_ratio(
                observed_sr=val_sr, returns=val_r_vec, n_trials=total_strategies_evaluated,
                var_returns=np.var(val_r_vec), skew_returns=scipy.stats.skew(val_r_vec), kurt_returns=scipy.stats.kurtosis(val_r_vec),
                annualization_factor=config.ANNUALIZATION_FACTOR
            )

            # Test Stats
            t_stats = train_map[s.name]
            v_stats = val_map[s.name]
            te_stats = test_map.get(s.name, {'total_return': 0.0, 'sortino': 0.0, 'trades': 0})

            test_ret = te_stats['total_return']
            
            # --- FINAL GATE: Test Performance Check ---
            if test_ret <= 0:
                    continue

            test_r_vec = test_returns[:, i]
            test_sr = estimated_sharpe_ratio(test_r_vec, config.ANNUALIZATION_FACTOR)
            
            # PSR calculation on Test Set (Purely for final reporting)
            psr_test = deflated_sharpe_ratio(
                observed_sr=test_sr, returns=test_r_vec, n_trials=1,
                var_returns=np.var(test_r_vec), skew_returns=scipy.stats.skew(test_r_vec), kurt_returns=scipy.stats.kurtosis(test_r_vec),
                annualization_factor=config.ANNUALIZATION_FACTOR
            )

            strat_data = s.to_dict()
            strat_data['test_sortino'] = float(te_stats['sortino'])
            strat_data['test_return'] = test_ret
            strat_data['test_trades'] = int(te_stats['trades'])
            strat_data['train_return'] = t_stats['total_return']
            strat_data['val_return'] = v_stats['total_return']
            strat_data['robust_score'] = float(val_fitness_map.get(s.name, -999.0)) 
            strat_data['dsr_val'] = float(dsr_val)
            strat_data['psr_test'] = float(psr_test)
            strat_data['training_id'] = training_id
            strat_data['generation'] = getattr(s, 'generation_found', -1)
            
            filtered_candidates.append(s)
            filtered_data.append(strat_data)

    # --- PORTFOLIO & SAVING ---
    output = filtered_data
    
    if not output:
        print(f"❌ No strategies passed the strict Train/Val/Test filters (Threshold: {config.MIN_RETURN_THRESHOLD*100:.2f}%).")
        if best_rejected_name:
            print(f"   👀 Closest Candidate: {best_rejected_name}")
            print(f"      Stats: {best_rejected_details}")
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
    
    new_inbox_count = 0
    for s_data in output:
        if not any(x['name'] == s_data['name'] for x in inbox_data):
                s_data['horizon'] = horizon 
                inbox_data.append(s_data)
                new_inbox_count += 1
    
    if new_inbox_count > 0:
        with open(inbox_path, "w") as f: json.dump(inbox_data, f, indent=4)
        print(f"📦 Added {new_inbox_count} new strategies to Inbox: {inbox_path}")
        print('\a' * 3) # 🔔 Audible Alert
