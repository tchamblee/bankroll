import os
import json
import pandas as pd
import numpy as np
import config
from genome import Strategy
from .engine import BacktestEngine

def find_strategy_in_files(strategy_name):
    """Searches all strategy output files for a strategy with the given name."""
    search_patterns = [
        "found_strategies.json",
        "candidates.json",
        "apex_strategies_*_top5_unique.json",
        "apex_strategies_*_top10.json",
        "apex_strategies_*.json",
        "optimized_*.json"
    ]
    
    import glob
    for pattern in search_patterns:
        files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], pattern))
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for s_dict in data:
                        if s_dict.get('name') == strategy_name:
                            # Found it!
                            # Ensure horizon is set if missing (infer from filename)
                            if 'horizon' not in s_dict:
                                # extracting horizon from filename e.g., apex_strategies_90_...
                                parts = os.path.basename(file_path).split('_')
                                for p in parts:
                                    if p.isdigit():
                                        s_dict['horizon'] = int(p)
                                        break
                            
                            # Ensure metrics are present
                            if 'metrics' not in s_dict:
                                s_dict['metrics'] = {
                                    'sortino_oos': s_dict.get('test_sortino', 0),
                                    'robust_return': s_dict.get('test_return', s_dict.get('robust_return', 0))
                                }
                                
                            return s_dict
            except Exception:
                continue
    return None

def refresh_strategies(strategies_data):
    """
    Re-simulates strategies to ensure fresh performance metrics.
    Updates the dictionaries in-place and returns the list.
    """
    if not strategies_data:
        return []
    
    print(f"🔄 Refreshing metrics for {len(strategies_data)} strategies...")
    
    # Load Data
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("⚠️  Feature Matrix not found. Cannot refresh metrics.")
        return strategies_data
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    engine = BacktestEngine(df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # Convert dicts to Strategy objects
    strat_objects = []
    
    # Pre-clean strategies_data to ensure valid dicts
    # We will map by name
    
    for s_data in strategies_data:
        try:
            strat = Strategy.from_dict(s_data)
            # Ensure horizon is set
            strat.horizon = s_data.get('horizon', config.DEFAULT_TIME_LIMIT)
            strat_objects.append(strat)
        except Exception as e:
            print(f"⚠️  Skipping invalid strategy {s_data.get('name', '?')}: {e}")

    if not strat_objects:
        return strategies_data

    # Group by horizon for batch processing
    from collections import defaultdict
    by_horizon = defaultdict(list)
    for s in strat_objects:
        h = getattr(s, 'horizon', config.DEFAULT_TIME_LIMIT)
        by_horizon[h].append(s)
        
    for h, group in by_horizon.items():
        # Test (OOS)
        stats_test, net_returns_matrix = engine.evaluate_population(
            group, set_type='test', return_series=True, time_limit=h
        )
        
        # Train
        stats_train = engine.evaluate_population(group, set_type='train', time_limit=h)
        
        # Validation
        stats_val = engine.evaluate_population(group, set_type='validation', time_limit=h)
        
        # Create lookup maps
        test_map = {row['id']: row for _, row in stats_test.iterrows()}
        train_map = {row['id']: row for _, row in stats_train.iterrows()}
        val_map = {row['id']: row for _, row in stats_val.iterrows()}
        
        # Update original data dicts
        for s in group:
            # Find the original dict object
            target_dict = next((d for d in strategies_data if d.get('name') == s.name), None)
            if not target_dict: continue
            
            if s.name in test_map and s.name in train_map and s.name in val_map:
                t_stats = train_map[s.name]
                v_stats = val_map[s.name]
                te_stats = test_map[s.name]
                
                # --- Update Flat Keys (Inbox Style) ---
                target_dict['train_return'] = float(t_stats['total_return'])
                target_dict['val_return'] = float(v_stats['total_return'])
                target_dict['test_return'] = float(te_stats['total_return'])
                target_dict['test_sortino'] = float(te_stats['sortino'])
                target_dict['test_trades'] = int(te_stats['trades'])
                target_dict['train_trades'] = int(t_stats['trades'])
                target_dict['val_trades'] = int(v_stats['trades'])
                
                # Additional Helper for Reporting
                returns = net_returns_matrix[:, group.index(s)] # Need correct index in group
                # Wait, 'net_returns_matrix' columns correspond to 'group' list order.
                # 'group' list order is preserved.
                
                # Calculate Sharpe/Drawdown if possible (fast approx)
                # But evaluate_population doesn't return full series by default unless return_series=True
                # We have return_series=True for Test.
                
                # Calculate Sharpe/DD for Test
                test_ret_vec = returns # already extracted column? No.
                # net_returns_matrix is (bars x strategies)
                test_ret_vec = net_returns_matrix[:, group.index(s)]
                
                ann_factor = config.ANNUALIZATION_FACTOR
                avg_ret = np.mean(test_ret_vec)
                std_ret = np.std(test_ret_vec)
                sharpe = (avg_ret * ann_factor) / (std_ret * np.sqrt(ann_factor) + 1e-9)
                
                cum_pnl = np.cumsum(test_ret_vec)
                peak = np.maximum.accumulate(cum_pnl)
                dd = cum_pnl - peak
                max_dd = np.min(dd) # Simple PnL drawdown (approx)
                
                target_dict['test_sharpe'] = float(sharpe)
                target_dict['max_drawdown'] = float(max_dd)
                target_dict['test_ann_return'] = float(avg_ret * ann_factor)

                # --- Update Structured Stats (Candidate/Optimizer Style) ---
                target_dict['train_stats'] = {'ret': float(t_stats['total_return']), 'sortino': float(t_stats['sortino']), 'trades': int(t_stats['trades'])}
                target_dict['val_stats'] = {'ret': float(v_stats['total_return']), 'sortino': float(v_stats['sortino']), 'trades': int(v_stats['trades'])}
                target_dict['test_stats'] = {'ret': float(te_stats['total_return']), 'sortino': float(te_stats['sortino']), 'trades': int(te_stats['trades'])}
                
                # --- Legacy/Report fields ---
                target_dict['metrics'] = {
                    'sortino_oos': float(te_stats['sortino']),
                    'robust_return': float(te_stats['total_return'])
                }

    engine.shutdown()
    print("✅ Metrics refreshed.")
    return strategies_data
