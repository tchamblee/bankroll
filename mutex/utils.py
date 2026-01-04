import os
import json
import config
from genome import Strategy

def load_all_candidates():
    """
    Loads candidates from candidates.json.
    """
    candidates_path = config.CANDIDATES_FILE
    if not os.path.exists(candidates_path):
        return {}

    with open(candidates_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Error decoding candidates.json")
            return []

    strategies = []
    for s_dict in data:
        try:
            strat = Strategy.from_dict(s_dict)
            # Hydrate Params
            strat.horizon = s_dict.get('horizon', config.DEFAULT_TIME_LIMIT)
            strat.stop_loss_pct = s_dict.get('stop_loss_pct', config.DEFAULT_STOP_LOSS)
            strat.take_profit_pct = s_dict.get('take_profit_pct', config.DEFAULT_TAKE_PROFIT)
            
            # Store metrics for sorting/ranking
            # Prioritize: 1. Robust Score (WFV), 2. Test Sortino
            
            # Robust Score
            strat.robust_score = s_dict.get('robust_score', 0.0)
            
            # Test Sortino
            test_sortino = 0.0
            if 'test_sortino' in s_dict:
                test_sortino = s_dict['test_sortino']
            elif 'test_stats' in s_dict:
                test_sortino = s_dict['test_stats'].get('sortino', 0)
            elif 'metrics' in s_dict:
                test_sortino = s_dict['metrics'].get('sortino_oos', 0)
            
            strat.test_sortino = test_sortino
            
            # Fitness logic: Use robust_score if available, else test_sortino
            if strat.robust_score != 0:
                strat.fitness = strat.robust_score
            else:
                strat.fitness = strat.test_sortino
            
            strategies.append(strat)
        except Exception as e:
            print(f"⚠️ Error loading strategy {s_dict.get('name', 'Unknown')}: {e}")
            continue
            
    print(f"✅ Loaded {len(strategies)} candidates from {candidates_path}")
    return strategies
