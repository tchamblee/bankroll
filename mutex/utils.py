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
            if 'test_stats' in s_dict:
                 strat.fitness = s_dict['test_stats'].get('sortino', 0)
            elif 'metrics' in s_dict:
                 strat.fitness = s_dict['metrics'].get('sortino_oos', 0)
            else:
                 strat.fitness = 0.0
            
            strategies.append(strat)
        except Exception as e:
            print(f"⚠️ Error loading strategy {s_dict.get('name', 'Unknown')}: {e}")
            continue
            
    print(f"✅ Loaded {len(strategies)} candidates from {candidates_path}")
    return strategies
