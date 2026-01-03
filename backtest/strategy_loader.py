import os
import json
import glob
import config
from genome import Strategy

def load_strategies(source_type, horizon=None, load_metrics=True):
    """
    Loads strategies from various sources.
    
    Args:
        source_type (str): 'inbox', 'mutex', 'apex', 'champion'
        horizon (int, optional): Required for 'apex' or 'champion' sources.
        load_metrics (bool): Whether to try and hydrate metric fields.
        
    Returns:
        tuple: (List[Strategy], List[dict]) -> (Strategy Objects, Original Dicts)
    """
    file_path = None
    
    if source_type == 'inbox':
        file_path = config.DIRS['STRATEGY_INBOX']
        
    elif source_type == 'mutex':
        file_path = config.MUTEX_PORTFOLIO_FILE
        
    elif source_type == 'apex':
        if horizon is None:
            raise ValueError("Horizon required for apex strategies.")
        
        # Priority: Top 5 Unique -> Top 10 -> Raw Apex
        p1 = config.APEX_FILE_TEMPLATE.format(horizon).replace(".json", "_top5_unique.json")
        p2 = config.APEX_FILE_TEMPLATE.format(horizon).replace(".json", "_top10.json")
        p3 = config.APEX_FILE_TEMPLATE.format(horizon)
        
        if os.path.exists(p1): file_path = p1
        elif os.path.exists(p2): file_path = p2
        elif os.path.exists(p3): file_path = p3
        
    elif source_type == 'champion':
        # Same as apex, but we expect the caller to just take the first one
        # Re-using logic
        return load_strategies('apex', horizon, load_metrics)
        
    elif source_type == 'all_apex':
        # Used for report_top_strategies where we scan specific file pattern
        if horizon is None:
             raise ValueError("Horizon required for all_apex.")
        file_path = config.APEX_FILE_TEMPLATE.format(horizon)

    else:
        # Try generic path if source_type looks like a filename
        potential_path = os.path.join(config.DIRS['STRATEGIES_DIR'], source_type)
        if os.path.exists(potential_path):
            file_path = potential_path
            
    if not file_path or not os.path.exists(file_path):
        return [], []
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return [], []
        
    if not isinstance(data, list):
        data = [data] # Handle single strategy files
        
    strategies = []
    strategy_dicts = []
    
    for d in data:
        try:
            strat = Strategy.from_dict(d)
            
            # Ensure horizon is set
            strat.horizon = d.get('horizon', horizon if horizon else config.DEFAULT_TIME_LIMIT)
            strat.generation_found = d.get('generation', '?')
            
            # Hydrate metrics
            if load_metrics:
                _hydrate_metrics(strat, d)
                
            # Store original dict in .data for convenience (used by reports)
            strat.data = d
            
            strategies.append(strat)
            strategy_dicts.append(d)
        except Exception:
            # Skip malformed
            pass
            
    return strategies, strategy_dicts

def _hydrate_metrics(strat, data_dict):
    """Helper to standardize metric access on the Strategy object."""
    metrics = data_dict.get('metrics', {})
    
    # Robust Return
    strat.robust_return = metrics.get('robust_return', 
                                    data_dict.get('robust_return', 
                                    data_dict.get('test_return', -999)))
    
    # Full Return
    strat.full_return = metrics.get('full_return', 
                                  data_dict.get('full_return', -999))
    
    # Sortino
    strat.sortino_oos = metrics.get('sortino_oos',
                                  data_dict.get('test_sortino', -999))
