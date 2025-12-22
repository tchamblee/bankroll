import json
import os
import sys
import pandas as pd
import numpy as np
import config
from backtest_engine import BacktestEngine
from strategy_genome import Strategy

def main():
    print("DEBUGGING CHILD_7803 (H240)")
    
    # Create Backtester
    backtester = BacktestEngine(base_df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # 3. Prepare Single Strategy
    with open("output/strategies/apex_strategies_240.json", "r") as f:
        data = json.load(f)
        
    strat_data = next((s for s in data if s["name"] == "Child_7803"), None)
    if not strat_data:
        print("Strategy not found!")
        return
        
    strat = Strategy.from_dict(strat_data)
    
    # Run Evaluation
    print("Evaluating Validation Set...")
    res_val = backtester.evaluate_population([strat], set_type='validation')
    print(f"Val Return: {res_val.iloc[0]['total_return']:.4f}")
    print(f"Val Sortino: {res_val.iloc[0]['sortino']:.4f}")
    
    print("\nEvaluating Test Set...")
    res_test = backtester.evaluate_population([strat], set_type='test')
    print(f"Test Return: {res_test.iloc[0]['total_return']:.4f}")
    print(f"Test Sortino: {res_test.iloc[0]['sortino']:.4f}")
    
    print("\nEvaluating Full Set...")
    full_signals = backtester.generate_signal_matrix([strat])
    net_returns_full, _ = backtester.run_simulation_batch(
        full_signals, 
        backtester.close_vec, 
        backtester.times_vec, 
        time_limit=120
    )
    # Exclude Warmup
    full_ret = np.sum(net_returns_full[3200:], axis=0)[0]
    print(f"Full Return: {full_ret:.4f}")

if __name__ == "__main__":
    main()
