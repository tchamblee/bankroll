import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import itertools
import time
from numba import jit
import scipy.stats
from scipy.optimize import minimize
import config
from genome import Strategy
from backtest import BacktestEngine
from backtest.statistics import deflated_sharpe_ratio, estimated_sharpe_ratio
from trade_simulator import TradeSimulator

# ... (Previous Helper Functions) ...

def optimize_weighted_portfolio(candidates, backtester):
    print("\n" + "="*80)
    print("⚖️  WEIGHTED PORTFOLIO OPTIMIZATION (Mean-Variance)")
    print("="*80)
    
    if len(candidates) < 2:
        print("Not enough candidates for weighted optimization.")
        return

    # 1. Generate Returns Matrix (Train + Val)
    # We need independent returns for each strategy to correlate them
    opt_start = 0
    opt_end = backtester.val_idx
    
    # Generate full signals first
    backtester.ensure_context(candidates)
    raw_sig = backtester.generate_signal_matrix(candidates)
    shifted_sig = np.vstack([np.zeros((1, len(candidates)), dtype=raw_sig.dtype), raw_sig[:-1]])
    
    opt_sig = shifted_sig[opt_start:opt_end]
    
    prices = backtester.open_vec[opt_start:opt_end].astype(np.float64)
    times = backtester.times_vec.iloc[opt_start:opt_end] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[opt_start:opt_end]
    highs = backtester.high_vec[opt_start:opt_end].astype(np.float64)
    lows = backtester.low_vec[opt_start:opt_end].astype(np.float64)
    atr = backtester.atr_vec[opt_start:opt_end].astype(np.float64)

    print("  Simulating individual streams...")
    rets_matrix, _ = backtester.run_simulation_batch(
        opt_sig, candidates, prices, times, 
        time_limit=config.DEFAULT_TIME_LIMIT, # Use default or per-strategy? Batch handles per-strategy in params
        highs=highs, lows=lows, atr=atr
    )
    
    # 2. Optimization Objective
    # Maximize Sharpe = Mean / Std
    n_assets = len(candidates)
    
    def objective(weights):
        portfolio_rets = np.dot(rets_matrix, weights)
        mean = np.mean(portfolio_rets)
        std = np.std(portfolio_rets) + 1e-9
        sharpe = mean / std
        return -sharpe # Minimize neg sharpe

    # Constraints: Sum weights = 1, 0 <= w <= 0.4 (Diversification constraint)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0.0, 0.4) for _ in range(n_assets))
    
    # Initial Guess (Equal Weight)
    init_weights = np.array([1.0/n_assets] * n_assets)
    
    print("  Optimizing weights...")
    result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        weights = result.x
        # Filter low weights
        final_portfolio = []
        for i, w in enumerate(weights):
            if w > 0.01:
                s_dict = candidates[i].to_dict()
                s_dict['weight'] = float(w)
                s_dict['horizon'] = candidates[i].horizon
                final_portfolio.append(s_dict)
        
        final_portfolio.sort(key=lambda x: x['weight'], reverse=True)
        
        # OOS Performance check
        oos_start = backtester.val_idx
        oos_sig = shifted_sig[oos_start:]
        oos_prices = backtester.open_vec[oos_start:].astype(np.float64)
        oos_times = backtester.times_vec.iloc[oos_start:] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[oos_start:]
        oos_highs = backtester.high_vec[oos_start:].astype(np.float64)
        oos_lows = backtester.low_vec[oos_start:].astype(np.float64)
        oos_atr = backtester.atr_vec[oos_start:].astype(np.float64)

        oos_rets_matrix, _ = backtester.run_simulation_batch(
            oos_sig, candidates, oos_prices, oos_times, 
            highs=oos_highs, lows=oos_lows, atr=oos_atr
        )
        
        oos_port_rets = np.dot(oos_rets_matrix, weights)
        oos_profit = np.sum(oos_port_rets) * config.ACCOUNT_SIZE
        oos_sortino = (np.mean(oos_port_rets) / (np.std(np.minimum(oos_port_rets, 0)) + 1e-9)) * np.sqrt(config.ANNUALIZATION_FACTOR)
        
        print(f"\n🏆 WEIGHTED PORTFOLIO RESULT")
        print(f"  Strategies: {len(final_portfolio)}")
        print(f"  OOS Profit: ${oos_profit:,.2f}")
        print(f"  OOS Sortino: {oos_sortino:.2f}")
        
        out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "weighted_portfolio.json")
        with open(out_path, 'w') as f:
            json.dump(final_portfolio, f, indent=4)
        print(f"💾 Saved to {out_path}")
        
        return final_portfolio
    else:
        print("Optimization failed.")
        return []

def run_mutex_backtest():
    # 1. Setup
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # 2. Load Candidates
    candidates = load_all_candidates()
    if not candidates:
        print("❌ No candidates found.")
        return

    # 3. Optimize Portfolio (Mutex)
    best_portfolio, stats = optimize_mutex_portfolio(candidates, backtester)
    
    if best_portfolio:
        # Save Mutex Result
        mutex_data = []
        for s in best_portfolio:
            s_dict = s.to_dict()
            s_dict['horizon'] = s.horizon
            s_dict['training_id'] = getattr(s, 'training_id', 'legacy')
            mutex_data.append(s_dict)

        mutex_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
        with open(mutex_path, 'w') as f:
            json.dump(mutex_data, f, indent=4)
        print(f"\n💾 Saved Optimized Mutex Portfolio to {mutex_path}")
        
        # Visual Validation (Re-run best for plotting)
        # ... (Existing Plotting Logic) ...
        oos_start = backtester.val_idx
        prices = backtester.open_vec[oos_start:].astype(np.float64)
        highs = backtester.high_vec[oos_start:].astype(np.float64)
        lows = backtester.low_vec[oos_start:].astype(np.float64)
        atr = backtester.atr_vec[oos_start:].astype(np.float64)
        times = backtester.times_vec.iloc[oos_start:] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[oos_start:]
        
        if hasattr(times, 'dt'):
            hours = times.dt.hour.values.astype(np.int8)
            weekdays = times.dt.dayofweek.values.astype(np.int8)
        else:
            dt_idx = pd.to_datetime(times)
            hours = dt_idx.hour.values.astype(np.int8)
            weekdays = dt_idx.dayofweek.values.astype(np.int8)
            
        backtester.ensure_context(best_portfolio)
        raw_sig = backtester.generate_signal_matrix(best_portfolio)
        shifted_sig = np.vstack([np.zeros((1, len(best_portfolio)), dtype=raw_sig.dtype), raw_sig[:-1]])
        oos_sig = shifted_sig[oos_start:]
        
        horizons = np.array([s.horizon for s in best_portfolio], dtype=np.int64)
        sl_mults = np.array([getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for s in best_portfolio], dtype=np.float64)
        tp_mults = np.array([getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for s in best_portfolio], dtype=np.float64)
        
        rets, _, _ = _jit_simulate_mutex_custom(
            oos_sig.astype(np.float64), 
            prices, highs, lows, atr, hours, weekdays, 
            horizons, sl_mults, tp_mults, 
            config.STANDARD_LOT_SIZE, 
            config.SPREAD_BPS / 10000.0, 
            config.COST_BPS / 10000.0, 
            config.ACCOUNT_SIZE, 
            config.TRADING_END_HOUR, 
            config.STOP_LOSS_COOLDOWN_BARS
        )
            
        cum_profit = np.cumsum(rets) * config.ACCOUNT_SIZE
        total_oos_profit = cum_profit[-1] if len(cum_profit) > 0 else 0.0
        
        avg_oos = np.mean(rets)
        downside_oos = np.sqrt(np.mean(np.minimum(rets, 0)**2)) + 1e-9
        sortino_oos = (avg_oos / downside_oos) * np.sqrt(config.ANNUALIZATION_FACTOR)
        
        print("\n" + "="*80)
        print("🧪 OUT-OF-SAMPLE (TEST SET) PERFORMANCE")
        print("="*80)
        print(f"  Range:      Index {oos_start} to End ({len(rets)} bars)")
        print(f"  Profit:     ${total_oos_profit:,.2f}")
        print(f"  Sortino:    {sortino_oos:.2f}")
        print(f"  Return:     {(total_oos_profit/config.ACCOUNT_SIZE)*100:.1f}%")
        print("="*80 + "\n")

        plt.figure(figsize=(15, 8))
        plt.plot(range(len(cum_profit)), cum_profit, label='Optimized Mutex', color='green', linewidth=2)
        plt.title(f"Optimized Mutex Portfolio (OOS)\nSortino: {sortino_oos:.2f} | Profit: ${total_oos_profit:,.0f}")
        plt.ylabel("Cumulative Profit ($)")
        plt.xlabel("Bars (OOS)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        out_path = os.path.join(config.DIRS['PLOTS_DIR'], "mutex_optimized.png")
        plt.savefig(out_path)
        print(f"📸 Saved Performance Chart to {out_path}")
    
    # 4. Optimize Weighted Portfolio (New Opportunity)
    # Use top candidates (already loaded and sorted)
    # Limiting to top 20 to keep optimization fast
    optimize_weighted_portfolio(candidates[:20], backtester)

if __name__ == "__main__":
    run_mutex_backtest()