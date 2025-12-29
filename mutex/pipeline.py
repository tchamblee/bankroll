import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from backtest import BacktestEngine
from .simulator import _jit_simulate_mutex_custom
from .optimization import optimize_mutex_portfolio, optimize_hrp_portfolio
from .utils import load_all_candidates

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
    
    # 4. Optimize HRP Portfolio (Robust)
    optimize_hrp_portfolio(candidates[:100], backtester)
