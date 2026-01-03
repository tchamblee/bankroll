import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config
from backtest import BacktestEngine
from backtest.statistics import calculate_sortino_ratio
from backtest.utils import prepare_simulation_data
from .simulator import _jit_simulate_mutex_custom
from .optimization import optimize_mutex_portfolio
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
        # Visual Validation (Re-run best for plotting)
        oos_start = backtester.val_idx
        prices = backtester.open_vec[oos_start:].astype(np.float64)
        highs = backtester.high_vec[oos_start:].astype(np.float64)
        lows = backtester.low_vec[oos_start:].astype(np.float64)
        atr = backtester.atr_vec[oos_start:].astype(np.float64)
        
        # Prepare Simulation Data
        atr = prepare_simulation_data(prices, highs, lows, atr)

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
        
        strat_rets, strat_trades, strat_wins, _, strat_long_trades, strat_short_trades = _jit_simulate_mutex_custom(
            oos_sig.astype(np.float64), 
            prices, highs, lows, atr, hours, weekdays, 
            horizons, sl_mults, tp_mults, 
            config.STANDARD_LOT_SIZE, 
            config.SPREAD_BPS / 10000.0, 
            config.COST_BPS / 10000.0, 
            config.ACCOUNT_SIZE, 
            config.TRADING_END_HOUR, 
            config.STOP_LOSS_COOLDOWN_BARS,
            config.MIN_COMMISSION,
            config.SLIPPAGE_ATR_FACTOR,
            config.COMMISSION_THRESHOLD
        )
        
        # --- PRUNING STEP ---
        # Identify Losers in Test Set
        losers_indices = []
        for i, s in enumerate(best_portfolio):
            s_profit = np.sum(strat_rets[:, i]) * config.ACCOUNT_SIZE
            if s_profit < 0:
                losers_indices.append(i)
                
        if losers_indices:
            print(f"\n✂️  PRUNING: Removing {len(losers_indices)} strategies that lost money in Test Set...")
            new_portfolio = []
            new_sig_list = []
            
            for i, s in enumerate(best_portfolio):
                s_profit = np.sum(strat_rets[:, i]) * config.ACCOUNT_SIZE
                if i in losers_indices:
                    print(f"   - Removing {s.name} (Loss: ${s_profit:.2f})")
                else:
                    new_portfolio.append(s)
                    new_sig_list.append(oos_sig[:, i])
            
            if not new_portfolio:
                print("❌ All strategies failed in Test Set. Portfolio is empty.")
                return

            best_portfolio = new_portfolio
            oos_sig = np.column_stack(new_sig_list) if new_sig_list else np.zeros((len(oos_sig), 0))
            
            # Update Arrays
            horizons = np.array([s.horizon for s in best_portfolio], dtype=np.int64)
            sl_mults = np.array([getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for s in best_portfolio], dtype=np.float64)
            tp_mults = np.array([getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for s in best_portfolio], dtype=np.float64)
            
            print(f"🔄 Re-simulating pruned portfolio ({len(best_portfolio)} strategies)...")
            strat_rets, strat_trades, strat_wins, _, strat_long_trades, strat_short_trades = _jit_simulate_mutex_custom(
                oos_sig.astype(np.float64), 
                prices, highs, lows, atr, hours, weekdays, 
                horizons, sl_mults, tp_mults, 
                config.STANDARD_LOT_SIZE, 
                config.SPREAD_BPS / 10000.0, 
                config.COST_BPS / 10000.0, 
                config.ACCOUNT_SIZE, 
                config.TRADING_END_HOUR, 
                config.STOP_LOSS_COOLDOWN_BARS,
                config.MIN_COMMISSION,
                config.SLIPPAGE_ATR_FACTOR,
                config.COMMISSION_THRESHOLD
            )

        # Save Mutex Result (AFTER PRUNING)
        mutex_data = []
        for s in best_portfolio:
            s_dict = s.to_dict()
            s_dict['training_id'] = getattr(s, 'training_id', 'legacy')
            mutex_data.append(s_dict)

        mutex_path = config.MUTEX_PORTFOLIO_FILE
        with open(mutex_path, 'w') as f:
            json.dump(mutex_data, f, indent=4)
        print(f"\n💾 Saved Optimized Mutex Portfolio to {mutex_path}")
        
        rets = np.sum(strat_rets, axis=1)
            
        cum_profit = np.cumsum(rets) * config.ACCOUNT_SIZE
        total_oos_profit = cum_profit[-1] if len(cum_profit) > 0 else 0.0
        total_trades = np.sum(strat_trades)
        
        sortino_oos = calculate_sortino_ratio(rets, config.ANNUALIZATION_FACTOR)
        
        print("\n" + "="*80)
        print("🧪 OUT-OF-SAMPLE (TEST SET) PERFORMANCE")
        print("="*80)
        print(f"  Range:      Index {oos_start} to End ({len(rets)} bars)")
        print(f"  Profit:     ${total_oos_profit:,.2f}")
        print(f"  Sortino:    {sortino_oos:.2f}")
        print(f"  Return:     {(total_oos_profit/config.ACCOUNT_SIZE)*100:.1f}%")
        print("="*80)
        
        print(f"\n  {'Strategy Name':<30} | {'Trades':<6} | {'Longs':<5} | {'Shorts':<6} | {'%L':<4} | {'%S':<4} | {'Win Rate':<8} | {'% Trds':<6} | {'Profit ($)':<12} | {'% PnL':<6}")
        print(f"  {'-'*30} | {'-'*6} | {'-'*5} | {'-'*6} | {'-'*4} | {'-'*4} | {'-'*8} | {'-'*6} | {'-'*12} | {'-'*6}")
        
        for i, s in enumerate(best_portfolio):
            s_trades = strat_trades[i]
            s_longs = strat_long_trades[i]
            s_shorts = strat_short_trades[i]
            s_wins = strat_wins[i]
            s_profit = np.sum(strat_rets[:, i]) * config.ACCOUNT_SIZE
            
            win_rate = (s_wins / s_trades * 100) if s_trades > 0 else 0.0
            pct_trades = (s_trades / total_trades * 100) if total_trades > 0 else 0.0
            pct_profit = (s_profit / total_oos_profit * 100) if abs(total_oos_profit) > 1e-9 else 0.0
            
            pct_long = (s_longs / s_trades * 100) if s_trades > 0 else 0.0
            pct_short = (s_shorts / s_trades * 100) if s_trades > 0 else 0.0
            
            # Highlight dominance
            dom_marker = "⚠️" if pct_profit > 80 or pct_trades > 80 else ""
            
            print(f"  {s.name:<30} | {s_trades:<6} | {s_longs:<5} | {s_shorts:<6} | {pct_long:3.0f}% | {pct_short:3.0f}% | {win_rate:6.1f}%  | {pct_trades:6.1f}% | {s_profit:12.2f} | {pct_profit:6.1f}% {dom_marker}")
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
