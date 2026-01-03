import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from feature_engine import FeatureEngine
from genome import Strategy
from backtest import BacktestEngine
from backtest.statistics import calculate_sortino_ratio
from backtest.strategy_loader import load_strategies
from mutex.simulator import _jit_simulate_mutex_custom
import config

def plot_performance(df, strategies):
    backtester = BacktestEngine(df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # 1. Generate Full Signal Matrix
    full_signal_matrix = backtester.generate_signal_matrix(strategies)
    
    # --- FIX: LOOKAHEAD BIAS (Next Open Execution) ---
    # Shift signals forward by 1
    full_signal_matrix = np.vstack([np.zeros((1, full_signal_matrix.shape[1]), dtype=full_signal_matrix.dtype), full_signal_matrix[:-1]])
    
    # Slice for OOS (Test Set)
    test_start = backtester.val_idx
    
    oos_signals = full_signal_matrix[test_start:]
    oos_prices = backtester.open_vec.flatten()[test_start:]
    oos_times = backtester.times_vec.iloc[test_start:] if hasattr(backtester.times_vec, 'iloc') else backtester.times_vec[test_start:]
    oos_highs = backtester.high_vec[test_start:]
    oos_lows = backtester.low_vec[test_start:]
    oos_atr = backtester.atr_vec[test_start:]
    
    # 2. Run Simulation via BacktestEngine's wrapper (consistent logic)
    print(f"Running OOS Simulation for Plotting (Bars: {len(oos_prices)})")
    net_returns, trades_count = backtester.run_simulation_batch(
        oos_signals, 
        strategies,
        oos_prices, 
        oos_times, 
        time_limit=config.DEFAULT_TIME_LIMIT,
        highs=oos_highs,
        lows=oos_lows,
        atr=oos_atr
    )
    
    # 3. Compute Cumulative Equity Curve (Starting from 0)
    cumulative = np.cumsum(net_returns, axis=0)
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    print(f"\n{'Strategy':<30} | {'Sortino':<8} | {'OOS Ret %':<12} | {'Trades':<8}")
    print("-" * 75)
    
    for i, strat in enumerate(strategies):
        total_ret_pct = cumulative[-1, i]
        n_trades = trades_count[i]
        
        # Calculate Sortino for OOS
        sortino = calculate_sortino_ratio(net_returns[:, i], config.ANNUALIZATION_FACTOR)
        
        print(f"{strat.name:<30} | {sortino:<8.2f} | {total_ret_pct*100:<12.2f}% | {int(n_trades):<8}")
        
        label = f"{strat.name} (Sortino: {sortino:.1f} | Ret: {total_ret_pct*100:.1f}% | Tr: {int(n_trades)})"
        plt.plot(cumulative[:, i], label=label)
        
    # Plot Buy & Hold (Benchmarks)
    # Using log returns for benchmark on same slice
    returns_vec = backtester.raw_data[backtester.target_col].values[test_start:]
    asset_cum = np.cumsum(returns_vec)
    plt.plot(asset_cum, label="Market Bench (Log Ret)", color='black', alpha=0.3, linestyle='--')
    
    plt.title(f"Apex Trading OOS Performance (Test Set Only)\n${int(config.ACCOUNT_SIZE/1000)}k Account, {config.MIN_LOTS}-{config.MAX_LOTS} Lots", fontsize=16)
    plt.ylabel("Cumulative OOS Return (%)")
    plt.xlabel("Bar Index (OOS Period)")
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    os.makedirs(config.DIRS['PLOTS_DIR'], exist_ok=True)
    output_path = os.path.join(config.DIRS['PLOTS_DIR'], "strategy_performance_oos.png")
    plt.savefig(output_path)
    print(f"📸 Saved OOS Performance Chart to {output_path}")

def filter_top_strategies(df, strategies, top_n=20, chunk_size=1000):
    backtester = BacktestEngine(df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    all_results = []
    
    # Chunk Processing
    total_chunks = (len(strategies) + chunk_size - 1) // chunk_size
    for i in range(0, len(strategies), chunk_size):
        chunk = strategies[i:i+chunk_size]
        
        try:
            results_df = backtester.evaluate_population(chunk, set_type='test')
            
            for idx, row in results_df.iterrows():
                all_results.append({
                    'strategy': chunk[idx],
                    'sortino': row['sortino'],
                    'sharpe': row['sharpe'],
                    'total_return': row['total_return']
                })
        except Exception as e:
            print(f"     ⚠️ Error in batch: {e}")
        
        backtester.reset_jit_context()
        
    sorted_results = sorted(all_results, key=lambda x: x['sortino'], reverse=True)
    top_performers = [x['strategy'] for x in sorted_results[:top_n]]
    
    
    return top_performers

def print_mutex_breakdown(strategies, backtester):
    """
    Prints the Mutex portfolio contribution using the JIT simulator.
    """
    print(f"\nRunning Mutex Contribution Analysis (OOS) via JIT...")
    
    # Prepare Data
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

    # Generate Signals
    backtester.ensure_context(strategies)
    raw_sig = backtester.generate_signal_matrix(strategies)
    # Next Open execution: Shift signals
    shifted_sig = np.vstack([np.zeros((1, len(strategies)), dtype=raw_sig.dtype), raw_sig[:-1]])
    sig_matrix = shifted_sig[oos_start:]
    
    # Params
    horizons = np.array([s.horizon for s in strategies], dtype=np.int64)
    sl_mults = np.array([getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for s in strategies], dtype=np.float64)
    tp_mults = np.array([getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for s in strategies], dtype=np.float64)
    
    # JIT Simulation
    strat_rets, strat_trades, strat_wins, _ = _jit_simulate_mutex_custom(
        sig_matrix.astype(np.float64), 
        prices, highs, lows, atr, 
        hours, weekdays, 
        horizons, sl_mults, tp_mults, 
        config.STANDARD_LOT_SIZE, 
        config.SPREAD_BPS / 10000.0, 
        config.COST_BPS / 10000.0, 
        config.ACCOUNT_SIZE, 
        config.TRADING_END_HOUR, 
        config.STOP_LOSS_COOLDOWN_BARS,
        config.MIN_COMMISSION,
        config.SLIPPAGE_ATR_FACTOR
    )
    
    # Compute Stats
    strat_pnl = np.sum(strat_rets, axis=0) * config.ACCOUNT_SIZE
    total_pnl = np.sum(strat_pnl)
    total_trades = np.sum(strat_trades)
    
    # Print Table
    print("\n" + "="*80)
    print(f"MUTEX CONTRIBUTION ANALYSIS (Actual Execution)")
    print(f"{'{Strategy':<20} | {'Trades':<8} | {'PnL ($)':<12} | {'Win Rate':<10} | {'Avg ($)':<10}")
    print("-" * 80)
    
    for i, s in enumerate(strategies):
        pnl = strat_pnl[i]
        tr = strat_trades[i]
        wr = (strat_wins[i] / tr * 100) if tr > 0 else 0
        avg = (pnl / tr) if tr > 0 else 0
        
        # Highlight 'Losers' turned Winners
        prefix = "✅" if pnl > 0 else "❌"
        
        print(f"{prefix} {s.name:<18} | {int(tr):<8} | {pnl:>12.2f} | {wr:>9.1f}% | {avg:>9.2f}")
        
    print("-" * 80)
    print(f"TOTAL                | {int(total_trades):<8} | {total_pnl:>12.2f} |")
    print("="*80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize Strategy Performance')
    parser.add_argument('--file', type=str, help='Path to specific strategy JSON file')
    args = parser.parse_args()

    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    strategies = []
    is_mutex_run = False
    
    if args.file:
        print(f"Loading strategies from {args.file}...")
        loaded, _ = load_strategies(args.file)
        if loaded:
            strategies.extend(loaded)
        else:
            print(f"❌ Failed to load any strategies from {args.file}")
            sys.exit(1)

    elif os.path.exists(config.MUTEX_PORTFOLIO_FILE):
        strategies = load_strategies(config.MUTEX_PORTFOLIO_FILE)
        title = "Mutex Portfolio OOS Performance"
    else:
        print("Mutex Portfolio not found. Scanning for Apex strategies...")
        for h in config.PREDICTION_HORIZONS:
            loaded, _ = load_strategies('all_apex', horizon=h)
            strategies.extend(loaded)
            
    if not strategies:
        print("No strategies found.")
        sys.exit(0)
    
    # If Mutex, run the detailed breakdown
    if is_mutex_run:
        backtester = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
        print_mutex_breakdown(strategies, backtester)
    
    # Pass df directly
    top_strategies = filter_top_strategies(df, strategies, top_n=20)
    
    if top_strategies:
        plot_performance(df, top_strategies)
    else:
        print("No viable strategies found after filtering.")
