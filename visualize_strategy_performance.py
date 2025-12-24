import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import re
from feature_engine import FeatureEngine
from genome import Strategy, RelationalGene, DeltaGene, ZScoreGene, TimeGene, ConsecutiveGene
from backtest import BacktestEngine
import config

def parse_gene_string(gene_str):
    match = re.match(r"(.+)\s+([<>=!]+)\s+(.+)", gene_str)
    if not match: return None
    left, op, right = match.groups()
    left, op, right = left.strip(), op.strip(), right.strip()
    
    if left.startswith("Consecutive("):
        direction = left.split("(")[1].split(")")[0]
        return ConsecutiveGene(direction, op, int(re.sub(r"[^0-9]", "", right)))
    if left.startswith("Time("):
        mode = left.split("(")[1].split(")")[0]
        return TimeGene(mode, op, int(re.sub(r"[^0-9]", "", right)))
    if left.startswith("Z("):
        try:
            inner = left.split("Z(")[1].split(")")[0]
            feature, window = [x.strip() for x in inner.split(",")]
            if right.endswith('σ'): right = right[:-1]
            return ZScoreGene(feature, op, float(right), int(window))
        except: return None
    if left.startswith("Delta("):
        try:
            inner = left.split("Delta(")[1].split(")")[0]
            feature, lookback = [x.strip() for x in inner.split(",")]
            return DeltaGene(feature, op, float(right), int(lookback))
        except: return None
    try:
        float(right) # Check if it is a number
        return None # StaticGene is deprecated
    except ValueError:
        return RelationalGene(left, op, right)

def reconstruct_strategy(strat_dict):
    # 1. New JSON Format (Native Serialization)
    if 'long_genes' in strat_dict:
        try:
            return Strategy.from_dict(strat_dict)
        except Exception as e:
            print(f"Error hydrating strategy {strat_dict.get('name')}: {e}")
            return None

    # 2. Legacy String Format (Regex Parsing)
    if 'logic' not in strat_dict: return None
    
    logic = strat_dict['logic']
    name = strat_dict['name']
    try:
        match = re.search(r"(])[(](.*?)[)]", logic)
        logic_type = match.group(1) if match else "AND"
        min_con = int(logic_type.split("(")[1].split(")")[0]) if logic_type.startswith("VOTE(") else None
        sep = " + " if logic_type == "VOTE" else " AND "
        parts = logic.split(" | ")
        if len(parts) != 2: return None
        long_block, short_block = parts[0], parts[1]
        
        def extract_content(block, tag):
            if f"{tag}:(" in block:
                start = block.find(f"{tag}:(") + len(tag) + 2
                return block[start:block.rfind(")")]
            return "None"

        long_part = extract_content(long_block, "LONG")
        short_part = extract_content(short_block, "SHORT")
        
        l_genes = [parse_gene_string(g) for g in long_part.split(sep) if g != "None"]
        s_genes = [parse_gene_string(g) for g in short_part.split(sep) if g != "None"]
        return Strategy(name=name, long_genes=[g for g in l_genes if g], short_genes=[g for g in s_genes if g], min_concordance=min_con)
    except Exception as e:
        print(f"Error parsing {name}: {e}")
        return None

def plot_performance(engine, strategies):
    backtester = BacktestEngine(engine.bars, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
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
        avg = np.mean(net_returns[:, i])
        downside = np.std(np.minimum(net_returns[:, i], 0)) + 1e-9
        sortino = (avg / downside) * np.sqrt(config.ANNUALIZATION_FACTOR)
        
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

class MockEngine:
    def __init__(self, df):
        self.bars = df

def filter_top_strategies(engine, strategies, top_n=20, chunk_size=1000):
    backtester = BacktestEngine(engine.bars, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
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

def simulate_mutex_breakdown(strategies, backtester):
    """
    Simulates the Mutex portfolio and tracks the contribution of each strategy
    (PnL and Trades) considering the priority locking.
    """
    print(f"\nRunning Mutex Contribution Analysis (OOS)...")
    
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
    
    n_bars, n_strats = sig_matrix.shape
    
    # Tracking
    strat_pnl = np.zeros(n_strats)
    strat_trades = np.zeros(n_strats)
    strat_wins = np.zeros(n_strats)
    
    # Simulation Loop
    position = 0.0
    entry_price = 0.0
    entry_idx = 0
    entry_atr = 0.0
    active_strat_idx = -1
    
    current_horizon = 0
    current_sl_mult = 0.0
    current_tp_mult = 0.0
    
    prev_pos = 0.0
    strat_cooldowns = np.zeros(n_strats, dtype=np.int64)
    
    end_hour = config.TRADING_END_HOUR
    cooldown_bars = config.STOP_LOSS_COOLDOWN_BARS
    lot_size = config.STANDARD_LOT_SIZE
    spread = config.SPREAD_BPS / 10000.0
    comm = config.COST_BPS / 10000.0
    
    for i in range(n_bars):
        # Decrement Cooldowns
        for s in range(n_strats):
            if strat_cooldowns[s] > 0: strat_cooldowns[s] -= 1
                
        # 1. Check Barriers / Exits
        exit_trade = False
        barrier_price = 0.0
        exit_reason = 0 
        
        if hours[i] >= end_hour or weekdays[i] >= 5:
            exit_trade = True
            exit_reason = 1
        elif position != 0:
            if (i - entry_idx) >= current_horizon:
                exit_trade = True
                exit_reason = 1
            
            if not exit_trade:
                h_prev = highs[i-1] if i > 0 else prices[i]
                l_prev = lows[i-1] if i > 0 else prices[i]
                sl_dist = entry_atr * current_sl_mult
                tp_dist = entry_atr * current_tp_mult
                
                if position > 0:
                    if current_sl_mult > 0 and l_prev <= (entry_price - sl_dist):
                        exit_trade = True
                        barrier_price = entry_price - sl_dist
                        exit_reason = 2 
                    elif current_tp_mult > 0 and h_prev >= (entry_price + tp_dist):
                        exit_trade = True
                        barrier_price = entry_price + tp_dist
                        exit_reason = 3
                else:
                    if current_sl_mult > 0 and h_prev >= (entry_price + sl_dist):
                        exit_trade = True
                        barrier_price = entry_price + sl_dist
                        exit_reason = 2
                    elif current_tp_mult > 0 and l_prev <= (entry_price - tp_dist):
                        exit_trade = True
                        barrier_price = entry_price - tp_dist
                        exit_reason = 3
            
            # Reversal
            if not exit_trade and active_strat_idx >= 0:
                sig = sig_matrix[i, active_strat_idx]
                if sig != 0 and np.sign(sig) != np.sign(position):
                     # Close current (Simplified for PnL attribution)
                     curr_price = prices[i]
                     pos_change = abs(0.0 - position)
                     cost = pos_change * lot_size * curr_price * (0.5 * spread + comm)
                     gross_pnl = position * lot_size * (curr_price - entry_price)
                     
                     pnl = gross_pnl - cost
                     strat_pnl[active_strat_idx] += pnl
                     if pnl > 0: strat_wins[active_strat_idx] += 1
                     
                     # New Entry
                     position = float(sig)
                     entry_price = prices[i]
                     entry_idx = i
                     entry_atr = atr[i]
                     strat_trades[active_strat_idx] += 1
                     continue 
                        
        if exit_trade:
            curr_price = prices[i]
            if barrier_price != 0.0: curr_price = barrier_price
            
            gross = position * lot_size * (curr_price - entry_price)
            # Roundtrip costs
            cost_entry = abs(position) * lot_size * entry_price * (0.5 * spread + comm)
            cost_exit = abs(position) * lot_size * curr_price * (0.5 * spread + comm)
            net_pnl = gross - cost_entry - cost_exit
            
            if active_strat_idx >= 0:
                strat_pnl[active_strat_idx] += net_pnl
                if net_pnl > 0: strat_wins[active_strat_idx] += 1
                if exit_reason == 2: strat_cooldowns[active_strat_idx] = cooldown_bars
            
            position = 0.0
            active_strat_idx = -1
            
        # 2. Check Entries
        if position == 0.0 and not (hours[i] >= end_hour or weekdays[i] >= 5):
            for s_idx in range(n_strats):
                if strat_cooldowns[s_idx] > 0: continue
                sig = sig_matrix[i, s_idx]
                if sig != 0:
                    position = float(sig)
                    entry_price = prices[i]
                    entry_idx = i
                    entry_atr = atr[i]
                    active_strat_idx = s_idx
                    current_horizon = horizons[s_idx]
                    current_sl_mult = sl_mults[s_idx]
                    current_tp_mult = tp_mults[s_idx]
                    
                    strat_trades[s_idx] += 1
                    break
                    
    # Print Table
    print("\n" + "="*80)
    print(f"MUTEX CONTRIBUTION ANALYSIS (Actual Execution)")
    print(f"{'Strategy':<20} | {'Trades':<8} | {'PnL ($)':<12} | {'Win Rate':<10} | {'Avg ($)':<10}")
    print("-" * 80)
    
    total_pnl = 0
    total_trades = 0
    
    for i, s in enumerate(strategies):
        pnl = strat_pnl[i]
        tr = strat_trades[i]
        wr = (strat_wins[i] / tr * 100) if tr > 0 else 0
        avg = (pnl / tr) if tr > 0 else 0
        
        # Highlight 'Losers' turned Winners
        prefix = "✅" if pnl > 0 else "❌"
        
        print(f"{prefix} {s.name:<18} | {int(tr):<8} | {pnl:>12.2f} | {wr:>9.1f}% | {avg:>9.2f}")
        
        total_pnl += pnl
        total_trades += tr
        
    print("-" * 80)
    print(f"TOTAL                | {int(total_trades):<8} | {total_pnl:>12.2f} |")
    print("="*80)

if __name__ == "__main__":
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    engine = MockEngine(df)
    
    import glob
    mutex_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
    
    all_strategies_data = []
    is_mutex_run = False
    
    if os.path.exists(mutex_path):
        print(f"Loading Mutex Portfolio from {mutex_path}...")
        try:
            with open(mutex_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_strategies_data.extend(data)
                    is_mutex_run = True # Flag to run contribution analysis
                elif isinstance(data, dict):
                    all_strategies_data.append(data)
        except Exception as e:
            print(f"Error loading mutex portfolio: {e}")
    else:
        print("Mutex Portfolio not found. Scanning for Apex strategies...")
        apex_files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], "apex_strategies_*.json"))
        portfolio_files = glob.glob(os.path.join(config.DIRS['STRATEGIES_DIR'], "apex_portfolio_*.json"))
        
        for fpath in apex_files + portfolio_files:
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_strategies_data.extend(data)
                    elif isinstance(data, dict):
                        all_strategies_data.append(data)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
            
    strategies = []
    seen = set()
    for d in all_strategies_data:
        s = reconstruct_strategy(d)
        if s:
            # Hydrate params if available
            s.horizon = d.get('horizon', 120)
            s.stop_loss_pct = d.get('stop_loss_pct', config.DEFAULT_STOP_LOSS)
            s.take_profit_pct = d.get('take_profit_pct', config.DEFAULT_TAKE_PROFIT)
            
            if str(s) not in seen:
                strategies.append(s)
                seen.add(str(s))
            
    if not strategies:
        print("No strategies found.")
        sys.exit(0)
    
    # If Mutex, run the detailed breakdown
    if is_mutex_run:
        backtester = BacktestEngine(df, annualization_factor=config.ANNUALIZATION_FACTOR)
        simulate_mutex_breakdown(strategies, backtester)
    
    top_strategies = filter_top_strategies(engine, strategies, top_n=20)
    
    if top_strategies:
        plot_performance(engine, top_strategies)
    else:
        print("No viable strategies found after filtering.")