import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import shutil
import config
from genome import Strategy
from backtest import BacktestEngine
from trade_simulator import TradeSimulator

def plot_trade(trade, prices, times, output_dir, trade_id, horizon):
    # Adapter for Trade object
    start_idx = trade.entry_idx
    end_idx = trade.exit_idx if trade.exit_idx is not None else len(prices) - 1
    
    trade_type = "Long" if trade.direction > 0 else "Short"
    
    # Context window
    context = 20 # bars
    start = max(0, start_idx - context)
    end = min(len(prices), end_idx + context)
    
    # Handle Series vs Array
    if hasattr(times, 'iloc'):
        section_times = times.iloc[start:end]
    else:
        section_times = times[start:end]
        
    section_prices = prices[start:end]
    
    plt.figure(figsize=(12, 6))
    
    # 1. Plot Price Line
    plt.plot(section_times, section_prices, color='black', alpha=0.6, linewidth=1.5, label='EUR/USD')
    
    # 2. Shade Active Region
    # Create mask for active period
    # Need to match index if times is Series
    if hasattr(section_times, 'index'):
        active_mask = (section_times.index >= start_idx) & (section_times.index <= end_idx)
    else:
        # Assuming times is list-like aligned with prices
        # Create a boolean mask manually
        active_mask = np.zeros(len(section_times), dtype=bool)
        # local indices
        s_loc = max(0, start_idx - start)
        e_loc = min(len(section_times), end_idx - start + 1)
        active_mask[s_loc:e_loc] = True
    
    # Calculate dynamic Y-limits
    p_min = section_prices.min()
    p_max = section_prices.max()
    p_range = p_max - p_min
    if p_range == 0: p_range = p_min * 0.001
    
    y_bottom = p_min - (p_range * 0.15) # 15% padding
    y_top = p_max + (p_range * 0.15)
    
    plt.ylim(y_bottom, y_top)
    
    if trade_type == 'Long':
        plt.fill_between(section_times, section_prices, y_bottom, where=active_mask, color='green', alpha=0.1)
    else:
        plt.fill_between(section_times, section_prices, y_top, where=active_mask, color='red', alpha=0.1)

    # 3. Mark Events
    for event in trade.events:
        idx = event['idx']
        if idx < start or idx >= end: continue
        
        local_idx = idx - start
        if local_idx < 0 or local_idx >= len(section_times): continue
        
        if hasattr(section_times, 'iloc'):
            t = section_times.iloc[local_idx]
        else:
            t = section_times[local_idx]
            
        p = section_prices[local_idx]
        
        action = event['action']
        size = event.get('size', 0)
        
        if "ENTRY" in action:
            marker = '^' if trade_type == 'Long' else 'v'
            color = 'green' if trade_type == 'Long' else 'red'
            plt.scatter(t, p, color=color, marker=marker, s=100, zorder=5)
            plt.annotate(f"{action}\n{size}L", (t, p), xytext=(0, 20 if trade_type=='Long' else -20), 
                         textcoords='offset points', ha='center', color=color, fontweight='bold')
            
        elif "EXIT" in action or "SL" in action or "TP" in action or "TIME" in action:
            plt.scatter(t, p, color='blue', marker='x', s=100, zorder=5)
            plt.annotate(f"{action}", (t, p), xytext=(0, 10), textcoords='offset points', ha='center', color='blue')
            
    # 4. Title & PnL
    pnl_color = 'green' if trade.pnl > 0 else 'red'
    plt.title(f"Horizon {horizon} | Trade #{trade_id} | Type: {trade_type} | PnL: ${trade.pnl:.2f}", fontsize=14, color='black')
    
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    # Save
    fname = f"trade_{trade_id:04d}_{trade_type}.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

def main():
    print("======================================================")
    print("🌍 GENERATING TRADE ATLAS (Mutex Portfolio) 🌍")
    print("======================================================\n")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return

    if args.mutex:
        portfolio_path = config.MUTEX_PORTFOLIO_FILE
        if os.path.exists(portfolio_path):
            print(f"Generating Atlas for Mutex Portfolio: {portfolio_path}")
    
    with open(portfolio_path, 'r') as f:
         portfolio_data = json.load(f)
         
    strategies = [Strategy.from_dict(d) for d in portfolio_data]
    # Re-attach horizon info if missing in Strategy object (it's not a standard field)
    for s, d in zip(strategies, portfolio_data):
        s.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)

    # Load Full Data
    print("Loading Market Data...")
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(df)
    
    if 'time_start' in df.columns:
        times = df['time_start']
    else:
        times = pd.Series(df.index)

    # Initialize Simulator
    simulator = TradeSimulator(
        prices=backtester.close_vec.flatten(),
        times=times,
        spread_bps=config.SPREAD_BPS, 
        cost_bps=config.COST_BPS
    )

    # Output Root
    atlas_root = os.path.join(config.DIRS['OUTPUT_DIR'], "trade_atlas")
    if not os.path.exists(atlas_root): os.makedirs(atlas_root)
    
    # Clean previous atlas
    # shutil.rmtree(atlas_root)
    # os.makedirs(atlas_root)

    print(f"Generating Atlas for {len(strategies)} Strategies in Mutex Portfolio...")
    print(f"NOTE: Only plotting trades in OOS period (Start Index: {backtester.val_idx})")

    for i, strat in enumerate(strategies):
        h = getattr(strat, 'horizon', 120)
        print(f"\n[{i+1}/{len(strategies)}] Analyzing {strat.name} (H{h})...")
        
        # 2. Generate Signals
        signals = backtester.generate_signal_matrix([strat], horizon=h).flatten()
        
        # 3. Simulate Trades
        sl_pct = getattr(strat, 'stop_loss_pct', config.DEFAULT_STOP_LOSS)
        tp_pct = getattr(strat, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT)
        
        trades, _ = simulator.simulate(
            signals, 
            stop_loss_pct=sl_pct, 
            take_profit_pct=tp_pct, 
            time_limit_bars=h,
            highs=backtester.high_vec.flatten(),
            lows=backtester.low_vec.flatten(),
            atr=backtester.atr_vec.flatten()
        )
        
        # FILTER: OOS Only
        oos_start = backtester.val_idx
        valid_trades = [t for t in trades if t.entry_idx >= oos_start]
        
        print(f"  Found {len(trades)} total trades. {len(valid_trades)} in OOS period.")
        
        if not valid_trades:
            print("  No OOS trades to plot.")
            continue

        # 4. Generate Plots
        strat_dir = os.path.join(atlas_root, f"{strat.name}_H{h}")
        if not os.path.exists(strat_dir): os.makedirs(strat_dir)
        else:
            shutil.rmtree(strat_dir)
            os.makedirs(strat_dir)
        
        # Limit to first 100 trades to prevent excessive rendering
        plot_limit = 100
        for j, trade in enumerate(valid_trades[:plot_limit]):
            plot_trade(trade, backtester.close_vec.flatten(), times, strat_dir, j+1, h)
            if (j+1) % 50 == 0:
                print(f"    ... Plotted {j+1}/{min(len(valid_trades), plot_limit)} trades")
                
        print(f"  ✅ Saved {min(len(valid_trades), plot_limit)} plots to {strat_dir}")

    # ==========================================================================
    #  MUTEX COMBINED ATLAS
    # ==========================================================================
    print("\n" + "="*60)
    print("🧩 GENERATING COMBINED MUTEX PORTFOLIO ATLAS")
    print("="*60)
    
    # 1. Generate Signal Matrix for all
    print("  Generating Combined Signal Matrix...")
    # Shift signals for Next-Open Execution logic, matching run_mutex_strategy behavior
    raw_sig_matrix = backtester.generate_signal_matrix(strategies)
    sig_matrix = np.vstack([np.zeros((1, len(strategies)), dtype=raw_sig_matrix.dtype), raw_sig_matrix[:-1]])
    
    # 2. Extract Params
    horizons = [getattr(s, 'horizon', 120) for s in strategies]
    sl_mults = [getattr(s, 'stop_loss_pct', config.DEFAULT_STOP_LOSS) for s in strategies]
    tp_mults = [getattr(s, 'take_profit_pct', config.DEFAULT_TAKE_PROFIT) for s in strategies]
    
    # 3. Simulate Mutex Execution
    print("  Simulating Mutex Priority Logic...")
    from trade_simulator import Trade
    
    prices = backtester.open_vec.flatten() # Executing at Open
    highs = backtester.high_vec.flatten()
    lows = backtester.low_vec.flatten()
    atr = backtester.atr_vec.flatten()
    
    if hasattr(times, 'dt'):
        hours = times.dt.hour.values
        weekdays = times.dt.dayofweek.values
    else:
        dt = pd.to_datetime(times)
        hours = dt.hour.values
        weekdays = dt.dayofweek.values
        
    n_bars = len(prices)
    n_strats = len(strategies)
    
    mutex_trades = []
    
    # State
    position = 0.0
    active_strat_idx = -1
    entry_idx = 0
    entry_price = 0.0
    entry_atr = 0.0
    
    current_horizon = 0
    current_sl_mult = 0.0
    current_tp_mult = 0.0
    
    # Cooldown State
    strat_cooldowns = np.zeros(n_strats, dtype=int)
    
    for i in range(n_bars):
        # Decrement Cooldowns
        for s in range(n_strats):
            if strat_cooldowns[s] > 0:
                strat_cooldowns[s] -= 1
                
        # 1. Check Exits
        exit_trade = False
        exit_reason = ""
        barrier_price = 0.0
        
        # Force Close
        if hours[i] >= config.TRADING_END_HOUR or weekdays[i] >= 5:
            if position != 0:
                exit_trade = True
                exit_reason = "EOD"
                
        elif position != 0:
            # Time Limit
            if (i - entry_idx) >= current_horizon:
                exit_trade = True
                exit_reason = "TIME"
            
            # SL/TP
            if not exit_trade:
                h_prev = highs[i-1] if i > 0 else prices[i]
                l_prev = lows[i-1] if i > 0 else prices[i]
                
                sl_dist = entry_atr * current_sl_mult
                tp_dist = entry_atr * current_tp_mult
                
                if position > 0:
                    if current_sl_mult > 0 and l_prev <= (entry_price - sl_dist):
                        exit_trade = True
                        exit_reason = "SL"
                        barrier_price = entry_price - sl_dist
                    elif current_tp_mult > 0 and h_prev >= (entry_price + tp_dist):
                        exit_trade = True
                        exit_reason = "TP"
                        barrier_price = entry_price + tp_dist
                else:
                    if current_sl_mult > 0 and h_prev >= (entry_price + sl_dist):
                        exit_trade = True
                        exit_reason = "SL"
                        barrier_price = entry_price + sl_dist
                    elif current_tp_mult > 0 and l_prev <= (entry_price - tp_dist):
                        exit_trade = True
                        exit_reason = "TP"
                        barrier_price = entry_price - tp_dist

            # Reversal
            if not exit_trade and active_strat_idx >= 0:
                sig = sig_matrix[i, active_strat_idx]
                if sig != 0 and np.sign(sig) != np.sign(position):
                    # Close current to flip
                    # We treat flip as Exit -> New Entry
                    # Close first
                    exit_trade = True
                    exit_reason = "FLIP"
                    # Note: We will re-enter in the Entry block below in same step if needed, 
                    # but simplest is to close now and let next bar pick up or handle flip logic.
                    # Actually, standard logic is atomic flip. 
                    # For Atlas, let's just close.
                    
        if exit_trade:
            # Record Trade
            exit_p = barrier_price if barrier_price != 0 else prices[i]
            pnl = (exit_p - entry_price) * abs(position) * np.sign(position) * config.STANDARD_LOT_SIZE
            
            t = Trade(
                entry_idx=entry_idx,
                entry_price=entry_price,
                direction=int(position),
                size=abs(position),
                exit_idx=i,
                exit_price=exit_p,
                exit_reason=exit_reason,
                pnl=pnl,
                events=[
                    {'idx': entry_idx, 'action': f"ENTRY (Strat {active_strat_idx})", 'size': abs(position)},
                    {'idx': i, 'action': f"EXIT ({exit_reason})", 'price': exit_p}
                ]
            )
            mutex_trades.append(t)
            
            # Apply Cooldown if SL
            if exit_reason == "SL" and active_strat_idx >= 0:
                strat_cooldowns[active_strat_idx] = config.STOP_LOSS_COOLDOWN_BARS
            
            position = 0.0
            active_strat_idx = -1
            
        # 2. Check Entries
        if position == 0.0 and not (hours[i] >= config.TRADING_END_HOUR or weekdays[i] >= 5):
            for s_idx in range(n_strats):
                # Check Cooldown
                if strat_cooldowns[s_idx] > 0:
                    continue
                
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
                    break
                    
    # Filter OOS
    oos_start = backtester.val_idx
    valid_mutex_trades = [t for t in mutex_trades if t.entry_idx >= oos_start]
    
    print(f"  Generated {len(mutex_trades)} Mutex trades. {len(valid_mutex_trades)} in OOS period.")
    
    if valid_mutex_trades:
        mutex_dir = os.path.join(atlas_root, "MUTEX_PORTFOLIO_COMBINED")
        if os.path.exists(mutex_dir): shutil.rmtree(mutex_dir)
        os.makedirs(mutex_dir)
        
        plot_limit = 200
        for j, trade in enumerate(valid_mutex_trades[:plot_limit]):
            # Use 'Mutex' as horizon label for simplicity, or the specific trade horizon
            # trade doesn't store horizon, but we can infer or just pass 'Mixed'
            plot_trade(trade, backtester.close_vec.flatten(), times, mutex_dir, j+1, "MUTEX")
            if (j+1) % 50 == 0:
                print(f"    ... Plotted {j+1}/{min(len(valid_mutex_trades), plot_limit)} trades")
                
        print(f"  ✅ Saved {min(len(valid_mutex_trades), plot_limit)} Combined plots to {mutex_dir}")

if __name__ == "__main__":
    main()
