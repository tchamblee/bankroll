import numpy as np
from numba import jit

@jit(nopython=True, nogil=True, cache=True)
def _jit_simulate_mutex_custom(signals: np.ndarray, prices: np.ndarray, 
                               highs: np.ndarray, lows: np.ndarray, atr_vec: np.ndarray,
                               hours: np.ndarray, weekdays: np.ndarray,
                               horizons: np.ndarray, sl_mults: np.ndarray, tp_mults: np.ndarray,
                               lot_size: float, spread_pct: float, comm_pct: float, 
                               account_size: float, end_hour: int, cooldown_bars: int):
    """
    Simulates a portfolio of strategies running concurrently on one account.
    Each strategy manages its own position logic (Horizon, SL, TP).
    Returns breakdown of returns and stats per strategy.
    """
    n_bars, n_strats = signals.shape
    
    # State tracking per strategy
    # 0: Flat, 1: Long, -1: Short
    positions = np.zeros(n_strats, dtype=np.float64) 
    entry_prices = np.zeros(n_strats, dtype=np.float64)
    entry_indices = np.zeros(n_strats, dtype=np.int64)
    entry_atrs = np.zeros(n_strats, dtype=np.float64)
    
    # Stats tracking
    strat_returns = np.zeros((n_bars, n_strats), dtype=np.float64)
    strat_trades = np.zeros(n_strats, dtype=np.int64)
    strat_wins = np.zeros(n_strats, dtype=np.int64)
    
    # Cooldown tracking
    cooldowns = np.zeros(n_strats, dtype=np.int64)
    
    # Loop over bars
    for i in range(1, n_bars):
        
        # Check Force Close (EOD or Weekend)
        force_close = (hours[i] >= end_hour) or (weekdays[i] >= 5)
        
        # MUTEX: Determine who currently holds the lock (if anyone)
        # We look at 'positions' state at start of bar (carried from i-1)
        active_strat = -1
        for s in range(n_strats):
            if positions[s] != 0:
                active_strat = s
                break
        
        # Loop over strategies
        for s in range(n_strats):
            # MUTEX LOCK: If someone else holds the position, skip this strategy entirely
            if active_strat != -1 and active_strat != s:
                continue
                
            # Manage Cooldown
            if cooldowns[s] > 0:
                cooldowns[s] -= 1
                
            curr_pos = positions[s]
            
            # --- 1. EXIT CHECKS (If in position) ---
            exit_signal = False
            is_sl_hit = False
            
            # Check Barriers from PREVIOUS bar (i-1) to avoid look-ahead
            # We entered at or before Open[i-1]. We need to see if i-1 killed us.
            if curr_pos != 0 and i > 0:
                # Check Time Limit (Current time i vs Entry)
                if (i - entry_indices[s]) >= horizons[s]:
                    exit_signal = True
                
                # Check SL/TP against PREVIOUS bar volatility
                if not exit_signal:
                    e_price = entry_prices[s]
                    sl_dist = entry_atrs[s] * sl_mults[s]
                    tp_dist = entry_atrs[s] * tp_mults[s]
                    
                    # Use i-1 for High/Low checks
                    if curr_pos > 0:
                        if lows[i-1] <= (e_price - sl_dist):
                            exit_signal = True
                            is_sl_hit = True
                        elif tp_mults[s] > 0 and highs[i-1] >= (e_price + tp_dist):
                            exit_signal = True
                    else:
                        if highs[i-1] >= (e_price + sl_dist):
                            exit_signal = True
                            is_sl_hit = True
                        elif tp_mults[s] > 0 and lows[i-1] <= (e_price - tp_dist):
                            exit_signal = True

                if force_close:
                    exit_signal = True
                    # Don't trigger cooldown on EOD
                    is_sl_hit = False 
            
            # --- 2. EXECUTION LOGIC ---
            target_pos = curr_pos
            
            if exit_signal:
                target_pos = 0.0
                if is_sl_hit:
                    cooldowns[s] = cooldown_bars
            elif curr_pos == 0:
                # Check Entry (Signal at i is for Open[i])
                # MUTEX: Only allowed if active_strat is -1 (Free)
                if active_strat == -1:
                    sig = signals[i, s]
                    if sig != 0 and cooldowns[s] == 0 and not force_close:
                        target_pos = float(sig)
                        # MUTEX: We claim the lock immediately for this bar loop
                        active_strat = s
            
            # --- 3. STATE UPDATE & COST ---
            if target_pos != curr_pos:
                # Determine Execution Price
                exec_price = prices[i] # Default to Open[i]
                
                if exit_signal and curr_pos != 0:
                    # If barrier hit in i-1, we assume we exited AT the barrier (plus slippage later)
                    if is_sl_hit or (target_pos == 0 and not force_close and (i - entry_indices[s]) < horizons[s]):
                        e_price = entry_prices[s]
                        sl_dist = entry_atrs[s] * sl_mults[s]
                        tp_dist = entry_atrs[s] * tp_mults[s]
                        
                        if is_sl_hit:
                             if curr_pos > 0: exec_price = e_price - sl_dist
                             else: exec_price = e_price + sl_dist
                        else:
                            # TP Hit 
                            if curr_pos > 0: exec_price = e_price + tp_dist
                            else: exec_price = e_price - tp_dist
                            
                change = abs(target_pos - curr_pos)
                
                # Costs
                # Spread
                cost_spread = change * lot_size * exec_price * (0.5 * spread_pct)
                # Comm
                raw_comm = change * lot_size * exec_price * comm_pct
                comm = max(2.0, raw_comm) if change > 0.5 else 0.0 
                # Slippage (10% ATR)
                slip = 0.1 * atr_vec[i] * lot_size * change
                
                total_cost = cost_spread + comm + slip
                
                # PnL from Trade Close
                if curr_pos != 0 and target_pos == 0:
                    trade_pnl = (exec_price - entry_prices[s]) * curr_pos * lot_size
                    net_pnl = trade_pnl - total_cost
                    strat_returns[i, s] = net_pnl / account_size
                    
                    if net_pnl > 0:
                        strat_wins[s] += 1
                        
                elif curr_pos == 0 and target_pos != 0:
                    # Entry
                    strat_returns[i, s] = -total_cost / account_size
                    entry_prices[s] = exec_price
                    entry_indices[s] = i
                    entry_atrs[s] = atr_vec[i]
                    strat_trades[s] += 1
                
                positions[s] = target_pos
        
    return strat_returns, strat_trades, strat_wins, positions.astype(np.int64)
