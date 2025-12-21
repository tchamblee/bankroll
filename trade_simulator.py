import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from numba import jit

@jit(nopython=True)
def _jit_simulate_fast(signals: np.ndarray, prices: np.ndarray, lot_size: float, spread_pct: float, comm_pct: float, account_size: float, stop_loss_pct: float, time_limit_bars: int) -> Tuple[np.ndarray, int]:
    """
    JIT-Compiled Event-Driven Simulation.
    """
    n = len(signals)
    
    # 1. Identify Potential Events (Signal Changes)
    # np.diff is supported in numba
    sig_diff = np.diff(signals) # returns length n-1
    # We need to detect change from 0 to 1st element too? 
    # Logic in python was np.diff(signals, prepend=0). Prepend not supported in all numba versions efficiently.
    # Manual approach for diff including first element
    
    # Actually, let's just loop. Numba is fast at looping.
    # We don't need complex vectorization tricks inside Numba.
    
    realized_signals = signals.copy().astype(np.float64) # Use float for 0.0 placeholders if needed, though signals are int usually
    
    # Tracking trade count
    trade_count = 0
    
    # We will reconstruct the realized path.
    # To do this efficiently, we jump from event to event? 
    # Or just iterate? iterating 20k bars in C++ speed is instant.
    # Let's try a single pass state machine loop. It is O(N) but N is small (20k).
    
    position = 0.0 # float
    entry_price = 0.0
    entry_idx = 0
    
    # Iterate all bars
    for i in range(n):
        # Check Exits on current position BEFORE processing new signal
        if position != 0.0:
            # Check Time Exit
            if time_limit_bars > 0:
                if (i - entry_idx) >= time_limit_bars:
                    position = 0.0 # Close
                    realized_signals[i] = 0.0
            
            # Check Stop Loss (if position still open)
            if position != 0.0 and stop_loss_pct > 0.0:
                current_price = prices[i]
                if position > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                    if pnl_pct <= -stop_loss_pct:
                        position = 0.0
                        realized_signals[i] = 0.0
                else: # Short
                    pnl_pct = (current_price - entry_price) / entry_price * -1.0
                    if pnl_pct <= -stop_loss_pct:
                        position = 0.0
                        realized_signals[i] = 0.0
        
        # Check New Signal (if not forced closed by barrier in this same bar - assuming barrier takes precedence or signal?)
        # Convention: Signal is the DESIRED position at end of bar.
        # But if we hit SL during the bar, we are flat. 
        # If the strategy wants to re-enter, it will signal again next bar.
        # But `signals` array is pre-computed. 
        # If we close due to SL, we must overwrite the signal to 0 for this bar and subsequent bars until signal changes?
        # Actually, if we hit SL, we are OUT. 
        # If the underlying signal is still holding the position, we must ignore it until it flips?
        # Or does the strategy just re-enter? 
        # Standard: Once stopped out, wait for a change in signal (entry condition) to re-enter? 
        # Or simple: just force 0.
        
        # Let's stick to the previous logic: "realized_signals[exit_idx:end] = 0"
        # This implies we stay out until the signal array naturally changes (event based).
        
        # To replicate "wait for next signal change", we need to track if we are in "lockout" mode.
        # BUT the Python version simply zeroed out the realized_signals array forward until the next event index.
        # Let's do that.
        
        target_pos = signals[i]
        
        # If we just closed due to SL/Time (position became 0 inside the loop above), 
        # we effectively override the target_pos for *this* bar to 0.
        # And we need to ensure we don't re-enter until target_pos changes from what it is now?
        # Actually, strict replication of `simulate_fast` logic:
        # It finds intervals [start, end) of constant signal.
        # It checks if SL/Time is hit within that interval.
        # If hit at exit_idx, it sets realized_signals[exit_idx:end] = 0.
        
        # Let's implement that exact logic in Numba.
        pass # Placeholder for the loop structure

    # Re-implementation of Event-Interval Logic
    # 1. Find Events
    # Numba doesn't like dynamic lists.
    # We can iterate manually.
    
    current_sig_val = signals[0]
    start_idx = 0
    
    for i in range(1, n + 1): # Go up to n to handle last segment
        # Check if segment ended
        if i == n or signals[i] != current_sig_val:
            end_idx = i
            
            # Process Segment [start_idx, end_idx)
            pos = float(current_sig_val)
            
            if pos != 0.0:
                seg_len = end_idx - start_idx
                # Check barriers
                
                # We need entry price. 
                # Entry price is prices[start_idx].
                entry_pr = prices[start_idx]
                
                # 1. Time Limit
                limit_idx = start_idx + time_limit_bars if time_limit_bars > 0 else 999999999
                
                # 2. Stop Loss
                sl_idx = 999999999
                if stop_loss_pct > 0.0:
                    if pos > 0:
                        thresh = entry_pr * (1.0 - stop_loss_pct)
                        for k in range(start_idx, end_idx):
                            if prices[k] < thresh:
                                sl_idx = k
                                break
                    else:
                        thresh = entry_pr * (1.0 + stop_loss_pct)
                        for k in range(start_idx, end_idx):
                            if prices[k] > thresh:
                                sl_idx = k
                                break
                
                # Determine Exit
                first_exit = min(limit_idx, sl_idx)
                
                if first_exit < end_idx:
                    # We exited early. Zero out the rest of this segment.
                    # Numba supports slice assignment
                    realized_signals[first_exit:end_idx] = 0.0
            
            # Prepare next segment
            if i < n:
                current_sig_val = signals[i]
                start_idx = i
                
    # End Loop
    
    # 3. Calculate PnL
    # Now we have the final realized_signals curve.
    # Calculate PnL vector.
    
    net_returns = np.zeros(n, dtype=np.float64)
    
    prev_pos = 0.0 # start flat
    # We need to handle index 0 specifically or use prev_pos variable
    
    # Single pass PnL
    for i in range(n):
        curr_pos = realized_signals[i]
        
        # 1. MTM PnL (Holding period return)
        # return on CURRENT bar is based on position held FROM prev bar TO current bar?
        # Typically: PnL[i] = pos[i-1] * (price[i] - price[i-1])
        # Numba loop i starts at 0.
        
        if i > 0:
            price_change = prices[i] - prices[i-1]
            gross_pnl = prev_pos * lot_size * price_change
            
            # Cost
            # If position changed FROM i-1 TO i
            pos_change = abs(curr_pos - prev_pos)
            cost = pos_change * lot_size * prices[i] * (spread_pct + comm_pct)
            
            net_pnl = gross_pnl - cost
            net_returns[i] = net_pnl / account_size
            
            if pos_change > 0:
                trade_count += 1
        else:
            # First bar. Assume we start flat from "before".
            # If signal[0] is not 0, we entered at Open? Or Close of 0?
            # Backtester assumes signals generated at Close, executed at Close (or Next Open).
            # Here logic assumes execution at Close of bar i?
            # "pos_lag * price_diff" in original code.
            # pos_lag[i] is realized_signals[i-1].
            # So PnL at i is driven by position established at i-1.
            
            # At i=0, pos_lag is 0. So no Gross PnL.
            # But if realized_signals[0] != 0, we have a position change (0 -> 1).
            # So we pay cost.
            
            pos_change = abs(curr_pos - 0.0)
            cost = pos_change * lot_size * prices[i] * (spread_pct + comm_pct)
            net_returns[i] = -cost / account_size
            if pos_change > 0:
                trade_count += 1
                
        prev_pos = curr_pos

    return net_returns, trade_count

@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    direction: int # 1 or -1
    size: float
    
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: str = "OPEN"
    
    pnl: float = 0.0
    return_pct: float = 0.0
    
    events: List[Dict] = field(default_factory=list)

class TradeSimulator:
    """
    Centralized logic for simulating trades with Triple Barrier constraints.
    Ensures consistency between Backtesting and Reporting/Visualization.
    """
    def __init__(self, 
                 prices: np.ndarray, 
                 times: pd.Series,
                 spread_bps: float = 1.0, 
                 cost_bps: float = 0.5,
                 lot_size: float = 100000.0,
                 account_size: float = 30000.0):
        
        self.prices = prices.astype(np.float64) # Ensure precision
        self.times = times.values if hasattr(times, 'values') else times
        self.n_bars = len(prices)
        
        # Cost Model
        self.spread_pct = spread_bps / 10000.0
        self.comm_pct = cost_bps / 10000.0
        self.lot_size = lot_size
        self.account_size = account_size

    def simulate(self, 
                 signals: np.ndarray, 
                 stop_loss_pct: float = 0.005, 
                 take_profit_pct: Optional[float] = None,
                 time_limit_bars: Optional[int] = None) -> Tuple[List[Trade], np.ndarray]:
        """
        Simulates trading a signal vector with barriers.
        Detailed version returning Trade objects for visualization.
        """
        # Prepare Result Containers
        trades = []
        equity_curve = np.zeros(self.n_bars)
        equity_curve[0] = self.account_size
        
        # State
        current_equity = self.account_size
        position = 0 # Current lots/direction
        entry_price = 0.0
        trade_start_idx = 0
        current_trade = None
        
        for i in range(1, self.n_bars):
            price = self.prices[i]
            
            step_pnl = 0.0
            if position != 0:
                step_pnl = position * self.lot_size * (price - self.prices[i-1])
            
            exit_signal = False
            exit_reason = ""
            
            if position != 0:
                if time_limit_bars and (i - trade_start_idx >= time_limit_bars):
                    exit_signal = True
                    exit_reason = "TIME"
                elif stop_loss_pct:
                    pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                    if pnl_pct <= -stop_loss_pct:
                        exit_signal = True
                        exit_reason = "SL"
                
                if take_profit_pct and not exit_signal:
                    pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                    if pnl_pct >= take_profit_pct:
                        exit_signal = True
                        exit_reason = "TP"
            
            target_pos = signals[i]
            if exit_signal:
                target_pos = 0 
            
            if target_pos != position:
                change = target_pos - position
                cost = abs(change) * self.lot_size * price * (self.spread_pct + self.comm_pct)
                step_pnl -= cost
                
                if position != 0:
                    if current_trade:
                        current_trade.exit_idx = i
                        current_trade.exit_price = price
                        current_trade.exit_reason = exit_reason if exit_signal else "SIGNAL"
                        gross = (price - current_trade.entry_price) * current_trade.size * current_trade.direction * self.lot_size
                        current_trade.pnl = gross 
                        trades.append(current_trade)
                        current_trade = None
                
                if target_pos != 0:
                    direction = 1 if target_pos > 0 else -1
                    size = abs(target_pos)
                    current_trade = Trade(
                        entry_idx=i,
                        entry_price=price,
                        direction=direction,
                        size=size,
                        events=[{'idx': i, 'action': 'ENTRY', 'price': price, 'size': size}]
                    )
                    trade_start_idx = i
                    entry_price = price
                    
                position = target_pos
                
            current_equity += step_pnl
            equity_curve[i] = current_equity
            
        return trades, equity_curve

    def simulate_fast(self, signals: np.ndarray, stop_loss_pct: float = 0.005, time_limit_bars: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        High-Performance Simulation: Event-Driven.
        Uses Numba JIT.
        """
        limit_val = time_limit_bars if time_limit_bars is not None else 0
        sl_val = stop_loss_pct if stop_loss_pct is not None else 0.0
        
        return _jit_simulate_fast(
            signals.astype(np.float64),
            self.prices,
            self.lot_size,
            self.spread_pct,
            self.comm_pct,
            self.account_size,
            sl_val,
            limit_val
        )

