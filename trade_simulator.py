import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from numba import jit
import config

@jit(nopython=True)
def _jit_simulate_fast(signals: np.ndarray, prices: np.ndarray, 
                       highs: np.ndarray, lows: np.ndarray,
                       hours: np.ndarray, weekdays: np.ndarray,
                       lot_size: float, spread_pct: float, comm_pct: float, 
                       account_size: float, stop_loss_pct: float, 
                       take_profit_pct: float, time_limit_bars: int) -> Tuple[np.ndarray, int]:
    """
    JIT-Compiled Event-Driven Simulation.
    Supports Intra-Bar Barrier Checking (High/Low) and Next-Open Execution.
    """
    n = len(signals)
    
    realized_signals = signals.copy().astype(np.float64) 
    trade_count = 0
    
    # Position entering step i (from i-1)
    position = 0.0 
    entry_price = 0.0
    entry_idx = 0
    
    # We need to track if we were stopped out in the previous interval
    # to correct the PnL calculation in the second loop.
    # Arrays to store the effective exit price for each bar if a barrier was hit.
    # 0.0 means no barrier hit (use standard Open-Open).
    barrier_exit_prices = np.zeros(n, dtype=np.float64)
    
    # Iterate all bars
    for i in range(n):
        # --- 1. CHECK BARRIERS FROM PREVIOUS INTERVAL [i-1 to i] ---
        # We are standing at Open[i]. We just traversed Bar i-1.
        # Did we survive Bar i-1?
        
        barrier_hit = False
        
        if i > 0 and position != 0.0:
            # Check High/Low of Bar i-1
            # Note: signals[i] is the decision for Open[i]. 
            # 'position' is what we held coming into i.
            
            # SL/TP Checks on Bar i-1
            # We use entry_price (which was Open[entry_idx] or similar)
            
            h_prev = highs[i-1]
            l_prev = lows[i-1]
            
            # Long Position
            if position > 0:
                # Stop Loss (Low Check)
                if stop_loss_pct > 0.0:
                    sl_price = entry_price * (1.0 - stop_loss_pct)
                    if l_prev <= sl_price:
                        # HIT SL
                        position = 0.0
                        realized_signals[i] = 0.0 # Forced flat
                        barrier_exit_prices[i] = sl_price
                        barrier_hit = True
                        
                # Take Profit (High Check) - Only if SL not hit (simplified)
                if not barrier_hit and take_profit_pct > 0.0:
                    tp_price = entry_price * (1.0 + take_profit_pct)
                    if h_prev >= tp_price:
                        # HIT TP
                        position = 0.0
                        realized_signals[i] = 0.0
                        barrier_exit_prices[i] = tp_price
                        barrier_hit = True

            # Short Position
            elif position < 0:
                # Stop Loss (High Check)
                if stop_loss_pct > 0.0:
                    sl_price = entry_price * (1.0 + stop_loss_pct)
                    if h_prev >= sl_price:
                        # HIT SL
                        position = 0.0
                        realized_signals[i] = 0.0
                        barrier_exit_prices[i] = sl_price
                        barrier_hit = True
                        
                # Take Profit (Low Check)
                if not barrier_hit and take_profit_pct > 0.0:
                    tp_price = entry_price * (1.0 - take_profit_pct)
                    if l_prev <= tp_price:
                        # HIT TP
                        position = 0.0
                        realized_signals[i] = 0.0
                        barrier_exit_prices[i] = tp_price
                        barrier_hit = True

        # --- 2. FORCE CLOSE (EOD/Time) ---
        # Logic remains similar, but applied to current time i
        force_close = False
        if hours[i] >= config.TRADING_END_HOUR or weekdays[i] >= 5:
            force_close = True
            
        if position != 0.0 and not barrier_hit:
            if force_close:
                position = 0.0
                realized_signals[i] = 0.0
            elif time_limit_bars > 0:
                 if (i - entry_idx) >= time_limit_bars:
                        position = 0.0
                        realized_signals[i] = 0.0

        # --- 3. NEW ENTRY ---
        # Only if we aren't forced closed and didn't just hit a barrier
        if force_close:
            realized_signals[i] = 0.0
        elif position == 0.0 and realized_signals[i] != 0.0:
            # Entering new trade at Open[i]
            position = realized_signals[i]
            entry_price = prices[i]
            entry_idx = i
            
        # Update current position state for next iteration
        # Note: 'position' variable here effectively tracks realized_signals[i]
        # unless we modify it in next loop.
        
    # --- COST & PNL CALCULATION ---
    net_returns = np.zeros(n, dtype=np.float64)
    prev_pos = 0.0 
    
    for i in range(n):
        curr_pos = realized_signals[i]
        
        # Determine Exit Price
        # Standard: Open[i]
        # Barrier Hit: barrier_exit_prices[i]
        
        current_price = prices[i]
        executed_at_barrier = False
        
        if barrier_exit_prices[i] != 0.0:
            current_price = barrier_exit_prices[i]
            executed_at_barrier = True
            
        if i > 0:
            # PnL from holding prev_pos from i-1 to i
            # Price Change: (Current Price - Prev Price)
            # Prev Price was Open[i-1] (or entry if i-1 was entry?)
            # Actually, simplify: Mark-to-Market PnL
            
            # If we hit a barrier at i, it means we exited AT the barrier price during the interval.
            # So the move is (BarrierPrice - Open[i-1]).
            # And then we are flat (curr_pos=0).
            
            # If we didn't hit barrier, move is (Open[i] - Open[i-1]).
            
            price_change = current_price - prices[i-1]
            gross_pnl = prev_pos * lot_size * price_change
            
            # Cost Logic
            # If we changed position, we paid spread/comm.
            # If we exited at Barrier, we paid cost at Barrier Price.
            
            pos_change = abs(curr_pos - prev_pos)
            
            cost_price = current_price # Price where execution happened
            cost = pos_change * lot_size * cost_price * (spread_pct + comm_pct)
            
            net_pnl = gross_pnl - cost
            net_returns[i] = net_pnl / account_size
            
            if pos_change > 0:
                trade_count += 1
                
        else:
            # First bar
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
                 spread_bps: float = config.SPREAD_BPS, 
                 cost_bps: float = config.COST_BPS,
                 lot_size: float = config.STANDARD_LOT_SIZE,
                 account_size: float = config.ACCOUNT_SIZE):
        
        self.prices = prices.astype(np.float64) # Ensure precision
        self.times = times.values if hasattr(times, 'values') else times
        self.n_bars = len(prices)
        
        # Extract Time Features for Filtering
        if hasattr(times, 'dt'): # Pandas Series
            self.hours = times.dt.hour.values.astype(np.int8)
            self.weekdays = times.dt.dayofweek.values.astype(np.int8)
        else:
            # Numpy array or similar
            dt_index = pd.to_datetime(times)
            self.hours = dt_index.hour.values.astype(np.int8)
            self.weekdays = dt_index.dayofweek.values.astype(np.int8)
        
        # Cost Model
        self.spread_pct = spread_bps / 10000.0
        self.comm_pct = cost_bps / 10000.0
        self.lot_size = lot_size
        self.account_size = account_size

    def simulate(self, 
                 signals: np.ndarray, 
                 stop_loss_pct: float = config.DEFAULT_STOP_LOSS, 
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
            hour = self.hours[i]
            weekday = self.weekdays[i]
            
            # --- FORCE CLOSE LOGIC ---
            force_close = (hour >= config.TRADING_END_HOUR) or (weekday >= 5)
            
            step_pnl = 0.0
            if position != 0:
                step_pnl = position * self.lot_size * (price - self.prices[i-1])
            
            exit_signal = False
            exit_reason = ""
            
            if position != 0:
                if force_close:
                    exit_signal = True
                    exit_reason = "EOD" # End of Day/Session
                elif time_limit_bars and (i - trade_start_idx >= time_limit_bars):
                    exit_signal = True
                    exit_reason = "TIME"
                elif stop_loss_pct:
                    pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                    if pnl_pct <= -stop_loss_pct:
                        exit_signal = True
                        exit_reason = "SL"
                
                if take_profit_pct and not exit_signal and not force_close:
                    pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                    if pnl_pct >= take_profit_pct:
                        exit_signal = True
                        exit_reason = "TP"
            
            target_pos = signals[i]
            if force_close:
                target_pos = 0 
                if position != 0:
                    exit_signal = True
                    exit_reason = "EOD"
            elif exit_signal:
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

    def simulate_fast(self, signals: np.ndarray, stop_loss_pct: float = config.DEFAULT_STOP_LOSS, take_profit_pct: Optional[float] = None, time_limit_bars: Optional[int] = None,
                      hours: Optional[np.ndarray] = None, weekdays: Optional[np.ndarray] = None,
                      highs: Optional[np.ndarray] = None, lows: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        High-Performance Simulation: Event-Driven.
        Uses Numba JIT. Supports Intra-Bar High/Low checks.
        """
        limit_val = time_limit_bars if time_limit_bars is not None else 0
        sl_val = stop_loss_pct if stop_loss_pct is not None else 0.0
        tp_val = take_profit_pct if take_profit_pct is not None else 0.0
        
        # Use pre-calculated or passed arrays
        h_arr = hours if hours is not None else self.hours
        w_arr = weekdays if weekdays is not None else self.weekdays
        
        # Highs/Lows defaults
        high_vec = highs.astype(np.float64) if highs is not None else self.prices
        low_vec = lows.astype(np.float64) if lows is not None else self.prices
        
        return _jit_simulate_fast(
            signals.astype(np.float64),
            self.prices,
            high_vec,
            low_vec,
            h_arr,
            w_arr,
            self.lot_size,
            self.spread_pct,
            self.comm_pct,
            self.account_size,
            sl_val,
            tp_val,
            limit_val
        )