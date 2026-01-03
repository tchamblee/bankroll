import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from numba import jit
import config

@jit(nopython=True, nogil=True, cache=True)
def _jit_simulate_fast(signals: np.ndarray, prices: np.ndarray, 
                       highs: np.ndarray, lows: np.ndarray, atr_vec: np.ndarray,
                       hours: np.ndarray, weekdays: np.ndarray,
                       lot_size: float, spread_pct: float, comm_pct: float, min_comm: float,
                       account_size: float, sl_mult: float, 
                       tp_mult: float, time_limit_bars: int, cooldown_bars: int,
                       slippage_factor: float) -> Tuple[np.ndarray, int]:
    """
    JIT-Compiled Event-Driven Simulation with Dynamic ATR Barriers and Cooldown.
    """
    n = len(signals)
    
    realized_signals = signals.copy().astype(np.float64) 
    trade_count = 0
    
    # Position entering step i (from i-1)
    position = 0.0 
    entry_price = 0.0
    entry_atr = 0.0
    entry_idx = 0
    
    # Cooldown tracking
    cooldown = 0
    
    # Barrier Exit Prices
    barrier_exit_prices = np.zeros(n, dtype=np.float64)
    
    # Iterate all bars
    for i in range(n):
        if cooldown > 0:
            cooldown -= 1
            
        # --- 1. CHECK BARRIERS FROM PREVIOUS INTERVAL [i-1 to i] ---
        barrier_hit = False
        is_sl_hit = False
        
        if i > 0 and position != 0.0:
            h_prev = highs[i-1]
            l_prev = lows[i-1]
            
            # Dynamic Distances based on Entry Volatility
            sl_dist = entry_atr * sl_mult
            tp_dist = entry_atr * tp_mult
            
            # Long Position
            if position > 0:
                # Stop Loss (Low Check)
                if sl_mult > 0.0:
                    sl_price = entry_price - sl_dist
                    if l_prev <= sl_price:
                        position = 0.0
                        realized_signals[i] = 0.0
                        barrier_exit_prices[i] = sl_price
                        barrier_hit = True
                        is_sl_hit = True
                        
                # Take Profit (High Check)
                if not barrier_hit and tp_mult > 0.0:
                    tp_price = entry_price + tp_dist
                    if h_prev >= tp_price:
                        position = 0.0
                        realized_signals[i] = 0.0
                        barrier_exit_prices[i] = tp_price
                        barrier_hit = True

            # Short Position
            elif position < 0:
                # Stop Loss (High Check)
                if sl_mult > 0.0:
                    sl_price = entry_price + sl_dist
                    if h_prev >= sl_price:
                        position = 0.0
                        realized_signals[i] = 0.0
                        barrier_exit_prices[i] = sl_price
                        barrier_hit = True
                        is_sl_hit = True
                        
                # Take Profit (Low Check)
                if not barrier_hit and tp_mult > 0.0:
                    tp_price = entry_price - tp_dist
                    if l_prev <= tp_price:
                        position = 0.0
                        realized_signals[i] = 0.0
                        barrier_exit_prices[i] = tp_price
                        barrier_hit = True

        # --- 2. FORCE CLOSE (EOD/Time) ---
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

        # --- 3. NEW ENTRY / REVERSAL / HOLD ---
        if barrier_hit:
            if is_sl_hit:
                cooldown = cooldown_bars
                
        if force_close:
            realized_signals[i] = 0.0
            position = 0.0
        elif position == 0.0:
            # New Entry
            if realized_signals[i] != 0.0 and cooldown == 0:
                position = realized_signals[i]
                entry_price = prices[i]
                entry_atr = atr_vec[i]
                entry_idx = i
        else:
            # Position Active: Check for Reversal
            # We ignore 0 signals (Latching/Holding)
            # We only act if signal is non-zero and opposite sign
            sig = realized_signals[i]
            if sig != 0.0 and np.sign(sig) != np.sign(position):
                # Reversal
                position = sig
                entry_price = prices[i]
                entry_atr = atr_vec[i]
                entry_idx = i
            
        # Update realized signal to reflect held position
        realized_signals[i] = position
        
        # Update loop state...
        
    # --- COST & PNL CALCULATION ---
    net_returns = np.zeros(n, dtype=np.float64)
    prev_pos = 0.0 
    
    for i in range(n):
        curr_pos = realized_signals[i]
        current_price = prices[i]
        
        if barrier_exit_prices[i] != 0.0:
            current_price = barrier_exit_prices[i]
            
        if i > 0:
            price_change = current_price - prices[i-1]
            gross_pnl = prev_pos * lot_size * price_change
            pos_change = abs(curr_pos - prev_pos)
            
            # Dynamic Slippage (Factor of ATR)
            slippage = slippage_factor * atr_vec[i] * lot_size * pos_change
            
            # Spread Cost
            spread_cost = pos_change * lot_size * current_price * (0.5 * spread_pct)
            
            # Commission (Max of Variable vs Min)
            raw_comm = pos_change * lot_size * current_price * comm_pct
            # If pos_change > 0 (trade happened), apply min. Else 0.
            # However, pos_change is float. If partial trade?
            # Assuming 'min_comm' applies per order.
            # If pos_change is significant (e.g. > 0.01 lots), we charge min.
            
            comm = 0.0
            if pos_change > config.COMMISSION_THRESHOLD:
                comm = max(min_comm, raw_comm)
            
            cost = spread_cost + comm + slippage
            
            net_pnl = gross_pnl - cost
            net_returns[i] = net_pnl / account_size
            if pos_change > 0: trade_count += 1
        else:
            pos_change = abs(curr_pos - 0.0)
            
            # Dynamic Slippage (Factor of ATR)
            slippage = slippage_factor * atr_vec[i] * lot_size * pos_change
            
            spread_cost = pos_change * lot_size * prices[i] * (0.5 * spread_pct)
            raw_comm = pos_change * lot_size * prices[i] * comm_pct
            
            comm = 0.0
            if pos_change > config.COMMISSION_THRESHOLD:
                comm = max(min_comm, raw_comm)
            
            cost = spread_cost + comm + slippage
            
            net_returns[i] = -cost / account_size
            if pos_change > 0: trade_count += 1
                
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
    def __init__(self, 
                 prices: np.ndarray, 
                 times: pd.Series,
                 spread_bps: float = config.SPREAD_BPS, 
                 cost_bps: float = config.COST_BPS,
                 min_comm: float = config.MIN_COMMISSION,
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
        self.min_comm = min_comm
        self.lot_size = lot_size
        self.account_size = account_size

    def simulate(self, 
                 signals: np.ndarray, 
                 stop_loss_pct: float = config.DEFAULT_STOP_LOSS, 
                 take_profit_pct: Optional[float] = None,
                 time_limit_bars: Optional[int] = None,
                 highs: Optional[np.ndarray] = None,
                 lows: Optional[np.ndarray] = None,
                 atr: Optional[np.ndarray] = None) -> Tuple[List[Trade], np.ndarray]:
        """
        Simulates trading a signal vector with barriers.
        Detailed version returning Trade objects for visualization.
        
        If 'atr' is provided, stop_loss_pct and take_profit_pct are interpreted as ATR Multipliers.
        Otherwise, they are interpreted as raw Percentage of Entry Price.
        """
        # Prepare Result Containers
        trades = []
        equity_curve = np.zeros(self.n_bars)
        equity_curve[0] = self.account_size
        
        # Data Checks
        use_atr = atr is not None
        h_vec = highs if highs is not None else self.prices
        l_vec = lows if lows is not None else self.prices
        atr_vec = atr if atr is not None else np.zeros(self.n_bars)
        
        # State
        current_equity = self.account_size
        position = 0 # Current lots/direction
        entry_price = 0.0
        entry_atr = 0.0
        trade_start_idx = 0
        current_trade = None
        
        for i in range(1, self.n_bars):
            price = self.prices[i] # Open/Close depending on setup. Usually Close for MTM, but exec at Open.
            # Assuming 'prices' passed to __init__ are the execution prices (Open) or Close.
            # Standard backtest uses Next Open.
            
            hour = self.hours[i]
            weekday = self.weekdays[i]
            
            # --- FORCE CLOSE LOGIC ---
            force_close = (hour >= config.TRADING_END_HOUR) or (weekday >= 5)
            
            step_pnl = 0.0
            if position != 0:
                step_pnl = position * self.lot_size * (price - self.prices[i-1])
            
            exit_signal = False
            exit_reason = ""
            barrier_exit_price = 0.0
            
            if position != 0:
                if force_close:
                    exit_signal = True
                    exit_reason = "EOD" # End of Day/Session
                elif time_limit_bars and (i - trade_start_idx >= time_limit_bars):
                    exit_signal = True
                    exit_reason = "TIME"
                else:
                    # Barrier Checks (SL/TP)
                    # Check against PREVIOUS bar High/Low (i-1) to avoid lookahead
                    # (We are at step i, deciding to exit based on what happened since entry)
                    # Actually, if we are at step i (Open), we check if price hit SL during i-1.
                    
                    h_prev = h_vec[i-1]
                    l_prev = l_vec[i-1]
                    
                    sl_dist = 0.0
                    tp_dist = 0.0
                    
                    if use_atr:
                        sl_dist = entry_atr * stop_loss_pct
                        tp_dist = entry_atr * take_profit_pct if take_profit_pct else 0.0
                    else:
                        sl_dist = entry_price * stop_loss_pct
                        tp_dist = entry_price * take_profit_pct if take_profit_pct else 0.0

                    if position > 0:
                        # Long SL (Low check)
                        if stop_loss_pct and l_prev <= (entry_price - sl_dist):
                            exit_signal = True
                            exit_reason = "SL"
                            barrier_exit_price = entry_price - sl_dist
                        # Long TP (High check)
                        elif take_profit_pct and h_prev >= (entry_price + tp_dist):
                            exit_signal = True
                            exit_reason = "TP"
                            barrier_exit_price = entry_price + tp_dist
                            
                    elif position < 0:
                        # Short SL (High check)
                        if stop_loss_pct and h_prev >= (entry_price + sl_dist):
                            exit_signal = True
                            exit_reason = "SL"
                            barrier_exit_price = entry_price + sl_dist
                        # Short TP (Low check)
                        elif take_profit_pct and l_prev <= (entry_price - tp_dist):
                            exit_signal = True
                            exit_reason = "TP"
                            barrier_exit_price = entry_price - tp_dist
            
            target_pos = signals[i]
            
            # --- LATCHING LOGIC ---
            # If barrier exit or force close, we must go to 0.
            if force_close:
                target_pos = 0 
                if position != 0:
                    exit_signal = True
                    exit_reason = "EOD"
            elif exit_signal:
                target_pos = 0
            else:
                # Normal state: Check for Hold or Reversal
                if position != 0:
                    if target_pos == 0:
                        # Signal Flat -> HOLD
                        target_pos = position
                    elif np.sign(target_pos) == np.sign(position):
                         # Signal Same -> HOLD
                         target_pos = position
                    else:
                        # Signal Flip -> REVERSE (Allow target_pos to be different)
                        pass
            
            if target_pos != position:
                change = target_pos - position
                
                # Execution Price
                # If barrier exit, use barrier price. Else use current bar price (Open)
                exec_price = barrier_exit_price if (exit_signal and barrier_exit_price != 0) else price
                
                # Cost Calculation
                spread_cost = abs(change) * self.lot_size * exec_price * (0.5 * self.spread_pct)
                raw_comm = abs(change) * self.lot_size * exec_price * self.comm_pct
                
                # Dynamic Slippage (Factor of ATR)
                current_atr = atr_vec[i] if use_atr else (exec_price * 0.001)
                slippage = config.SLIPPAGE_ATR_FACTOR * current_atr * self.lot_size * abs(change)
                
                comm = 0.0
                if abs(change) > config.COMMISSION_THRESHOLD:
                     comm = max(self.min_comm, raw_comm)
                
                cost = spread_cost + comm + slippage
                step_pnl -= cost
                
                if position != 0:
                    if current_trade:
                        current_trade.exit_idx = i
                        current_trade.exit_price = exec_price
                        current_trade.exit_reason = exit_reason if exit_signal else "SIGNAL"
                        gross = (exec_price - current_trade.entry_price) * current_trade.size * current_trade.direction * self.lot_size
                        current_trade.pnl = gross 
                        trades.append(current_trade)
                        current_trade = None
                
                if target_pos != 0:
                    direction = 1 if target_pos > 0 else -1
                    size = abs(target_pos)
                    current_trade = Trade(
                        entry_idx=i,
                        entry_price=exec_price,
                        direction=direction,
                        size=size,
                        events=[{'idx': i, 'action': 'ENTRY', 'price': exec_price, 'size': size}]
                    )
                    trade_start_idx = i
                    entry_price = exec_price
                    entry_atr = atr_vec[i] if use_atr else 0.0
                    
                position = target_pos
                
            current_equity += step_pnl
            equity_curve[i] = current_equity
            
        return trades, equity_curve

    def simulate_fast(self, signals: np.ndarray, stop_loss_pct: float = config.DEFAULT_STOP_LOSS, take_profit_pct: Optional[float] = None, time_limit_bars: Optional[int] = None,
                      hours: Optional[np.ndarray] = None, weekdays: Optional[np.ndarray] = None,
                      highs: Optional[np.ndarray] = None, lows: Optional[np.ndarray] = None,
                      atr: Optional[np.ndarray] = None,
                      cooldown_bars: int = config.STOP_LOSS_COOLDOWN_BARS) -> Tuple[np.ndarray, int]:
        """
        High-Performance Simulation: Event-Driven.
        """
        limit_val = time_limit_bars if time_limit_bars is not None else 0
        
        # Interpret inputs as Multipliers
        sl_mult = stop_loss_pct if stop_loss_pct is not None else 0.0
        tp_mult = take_profit_pct if take_profit_pct is not None else 0.0
        
        h_arr = hours if hours is not None else self.hours
        w_arr = weekdays if weekdays is not None else self.weekdays
        high_vec = highs.astype(np.float64) if highs is not None else self.prices
        low_vec = lows.astype(np.float64) if lows is not None else self.prices
        
        # ATR Handling
        if atr is None:
            atr_vec = self.prices * 0.001 
        else:
            atr_vec = atr.astype(np.float64)
        
        return _jit_simulate_fast(
            signals.astype(np.float64),
            self.prices,
            high_vec,
            low_vec,
            atr_vec,
            h_arr,
            w_arr,
            self.lot_size,
            self.spread_pct,
            self.comm_pct,
            self.min_comm,
            self.account_size,
            sl_mult,
            tp_mult,
            limit_val,
            cooldown_bars,
            config.SLIPPAGE_ATR_FACTOR
        )
