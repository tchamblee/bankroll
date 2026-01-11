import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from numba import jit
import config
from utils import trading as utrade

@jit(nopython=True, nogil=True, cache=True)
def _jit_simulate_fast(signals: np.ndarray, prices: np.ndarray,
                       highs: np.ndarray, lows: np.ndarray, atr_vec: np.ndarray,
                       hours: np.ndarray, weekdays: np.ndarray,
                       lot_size: float, spread_pct: float, comm_pct: float, min_comm: float,
                       account_size: float, sl_mult: float,
                       tp_mult: float, time_limit_bars: int, cooldown_bars: int,
                       slippage_factor: float,
                       vol_targeting: bool, target_risk_dollars: float, min_lots: float, max_lots: float,
                       limit_dist_atr: np.ndarray,
                       sl_long: float, sl_short: float, tp_long: float, tp_short: float) -> Tuple[np.ndarray, int]:
    """
    JIT-Compiled Event-Driven Simulation with Dynamic ATR Barriers, Cooldown, and Limit Orders.
    """
    n = len(signals)
    
    realized_signals = signals.copy().astype(np.float64) 
    trade_count = 0
    
    # Position entering step i (from i-1)
    position = 0.0 # This is now SIGNED LOTS
    entry_price = 0.0
    entry_atr = 0.0
    entry_idx = 0
    
    # Cooldown tracking
    cooldown = 0
    
    # Barrier Exit Prices
    barrier_exit_prices = np.zeros(n, dtype=np.float64)

    # Entry Price Per Bar (for accurate limit order PnL calculation)
    # Stores the actual entry price on the bar where entry occurred
    entry_price_per_bar = np.zeros(n, dtype=np.float64)

    # Limit Fill Tracking (0=Market, 1=Limit)
    # Used for Cost Calculation
    entry_fill_type = 0 
    
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

            if position > 0:
                # Use long-specific barriers
                hit, exit_p, code = utrade.check_barrier_long(
                    entry_price, entry_atr, l_prev, h_prev, sl_long, tp_long
                )
                if hit:
                    position = 0.0
                    realized_signals[i] = 0.0
                    barrier_exit_prices[i] = exit_p
                    barrier_hit = True
                    if code == 1: is_sl_hit = True

            elif position < 0:
                # Use short-specific barriers
                hit, exit_p, code = utrade.check_barrier_short(
                    entry_price, entry_atr, l_prev, h_prev, sl_short, tp_short
                )
                if hit:
                    position = 0.0
                    realized_signals[i] = 0.0
                    barrier_exit_prices[i] = exit_p
                    barrier_hit = True
                    if code == 1: is_sl_hit = True

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
            entry_fill_type = 0
        elif position == 0.0:
            # New Entry Logic
            sig = realized_signals[i]
            if sig != 0.0 and cooldown == 0:
                # --- LIMIT ORDER LOGIC ---
                limit_dist = limit_dist_atr[i]
                filled = True
                fill_price = prices[i] # Default Market (Open)
                fill_type = 0 # Market
                
                if limit_dist > 0:
                    # Limit Order Attempt
                    current_atr = atr_vec[i] if atr_vec[i] > 1e-6 else 1e-6
                    dist_price = limit_dist * current_atr
                    
                    if sig > 0:
                        # Buy Limit at Open - Dist
                        target_price = prices[i] - dist_price
                        # Check Low
                        if lows[i] <= target_price:
                            fill_price = target_price
                            fill_type = 1 # Limit
                        else:
                            filled = False
                            
                    elif sig < 0:
                        # Sell Limit at Open + Dist
                        target_price = prices[i] + dist_price
                        # Check High
                        if highs[i] >= target_price:
                            fill_price = target_price
                            fill_type = 1 # Limit
                        else:
                            filled = False
                
                if filled:
                    # Calculate Size using direction-specific SL for vol targeting
                    size = 1.0
                    if vol_targeting:
                        # Use appropriate SL based on trade direction
                        if sig > 0:
                            eff_sl = sl_long if sl_long > 0 else 1.0
                        else:
                            eff_sl = sl_short if sl_short > 0 else 1.0
                        eff_atr = atr_vec[i] if atr_vec[i] > 1e-6 else 1e-6
                        calc_size = target_risk_dollars / (lot_size * eff_sl * eff_atr)
                        size = min(max(calc_size, min_lots), max_lots)

                    position = np.sign(sig) * size
                    entry_price = fill_price
                    entry_atr = atr_vec[i]
                    entry_idx = i
                    entry_fill_type = fill_type
                    # Record actual entry price for accurate PnL calculation
                    entry_price_per_bar[i] = fill_price
                else:
                    # Missed Limit Order
                    realized_signals[i] = 0.0
                    position = 0.0
                    
        else:
            # Position Active: Check for Reversal
            sig = realized_signals[i]
            if sig != 0.0 and np.sign(sig) != np.sign(position):
                # Reversal Logic
                # Currently treating reversals as Market Orders for exit of old + Market/Limit for new?
                # For simplicity: Reversals are Market Orders (Simulating "Get me out and flip now")
                # UNLESS we want to support Limit Reversals? Too complex for v1.
                # Let's enforce Market Fill on Reversal for safety.

                size = 1.0
                if vol_targeting:
                    # Use appropriate SL based on new trade direction
                    if sig > 0:
                        eff_sl = sl_long if sl_long > 0 else 1.0
                    else:
                        eff_sl = sl_short if sl_short > 0 else 1.0
                    eff_atr = atr_vec[i] if atr_vec[i] > 1e-6 else 1e-6
                    calc_size = target_risk_dollars / (lot_size * eff_sl * eff_atr)
                    size = min(max(calc_size, min_lots), max_lots)

                position = np.sign(sig) * size
                entry_price = prices[i]
                entry_atr = atr_vec[i]
                entry_idx = i
                entry_fill_type = 0 # Market Reversal
                # Record actual entry price for accurate PnL calculation
                entry_price_per_bar[i] = prices[i]
            else:
                # Hold
                pass
            
        realized_signals[i] = position
        
    # --- COST & PNL CALCULATION ---
    net_returns = np.zeros(n, dtype=np.float64)
    prev_pos = 0.0 
    
    for i in range(n):
        curr_pos = realized_signals[i]
        current_price = prices[i]
        
        # Determine Execution Price for PnL
        # If we entered this bar (prev_pos == 0 and curr_pos != 0), use entry_price (which might be limit)
        # If we exited this bar (prev_pos != 0 and curr_pos != prev_pos), check exit price
        
        exec_price = current_price # Default
        
        # Override price if Barrier Exit
        if barrier_exit_prices[i] != 0.0:
            exec_price = barrier_exit_prices[i]
            
        # PnL Logic
        if i > 0:
            # Mark-to-Market PnL Calculation
            # Standard: PnL = (Price[i] - Price[i-1]) * Position
            # FIX: If we entered on bar i-1 (via limit or market), use actual entry price
            # instead of prices[i-1] to correctly credit/debit the fill price difference.

            # Determine the reference price for previous bar
            prev_price = prices[i-1]
            if entry_price_per_bar[i-1] != 0.0:
                # We entered on bar i-1, use actual entry price
                prev_price = entry_price_per_bar[i-1]

            # Gross PnL from holding prev_pos
            price_change = exec_price - prev_price
            gross_pnl = prev_pos * lot_size * price_change
            
            # Transaction Cost
            pos_change = abs(curr_pos - prev_pos)
            
            cost = 0.0
            if pos_change > 0:
                # Determine Effective Spread for this trade
                # If entering (prev=0 -> curr!=0) and it was a Limit Fill (fill_type=1) -> Spread = 0
                # But here we iterate linearly. We need to know if *this specific transaction* was limit.
                
                # Simplified: If we are entering a new position (prev=0)
                # We can check limit_dist_atr[i]. If > 0 and we filled, spread=0.
                
                # However, loop variables don't track "Did we just fill limit?".
                # But we have entry_price! 
                # If we just entered, the PnL math handles the price diff.
                # Cost is purely Spread + Comm.
                
                # If Entering:
                is_limit_entry = False
                if prev_pos == 0 and curr_pos != 0:
                    # Check if entry price differs from Open (Market)
                    # OR check limit_dist_atr
                    if limit_dist_atr[i] > 0:
                         is_limit_entry = True
                
                eff_spread = 0.0 if is_limit_entry else spread_pct
                
                cost = utrade.calculate_cost(
                    pos_change, exec_price, atr_vec[i], lot_size, 
                    eff_spread, comm_pct, min_comm, slippage_factor, 
                    config.COMMISSION_THRESHOLD
                )
                if prev_pos == 0: trade_count += 1
            
            net_pnl = gross_pnl - cost
            net_returns[i] = net_pnl / account_size
            
        else:
            # First bar
            pos_change = abs(curr_pos - 0.0)
            cost = 0.0
            if pos_change > 0:
                is_limit_entry = (limit_dist_atr[i] > 0)
                eff_spread = 0.0 if is_limit_entry else spread_pct
                
                cost = utrade.calculate_cost(
                    pos_change, prices[i], atr_vec[i], lot_size, 
                    eff_spread, comm_pct, min_comm, slippage_factor, 
                    config.COMMISSION_THRESHOLD
                )
                trade_count += 1
            
            net_returns[i] = -cost / account_size
                
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
                 atr: Optional[np.ndarray] = None,
                 cooldown_bars: int = config.STOP_LOSS_COOLDOWN_BARS,
                 vol_targeting: bool = True,
                 target_risk_pct: float = config.RISK_PER_TRADE_PERCENT,
                 limit_dist_atr: Optional[np.ndarray] = None) -> Tuple[List[Trade], np.ndarray]:
        """
        Simulates trading a signal vector with barriers.
        Detailed version returning Trade objects for visualization.
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
        limit_vec = limit_dist_atr if limit_dist_atr is not None else np.zeros(self.n_bars)
        
        # State
        current_equity = self.account_size
        position = 0 # Current lots/direction
        entry_price = 0.0
        entry_atr = 0.0
        trade_start_idx = 0
        current_trade = None
        cooldown = 0
        
        sl_mult = stop_loss_pct if stop_loss_pct else 0.0
        tp_mult = take_profit_pct if take_profit_pct else 0.0
        
        for i in range(0, self.n_bars):
            if cooldown > 0:
                cooldown -= 1
            
            price = self.prices[i] 
            
            hour = self.hours[i]
            weekday = self.weekdays[i]
            
            # --- FORCE CLOSE LOGIC ---
            force_close = (hour >= config.TRADING_END_HOUR) or (weekday >= 5)
            
            step_pnl = 0.0
            if i > 0 and position != 0:
                step_pnl = position * self.lot_size * (price - self.prices[i-1])
            
            exit_signal = False
            exit_reason = ""
            barrier_exit_price = 0.0
            is_sl = False
            
            if position != 0:
                if force_close:
                    exit_signal = True
                    exit_reason = "EOD" 
                elif time_limit_bars and (i - trade_start_idx >= time_limit_bars):
                    exit_signal = True
                    exit_reason = "TIME"
                elif i > 0:
                    # Barrier Checks (SL/TP)
                    h_prev = h_vec[i-1]
                    l_prev = l_vec[i-1]
                    
                    # Adapt to ATR vs Percentage
                    eff_atr = entry_atr if use_atr else entry_price
                    
                    if position > 0:
                        hit, exit_p, code = utrade.check_barrier_long(entry_price, eff_atr, l_prev, h_prev, sl_mult, tp_mult)
                        if hit:
                            exit_signal = True
                            barrier_exit_price = exit_p
                            exit_reason = "SL" if code == 1 else "TP"
                            if code == 1: is_sl = True
                            
                    elif position < 0:
                        hit, exit_p, code = utrade.check_barrier_short(entry_price, eff_atr, l_prev, h_prev, sl_mult, tp_mult)
                        if hit:
                            exit_signal = True
                            barrier_exit_price = exit_p
                            exit_reason = "SL" if code == 1 else "TP"
                            if code == 1: is_sl = True
            
            target_pos = signals[i]
            
            # --- LATCHING LOGIC ---
            if force_close:
                target_pos = 0 
                if position != 0:
                    exit_signal = True
                    exit_reason = "EOD"
            elif exit_signal:
                target_pos = 0
                if is_sl:
                    cooldown = cooldown_bars
            else:
                if position != 0:
                    if target_pos == 0: target_pos = position
                    elif np.sign(target_pos) == np.sign(position): target_pos = position
                
                # Cooldown Block
                if position == 0 and cooldown > 0:
                    target_pos = 0
            
            # --- POSITION SIZING LOGIC ---
            target_size = 0.0
            
            # Limit Execution Logic
            # If target_pos != 0 and position == 0, check Limit
            fill_price = price # Default to Market (Open)
            is_limit_fill = False
            limit_dist = limit_vec[i]
            
            if target_pos != 0 and position == 0:
                if limit_dist > 0:
                    # Attempt Limit
                    current_atr = atr_vec[i] if atr_vec[i] > 1e-6 else 1e-6
                    dist_price = limit_dist * current_atr
                    
                    if target_pos > 0:
                        # Buy Limit
                        limit_p = price - dist_price
                        if l_vec[i] <= limit_p:
                            fill_price = limit_p
                            is_limit_fill = True
                        else:
                            # Missed
                            target_pos = 0
                    elif target_pos < 0:
                        # Sell Limit
                        limit_p = price + dist_price
                        if h_vec[i] >= limit_p:
                            fill_price = limit_p
                            is_limit_fill = True
                        else:
                            # Missed
                            target_pos = 0
            
            if target_pos != 0:
                if position == 0:
                     # New Entry - Calculate Size
                     if vol_targeting and use_atr:
                         # Risk Parity Sizing
                         eff_sl = sl_mult if sl_mult > 0 else 1.0
                         eff_atr = atr_vec[i] if atr_vec[i] > 1e-6 else 1e-6
                         target_risk_dollars = self.account_size * target_risk_pct
                         
                         calc_size = target_risk_dollars / (self.lot_size * eff_sl * eff_atr)
                         target_size = min(max(calc_size, config.MIN_LOTS), config.MAX_LOTS)
                     else:
                         target_size = 1.0 # Fixed Lot
                         
                     # Apply Direction
                     target_pos = np.sign(target_pos) * target_size
                else:
                     # Existing Position - Maintain Size
                     target_size = abs(position)
                     target_pos = np.sign(target_pos) * target_size

            if target_pos != position:
                # change is now difference in signed lots
                change = target_pos - position
                
                # Use calculated fill_price for entry, or exit price
                # If we are exiting (target=0), use barrier_exit or current market price
                exec_price = barrier_exit_price if (exit_signal and barrier_exit_price != 0) else fill_price
                
                # Adjustment for difference between Open Price (booked in step_pnl) and Exec Price
                # Only applies to the position we are holding (and potentially closing)
                if position != 0:
                    pnl_adj = position * self.lot_size * (exec_price - price)
                    step_pnl += pnl_adj
                
                # Cost Calculation via Shared Logic
                # Use lots traded (abs(change))
                current_atr = atr_vec[i] if use_atr else (exec_price * 0.001)
                
                # Spread is 0 if this is a limit entry
                # Check: Is this an entry? (position==0 -> target!=0) AND is it limit fill?
                eff_spread = self.spread_pct
                if position == 0 and target_pos != 0 and is_limit_fill:
                    eff_spread = 0.0
                
                cost = utrade.calculate_cost(
                    abs(change), exec_price, current_atr, self.lot_size, 
                    eff_spread, self.comm_pct, self.min_comm, 
                    config.SLIPPAGE_ATR_FACTOR, config.COMMISSION_THRESHOLD
                )
                
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
                        events=[{'idx': i, 'action': 'ENTRY', 'price': exec_price, 'size': size, 'limit': is_limit_fill}]
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
                      cooldown_bars: int = config.STOP_LOSS_COOLDOWN_BARS,
                      vol_targeting: bool = True,
                      target_risk_pct: float = config.RISK_PER_TRADE_PERCENT,
                      limit_dist_atr: Optional[np.ndarray] = None,
                      sl_long: Optional[float] = None, sl_short: Optional[float] = None,
                      tp_long: Optional[float] = None, tp_short: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        High-Performance Simulation: Event-Driven.
        """
        limit_val = time_limit_bars if time_limit_bars is not None else 0

        sl_mult = stop_loss_pct if stop_loss_pct is not None else 0.0
        tp_mult = take_profit_pct if take_profit_pct is not None else 0.0

        # Directional barriers (default to symmetric if not specified)
        eff_sl_long = sl_long if sl_long is not None else sl_mult
        eff_sl_short = sl_short if sl_short is not None else sl_mult
        eff_tp_long = tp_long if tp_long is not None else tp_mult
        eff_tp_short = tp_short if tp_short is not None else tp_mult

        h_arr = hours if hours is not None else self.hours
        w_arr = weekdays if weekdays is not None else self.weekdays
        high_vec = highs.astype(np.float64) if highs is not None else self.prices
        low_vec = lows.astype(np.float64) if lows is not None else self.prices

        if atr is None:
            atr_vec = self.prices * 0.001
        else:
            atr_vec = atr.astype(np.float64)

        target_risk_dollars = self.account_size * target_risk_pct

        # Prepare Limit Distance Vector
        if limit_dist_atr is None:
            limit_dist_vec = np.zeros(len(signals), dtype=np.float64)
        else:
            limit_dist_vec = limit_dist_atr.astype(np.float64)

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
            config.SLIPPAGE_ATR_FACTOR,
            vol_targeting,
            target_risk_dollars,
            float(config.MIN_LOTS),
            float(config.MAX_LOTS),
            limit_dist_vec,
            eff_sl_long,
            eff_sl_short,
            eff_tp_long,
            eff_tp_short
        )