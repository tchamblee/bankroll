import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

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
        Iterates only over signal changes (trades), skipping idle bars.
        Returns:
            net_returns: Array of % returns per bar (relative to account size).
            trades_count: Total number of executed trades.
        """
        n = len(signals)
        prices = self.prices
        
        # 1. Identify Potential Events (Signal Changes)
        sig_diff = np.diff(signals, prepend=0)
        event_indices = np.nonzero(sig_diff)[0]
        
        if len(event_indices) == 0:
            return np.zeros(n), 0
            
        # 2. Event Loop - Reconstruct Realized Signal Path
        realized_signals = signals.copy().astype(np.float32)
        
        # Add end index sentinel
        events = np.append(event_indices, n)
        
        # Iterate intervals [start, end) where signal is constant
        # Python loop over trades (events) is fast enough (100-200 iterations vs 20000)
        
        # Pre-calc Limit for Time Barrier
        # Pre-calc SL Thresholds? Price dependent.
        
        limit_val = time_limit_bars if time_limit_bars else 999999
        
        for i in range(len(events) - 1):
            start = events[i]
            end = events[i+1]
            
            pos = signals[start]
            if pos == 0: continue
            
            entry_price = prices[start]
            limit_idx = start + limit_val
            
            # SL Check
            # Need min/max price in valid range
            search_end = min(end, limit_idx)
            if search_end > n: search_end = n
            
            segment = prices[start:search_end]
            
            exit_idx = -1
            
            if stop_loss_pct:
                if pos > 0:
                    thresh = entry_price * (1 - stop_loss_pct)
                    # Vectorized search for first violation
                    hits = np.where(segment < thresh)[0]
                else:
                    thresh = entry_price * (1 + stop_loss_pct)
                    hits = np.where(segment > thresh)[0]
                    
                if len(hits) > 0:
                    exit_idx = start + hits[0]
            
            # Time vs SL
            if time_limit_bars and limit_idx < end:
                if exit_idx != -1:
                    exit_idx = min(exit_idx, limit_idx)
                else:
                    exit_idx = limit_idx
            
            # Apply Exit
            if exit_idx != -1:
                realized_signals[exit_idx:end] = 0
                
        # 3. Calculate PnL (Vectorized)
        pos_lag = np.roll(realized_signals, 1)
        pos_lag[0] = 0
        
        price_diff = np.diff(prices, prepend=prices[0])
        gross_pnl = pos_lag * self.lot_size * price_diff
        
        lots_diff = np.abs(realized_signals - pos_lag)
        costs = lots_diff * self.lot_size * prices * (self.spread_pct + self.comm_pct)
        
        net_pnl = gross_pnl - costs
        net_returns = net_pnl / self.account_size
        
        trades_count = np.sum(lots_diff > 0)
        
        return net_returns, trades_count

