import numpy as np
from numba import jit

@jit(nopython=True, nogil=True, cache=True)
def calculate_cost(pos_change: float, price: float, atr: float, lot_size: float, 
                   spread_pct: float, comm_pct: float, min_comm: float, 
                   slippage_factor: float, comm_threshold: float = 1e-6) -> float:
    """
    Calculates total transaction cost (Spread + Commission + Slippage).
    """
    if pos_change < comm_threshold:
        return 0.0
        
    # Spread Cost (Half Spread)
    spread_cost = pos_change * lot_size * price * (0.5 * spread_pct)
    
    # Commission (Max of Variable vs Min)
    raw_comm = pos_change * lot_size * price * comm_pct
    comm = max(min_comm, raw_comm)
    
    # Dynamic Slippage (Factor of ATR)
    slippage = slippage_factor * atr * lot_size * pos_change
    
    return spread_cost + comm + slippage

@jit(nopython=True, nogil=True, cache=True)
def check_barrier_long(entry_price: float, entry_atr: float, 
                       low_prev: float, high_prev: float, 
                       sl_mult: float, tp_mult: float):
    """
    Checks if Long position hit barriers.
    Returns: (hit: bool, exit_price: float, reason_code: int)
    Reason Code: 0=None, 1=SL, 2=TP
    """
    # Stop Loss (Low Check)
    if sl_mult > 0:
        sl_price = entry_price - (entry_atr * sl_mult)
        if low_prev <= sl_price:
            return True, sl_price, 1
            
    # Take Profit (High Check)
    if tp_mult > 0:
        tp_price = entry_price + (entry_atr * tp_mult)
        if high_prev >= tp_price:
            return True, tp_price, 2
            
    return False, 0.0, 0

@jit(nopython=True, nogil=True, cache=True)
def check_barrier_short(entry_price: float, entry_atr: float, 
                        low_prev: float, high_prev: float, 
                        sl_mult: float, tp_mult: float):
    """
    Checks if Short position hit barriers.
    Returns: (hit: bool, exit_price: float, reason_code: int)
    Reason Code: 0=None, 1=SL, 2=TP
    """
    # Stop Loss (High Check)
    if sl_mult > 0:
        sl_price = entry_price + (entry_atr * sl_mult)
        if high_prev >= sl_price:
            return True, sl_price, 1
            
    # Take Profit (Low Check)
    if tp_mult > 0:
        tp_price = entry_price - (entry_atr * tp_mult)
        if low_prev <= tp_price:
            return True, tp_price, 2
            
    return False, 0.0, 0
