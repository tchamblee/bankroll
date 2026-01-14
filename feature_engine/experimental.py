import numpy as np
import pandas as pd
from joblib import Parallel, delayed

def _calc_choppiness(high, low, close, window):
    """
    Choppiness Index = 100 * LOG10( SUM(ATR(1), n) / ( MaxHigh(n) - MinLow(n) ) ) / LOG10(n)
    Values: 0-100. Lower = Strong Trend. Higher = Consolidation.
    """
    # True Range (1-period)
    tr1 = np.maximum(high - low, np.abs(high - close.shift(1)))
    tr_sum = tr1.rolling(window).sum()
    
    # Range
    max_hi = high.rolling(window).max()
    min_lo = low.rolling(window).min()
    range_hl = max_hi - min_lo
    
    # Avoid division by zero
    range_hl = range_hl.replace(0, np.nan)
    
    chop = 100 * np.log10(tr_sum / range_hl) / np.log10(window)
    return chop

def _calc_vortex(high, low, close, window):
    """
    Vortex Indicator (VI+ and VI-).
    Returns VI_diff (VI+ - VI-)
    """
    # VM+ = Abs(CurrentHigh - PriorLow)
    vm_plus = (high - low.shift(1)).abs()
    
    # VM- = Abs(CurrentLow - PriorHigh)
    vm_minus = (low - high.shift(1)).abs()
    
    # TR (1-period)
    tr1 = np.maximum(high - low, np.abs(high - close.shift(1)))
    
    # Sums
    vm_plus_sum = vm_plus.rolling(window).sum()
    vm_minus_sum = vm_minus.rolling(window).sum()
    tr_sum = tr1.rolling(window).sum()
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    return vi_plus - vi_minus

def _calc_eom(high, low, volume, window):
    """
    Ease of Movement (EOM).
    EOM = (MidPoint_Diff / Volume) * (High - Low)
    """
    # Distance moved (Midpoint diff)
    mid = (high + low) / 2
    dist = mid.diff(1)
    
    # Box Ratio = Volume / (High - Low)
    # Use small epsilon for zero-range bars to avoid NaN propagation in rolling windows
    hl_range = (high - low).replace(0, 1e-8)
    box_ratio = volume / hl_range
    
    eom_1 = dist / box_ratio
    
    # Smoothed EOM
    eom_smooth = eom_1.rolling(window).mean()
    return eom_smooth

def _calc_mfi(high, low, close, volume, window):
    """
    Money Flow Index (MFI).
    RSI-like indicator using Volume-Weighted Price.
    """
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Positive/Negative Flow
    # 1. Diff of Typical Price
    tp_diff = typical_price.diff()
    
    pos_flow = np.where(tp_diff > 0, raw_money_flow, 0)
    neg_flow = np.where(tp_diff < 0, raw_money_flow, 0)
    
    # Rolling Sums
    pos_sum = pd.Series(pos_flow, index=close.index).rolling(window).sum()
    neg_sum = pd.Series(neg_flow, index=close.index).rolling(window).sum()
    
    # MFI
    mfi_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfi_ratio))
    return mfi

def _calc_cmf(high, low, close, volume, window):
    """
    Chaikin Money Flow (CMF).
    Sum(AD, n) / Sum(Vol, n)
    """
    # Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    # Use small epsilon for zero-range bars to avoid NaN propagation in rolling windows
    hl_range = (high - low).replace(0, 1e-8)
    mf_mult = ((close - low) - (high - close)) / hl_range
    mf_vol = mf_mult * volume
    
    cmf = mf_vol.rolling(window).sum() / volume.rolling(window).sum()
    return cmf

def add_experimental_features(df, windows=[14, 100]):
    """
    Adds Choppiness Index, Vortex Indicator, Ease of Movement, MFI, and CMF.
    """
    if df is None: return None
    df = df.copy()
    
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # 1. Choppiness Index (Trend Quality)
    for w in windows:
        df[f'chop_{w}'] = _calc_choppiness(high, low, close, w)
        
    # 2. Vortex Indicator (Trend Reversal)
    for w in windows:
        df[f'vortex_{w}'] = _calc_vortex(high, low, close, w)
        
    # 3. Ease of Movement (Volume/Price Dynamics)
    for w in windows:
        df[f'eom_{w}'] = _calc_eom(high, low, volume, w)

    # 4. Money Flow Index (MFI)
    for w in windows:
        df[f'mfi_{w}'] = _calc_mfi(high, low, close, volume, w)

    # 5. Chaikin Money Flow (CMF)
    for w in windows:
        df[f'cmf_{w}'] = _calc_cmf(high, low, close, volume, w)

    return df
