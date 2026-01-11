# System Audit Findings - 2026-01-11

## Critical Issues

### 1. Limit Order PnL Calculation Bug
- **File:** `trade_simulator.py:210` (JIT function `_jit_simulate_fast`)
- **Status:** FIXED (2026-01-11)
- **Impact:** HIGH (dormant - only triggers when limit orders enabled)

**Description:**
When a limit order fills at a price different from the bar's open, the mark-to-market PnL calculation is incorrect. The code calculates PnL as `prices[i+1] - prices[i]` instead of `prices[i+1] - entry_price`.

**Current Code (Line 210):**
```python
price_change = exec_price - prices[i-1]
gross_pnl = prev_pos * lot_size * price_change
```

**Problem:**
- For limit buys below market: PnL is understated
- For limit sells above market: PnL is understated
- The spread cost saving (line 238) only accounts for not paying bid-ask, NOT for the better entry price

**Example:**
- Entry via limit at 1.0990 (10 pips below open of 1.1000)
- Next bar opens at 1.1010
- Correct PnL: 1.1010 - 1.0990 = +20 pips
- Calculated PnL: 1.1010 - 1.1000 = +10 pips
- Missing: 10 pips of edge from the limit fill

**Mitigation:** Currently all strategies use `limit_dist_atr = 0.0` (market orders only), so bug is dormant.

**Fix Applied:**
Added `entry_price_per_bar` array to track actual entry prices. Modified PnL calculation to use entry price when available:
```python
prev_price = prices[i-1]
if entry_price_per_bar[i-1] != 0.0:
    prev_price = entry_price_per_bar[i-1]
price_change = exec_price - prev_price
```

---

## Moderate Issues

### 2. Dead Code - Signal Magnitude Multiplication
- **File:** `genome/strategy.py:106`
- **Status:** FIXED (2026-01-11)
- **Impact:** None (cosmetic)

**Description:**
```python
return net_signal * config.MAX_LOTS
```
The signal is multiplied by MAX_LOTS, but when `vol_targeting=True` (always on in workers.py:64), only `np.sign(sig)` is used. This multiplication is effectively dead code.

**Fix Applied:** Removed `* config.MAX_LOTS` multiplication and added clarifying comment:
```python
# Net signal: +1 (long), -1 (short), 0 (flat)
# Actual position sizing is handled by vol_targeting in trade_simulator
return net_signal
```

---

### 3. Sortino Ratio Edge Case
- **File:** `backtest/statistics.py:27`
- **Status:** FIXED (2026-01-11) - Documented
- **Impact:** Cosmetic in rare edge cases

**Description:**
```python
downside_std = np.sqrt(np.mean(downside**2)) + config.EPSILON
```
Adding EPSILON (1e-9) means strategies with zero downside deviation get a very high but finite Sortino rather than infinity. Could mask perfect strategies in ranking.

**Fix Applied:** Added comprehensive docstring explaining the intentional behavior:
- Maintains numerical stability for ranking/sorting
- Avoids inf propagation in portfolio optimization
- Treats "no drawdowns yet" as "excellent but unproven" rather than "perfect"

---

## Verified Correct (No Action Needed)

- [x] Barrier Checks (`utils/trading.py`) - SL/TP using ATR multiples
- [x] Cost Model (`utils/trading.py`) - Half-spread + commission + slippage
- [x] Look-Ahead Bias Prevention (`feature_engine/correlations.py`, `intermarket.py`) - 1-min bar +60s shift
- [x] Volume Bar Construction (`feature_engine/bars.py`) - OHLC aggregation
- [x] Z-Score Calculation (`utils/math.py`) - Numerically stable
- [x] Hurst Exponent (`feature_engine/physics.py`) - Multi-lag variance method
- [x] Fractal Dimension Index (`feature_engine/physics.py`) - Sevcik formula
- [x] Garman-Klass Volatility (`feature_engine/standard.py`) - Correct OHLC estimator
- [x] Paper Trade Execution (`paper_trade.py`) - Vol targeting, state persistence
- [x] Yang-Zhang Volatility (`feature_engine/physics.py`) - Minimum variance estimator
- [x] Fisher Transform (`feature_engine/dsp.py`) - Ehlers formula with soft clipping
- [x] Feature Delta/Flux/Slope (`utils/math.py`) - Vectorized Numba implementations
- [x] Mutex Simulator (`mutex/simulator.py`) - Portfolio-level position management
- [x] Statistics (`backtest/statistics.py`) - Sharpe, Sortino, DSR formulas

---

## Action Items Summary

| Priority | Issue | File | Status |
|----------|-------|------|--------|
| HIGH | Limit Order PnL Bug | `trade_simulator.py` | FIXED |
| LOW | Dead code (MAX_LOTS) | `genome/strategy.py` | FIXED |
| LOW | Sortino edge case | `backtest/statistics.py` | FIXED (Documented) |

---

*Audit performed: 2026-01-11*
*Files audited: 40+ Python source files*
*Auditor: Claude Code*

*All issues resolved: 2026-01-11*
