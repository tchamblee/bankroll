# System Audit v2 - 2026-01-11
## Critical Errors & Alpha Refinement Opportunities

---

## Critical Issues

### 1. Walk-Forward Validation Feature Leakage
- **File:** `backtest/mixins/evaluation.py:134`
- **Status:** VERIFIED - NOT A BUG
- **Priority:** N/A

**Original Concern:**
Signals generated from `full_signal_matrix` computed over ALL data, then sliced for each fold.

**Verification Result (2026-01-11):**
This is NOT leakage. Rolling features (z-scores, correlations, slopes) use backward-looking windows - this is correct behavior. A z-score at bar 1000 with window 100 uses bars 900-1000, which is all PAST data.

Confirmed correct:
- No negative shifts (future-peeking) in feature_engine or utils/math.py
- Signal shift applied for next-bar execution (`signals[:-1]`)
- External data properly lagged (FRED +1d, COT +4d, intermarket +60s)

---

### 2. MeanReversionGene Direction/Placement Mismatch Risk
- **File:** `genome/factory.py:114-150, 349-374`
- **Status:** FIXED (2026-01-11)
- **Priority:** HIGH
- **Impact:** Inverted trade signals

**Description:**
MeanReversionGene has internal `direction` parameter ('long'/'short'), but when placed in a strategy's `long_genes` or `short_genes` list, the semantics may conflict:
- A MeanReversionGene with `direction='short'` in `long_genes` triggers on overbought conditions but generates a LONG signal.

**Fix Applied:**
Added `for_direction` parameter to `create_gene_from_pool()`. When creating genes for `long_genes`, pass `for_direction='long'`; for `short_genes`, pass `for_direction='short'`. MeanReversionGene now uses this direction instead of random selection.

---

### 3. Signal Safe-Entry Uses Future Horizon Knowledge
- **File:** `backtest/mixins/generation.py:53-54`
- **Status:** VERIFIED - NOT A BUG
- **Priority:** N/A

**Original Concern:**
Safe entry logic uses `horizon` to determine cutoff time, which seemed like "future knowledge."

**Verification Result (2026-01-11):**
This is NOT a bug. The `horizon` is a known strategy parameter (max hold time), not future information about any particular trade. The logic correctly prevents entering trades that couldn't complete before market close even in the worst case.

This is intentional conservative behavior that:
- Matches what you'd want in live trading
- Creates consistent backtest/live behavior
- Prevents overnight position risk

---

## Moderate Issues

### 4. COT Data Staleness Window Too Generous
- **File:** `feature_engine/cot.py:185`, `config.py:29`
- **Status:** FIXED (2026-01-11)
- **Priority:** MEDIUM

**Description:**
COT is weekly data. 30-day staleness meant potentially using 4-week-old positioning data.

**Fix Applied:**
- Reduced default staleness to 14 days (ensures data from last 2 reports max)
- Made configurable via `config.COT_STALENESS_DAYS`
- Uses `getattr(config, 'COT_STALENESS_DAYS', 14)` for backwards compatibility

---

### 5. Annualization Factor Not Empirically Validated
- **File:** `config.py:70`
- **Status:** FIXED (2026-01-11)
- **Priority:** MEDIUM

**Description:**
```python
ANNUALIZATION_FACTOR = 338363  # Approx 1342 bars/day * 252 days
```
Volume bars have variable frequency. This assumed constant may inflate/deflate Sharpe/Sortino ratios.

**Fix Applied:**
Calculated empirically: 234,461 bars / 160.9 trading days * 252 = 367,288. Updated config.py with validated value.

---

### 6. Stability Penalty Only Applied to Profitable Strategies
- **File:** `backtest/mixins/evaluation.py:76-77`
- **Status:** FIXED (2026-01-11)
- **Priority:** LOW

**Description:**
```python
if stability_ratio[j] > config.STABILITY_PENALTY_THRESHOLD and total_ret[j] > 0:
    final_sortino *= config.STABILITY_PENALTY_FACTOR
```
A strategy with negative returns and a huge single loss gets no penalty. Asymmetric treatment.

**Fix Applied:**
Changed stability_ratio to use `abs(total_ret)` in denominator, and removed the `total_ret > 0` condition. Now catches unstable strategies regardless of profitability.

---

### 7. Feature Selection Triangular Distribution Too Aggressive
- **File:** `genome/factory.py:107`
- **Status:** FIXED (2026-01-11)
- **Priority:** LOW

**Description:**
```python
idx = int(random.triangular(0, len(pool), 0))
```
Mode=0 causes top 10% of features to be selected ~50% of time, reducing exploration.

**Fix Applied:**
Changed mode from 0 to `len(pool) * 0.2` for gentler bias toward top features while maintaining diversity.

---

## Alpha Refinement Opportunities

### 8. Add Intraday Regime Feature
- **Status:** FIXED (2026-01-11)
- **Priority:** HIGH (Alpha)
- **Estimated Impact:** Medium

**Description:**
Markets transition from "London Open momentum" to "NY session mean-reversion" to "late-day chop." Current features are static within a day.

**Fix Applied:**
Added `intraday_regime` and `session_bucket` features to `feature_engine/seasonality.py`. Combines session bucket (0-4 for Asia/London/NY Overlap/NY Afternoon/Late) with rolling volatility percentile.

---

### 9. Add Asymmetric SL/TP by Direction
- **Status:** DEFERRED
- **Priority:** HIGH (Alpha)
- **Estimated Impact:** Medium-High
- **Reason:** Requires significant architectural changes across multiple files

**Description:**
EURUSD has structural skew - risk-off moves (USD strength) are faster/sharper than risk-on. Current symmetric SL/TP doesn't capture this.

**Implementation Design:**
1. **Strategy class** (`genome/strategy.py`):
   - Add `sl_long`, `sl_short`, `tp_long`, `tp_short` attributes
   - Default to existing `stop_loss_pct`/`take_profit_pct` for backwards compatibility
   - Update `to_dict()`, `from_dict()`, `get_hash()`, `__repr__()`
2. **GenomeFactory** (`genome/factory.py`):
   - Add separate mutation for directional barriers
   - Consider: 50% chance of symmetric (simpler), 50% chance of asymmetric
3. **Simulation mixin** (`backtest/mixins/simulation.py`):
   - Pass 4 barrier values instead of 2
4. **Trade simulator** (`backtest/trade_simulator.py`):
   - `calculate_barriers()` to accept direction-specific arrays
   - Use `signal > 0` to select long barriers, `signal < 0` for short barriers

---

### 10. Add Train Sortino Floor Filter
- **Status:** FIXED (2026-01-11)
- **Priority:** HIGH (Alpha)
- **Estimated Impact:** High (reduces false positives)

**Description:**
Tech_Fdi had train Sortino of 0.13 but made it to candidates. Strategies with poor training performance but good OOS are likely spurious.

**Fix Applied:**
- Added `MIN_HOF_SORTINO = 0.5` to config.py as early gate for HOF entry
- Updated `evolution/selection.py` to use this floor after early exploration phase
- Works as defense-in-depth before the final 0.9 filter in reporting.py

---

### 11. Add Volatility-Scaled Barriers
- **Status:** DEFERRED
- **Priority:** MEDIUM (Alpha)
- **Estimated Impact:** Medium
- **Reason:** Requires modification of Numba JIT functions (check_barrier_long/short)

**Description:**
Vol-targeting adapts position size at entry, but doesn't adapt within a trade. A position entered during low-vol that sees vol spike is over-risked.

**Implementation Design:**
1. Pass `current_atr` and `entry_atr` to barrier functions
2. Calculate `vol_ratio = current_atr / entry_atr`
3. If vol_ratio > 1.5, tighten SL by 20%: `effective_sl = sl_mult * 0.8`
4. Requires changes to `_simulate_batch_jit()` and `check_barrier_*()` in utils_trade.py

---

### 12. Add News Sentiment Decay Features
- **Status:** FIXED (2026-01-11)
- **Priority:** MEDIUM (Alpha)
- **Estimated Impact:** Medium

**Description:**
GDELT panic/sentiment features are static. News impact decays exponentially (half-life ~2-4 hours for FX).

**Fix Applied:**
Added `news_impact_decay_{2,4,8}h` features to `feature_engine/gdelt.py` using exponential weighted moving average with appropriate spans for each half-life.

---

### 13. Add Cross-Asset Momentum Confirmation
- **Status:** FIXED (2026-01-11)
- **Priority:** MEDIUM (Alpha)
- **Estimated Impact:** Medium

**Description:**
Intermarket features track correlation regimes but not momentum alignment. EURUSD long is higher-confidence when ES rallying AND ZN falling (risk-on).

**Fix Applied:**
Added `risk_on_momentum` feature to `feature_engine/intermarket.py`. Uses 50-bar returns of ES and ZN to compute risk-on/off score (+2/-2/0).

---

### 14. Add Autocorrelation Regime Gate
- **Status:** FIXED (2026-01-11)
- **Priority:** MEDIUM (Alpha)
- **Estimated Impact:** Medium

**Description:**
Markets alternate between trending (positive autocorr) and mean-reverting (negative autocorr). Current autocorr features are triggers, not regime gates.

**Fix Applied:**
Added `autocorr_regime_trending` and `autocorr_regime_reverting` binary features to `feature_engine/standard.py` (after autocorr_100 calculation). Returns 1.0 when in regime, 0.0 otherwise.

---

### 15. Develop Composite Profile Features
- **Status:** DEFERRED
- **Priority:** LOW (Alpha)
- **Estimated Impact:** Medium (per original reviewer suggestion)
- **Reason:** Requires significant new logic for intraday histogram tracking and multi-day aggregation

**Description:**
Current `profile.py` only calculates yesterday's market profile. Prop traders use:
- Developing Value Area (intraday)
- Weekly/Monthly composite profiles

**Implementation Design:**
1. **Developing Value Area** (complex - requires intraday state):
   - Track cumulative histogram within each day
   - Recalculate VPOC/VAH/VAL at each bar
   - Store `developing_vpoc`, `developing_vah`, `developing_val`

2. **Composite Profile** (simpler):
   - Aggregate histograms over multiple days (5-day, 20-day)
   - Store `weekly_vpoc`, `weekly_vah`, `weekly_val`, `monthly_*`

3. **Features to add:**
   - `dev_va_width`: Current developing value area width
   - `price_vs_dev_vpoc`: Current price vs developing VPOC
   - `weekly_in_value`: Binary - is price inside weekly value area

---

## Action Items Summary

| ID | Priority | Category | Issue/Opportunity | Status |
|----|----------|----------|-------------------|--------|
| 1 | HIGH | Bug | WFV feature leakage | VERIFIED (NOT A BUG) |
| 2 | HIGH | Bug | MeanReversionGene direction mismatch | FIXED |
| 3 | MEDIUM | Bug | Safe-entry future knowledge | VERIFIED (NOT A BUG) |
| 4 | MEDIUM | Bug | COT staleness threshold | FIXED |
| 5 | MEDIUM | Bug | Annualization factor validation | FIXED |
| 6 | LOW | Bug | Stability penalty asymmetry | FIXED |
| 7 | LOW | Bug | Feature selection bias | FIXED |
| 8 | HIGH | Alpha | Intraday regime feature | FIXED |
| 9 | HIGH | Alpha | Asymmetric SL/TP | DEFERRED |
| 10 | HIGH | Alpha | Train Sortino floor | FIXED |
| 11 | MEDIUM | Alpha | Volatility-scaled barriers | DEFERRED |
| 12 | MEDIUM | Alpha | News sentiment decay | FIXED |
| 13 | MEDIUM | Alpha | Cross-asset momentum | FIXED |
| 14 | MEDIUM | Alpha | Autocorrelation regime gate | FIXED |
| 15 | LOW | Alpha | Composite profile features | DEFERRED |

---

*Audit performed: 2026-01-11*
*All items processed: 2026-01-11*
