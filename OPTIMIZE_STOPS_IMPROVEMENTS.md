# optimize_stops.py Improvements

Audit Date: 2026-01-11

## Critical Priority

### 1. Fix Data Leakage in Selection
- **Status:** DONE
- **Location:** Lines 133, 166, 171, 177
- **Problem:** Selection and filtering uses test set performance. `min_sort` includes test sortino, and robustness filter checks `v_test['sortino'] > 0`. This is data leakage - shopping for variants that fit test.
- **Fix:** Use only train/val for selection and filtering. Report test as informational only.

### 2. Add Random Seed for Reproducibility
- **Status:** DONE
- **Location:** `__init__` method
- **Problem:** No seed initialization. While this script doesn't use randomness directly, adding seed ensures consistency with optimize_candidate.py pattern.
- **Fix:** Add optional seed parameter to `__init__`.

## Medium Priority

### 3. Make Robustness Thresholds Configurable
- **Status:** DONE
- **Location:** Line 171
- **Problem:** Hardcoded `v_train['sortino'] > 1.0 and v_val['sortino'] > 1.0` thresholds.
- **Fix:** Move to config.py as `OPTIMIZE_STOPS_MIN_SORTINO`.

### 4. Add Minimum Improvement Threshold
- **Status:** DONE
- **Location:** Selection logic
- **Problem:** Any tiny improvement over parent is accepted. Should require meaningful improvement to avoid noise.
- **Fix:** Use `config.OPTIMIZE_MIN_IMPROVEMENT` like optimize_candidate.py.

### 5. Add get_best_variant() Method
- **Status:** DONE
- **Location:** New method
- **Problem:** Only has `evaluate_and_report()` for CLI. No programmatic API for integration with other scripts.
- **Fix:** Add `get_best_variant()` that returns the best variant and its stats.

## Low Priority

### 6. Add Optional R:R Ratio Filter
- **Status:** DONE
- **Location:** `generate_grid()` method
- **Problem:** Currently allows TP < SL combinations (bad risk/reward ratio). Should optionally filter these.
- **Fix:** Add `min_rr_ratio` parameter to filter combinations where TP/SL < threshold.

### 7. Add Directional Stops Mode
- **Status:** DONE
- **Location:** New feature
- **Problem:** Strategy class supports directional barriers (sl_long, sl_short, tp_long, tp_short) but optimizer only sets symmetric stops.
- **Fix:** Add `--directional` mode that optimizes long/short stops separately.

### 8. Add Grid Size Warning
- **Status:** DONE
- **Location:** `generate_grid()` method
- **Problem:** Default grid = 361 variants. Large grids can be slow with no warning.
- **Fix:** Print estimated evaluation time or warn if grid > 500 variants.

---

## Progress Tracker

| # | Item | Priority | Status | Date Completed |
|---|------|----------|--------|----------------|
| 1 | Fix data leakage | Critical | DONE | 2026-01-11 |
| 2 | Add random seed | Critical | DONE | 2026-01-11 |
| 3 | Configurable thresholds | Medium | DONE | 2026-01-11 |
| 4 | Min improvement threshold | Medium | DONE | 2026-01-11 |
| 5 | get_best_variant() method | Medium | DONE | 2026-01-11 |
| 6 | R:R ratio filter | Low | DONE | 2026-01-11 |
| 7 | Directional stops mode | Low | DONE | 2026-01-11 |
| 8 | Grid size warning | Low | DONE | 2026-01-11 |
