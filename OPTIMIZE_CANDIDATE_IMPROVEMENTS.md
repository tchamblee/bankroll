# optimize_candidate.py Improvements

Audit Date: 2026-01-11

## Critical Priority

### 1. Fix Data Leakage in evaluate_and_report()
- **Status:** DONE
- **Location:** Lines 346-358
- **Problem:** The `evaluate_and_report()` method uses test set performance as a selection criterion, shopping around for variants that fit the test set best. This is data leakage.
- **Fix:** Change selection logic to use only train/val for selection, then report test as informational only (matching `get_best_variant()` behavior).
- **Code Change:**
  ```python
  # BEFORE (leakage):
  train_ok = res['train']['sortino'] >= parent['train']['sortino']
  val_ok = res['val']['sortino'] >= parent['val']['sortino']
  test_ok = res['test']['sortino'] >= parent['test']['sortino']  # BAD

  # AFTER (no leakage):
  train_ok = res['train']['sortino'] >= parent['train']['sortino']
  val_ok = res['val']['sortino'] >= parent['val']['sortino']
  # Test is reported but not used for selection
  ```

### 2. Re-evaluate After Stop Optimization
- **Status:** DONE
- **Location:** Lines 280-287
- **Problem:** After `optimize_stops()` changes SL/TP, the returned `best_stats` contains metrics from BEFORE the stop change. Caller receives misleading/stale statistics.
- **Fix:** Re-run evaluation after stop optimization and update stats, or return a flag indicating stats need refresh.
- **Code Change:**
  ```python
  # After setting new stops, re-evaluate:
  if (best_sl, best_tp) != (best_candidate.stop_loss_pct, best_candidate.take_profit_pct):
      best_candidate.stop_loss_pct = best_sl
      best_candidate.take_profit_pct = best_tp
      # Re-evaluate with new stops
      final_train = self.backtester.evaluate_population([best_candidate], set_type='train', time_limit=self.horizon)
      final_val = self.backtester.evaluate_population([best_candidate], set_type='validation', time_limit=self.horizon)
      final_test = self.backtester.evaluate_population([best_candidate], set_type='test', time_limit=self.horizon)
      # Update best_stats with fresh values
  ```

### 3. Add Random Seed for Reproducibility
- **Status:** DONE
- **Location:** Top of file / __init__
- **Problem:** `random.choice()` and `random.uniform()` used without seeding. Two runs on same strategy produce different results, making debugging and comparison impossible.
- **Fix:** Add optional seed parameter and set it at initialization.
- **Code Change:**
  ```python
  def __init__(self, target_name, source_file=None, horizon=180, strategy_dict=None,
               backtester=None, data=None, verbose=True, seed=None):
      # ... existing code ...
      if seed is not None:
          random.seed(seed)
          np.random.seed(seed)
      else:
          # Default reproducible seed based on strategy name
          random.seed(hash(target_name) % 2**32)
          np.random.seed(hash(target_name) % 2**32)
  ```

## Medium Priority

### 4. Add Complexity Penalty to Selection
- **Status:** DONE
- **Location:** `get_best_variant()` method, around line 230
- **Problem:** A 4-gene variant beating a 2-gene variant by 0.01 sortino wins, even though the simpler strategy is likely more robust. No preference for parsimony.
- **Fix:** Apply complexity penalty when comparing variants. Use existing `COMPLEXITY_PENALTY_PER_GENE` from config.
- **Code Change:**
  ```python
  # When calculating fitness:
  n_genes = len(variant.long_genes) + len(variant.short_genes)
  complexity_penalty = n_genes * config.COMPLEXITY_PENALTY_PER_GENE
  v_fitness = min(v_train['sortino'], v_val['sortino']) - complexity_penalty

  # Parent gets same treatment:
  parent_n_genes = len(self.parent_strategy.long_genes) + len(self.parent_strategy.short_genes)
  parent_penalty = parent_n_genes * config.COMPLEXITY_PENALTY_PER_GENE
  parent_fitness = min(parent_train['sortino'], parent_val['sortino']) - parent_penalty
  ```

### 5. Make Stop Grid Configurable
- **Status:** DONE
- **Location:** `optimize_stops()` method, lines 139-140
- **Problem:** Hardcoded SL/TP grid may not suit all strategies or market conditions.
- **Fix:** Move to config.py or accept as parameters.
- **Code Change:**
  ```python
  # In config.py:
  OPTIMIZE_SL_OPTIONS = [1.0, 1.5, 2.0, 2.5, 3.0]
  OPTIMIZE_TP_OPTIONS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]

  # In optimize_stops():
  sl_options = getattr(config, 'OPTIMIZE_SL_OPTIONS', [1.0, 1.5, 2.0, 2.5, 3.0])
  tp_options = getattr(config, 'OPTIMIZE_TP_OPTIONS', [2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
  ```

### 6. Make Jitter Range Configurable
- **Status:** DONE
- **Location:** `generate_variants()` method, lines 107, 112
- **Problem:** 5% jitter hardcoded. May be too small for some parameters, too large for others.
- **Fix:** Add config parameter or method argument.
- **Code Change:**
  ```python
  # In config.py:
  OPTIMIZE_JITTER_PCT = 0.05  # 5%

  # In generate_variants():
  jitter = config.OPTIMIZE_JITTER_PCT
  g.threshold *= random.uniform(1 - jitter, 1 + jitter)
  g.window = max(2, int(g.window * random.uniform(1 - jitter, 1 + jitter)))
  ```

## Low Priority

### 7. Add Concordance Increase Variant
- **Status:** DONE
- **Location:** `generate_variants()` method, after line 95
- **Problem:** Only tries lowering concordance ("Relaxed"). Raising concordance (stricter agreement) could produce more selective, higher-quality signals.
- **Fix:** Add "Strict" variant that increases concordance.
- **Code Change:**
  ```python
  # After Relaxation block:
  # 2b. Strictness - Higher Concordance (more agreement required)
  max_concordance = max(len(self.parent_strategy.long_genes), len(self.parent_strategy.short_genes))
  if self.parent_strategy.min_concordance < max_concordance:
      variant = copy.deepcopy(self.parent_strategy)
      variant.name = f"{self.target_name}_Strict"
      variant.min_concordance += 1
      self.variants.append(variant)
  ```

### 8. Add Gene Expansion Variants
- **Status:** DONE
- **Location:** `generate_variants()` method
- **Problem:** Only removes genes (simplification) and mutates existing ones. Never tries adding genes from the available pool.
- **Fix:** Add variants that include one additional gene.
- **Code Change:**
  ```python
  # 5. Expansion - Add 1 random gene (test if more complexity helps)
  for i in range(5):  # Try 5 expansions
      variant = copy.deepcopy(self.parent_strategy)
      variant.name = f"{self.target_name}_Expand_{i}"
      new_gene = self.factory.create_random_gene()
      if random.random() < 0.5:
          variant.long_genes.append(new_gene)
      else:
          variant.short_genes.append(new_gene)
      variant.recalculate_concordance()
      self.variants.append(variant)
  ```

### 9. Add Minimum Improvement Threshold
- **Status:** DONE
- **Location:** `get_best_variant()` method, around line 238
- **Problem:** Sortino 1.51 vs 1.50 is likely noise, not real improvement. No significance threshold.
- **Fix:** Require minimum improvement margin (e.g., 5% relative or 0.1 absolute).
- **Code Change:**
  ```python
  # In config.py:
  OPTIMIZE_MIN_IMPROVEMENT = 0.05  # 5% relative improvement required

  # In get_best_variant():
  improvement_threshold = parent_fitness * config.OPTIMIZE_MIN_IMPROVEMENT
  improved = v_fitness > (parent_fitness + improvement_threshold)
  ```

### 10. Consider Walk-Forward Validation
- **Status:** DONE
- **Location:** Overall architecture
- **Problem:** Single train/val/test split may be lucky. Walk-forward or CPCV would be more robust.
- **Fix:** This is a larger architectural change. Consider integrating with existing CPCV infrastructure in backtest module.
- **Notes:** Low priority because get_best_variant() already uses train+val for selection and test as gatekeeper, which is reasonably robust for now.

---

## Progress Tracker

| # | Item | Priority | Status | Date Completed |
|---|------|----------|--------|----------------|
| 1 | Fix data leakage | Critical | DONE | 2026-01-11 |
| 2 | Re-eval after stops | Critical | DONE | 2026-01-11 |
| 3 | Add random seed | Critical | DONE | 2026-01-11 |
| 4 | Complexity penalty | Medium | DONE | 2026-01-11 |
| 5 | Configurable stop grid | Medium | DONE | 2026-01-11 |
| 6 | Configurable jitter | Medium | DONE | 2026-01-11 |
| 7 | Concordance increase | Low | DONE | 2026-01-11 |
| 8 | Gene expansion | Low | DONE | 2026-01-11 |
| 9 | Min improvement threshold | Low | DONE | 2026-01-11 |
| 10 | Walk-forward validation | Low | DONE | 2026-01-11 |
