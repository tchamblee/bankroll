# Alpha Factory TODO List

## Phase 1: Feature Engineering (The "DNA")
- [ ] Build `feature_engine.py` to ingest raw ticks.
- [ ] Implement **Velocity** features (Rate of change: 1s, 1m, 1h).
- [ ] Implement **Efficiency** features (Kaufman Efficiency Ratio: Displacement / Path Length).
- [ ] Implement **Cyclicality** features (Hilbert Transform / Autocorrelation).
- [ ] Implement **Regime Detection** features (Volatility clusters, Trend strength).
- [ ] Normalize/Scale features for consistent "Gene" usage.

## Phase 2: The Genome Structure
- [ ] Define a standardized `Strategy` class/structure.
- [ ] Create a "Gene" representation (Feature, Operator, Threshold).
- [ ] Implement logic to translate a `Strategy` object into executable code/logic.

## Phase 3: The Engine (Backtester)
- [ ] Build a high-performance, vectorized backtester using `polars` or `numpy`.
- [ ] Implement `evaluate_strategy(strategy, data)` function.
- [ ] Define Fitness Metrics (Sharpe Ratio, Profit Factor, Sortino Ratio).
- [ ] Ensure execution speed allows for thousands of iterations per minute.

## Phase 4: The Evolutionary Loop
- [ ] Implement **Population Initialization** (Generate 1,000 random strategies).
- [ ] Implement **Fitness Function** (Run backtest, calculate score).
- [ ] Implement **Selection** (Tournament selection or Roulette wheel to pick parents).
- [ ] Implement **Crossover** (Combine logic of two parents).
- [ ] Implement **Mutation** (Randomly alter parameters or operators).
- [ ] Build the main `evolution_loop` (Generations control).
