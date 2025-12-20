# Alpha Factory TODO List

## Phase 1: Feature Engineering (The "DNA") - **COMPLETE ✅**
- [x] Build `feature_engine.py` to ingest raw ticks.
- [x] Implement **Physics** features (Velocity, GK Volatility, Entropy, Skew).
- [x] Implement **Microstructure** features (Order Book Pressure, Price-Flow Correlation).
- [x] Implement **Regime Detection** features (Hurst, Stability Scoring).
- [x] Validate features via "Hunger Games" (IC, P-Value, Stability) across multiple horizons (30, 60, 90).
- [x] Audit code for "Prop Desk" accuracy and redundancy.

## Phase 2: The Genome Structure (The "Blueprint") - **IN PROGRESS 🚀**
- [ ] Define `Gene` class: Tuple of (Feature, Operator, Threshold).
- [ ] Define `Strategy` class: Collection of Genes with logic (AND/OR).
- [ ] Implement **Vectorized Translator**: Convert a `Strategy` object into a Pandas/Numpy Boolean Mask for high-speed testing.
- [ ] Create `GenomeFactory` to generate random strategies based on the Survivor features.

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