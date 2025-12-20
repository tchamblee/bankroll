# Alpha Factory TODO List

## Phase 1: Feature Engineering (The "DNA") - **COMPLETE ✅**
- [x] Build `feature_engine.py` to ingest raw ticks.
- [x] Implement **Physics** features (Velocity, GK Volatility, Entropy, Skew).
- [x] Implement **Microstructure** features (Order Book Pressure, Price-Flow Correlation).
- [x] Implement **Regime Detection** features (Hurst, Stability Scoring).
- [x] Validate features via "Hunger Games" (IC, P-Value, Stability) across multiple horizons (30, 60, 90).
- [x] Audit code for "Prop Desk" accuracy and redundancy.

## Phase 2: The Genome Structure (The "Blueprint") - **COMPLETE ✅**
- [x] Define `Gene` class: Tuple of (Feature, Operator, Threshold).
- [x] Define `Strategy` class: Collection of Genes with logic (AND/OR).
- [x] Implement **Vectorized Translator**: Convert a `Strategy` object into a Pandas/Numpy Boolean Mask for high-speed testing.
- [x] Create `GenomeFactory` to generate random strategies based on the Survivor features.
- [x] Implement **Gated Logic** (Regime Filter + Trigger) for robust strategy structure.

## Phase 3: The Engine (Backtester) - **COMPLETE ✅**
- [x] Build a high-performance, vectorized backtester using `polars` or `numpy`.
- [x] Implement `evaluate_strategy(strategy, data)` function.
- [x] Define Fitness Metrics (Sharpe Ratio, Profit Factor, Sortino Ratio).
- [x] Ensure execution speed allows for thousands of iterations per minute (Achieved ~50k/min).

## Phase 4: The Evolutionary Loop - **COMPLETE ✅**
- [x] Implement **Population Initialization** (Generate 20,000 random strategies).
- [x] Implement **Fitness Function** (Run backtest, calculate score).
- [x] Implement **Selection** (Tournament selection or Roulette wheel to pick parents).
- [x] Implement **Crossover** (Combine logic of two parents).
- [x] Implement **Mutation** (Randomly alter parameters or operators).
- [x] Build the main `evolution_loop` (Generations control).
- [x] Implement **Walk-Forward Analysis** (Train/Validation/Test split).
- [x] Add **Visualization Suite** (Equity Curves, DNA Analysis).

## Phase 5: Production & Scale (Next Steps)
- [ ] **Parallelization:** Run Islands on multiple cores/machines.
- [ ] **Portfolio Construction:** Combine Apex Strategies into a non-correlated portfolio.
- [ ] **Live Execution:** Connect the Signal Engine to IBKR API.
