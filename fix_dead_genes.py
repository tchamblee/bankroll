import argparse
import json
import os
import random
import sys
import numpy as np
import pandas as pd
from typing import List

# Add project root to path
sys.path.append(os.getcwd())

import config
from backtest.engine import BacktestEngine
from backtest.statistics import calculate_sortino_ratio
from genome.strategy import Strategy
from genome.factory import GenomeFactory

# --- HELPERS ---

def load_strategies_from_file(filepath):
    """Manually loads strategies to avoid import issues or path constraints."""
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            data = [data]
            
        strategies = []
        for d in data:
            try:
                s = Strategy.from_dict(d)
                s.data = d # Keep original data just in case
                strategies.append(s)
            except Exception as e:
                print(f"⚠️ Failed to load strategy: {e}")
                
        return strategies
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return []

def load_candidates(filepath="output/strategies/candidates.json"):
    if not os.path.exists(filepath):
        print(f"❌ Candidates file not found: {filepath}")
        sys.exit(1)
    return load_strategies_from_file(filepath)

def save_candidates(strategies, filepath="output/strategies/candidates.json"):
    data = [s.to_dict() for s in strategies]
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 Saved {len(strategies)} strategies to {filepath}")

def run_backtest_side(engine, strategy, side='long', dataset='train'):
    """
    Runs a backtest on a specific side (long/short) and specific dataset.
    Returns (trades_count, total_pnl)
    """
    # Define indices
    if dataset == 'train':
        start, end = 0, engine.train_idx
    elif dataset == 'validation':
        start, end = engine.train_idx, engine.val_idx
    else:
        start, end = 0, len(engine.raw_data) # Full

    # 1. Generate Signals
    # Strategy generate_signal uses 'context' which engine constructs. 
    # But engine.generate_signal_matrix handles this for population.
    # It creates the JIT context internally.
    
    # We use generate_signal_matrix for consistency
    signals_global = engine.generate_signal_matrix([strategy], horizon=strategy.horizon) # (N, 1)
    
    # CRITICAL: Shift signals by 1 to enforce Next-Open execution
    signals_global = np.roll(signals_global, 1, axis=0)
    signals_global[0] = 0 
    
    # Slice
    signals_seg = signals_global[start:end]
    
    # Filter Side
    if side == 'long':
        signals_seg[signals_seg < 0] = 0
    elif side == 'short':
        signals_seg[signals_seg > 0] = 0
        
    if np.sum(np.abs(signals_seg)) == 0:
        return 0, 0.0

    # Prepare simulation data
    prices = engine.open_vec[start:end]
    times = engine.times_vec.iloc[start:end] if hasattr(engine.times_vec, 'iloc') else engine.times_vec[start:end]
    highs = engine.high_vec[start:end]
    lows = engine.low_vec[start:end]
    atr = engine.atr_vec[start:end] if engine.atr_vec is not None else None
    
    # Run Simulation
    net_returns, trades_count = engine.run_simulation_batch(
        signals_seg, [strategy], prices, times, 
        time_limit=strategy.horizon, highs=highs, lows=lows, atr=atr
    )
    
    total_ret = np.sum(net_returns)
    return int(trades_count[0]), total_ret

def evaluate_full_strategy(engine, strategy, dataset='validation'):
    """
    Runs a full (Long+Short) backtest on the specified dataset and returns Sortino Ratio.
    """
    if dataset == 'train':
        start, end = 0, engine.train_idx
    elif dataset == 'validation':
        start, end = engine.train_idx, engine.val_idx
    else:
        start, end = 0, len(engine.raw_data)

    signals_global = engine.generate_signal_matrix([strategy], horizon=strategy.horizon)
    signals_global = np.roll(signals_global, 1, axis=0)
    signals_global[0] = 0 
    
    signals_seg = signals_global[start:end]
    
    if np.sum(np.abs(signals_seg)) == 0:
        return 0.0
        
    prices = engine.open_vec[start:end]
    times = engine.times_vec.iloc[start:end] if hasattr(engine.times_vec, 'iloc') else engine.times_vec[start:end]
    highs = engine.high_vec[start:end]
    lows = engine.low_vec[start:end]
    atr = engine.atr_vec[start:end] if engine.atr_vec is not None else None
    
    net_returns, _ = engine.run_simulation_batch(
        signals_seg, [strategy], prices, times, 
        time_limit=strategy.horizon, highs=highs, lows=lows, atr=atr
    )
    
    # Calculate Sortino
    return calculate_sortino_ratio(net_returns.flatten())


def main():
    parser = argparse.ArgumentParser(description="Fix dead genes in a strategy.")
    parser.add_argument("name", help="Name of the strategy to fix")
    parser.add_argument("--candidates_file", default="output/strategies/candidates.json", help="Path to candidates file")
    
    args = parser.parse_args()
    
    print(f"🔍 Searching for strategy '{args.name}' in {args.candidates_file}...")
    candidates = load_candidates(args.candidates_file)
    
    target_idx = -1
    target_strat = None
    for i, s in enumerate(candidates):
        if s.name == args.name:
            target_idx = i
            target_strat = s
            break
            
    if target_strat is None:
        print(f"❌ Strategy '{args.name}' not found.")
        sys.exit(1)

    print(f"✅ Found: {target_strat.name}")
    
    # Initialize Engine
    print("⚙️ Initializing Backtest Engine...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    bars_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    engine = BacktestEngine(bars_df)
    
    # Initialize GenomeFactory
    print("🧬 Initializing GenomeFactory...")
    survivors_file = None
    possible_files = [f for f in os.listdir("output/features") if f.startswith("survivors_") and f.endswith(".json")]
    if possible_files:
        survivors_file = os.path.join("output/features", possible_files[0])
        print(f"   - Using survivors file: {survivors_file}")
    
    factory = GenomeFactory(survivors_file=survivors_file)
    factory.set_stats(engine.raw_data)
    feature_list = list(factory.feature_stats.keys())

    # 1. Audit Original (TRAIN)
    print("\n📊 Auditing Original Strategy (Train Set)...")
    l_count, l_pnl = run_backtest_side(engine, target_strat, 'long', 'train')
    s_count, s_pnl = run_backtest_side(engine, target_strat, 'short', 'train')
    
    print(f"   - Longs: {l_count} trades, PnL: {l_pnl:.2%}")
    print(f"   - Shorts: {s_count} trades, PnL: {s_pnl:.2%}")
    
    dead_long = l_count == 0
    dead_short = s_count == 0
    
    if not dead_long and not dead_short:
        print("✅ Strategy is already bidirectional in Train. No fix needed.")
        sys.exit(0)
        
    side_to_fix = "long" if dead_long else "short"
    print(f"\n⚠️ Dead {side_to_fix.upper()} side detected. Preparing candidates...")

    # --- Candidate A: Stripped Strategy ---
    stripped_strat = target_strat.copy()
    if dead_long:
        stripped_strat.long_genes = []
        stripped_strat.name += "_ShortOnly"
    else:
        stripped_strat.short_genes = []
        stripped_strat.name += "_LongOnly"
    stripped_strat.recalculate_concordance()
    
    # Evaluate Stripped (Validation)
    stripped_sortino = evaluate_full_strategy(engine, stripped_strat, 'validation')
    print(f"   📉 Stripped Strategy Sortino (Val): {stripped_sortino:.2f}")

    # --- Candidate B: Awakened Strategy ---
    best_mutant = None
    best_score = -999999
    
    MUTATION_ATTEMPTS = 50
    print(f"🧪 Generating {MUTATION_ATTEMPTS} mutants to awaken {side_to_fix} side...")
    
    for i in range(MUTATION_ATTEMPTS):
        mutant = target_strat.copy()
        mutant.name = f"{target_strat.name}_Mutant_{i}"
        
        # Determine gene list to modify
        if side_to_fix == "long":
            gene_list = mutant.long_genes
        else:
            gene_list = mutant.short_genes
            
        dice = random.random()
        
        if dice < 0.30:
            # Fresh Start
            new_genes = []
            count = max(len(gene_list), 2)
            new_genes.append(factory.create_gene_from_pool(factory.regime_pool))
            for _ in range(count - 1):
                new_genes.append(factory.create_gene_from_pool(factory.trigger_pool))
            
            if side_to_fix == "long":
                mutant.long_genes = new_genes
            else:
                mutant.short_genes = new_genes
                
        else:
            # Heavy Mutation
            for g in gene_list:
                if hasattr(g, 'mutate'):
                     g.mutate(feature_list)

        mutant.recalculate_concordance()
        
        # Test Mutant (Train)
        m_count, m_pnl = run_backtest_side(engine, mutant, side_to_fix, 'train')
        
        # Criteria: At least 15 trades and Positive PnL in Train
        # If it passes Train, we double check Validation to ensure it's not overfitting Train
        if m_count >= 15 and m_pnl > 0:
            # Check Validation robustness
            v_count, v_pnl = run_backtest_side(engine, mutant, side_to_fix, 'validation')
            
            if v_count >= 5 and v_pnl > -0.02: # Relaxed validation check: Just don't be catastrophic
                score = m_pnl + v_pnl
                print(f"   ✨ Candidate {i}: Train({m_count}, {m_pnl:.2%}) | Val({v_count}, {v_pnl:.2%})")
                
                if score > best_score:
                    best_score = score
                    best_mutant = mutant

    # --- Final Decision ---
    final_strategy = None
    
    if best_mutant:
        print(f"\n⚖️ Comparing Awakened vs Stripped...")
        awakened_sortino = evaluate_full_strategy(engine, best_mutant, 'validation')
        print(f"   - Awakened Sortino (Val): {awakened_sortino:.2f}")
        print(f"   - Stripped Sortino (Val): {stripped_sortino:.2f}")
        
        if stripped_sortino > awakened_sortino:
            print(f"   👉 Stripped Strategy wins! (Higher Sortino)")
            final_strategy = stripped_strat
        else:
            print(f"   👉 Awakened Strategy wins! (Awakened {side_to_fix.upper()} side)")
            final_strategy = best_mutant
            final_strategy.name = target_strat.name + "_Awakened"
    else:
        print(f"\n❌ FAILED to awaken {side_to_fix.upper()} side.")
        print("   👉 Defaulting to Stripped Strategy.")
        final_strategy = stripped_strat

    # 3. Action C: Insert
    print(f"\n💾 Inserting new strategy into Candidate List...")
    candidates.append(final_strategy)
    
    save_candidates(candidates, args.candidates_file)
    print("✅ Done.")

if __name__ == "__main__":
    main()