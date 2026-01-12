import numpy as np
import config
from collections import Counter


def extract_features_from_strategy(strategy):
    """
    Extracts all feature names used by a strategy's genes.
    Returns a set of feature strings.
    """
    features = set()
    all_genes = list(strategy.long_genes) + list(strategy.short_genes)

    for gene in all_genes:
        if hasattr(gene, 'feature') and gene.feature:
            features.add(gene.feature)
        if hasattr(gene, 'regime_feature') and gene.regime_feature:
            features.add(gene.regime_feature)
        if hasattr(gene, 'feature_left') and gene.feature_left:
            features.add(gene.feature_left)
        if hasattr(gene, 'feature_right') and gene.feature_right:
            features.add(gene.feature_right)
        if hasattr(gene, 'feature_a') and gene.feature_a:
            features.add(gene.feature_a)
        if hasattr(gene, 'feature_b') and gene.feature_b:
            features.add(gene.feature_b)
        if hasattr(gene, 'feature_short') and gene.feature_short:
            features.add(gene.feature_short)
        if hasattr(gene, 'feature_long') and gene.feature_long:
            features.add(gene.feature_long)

    return features


def extract_gene_types_from_strategy(strategy):
    """
    Extracts all gene types used by a strategy.
    Returns a list of gene type strings.
    """
    gene_types = []
    all_genes = list(strategy.long_genes) + list(strategy.short_genes)

    for gene in all_genes:
        if hasattr(gene, 'type'):
            gene_types.append(gene.type)

    return gene_types


def calculate_exploration_bonus(candidate, hof_features, hof_gene_type_counts, total_hof_genes):
    """
    Calculates a fitness bonus for using novel features or underrepresented gene types.

    Args:
        candidate: Strategy object
        hof_features: Set of all features used in current HOF
        hof_gene_type_counts: Counter of gene types in current HOF
        total_hof_genes: Total number of genes in HOF

    Returns:
        Float bonus to add to fitness (0.0 to 0.5)
    """
    bonus = 0.0

    # 1. Novel Feature Bonus (up to 0.3)
    # Reward strategies that use features not yet in HOF
    cand_features = extract_features_from_strategy(candidate)
    if cand_features:
        novel_features = cand_features - hof_features
        novelty_ratio = len(novel_features) / len(cand_features)
        bonus += novelty_ratio * 0.3  # Max 0.3 if 100% novel features

    # 2. Underrepresented Gene Type Bonus (up to 0.2)
    # Reward strategies using gene types that are rare in HOF
    cand_gene_types = extract_gene_types_from_strategy(candidate)
    if cand_gene_types and total_hof_genes > 0:
        # Calculate "rarity score" for each gene type
        rarity_scores = []
        for gtype in cand_gene_types:
            count = hof_gene_type_counts.get(gtype, 0)
            # Rarity = 1 - (proportion in HOF). Novel types get rarity = 1.0
            if total_hof_genes > 0:
                rarity = 1.0 - (count / total_hof_genes)
            else:
                rarity = 1.0
            rarity_scores.append(rarity)

        avg_rarity = sum(rarity_scores) / len(rarity_scores)
        bonus += avg_rarity * 0.2  # Max 0.2 if all gene types are novel

    return bonus


def update_hall_of_fame(hall_of_fame, backtester, candidates, gen, max_gens=100, horizon=None):
    """
    Updates the Hall of Fame with new candidates.
    Keeps all strategies that pass fitness and expectancy thresholds.
    """
    if not candidates: return
    
    limit_idx = backtester.val_idx
    cand_signals_full = backtester.generate_signal_matrix(candidates)
    cand_signals = cand_signals_full[:limit_idx]
    
    # FIX: Shift signals for Next-Open Execution (Prevent Peeking)
    cand_signals = np.vstack([np.zeros((1, cand_signals.shape[1]), dtype=cand_signals.dtype), cand_signals[:-1]])
    
    # Expectancy Filter (Churn Prevention)
    rets_batch, trades_batch = backtester.run_simulation_batch(
        cand_signals, 
        candidates, 
        backtester.open_vec[:limit_idx], 
        backtester.times_vec[:limit_idx],
        time_limit=horizon,
        highs=backtester.high_vec[:limit_idx],
        lows=backtester.low_vec[:limit_idx],
        atr=backtester.atr_vec[:limit_idx]
    )

    phase_1 = max(5, int(max_gens * 0.30))
    phase_2 = max(10, int(max_gens * 0.60))

    # --- BUILD HOF FEATURE/GENE TYPE INVENTORY ---
    # Used to calculate exploration bonus for novel strategies
    hof_features = set()
    hof_gene_type_counts = Counter()
    total_hof_genes = 0

    for item in hall_of_fame:
        strat = item['strat']
        hof_features.update(extract_features_from_strategy(strat))
        gene_types = extract_gene_types_from_strategy(strat)
        hof_gene_type_counts.update(gene_types)
        total_hof_genes += len(gene_types)

    # Structural Deduplication Set
    # Assume 'hall_of_fame' contains dicts with 'strat' key
    existing_hashes = set()
    for item in hall_of_fame:
        if hasattr(item['strat'], 'get_hash'):
            existing_hashes.add(item['strat'].get_hash())

    for i, cand in enumerate(candidates):
        # 1. Structural Check
        cand_hash = cand.get_hash()
        if cand_hash in existing_hashes:
            continue

        n_trades = trades_batch[i]
        total_ret = np.sum(rets_batch[:, i])

        # HOF Entry fitness thresholds (train Sortino)
        # Early: Allow exploration. Mid: Require MIN_HOF_SORTINO. Late: Same floor.
        # This prevents wasting resources on low-quality strategies that will fail final filter anyway.
        if gen < phase_1:
            min_fitness = -10.0  # Allow exploration
        else:
            min_fitness = getattr(config, 'MIN_HOF_SORTINO', 0.5)

        if cand.fitness < min_fitness:
            continue

        if gen < phase_1:
            thresh = -0.0005 # Allow loss (-5bps) in early game
        elif gen < phase_2:
            thresh = 0.0 # Breakeven
        else:
            thresh = 0.00005  # 0.5 bps

        if n_trades > 0:
            avg_ret = total_ret / n_trades
            if avg_ret < thresh:
                continue
        else:
            continue # No trades = No alpha

        cand_sig = cand_signals[:, i]
        cand_fit = cand.fitness

        # --- EXPLORATION BONUS ---
        # Reward strategies using novel features or underrepresented gene types
        # This prevents convergence on a single dominant pattern
        exploration_bonus = calculate_exploration_bonus(
            cand, hof_features, hof_gene_type_counts, total_hof_genes
        )
        adjusted_fit = cand_fit + exploration_bonus

        hall_of_fame.append({
            'strat': cand,
            'fit': adjusted_fit,  # Use adjusted fitness for ranking
            'base_fit': cand_fit,  # Keep original for reference
            'exploration_bonus': exploration_bonus,
            'sig': cand_sig,
            'gen': gen
        })
        existing_hashes.add(cand_hash) # Mark as seen

        # Update HOF inventory for subsequent candidates in this batch
        cand_features = extract_features_from_strategy(cand)
        hof_features.update(cand_features)
        cand_gene_types = extract_gene_types_from_strategy(cand)
        hof_gene_type_counts.update(cand_gene_types)
        total_hof_genes += len(cand_gene_types)

    # Sort HOF by Adjusted Fitness (includes exploration bonus)
    hall_of_fame.sort(key=lambda x: x['fit'], reverse=True)
    
    # Cap Size
    HOF_CAP = 300
    if len(hall_of_fame) > HOF_CAP:
        del hall_of_fame[HOF_CAP:]
