import random
import config
from genome import Strategy

def crossover_strategies(p1: Strategy, p2: Strategy) -> Strategy:
    child = Strategy(name=f"Child_{random.randint(1000,9999)}")

    n_long = random.randint(config.GENE_COUNT_MIN, config.GENE_COUNT_MAX)
    n_short = random.randint(config.GENE_COUNT_MIN, config.GENE_COUNT_MAX)

    combined_long = p1.long_genes + p2.long_genes
    combined_short = p1.short_genes + p2.short_genes

    # Sample without replacement if possible, else with replacement if not enough genes (unlikely given min counts)
    child.long_genes = [g.copy() for g in random.sample(combined_long, min(len(combined_long), n_long))]
    child.short_genes = [g.copy() for g in random.sample(combined_short, min(len(combined_short), n_short))]

    # Inherit Params (including directional barriers)
    donor = p1 if random.random() < 0.5 else p2
    child.stop_loss_pct = donor.stop_loss_pct
    child.take_profit_pct = donor.take_profit_pct
    child.limit_dist_atr = donor.limit_dist_atr
    child.sl_long = donor.sl_long
    child.sl_short = donor.sl_short
    child.tp_long = donor.tp_long
    child.tp_short = donor.tp_short

    child.recalculate_concordance()
    return child

def mutate_strategy(strategy: Strategy, available_features: list):
    """Mutates a strategy in place."""
    # Gene Mutation
    genes_to_mutate = strategy.long_genes + strategy.short_genes
    if genes_to_mutate:
        target_genes = random.sample(genes_to_mutate, min(len(genes_to_mutate), 2))
        for g in target_genes:
            g.mutate(available_features)

    # Param Mutation (20% chance)
    if random.random() < 0.20:
        # Include directional barrier options
        param_type = random.choice(['SL', 'TP', 'LIM', 'SL_LONG', 'SL_SHORT', 'TP_LONG', 'TP_SHORT', 'ASYM_TOGGLE'])
        if param_type == 'SL':
            strategy.stop_loss_pct = random.choice(config.STOP_LOSS_OPTIONS)
        elif param_type == 'TP':
            strategy.take_profit_pct = random.choice(config.TAKE_PROFIT_OPTIONS)
        elif param_type == 'LIM':
            strategy.limit_dist_atr = random.choice(config.LIMIT_DIST_OPTIONS)
        elif param_type == 'SL_LONG':
            strategy.sl_long = random.choice(config.STOP_LOSS_OPTIONS)
        elif param_type == 'SL_SHORT':
            strategy.sl_short = random.choice(config.STOP_LOSS_OPTIONS)
        elif param_type == 'TP_LONG':
            strategy.tp_long = random.choice(config.TAKE_PROFIT_OPTIONS)
        elif param_type == 'TP_SHORT':
            strategy.tp_short = random.choice(config.TAKE_PROFIT_OPTIONS)
        elif param_type == 'ASYM_TOGGLE':
            # Toggle between symmetric and asymmetric
            if strategy.is_asymmetric():
                # Reset to symmetric
                strategy.sl_long = strategy.sl_short = strategy.tp_long = strategy.tp_short = None
            else:
                # Make asymmetric
                strategy.sl_long = random.choice(config.STOP_LOSS_OPTIONS)
                strategy.sl_short = random.choice(config.STOP_LOSS_OPTIONS)
                strategy.tp_long = random.choice(config.TAKE_PROFIT_OPTIONS)
                strategy.tp_short = random.choice(config.TAKE_PROFIT_OPTIONS)

    strategy.cleanup()
    strategy.recalculate_concordance()
