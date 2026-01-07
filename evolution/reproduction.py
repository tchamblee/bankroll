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
    
    # Inherit Params
    if random.random() < 0.5:
        child.stop_loss_pct = p1.stop_loss_pct
        child.take_profit_pct = p1.take_profit_pct
        child.limit_dist_atr = p1.limit_dist_atr
    else:
        child.stop_loss_pct = p2.stop_loss_pct
        child.take_profit_pct = p2.take_profit_pct
        child.limit_dist_atr = p2.limit_dist_atr

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
        param_type = random.choice(['SL', 'TP', 'LIM'])
        if param_type == 'SL':
            strategy.stop_loss_pct = random.choice(config.STOP_LOSS_OPTIONS)
        elif param_type == 'TP':
            strategy.take_profit_pct = random.choice(config.TAKE_PROFIT_OPTIONS)
        elif param_type == 'LIM':
            strategy.limit_dist_atr = random.choice(config.LIMIT_DIST_OPTIONS)

    strategy.cleanup()
    strategy.recalculate_concordance()
