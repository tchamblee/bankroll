import numpy as np
import config

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

    for i, cand in enumerate(candidates):
        if cand.fitness < 0.1: continue # Ignore junk
        
        n_trades = trades_batch[i]
        total_ret = np.sum(rets_batch[:, i])
            
        # Relaxed HOF Entry for early generations
        min_fitness = -10.0 if gen < phase_1 else 0.1
        
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
        
        hall_of_fame.append({
            'strat': cand,
            'fit': cand_fit,
            'sig': cand_sig,
            'gen': gen
        })

    # Sort HOF by Fitness
    hall_of_fame.sort(key=lambda x: x['fit'], reverse=True)
    
    # Cap Size
    HOF_CAP = 300
    if len(hall_of_fame) > HOF_CAP:
        del hall_of_fame[HOF_CAP:]
