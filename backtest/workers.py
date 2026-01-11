import numpy as np
import config
from trade_simulator import TradeSimulator
from .context import LazyMMapContext

def _worker_generate_signals(strategies, context_dir):
    """
    Worker function to generate signals for a chunk of strategies.
    Executed in parallel processes.
    """
    # Use Lazy Context
    context = LazyMMapContext(context_dir)
    
    n_rows = len(context)
    n_strats = len(strategies)
    chunk_matrix = np.zeros((n_rows, n_strats), dtype=np.int8)
    gene_cache = {} 
    
    for i, strat in enumerate(strategies):
        chunk_matrix[:, i] = strat.generate_signal(context, cache=gene_cache)
        
    return chunk_matrix

def _worker_simulate(signals_chunk, params_chunk, prices, times, spread_bps, effective_cost_bps, min_comm, standard_lot, account_size, time_limit, hours, weekdays, highs, lows, atr=None):
    """
    Worker function to simulate a chunk of strategy signals.
    """
    simulator = TradeSimulator(
        prices=prices,
        times=times,
        spread_bps=spread_bps,
        cost_bps=effective_cost_bps,
        min_comm=min_comm,
        lot_size=standard_lot,
        account_size=account_size
    )
    
    n_bars, n_strats = signals_chunk.shape
    net_returns = np.zeros((n_bars, n_strats), dtype=np.float32)
    trades_count = np.zeros(n_strats, dtype=int)
    
    for i in range(n_strats):
        # Extract params
        start_idx = params_chunk[i].get('start_idx', 0)
        end_idx = params_chunk[i].get('end_idx', len(prices))
        tp = params_chunk[i].get('tp', config.DEFAULT_TAKE_PROFIT)
        sl = params_chunk[i].get('sl', config.DEFAULT_STOP_LOSS)
        limit_dist = params_chunk[i].get('limit_dist', 0.0)

        # Directional barriers (use symmetric if not provided)
        sl_long = params_chunk[i].get('sl_long', sl)
        sl_short = params_chunk[i].get('sl_short', sl)
        tp_long = params_chunk[i].get('tp_long', tp)
        tp_short = params_chunk[i].get('tp_short', tp)

        # Create Limit Distance Vector (Scalar -> Vector)
        # We only apply limit logic where signal != 0, but passing full vector is fine/fast.
        limit_vec = np.full(len(signals_chunk[:, i]), limit_dist, dtype=np.float64)

        net_rets, t_count = simulator.simulate_fast(
            signals_chunk[:, i],
            stop_loss_pct=sl,
            take_profit_pct=tp,
            time_limit_bars=time_limit,
            hours=hours,
            weekdays=weekdays,
            highs=highs,
            lows=lows,
            atr=atr,
            vol_targeting=True, # ENABLE VOL TARGETING FOR ALL STRATEGIES
            target_risk_pct=config.RISK_PER_TRADE_PERCENT,
            limit_dist_atr=limit_vec,
            sl_long=sl_long,
            sl_short=sl_short,
            tp_long=tp_long,
            tp_short=tp_short
        )
        net_returns[:, i] = net_rets
        trades_count[i] = t_count
        
    return net_returns, trades_count
