import numpy as np
import config
import math
from .genes import gene_from_dict

class Strategy:
    """
    Represents a Bidirectional Trading Strategy with Regime Filtering.
    """
    def __init__(self, name="Strategy", long_genes=None, short_genes=None, min_concordance=None,
                 stop_loss_pct=None, take_profit_pct=None, horizon=None, limit_dist_atr=0.0,
                 sl_long=None, sl_short=None, tp_long=None, tp_short=None):
        self.name = name
        self.long_genes = long_genes if long_genes else []
        self.short_genes = short_genes if short_genes else []
        self.min_concordance = min_concordance
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else config.DEFAULT_STOP_LOSS
        self.take_profit_pct = take_profit_pct if take_profit_pct is not None else config.DEFAULT_TAKE_PROFIT
        self.horizon = horizon if horizon is not None else 120 # Default
        self.limit_dist_atr = limit_dist_atr if limit_dist_atr is not None else 0.0
        self.fitness = 0.0

        # Directional barriers (None = use symmetric stop_loss_pct/take_profit_pct)
        self.sl_long = sl_long
        self.sl_short = sl_short
        self.tp_long = tp_long
        self.tp_short = tp_short

        self.cleanup()

    def cleanup(self):
        """Removes duplicate genes within each gene list."""
        def _dedup(genes):
            seen = set()
            unique = []
            for g in genes:
                s = str(g)
                if s not in seen:
                    seen.add(s)
                    unique.append(g)
            return unique
            
        self.long_genes = _dedup(self.long_genes)
        self.short_genes = _dedup(self.short_genes)
        
    def copy(self):
        """Creates a deep copy of the strategy."""
        # Genes usually have their own copy/to_dict methods, or are immutable enough.
        # But to be safe, we re-instantiate from dict logic or manual copy.
        # Using from_dict(to_dict) is cleanest for deep copy of nested genes.
        c = Strategy.from_dict(self.to_dict())
        c.generation_found = getattr(self, 'generation_found', '?')
        # Metrics are ephemeral, don't copy
        return c

    def recalculate_concordance(self):
        """
        Updates min_concordance based on gene count.
        Uses Majority Rule: >50% agreement required.
        For small sets (<=2), allows 1 (OR logic) for diversity.
        """
        n = max(len(self.long_genes), len(self.short_genes))
        if n <= 2:
            self.min_concordance = 1
        else:
            self.min_concordance = math.ceil(n * 0.51)

    def get_effective_sl(self, direction):
        """Returns the effective stop loss for the given direction ('long' or 'short')."""
        if direction == 'long':
            return self.sl_long if self.sl_long is not None else self.stop_loss_pct
        else:
            return self.sl_short if self.sl_short is not None else self.stop_loss_pct

    def get_effective_tp(self, direction):
        """Returns the effective take profit for the given direction ('long' or 'short')."""
        if direction == 'long':
            return self.tp_long if self.tp_long is not None else self.take_profit_pct
        else:
            return self.tp_short if self.tp_short is not None else self.take_profit_pct

    def is_asymmetric(self):
        """Returns True if strategy uses asymmetric barriers."""
        return any(x is not None for x in [self.sl_long, self.sl_short, self.tp_long, self.tp_short])

    def get_hash(self):
        """Returns a unique hash representing the strategy logic (ignoring name)."""
        # Sort genes to ensure Order Agnostic hashing (Gene A + Gene B == Gene B + Gene A)
        l_genes = sorted([str(g) for g in self.long_genes])
        s_genes = sorted([str(g) for g in self.short_genes])

        # Combine structural elements
        # Include directional barriers if asymmetric
        if self.is_asymmetric():
            sl_str = f"SLL:{self.get_effective_sl('long')}|SLS:{self.get_effective_sl('short')}"
            tp_str = f"TPL:{self.get_effective_tp('long')}|TPS:{self.get_effective_tp('short')}"
        else:
            sl_str = f"SL:{self.stop_loss_pct}"
            tp_str = f"TP:{self.take_profit_pct}"

        structure = f"L:{l_genes}|S:{s_genes}|{sl_str}|{tp_str}|H:{self.horizon}|LIM:{self.limit_dist_atr}"

        return structure

    def generate_signal(self, context: dict, cache: dict = None) -> np.array:
        n_rows = context.get('__len__', 0)
        if n_rows == 0 and len(context) > 0:
            for val in context.values():
                 if hasattr(val, 'shape'):
                     n_rows = val.shape[0]
                     break
        
        # Helper for Voting Logic
        def get_votes(genes):
            if not genes: return np.zeros(n_rows, dtype=np.float32)
            votes = np.zeros(n_rows, dtype=np.float32)
            for gene in genes:
                votes += gene.evaluate(context, cache)
            return votes

        l_votes = get_votes(self.long_genes)
        s_votes = get_votes(self.short_genes)
        
        # Concordance Logic
        l_thresh = self.min_concordance if self.min_concordance else len(self.long_genes)
        s_thresh = self.min_concordance if self.min_concordance else len(self.short_genes)
        
        if self.long_genes: l_thresh = max(1, min(l_thresh, len(self.long_genes)))
        if self.short_genes: s_thresh = max(1, min(s_thresh, len(self.short_genes)))
        
        go_long = l_votes >= l_thresh if self.long_genes else np.zeros(n_rows, dtype=bool)
        go_short = s_votes >= s_thresh if self.short_genes else np.zeros(n_rows, dtype=bool)

        # Net signal: +1 (long), -1 (short), 0 (flat)
        # Actual position sizing is handled by vol_targeting in trade_simulator
        net_signal = go_long.astype(int) - go_short.astype(int)

        return net_signal

    def to_dict(self):
        d = {
            'name': self.name,
            'long_genes': [g.to_dict() for g in self.long_genes],
            'short_genes': [g.to_dict() for g in self.short_genes],
            'min_concordance': self.min_concordance,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'horizon': self.horizon,
            'limit_dist_atr': self.limit_dist_atr
        }
        # Only include directional barriers if asymmetric (keeps backwards compatibility)
        if self.sl_long is not None:
            d['sl_long'] = self.sl_long
        if self.sl_short is not None:
            d['sl_short'] = self.sl_short
        if self.tp_long is not None:
            d['tp_long'] = self.tp_long
        if self.tp_short is not None:
            d['tp_short'] = self.tp_short
        return d

    @staticmethod
    def from_dict(d):
        return Strategy(
            name=d.get('name', 'Unknown'),
            long_genes=[gene_from_dict(g) for g in d.get('long_genes', [])],
            short_genes=[gene_from_dict(g) for g in d.get('short_genes', [])],
            min_concordance=d.get('min_concordance', 1),
            stop_loss_pct=d.get('stop_loss_pct', config.DEFAULT_STOP_LOSS),
            take_profit_pct=d.get('take_profit_pct', config.DEFAULT_TAKE_PROFIT),
            horizon=d.get('horizon', 120),
            limit_dist_atr=d.get('limit_dist_atr', 0.0),
            sl_long=d.get('sl_long'),
            sl_short=d.get('sl_short'),
            tp_long=d.get('tp_long'),
            tp_short=d.get('tp_short')
        )

    def __repr__(self):
        l_str = f" & ".join([str(g) for g in self.long_genes]) if self.long_genes else "None"
        s_str = f" & ".join([str(g) for g in self.short_genes]) if self.short_genes else "None"
        if self.is_asymmetric():
            barrier_str = f"SL_L:{self.get_effective_sl('long')} SL_S:{self.get_effective_sl('short')} TP_L:{self.get_effective_tp('long')} TP_S:{self.get_effective_tp('short')}"
        else:
            barrier_str = f"SL:{self.stop_loss_pct} TP:{self.take_profit_pct}"
        return f"[{self.name}] H:{self.horizon} {barrier_str} LIM:{self.limit_dist_atr} | LONG:({l_str}) | SHORT:({s_str})"
