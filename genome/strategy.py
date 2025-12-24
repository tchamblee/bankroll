import numpy as np
import config
import math
from .genes import gene_from_dict

class Strategy:
    """
    Represents a Bidirectional Trading Strategy with Regime Filtering.
    """
    def __init__(self, name="Strategy", long_genes=None, short_genes=None, regime_genes=None, min_concordance=None):
        self.name = name
        self.long_genes = long_genes if long_genes else []
        self.short_genes = short_genes if short_genes else []
        self.regime_genes = regime_genes if regime_genes else []
        self.min_concordance = min_concordance
        self.fitness = 0.0
        
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
        self.regime_genes = _dedup(self.regime_genes)
        
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

    def generate_signal(self, context: dict, cache: dict = None) -> np.array:
        n_rows = context.get('__len__', 0)
        if n_rows == 0 and len(context) > 0:
            for val in context.values():
                 if hasattr(val, 'shape'):
                     n_rows = val.shape[0]
                     break
        
        # Helper for Voting Logic
        def get_votes(genes):
            if not genes: return np.zeros(n_rows, dtype=int)
            votes = np.zeros(n_rows, dtype=int)
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
        
        net_signal = go_long.astype(int) - go_short.astype(int)
        
        # --- REGIME FILTER (VETO) ---
        if self.regime_genes:
            regime_mask = np.ones(n_rows, dtype=bool)
            for gene in self.regime_genes:
                regime_mask &= gene.evaluate(context, cache)
            
            # Apply Mask: If Regime is False, Signal is 0 (Flat)
            net_signal = net_signal * regime_mask.astype(int)

        return net_signal * config.MAX_LOTS

    def to_dict(self):
        return {
            'name': self.name,
            'long_genes': [g.to_dict() for g in self.long_genes],
            'short_genes': [g.to_dict() for g in self.short_genes],
            'regime_genes': [g.to_dict() for g in self.regime_genes],
            'min_concordance': self.min_concordance,
            'fitness': self.fitness
        }

    @classmethod
    def from_dict(cls, d):
        s = cls(
            name=d.get('name', 'Strategy'),
            long_genes=[gene_from_dict(g) for g in d.get('long_genes', [])],
            short_genes=[gene_from_dict(g) for g in d.get('short_genes', [])],
            regime_genes=[gene_from_dict(g) for g in d.get('regime_genes', [])],
            min_concordance=d.get('min_concordance')
        )
        s.fitness = d.get('fitness', 0.0)
        s.cleanup()
        return s

    def __repr__(self):
        l_str = f" + ".join([str(g) for g in self.long_genes]) if self.long_genes else "None"
        s_str = f" + ".join([str(g) for g in self.short_genes]) if self.short_genes else "None"
        r_str = f" & ".join([str(g) for g in self.regime_genes]) if self.regime_genes else "None"
        return f"[{self.name}] LONG:({l_str}) | SHORT:({s_str}) | REGIME:({r_str})"
