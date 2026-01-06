class GeneTranslator:
    """Translates Strategy Genes into Trader-Speak."""
    
    @staticmethod
    def translate_feature(feature_name):
        """Converts snake_case features to readable English."""
        f = feature_name.replace('_', ' ').title()
        # Common replacements
        replacements = {
            "Vol": "Volatility",
            "Ret": "Return",
            "Sma": "SMA",
            "Ema": "EMA",
            "Rsi": "RSI",
            "Std": "Standard Deviation",
            "Vwap": "VWAP",
            "Obv": "On-Balance Volume",
            "Log": "Logarithmic",
            "Autocorr": "Autocorrelation",
            "Fdi": "Fractal Dimension",
            "Hurst": "Hurst Exponent",
            "Yang Zhang": "Yang-Zhang",
            "Adx": "ADX",
            "Skew": "Skewness",
            "Kurt": "Kurtosis"
        }
        for k, v in replacements.items():
            f = f.replace(k, v)
        return f

    @staticmethod
    def translate_gene(gene_dict):
        """Translates a single gene dictionary into a sentence."""
        if hasattr(gene_dict, 'to_dict'):
            gene_dict = gene_dict.to_dict()
            
        g_type = gene_dict['type']
        
        if g_type == 'static':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            op = gene_dict['operator']
            val = f"{gene_dict['threshold']:.4f}"
            return f"{f} is {op} {val}"
            
        elif g_type == 'relational':
            f1 = GeneTranslator.translate_feature(gene_dict['feature_left'])
            f2 = GeneTranslator.translate_feature(gene_dict['feature_right'])
            op = gene_dict['operator']
            return f"{f1} is {op} {f2}"
            
        elif g_type == 'delta':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            lookback = gene_dict['lookback']
            op = gene_dict['operator']
            val = f"{gene_dict['threshold']:.4f}"
            return f"Change in {f} ({lookback} bars) is {op} {val}"
            
        elif g_type == 'zscore':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            win = gene_dict['window']
            op = gene_dict['operator']
            sigma = f"{gene_dict['threshold']:.2f}σ"
            return f"{f} ({win}-bar Z-Score) is {op} {sigma}"
            
        elif g_type == 'time':
            mode = gene_dict['mode'].title() # Hour or Weekday
            op = gene_dict['operator']
            val = gene_dict['value']
            return f"Current {mode} is {op} {val}"
            
        elif g_type == 'consecutive':
            direction = gene_dict['direction'].upper()
            count = gene_dict['count']
            op = gene_dict['operator']
            return f"Consecutive {direction} Candles {op} {count}"
            
        elif g_type == 'cross':
            f1 = GeneTranslator.translate_feature(gene_dict['feature_left'])
            f2 = GeneTranslator.translate_feature(gene_dict['feature_right'])
            direction = gene_dict['direction'].upper()
            return f"{f1} crosses {direction} {f2}"
            
        elif g_type == 'persistence':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            op = gene_dict['operator']
            thresh = f"{gene_dict['threshold']:.4f}"
            win = gene_dict['window']
            return f"{f} has been {op} {thresh} for {win} consecutive bars"

        elif g_type == 'squeeze':
            f_short = GeneTranslator.translate_feature(gene_dict['feature_short'])
            f_long = GeneTranslator.translate_feature(gene_dict['feature_long'])
            mult = f"{gene_dict['multiplier']:.2f}"
            return f"{f_short} is squeezed (< {mult}x) relative to {f_long}"

        elif g_type == 'range':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            min_v = gene_dict['min_val']
            max_v = gene_dict['max_val']
            return f"{f} is inside range [{min_v:.4f}, {max_v:.4f}]"

        elif g_type == 'correlation':
            f_left = GeneTranslator.translate_feature(gene_dict['feature_left'])
            f_right = GeneTranslator.translate_feature(gene_dict['feature_right'])
            win = gene_dict['window']
            op = gene_dict['operator']
            thresh = f"{gene_dict['threshold']:.2f}"
            return f"Correlation({f_left}, {f_right}, {win}) is {op} {thresh}"

        elif g_type == 'efficiency':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            win = gene_dict['window']
            op = gene_dict['operator']
            thresh = f"{gene_dict['threshold']:.2f}"
            return f"Efficiency({f}, {win}) is {op} {thresh}"

        elif g_type == 'event':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            op = gene_dict['operator']
            val = f"{gene_dict['threshold']:.4f}"
            win = gene_dict['window']
            return f"{f} was {op} {val} at least once in last {win} bars"

        elif g_type == 'soft_zscore':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            win = gene_dict['window']
            op = gene_dict['operator']
            sigma = f"{gene_dict['threshold']:.2f}σ"
            slope = f"{gene_dict['slope']:.1f}"
            return f"Soft {f} ({win}-bar Z-Score) is {op} {sigma} (Slope: {slope})"

        elif g_type == 'flux':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            lag = gene_dict['lag']
            op = gene_dict['operator']
            val = f"{gene_dict['threshold']:.4f}"
            return f"Flux (Accel) of {f} (Lag {lag}) is {op} {val}"

        elif g_type == 'extrema':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            mode = gene_dict['mode'].upper()
            win = gene_dict['window']
            return f"{f} is at {win}-bar {mode}"

        elif g_type == 'divergence':
            f1 = GeneTranslator.translate_feature(gene_dict['feature_a'])
            f2 = GeneTranslator.translate_feature(gene_dict['feature_b'])
            win = gene_dict['window']
            return f"Divergence between {f1} and {f2} (Window {win})"

        elif g_type == 'seasonality':
            op = gene_dict['operator']
            sigma = f"{gene_dict['threshold']:.2f}σ"
            return f"Seasonal Deviation is {op} {sigma}"

        elif g_type == 'mean_reversion':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            reg = GeneTranslator.translate_feature(gene_dict['regime_feature'])
            win = gene_dict['window']
            thresh = f"{gene_dict['threshold']:.2f}σ"
            reg_thresh = f"{gene_dict['regime_threshold']:.2f}"
            direction = "Buy Dip" if gene_dict['direction'] == 'long' else "Sell Rip"
            return f"{direction} on {f} ({win}-bar Z > {thresh}) when {reg} > {reg_thresh}"

        elif g_type == 'hysteresis':
            f = GeneTranslator.translate_feature(gene_dict['feature'])
            op = gene_dict['operator']
            op_str = "Higher" if op == '>' else "Lower"
            win = gene_dict['window']
            return f"Price is {op_str} than last time {f} was at this level (Window: {win})"
            
        return f"Unknown Rule (Type: {g_type})"

    @staticmethod
    def interpret_strategy_logic(strategy_dict):
        """Generates the 'Why it works' narrative."""
        long_genes = strategy_dict.get('long_genes', [])
        short_genes = strategy_dict.get('short_genes', [])
        
        narrative = []
        narrative.append("**Long Entry Logic:**")
        if not long_genes:
            narrative.append("- *No Long Entries defined.*")
        else:
            for g in long_genes:
                narrative.append(f"- {GeneTranslator.translate_gene(g)}")
                
        narrative.append("\n**Short Entry Logic:**")
        if not short_genes:
            narrative.append("- *No Short Entries defined.*")
        else:
            for g in short_genes:
                narrative.append(f"- {GeneTranslator.translate_gene(g)}")
                
        # Heuristic Analysis
        narrative.append("\n**Prop Desk Commentary:**")
        
        all_features = [g['feature'] for g in long_genes + short_genes if 'feature' in g]
        all_features += [g['feature_left'] for g in long_genes + short_genes if 'feature_left' in g]
        all_features += [g['feature_right'] for g in long_genes + short_genes if 'feature_right' in g]
        all_text = " ".join(all_features).lower()
        
        if "volatility" in all_text or "std" in all_text or "atr" in all_text:
            narrative.append("- This strategy explicitly factors in market **Volatility**. It likely adjusts its entry criteria based on how 'hot' or 'cold' the market action is.")
        
        if "hurst" in all_text or "fdi" in all_text or "efficiency" in all_text:
            narrative.append("- The presence of **Fractal/Chaos Metrics** (Hurst, FDI) suggests this strategy is Regime-Aware. It attempts to distinguish between Trending and Mean-Reverting market states before committing capital.")
            
        if "skew" in all_text or "kurt" in all_text:
            narrative.append("- Uses **Higher Moment Statistics** (Skew/Kurtosis) to detect tail risk or distributional shifts, potentially front-running crash or explosion events.")
            
        if "delta" in str(long_genes + short_genes):
            narrative.append("- Heavily relies on **Momentum/Rate-of-Change**. This is likely a trend-following or breakout component.")
            
        if "zscore" in str(long_genes + short_genes):
            narrative.append("- Uses **Statistical Mean Reversion**. Z-Scores indicate it looks for price extremes (overbought/oversold) relative to a rolling baseline.")

        return "\n".join(narrative)

def get_avg_sortino(c):
    """Calculates the average Sortino ratio across Train/Val/Test."""
    def get_sortino(prefix):
        stats = c.get(f'{prefix}_stats', {})
        if stats:
            return stats.get('sortino', 0)
        
        sort = c.get(f'{prefix}_sortino', 0)
        if prefix == 'test' and sort == 0:
             sort = c.get('test_sortino', 0)
        return sort

    t_s = get_sortino('train')
    v_s = get_sortino('val')
    te_s = get_sortino('test')
    
    if te_s == 0:
        m = c.get('metrics', {})
        te_s = m.get('sortino_oos', 0)

    return (t_s + v_s + te_s) / 3

def get_min_sortino(c):
    """Calculates the minimum Sortino ratio across Train/Val/Test."""
    def get_sortino(prefix):
        stats = c.get(f'{prefix}_stats', {})
        if stats:
            return stats.get('sortino', 0)
        
        sort = c.get(f'{prefix}_sortino', 0)
        if prefix == 'test' and sort == 0:
             sort = c.get('test_sortino', 0)
        return sort

    t_s = get_sortino('train')
    v_s = get_sortino('val')
    te_s = get_sortino('test')
    
    if te_s == 0:
        m = c.get('metrics', {})
        te_s = m.get('sortino_oos', 0)

    return min(t_s, v_s, te_s)

def print_candidate_table(candidates, title="CURRENT CANDIDATE LIST"):
    """Prints a standardized table of strategy candidates."""
    if not candidates:
        print(f"\n📋 {title} (Empty)")
        return

    print(f"\n📋 {title} ({len(candidates)} strategies)")
    # Headers
    header = f"{'Name':<50} | {'Hz':<3} | {'Trds':<4} | {'Tr S':<5} | {'Val S':<5} | {'Tst S':<5} | {'Min S':<5} | {'Tr %':<6} | {'Val %':<6} | {'Tst %':<6}"
    print(header)
    print("-" * len(header))
    
    for c in candidates:
        name = c.get('name', 'Unknown')
        horizon = str(c.get('horizon', '?'))
        
        # Helper to extract metrics from various possible schemas
        def get_m(prefix):
            # Try new stats dict first (from optimizer/backtester)
            stats = c.get(f'{prefix}_stats', {})
            if stats:
                return stats.get('ret', 0) * 100, stats.get('sortino', 0), int(stats.get('trades', 0))
            
            # Fallback to flat keys
            ret = c.get(f'{prefix}_return', 0) * 100
            sort = c.get(f'{prefix}_sortino', 0)
            trades = c.get(f'{prefix}_trades', 0)
            
            # Special case for Test/OOS mixup
            if prefix == 'test':
                 if sort == 0: sort = c.get('test_sortino', 0)
                 if trades == 0: trades = c.get('test_trades', 0)
            
            return ret, sort, trades

        t_r, t_s, t_tr = get_m('train')
        v_r, v_s, v_tr = get_m('val')
        te_r, te_s, te_tr = get_m('test')
        
        # Handle cases where robust_return/sortino_oos were used in metrics dict (older candidates)
        if te_r == 0 and te_s == 0:
            m = c.get('metrics', {})
            te_r = m.get('robust_return', 0) * 100
            te_s = m.get('sortino_oos', 0)

        avg_s = get_min_sortino(c)

        row = f"{name[:50]:<50} | {horizon:<3} | {te_tr:<4} | {t_s:5.2f} | {v_s:5.2f} | {te_s:5.2f} | {avg_s:5.2f} | {t_r:5.2f}% | {v_r:5.2f}% | {te_r:5.2f}%"
        print(row)
        
        if 'warning' in c:
            print(f"   ⚠️  {c['warning']}")
    print("-" * len(header))

def cluster_strategies(results_list, backtester, threshold=0.7, top_n=200):
    """
    Performs Hierarchical Correlation Clustering on strategy results.
    Returns a list of selected Strategy objects (best per cluster).
    
    results_list: list of dicts with 'Strategy' (obj) and 'Robust_Ret' (float)
    backtester: instance of BacktestEngine
    """
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # Sort by Robust Return to prioritize better strategies
    sorted_results = sorted(results_list, key=lambda x: x['Robust_Ret'], reverse=True)
    
    # Filter valid strategies
    valid_results = [r for r in sorted_results if r['Strategy'] is not None]
    
    if not valid_results:
        return []
        
    candidates = [r['Strategy'] for r in valid_results][:top_n]
    results_map = {s.name: r for s, r in zip(candidates, valid_results[:top_n])}
    
    if len(candidates) <= 1:
        return candidates

    # Generate Signals
    sig_matrix = backtester.generate_signal_matrix(candidates)
    
    # 1. Correlation Matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(sig_matrix, rowvar=False)
    
    # Fix NaNs (0 variance signals)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    
    # 2. Distance Matrix
    dist_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip(dist_matrix, 0, 1)
    
    # 3. Clustering
    condensed_dist = squareform(dist_matrix, checks=False)
    # Threshold: distance < (1 - 0.7) = 0.3
    t_val = 1.0 - threshold
    
    Z = linkage(condensed_dist, method='complete')
    cluster_labels = fcluster(Z, t=t_val, criterion='distance')
    
    # 4. Select Best per Cluster
    cluster_map = {}
    for i, label in enumerate(cluster_labels):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(candidates[i])
    
    selected_strats = []
    for label, members in cluster_map.items():
        # Pick member with highest Robust Return
        best_in_cluster = max(members, key=lambda s: results_map[s.name]['Robust_Ret'])
        selected_strats.append(best_in_cluster)
        
    return selected_strats
