import os
import json
import pandas as pd
import numpy as np
import config
from genome import Strategy
from backtest import BacktestEngine
import glob

def load_and_rank_strategies(horizon):
    """
    Consolidated Strategy Loading Logic.
    Matches the logic in generate_trade_atlas.py to ensure the same 'Champion' is selected.
    """
    # Prefer the Top 5 Unique file which contains pre-calculated metrics from report_top_strategies.py
    file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}_top5_unique.json")
    if not os.path.exists(file_path):
        # Fallback to Top 10 if Top 5 Unique doesn't exist (legacy support)
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}_top10.json")
    
    if not os.path.exists(file_path):
        # Final Fallback to raw file (metrics might be missing/defaulted to -999)
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{horizon}.json")
        
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    strategies = []
    # Preserve original dictionary data for reporting logic
    strategy_dicts = []
    
    for d in data:
        try:
            strat = Strategy.from_dict(d)
            # Hydrate metrics if available, or assume they are in the dict
            metrics = d.get('metrics', {})
            # Use metrics dict if available, otherwise check root
            strat.robust_return = metrics.get('robust_return', d.get('robust_return', -999))
            strat.full_return = metrics.get('full_return', d.get('full_return', -999))
            
            strategies.append(strat)
            strategy_dicts.append(d)
        except Exception as e:
            pass
            
    # Sort: Primary = Robust%, Secondary = Ret%(Full)
    # We zip them to sort both lists in sync
    combined = list(zip(strategies, strategy_dicts))
    combined.sort(key=lambda x: (x[0].robust_return, x[0].full_return), reverse=True)
    
    if not combined:
        return [], []
        
    return zip(*combined)

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
            
        return "Unknown Rule"

    @staticmethod
    def interpret_strategy_logic(strategy_dict):
        """Generates the 'Why it works' narrative."""
        long_genes = strategy_dict['long_genes']
        short_genes = strategy_dict['short_genes']
        
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

def generate_report(horizon):
    # Use standard loading/ranking logic
    strategies, strategy_dicts = load_and_rank_strategies(horizon)
    
    if not strategies:
        print(f"Skipping Horizon {horizon}: No valid strategies found.")
        return

    # Champion is the first one
    champion_strat = strategies[0]
    champion_data = strategy_dicts[0]
    
    print(f"Analyzing Champion for Horizon {horizon}: {champion_strat.name}")
    
    # Load Data
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("Error: Feature Matrix missing.")
        return
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # Backtest
    # We need to run the backtester to get the equity curve
    engine = BacktestEngine(
        df,
        cost_bps=config.COST_BPS,
        annualization_factor=config.ANNUALIZATION_FACTOR
    )
    
    # Evaluate on Test Set (OOS)
    stats_df, net_returns_matrix = engine.evaluate_population(
        [champion_strat], 
        set_type='test', 
        return_series=True, 
        time_limit=horizon
    )
    
    # Calculate Detailed Metrics
    returns = net_returns_matrix[:, 0]
        
    # Cumulative PnL
    cum_pnl = np.cumsum(returns * config.ACCOUNT_SIZE)
    equity_curve = config.ACCOUNT_SIZE + cum_pnl
    
    # Drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = np.min(drawdown)
    
    # Annualization
    # Let's use the engine's factor for annualization scalar.
    ann_factor = 181440 # From code memory
    
    avg_ret = np.mean(returns)
    std_ret = np.std(returns)
    
    ann_return = avg_ret * ann_factor
    ann_vol = std_ret * np.sqrt(ann_factor)
    sharpe = ann_return / (ann_vol + 1e-9)
    
    # Sourced from FRESH BACKTEST
    sortino = stats_df.iloc[0]['sortino']
    total_trades = int(stats_df.iloc[0]['trades'])
    
    # Pre-calculate Ratings
    rating_return = '⭐⭐⭐⭐⭐' if ann_return > 0.5 else '⭐⭐⭐'
    rating_sortino = '⭐⭐⭐⭐⭐' if sortino > 3.0 else '⭐⭐⭐'
    rating_dd = '⭐⭐⭐⭐⭐' if max_drawdown > -0.10 else '⭐⭐'
    rating_freq = 'High Freq' if total_trades > 500 else 'Med Freq'
    rating_sharpe = 'Excellent' if sharpe > 2.0 else 'Good'
    
    recovery_profile = 'Fast recovery suggested by high Sortino.' if sortino > 2.0 else 'Moderate recovery profile.'
    stability_profile = 'Consistent' if ann_vol < 0.20 else 'Volatile'
    
    edge_type = 'Robust' if sortino > 1.5 else 'Speculative'
    complexity = 'Parimonious (Simple)' if (len(champion_strat.long_genes) + len(champion_strat.short_genes)) < 4 else 'Complex'
    calmar = abs(ann_return/max_drawdown) if max_drawdown != 0 else 0
    rating_calmar = 'Exceptional' if calmar > 3.0 else 'Acceptable'
    
    # Markdown Construction
    md_content = f"""# 🏆 Prop Firm Analysis: {champion_strat.name}

**Asset Class:** FX/CFD (EUR/USD)  
**Prediction Horizon:** {horizon} Bars  
**Generation Discovered:** {champion_data.get('generation', 'N/A')}  
**Strategy ID:** `{champion_strat.name}`

---

## 🧬 Genetic Alpha Structure
*The strategy is composed of the following decision logic:*

{GeneTranslator.interpret_strategy_logic(champion_data)}

---

## 🏦 Prop Desk Performance Analysis
*Performance evaluated on Out-of-Sample (Test) Data.*

### 📊 Key Metrics
| Metric | Value | Prop Desk Rating |
| :--- | :--- | :--- |
| **Annualized Return** | `{ann_return*100:.2f}%` | {rating_return} |
| **Sortino Ratio** | `{sortino:.2f}` | {rating_sortino} |
| **Max Drawdown** | `{max_drawdown*100:.2f}%` | {rating_dd} |
| **Total Trades** | `{total_trades}` | {rating_freq} |
| **Sharpe Ratio** | `{sharpe:.2f}` | {rating_sharpe} |

### 📉 Risk Profile
* **Drawdown Depth:** The strategy experienced a maximum peak-to-valley decline of **{max_drawdown*100:.2f}%**.
* **Recovery:** {recovery_profile}
* **Stability:** The strategy shows a {stability_profile} equity growth trajectory.

### 💡 Trader's Verdict
> "This strategy demonstrates a **{edge_type}** edge. The use of {len(champion_strat.long_genes) + len(champion_strat.short_genes)} distinct genes suggests a {complexity} approach to market timing. 
> Given the Annualized Return of {ann_return*100:.0f}% against a Max DD of {max_drawdown*100:.1f}%, the **Calmar Ratio is {calmar:.2f}**, which is {rating_calmar} for a funded account."

---
*Generated by Gemini Alpha Factory on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}* 
"""

    report_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"REPORT_CHAMPION_{horizon}.md")
    with open(report_path, "w") as f:
        f.write(md_content)
        
    print(f"✅ Generated Prop Report: {report_path}")

if __name__ == "__main__":
    print("\n📜 Generating Prop Firm Strategy Reports...\n")
    for h in config.PREDICTION_HORIZONS:
        try:
            generate_report(h)
        except Exception as e:
            print(f"❌ Failed to generate report for Horizon {h}: {e}")
            # print stack trace for debug
            import traceback
            traceback.print_exc()

    print("\nDone.")