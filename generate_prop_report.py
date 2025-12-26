import os
import json
import pandas as pd
import numpy as np
import config
from genome import Strategy
from backtest import BacktestEngine
from backtest.utils import refresh_strategies
from backtest.reporting import GeneTranslator
import glob

def load_mutex_portfolio():
    """Loads the optimized mutex portfolio."""
    file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    strategies = []
    for d in data:
        try:
            strat = Strategy.from_dict(d)
            # Ensure horizon is set
            strat.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)
            strat.data = d # Store original dict
            strategies.append(strat)
        except Exception:
            pass
    return strategies

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
        return [], []
    
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

def generate_mutex_report():
    print("Generating Report for MUTEX PORTFOLIO...")
    strategies = load_mutex_portfolio()
    if not strategies:
        print("No mutex portfolio found. Skipping.")
        return

    # Load Data
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("Error: Feature Matrix missing.")
        return
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    engine = BacktestEngine(df, cost_bps=config.COST_BPS, annualization_factor=config.ANNUALIZATION_FACTOR)
    
    # Evaluate Portfolio Components Individually
    # Note: We are reporting on the components, not the mutex interaction itself (which is complex to summarize in a static report)
    # But we will add a section for "Synergy"
    
    full_md = "# 🏆 MUTEX PORTFOLIO REPORT\n\n"
    full_md += f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    full_md += f"**Components:** {len(strategies)}\n\n"
    full_md += "---\n\n"
    
    for i, strat in enumerate(strategies):
        horizon = getattr(strat, 'horizon', 120)
        
        stats_df, net_returns_matrix = engine.evaluate_population(
            [strat], 
            set_type='test', 
            return_series=True, 
            time_limit=horizon
        )
        
        returns = net_returns_matrix[:, 0]
        
        # Metrics
        ann_factor = config.ANNUALIZATION_FACTOR
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        ann_return = avg_ret * ann_factor
        sharpe = ann_return / (std_ret * np.sqrt(ann_factor) + 1e-9)
        
        sortino = stats_df.iloc[0]['sortino']
        total_trades = int(stats_df.iloc[0]['trades'])
        
        # Ratings
        rating_sortino = '⭐⭐⭐⭐⭐' if sortino > 3.0 else '⭐⭐⭐'
        
        full_md += f"## Strategy {i+1}: {strat.name} (H{horizon})\n\n"
        full_md += f"**Sortino:** `{sortino:.2f}` {rating_sortino} | **Sharpe:** `{sharpe:.2f}` | **Trades:** `{total_trades}`\n\n"
        
        full_md += "### 🧬 Logic\n"
        full_md += GeneTranslator.interpret_strategy_logic(strat.data)
        full_md += "\n\n---\n\n"

    report_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "REPORT_MUTEX_PORTFOLIO.md")
    with open(report_path, "w") as f:
        f.write(full_md)
        
    print(f"✅ Generated Mutex Report: {report_path}")

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
    ann_factor = config.ANNUALIZATION_FACTOR
    
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

def load_inbox_strategies():
    """Loads the strategies from the inbox."""
    file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json")
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    strategies = []
    for d in data:
        try:
            strat = Strategy.from_dict(d)
            # Ensure horizon is set
            strat.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)
            strat.data = d # Store original dict
            strategies.append(strat)
        except Exception:
            pass
    return strategies

def generate_inbox_report():
    print("Generating Report for INBOX STRATEGIES...")
    strategies = load_inbox_strategies()
    if not strategies:
        print("No inbox strategies found. Skipping.")
        return

    # Convert Strategy objects to dicts for refresh_strategies
    strat_dicts = [s.data for s in strategies]
    
    # REFRESH METRICS (Uses BacktestEngine internally)
    strat_dicts = refresh_strategies(strat_dicts)
    
    # Re-hydrate strategies with refreshed data
    # (refresh_strategies updates dicts in place, so strategies[i].data might be stale if we replaced it entirely,
    # but refresh_strategies modifies mutable dicts in list, so let's verify)
    # The list 'strat_dicts' now contains updated dicts.
    # We should re-create Strategy objects or just update their .data
    
    refreshed_strategies = []
    for d in strat_dicts:
        s = Strategy.from_dict(d)
        s.data = d
        s.horizon = d.get('horizon', config.DEFAULT_TIME_LIMIT)
        refreshed_strategies.append(s)
    
    full_md = "# 📥 INBOX STRATEGY AUDIT\n\n"
    full_md += f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    full_md += f"**Strategies:** {len(refreshed_strategies)}\n\n"
    full_md += "---\n\n"
    
    for i, strat in enumerate(refreshed_strategies):
        horizon = getattr(strat, 'horizon', 120)
        
        # Extract metrics from refreshed dict
        d = strat.data
        ret_train = d.get('train_return', 0)
        ret_val = d.get('val_return', 0)
        ret_test = d.get('test_return', 0)
        sortino = d.get('test_sortino', 0)
        total_trades = d.get('test_trades', 0)
        sharpe = d.get('test_sharpe', 0)
        max_drawdown = d.get('max_drawdown', 0)
        ann_return = d.get('test_ann_return', 0.0)
        
        # Ratings
        rating_sortino = '⭐⭐⭐⭐⭐' if sortino > 3.0 else '⭐⭐⭐'
        rating_return = '⭐⭐⭐⭐⭐' if ann_return > 0.5 else '⭐⭐⭐'
        rating_dd = '⭐⭐⭐⭐⭐' if max_drawdown > -0.10 else '⭐⭐'
        
        full_md += f"## {i+1}. {strat.name} (H{horizon})\n\n"
        full_md += f"### 📊 Performance\n"
        full_md += f"| Metric | Value | Rating |\n"
        full_md += f"| :--- | :--- | :--- |\n"
        full_md += f"| **Train Return** | `{ret_train*100:.2f}%` | |\n"
        full_md += f"| **Val Return** | `{ret_val*100:.2f}%` | |\n"
        full_md += f"| **Test Return** | `{ret_test*100:.2f}%` | |\n"
        full_md += f"| **Annualized Return** | `{ann_return*100:.2f}%` | {rating_return} |\n"
        full_md += f"| **Sortino Ratio** | `{sortino:.2f}` | {rating_sortino} |\n"
        full_md += f"| **Max Drawdown** | `{max_drawdown*100:.2f}%` | {rating_dd} |\n"
        full_md += f"| **Sharpe Ratio** | `{sharpe:.2f}` | |\n"
        full_md += f"| **Trades** | `{total_trades}` | |\n\n"
        
        full_md += "### 🧬 Logic\n"
        full_md += GeneTranslator.interpret_strategy_logic(strat.data)
        full_md += "\n\n---\n\n"

    # SAVE UPDATED METRICS TO JSON
    json_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json")
    with open(json_path, "w") as f:
        json.dump(strat_dicts, f, indent=4)
    print(f"💾 Updated {len(strat_dicts)} strategies in {json_path} with fresh metrics.")

    report_path = os.path.join(config.DIRS['STRATEGIES_DIR'], "REPORT_INBOX_AUDIT.md")
    with open(report_path, "w") as f:
        f.write(full_md)
        
    print(f"✅ Generated Inbox Report: {report_path}")

if __name__ == "__main__":
    print("\n📜 Generating Prop Firm Strategy Reports...\n")
    
    # 0. Inbox Audit (New)
    try:
        generate_inbox_report()
    except Exception as e:
        print(f"❌ Failed to generate Inbox Report: {e}")

    # 1. Mutex Portfolio Report (New)
    try:
        generate_mutex_report()
    except Exception as e:
        print(f"❌ Failed to generate Mutex Report: {e}")
    
    # 2. Individual Horizon Reports (Legacy)
    # for h in config.PREDICTION_HORIZONS:
    #     try:
    #         generate_report(h)
    #     except Exception as e:
    #         # print(f"❌ Failed to generate report for Horizon {h}: {e}")
    #         pass

    print("\nDone.")
