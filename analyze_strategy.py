import os
import json
import pandas as pd
import numpy as np
import config
import argparse
import sys
from genome import Strategy
from backtest import BacktestEngine
from backtest.reporting import GeneTranslator
import matplotlib.pyplot as plt

def find_strategy(strategy_name):
    """Searches for a strategy by name in common files."""
    search_paths = [
        os.path.join(config.DIRS['STRATEGIES_DIR'], "found_strategies.json"),
        os.path.join(config.DIRS['STRATEGIES_DIR'], "mutex_portfolio.json"),
        os.path.join(config.DIRS['STRATEGIES_DIR'], "candidates.json"),
    ]
    
    # Also check apex files
    for f in os.listdir(config.DIRS['STRATEGIES_DIR']):
        if f.startswith("apex_strategies") and f.endswith(".json"):
            search_paths.append(os.path.join(config.DIRS['STRATEGIES_DIR'], f))
            
    found_strat = None
    found_source = None
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            for d in data:
                if d.get('name') == strategy_name:
                    found_strat = d
                    found_source = path
                    break
        except:
            pass
            
        if found_strat:
            break
            
    return found_strat, found_source

def analyze_strategy(strategy_name):
    print(f"🔍 Searching for strategy: {strategy_name}...")
    strat_dict, source = find_strategy(strategy_name)
    
    if not strat_dict:
        print(f"❌ Strategy '{strategy_name}' not found in any standard strategy file.")
        return
        
    print(f"✅ Found in: {source}")
    
    # Load Strategy Object
    try:
        strat = Strategy.from_dict(strat_dict)
        strat.horizon = strat_dict.get('horizon', config.DEFAULT_TIME_LIMIT)
        # Hydrate extra params if needed
        strat.stop_loss_pct = strat_dict.get('stop_loss_pct', config.DEFAULT_STOP_LOSS)
        strat.take_profit_pct = strat_dict.get('take_profit_pct', config.DEFAULT_TAKE_PROFIT)
    except Exception as e:
        print(f"❌ Error hydrating strategy: {e}")
        return

    # Load Data
    print("📊 Loading Market Data...")
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix missing.")
        return
        
    df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    
    # Initialize Engine
    engine = BacktestEngine(
        df,
        cost_bps=config.COST_BPS,
        annualization_factor=config.ANNUALIZATION_FACTOR
    )
    
    print(f"🚀 Running Backtest (Horizon: {strat.horizon})...")
    
    # Evaluate (Full Set for Analysis)
    stats_df, net_returns_matrix = engine.evaluate_population(
        [strat], 
        set_type='test', # Use Test (OOS) for the report as standard
        return_series=True, 
        time_limit=strat.horizon
    )
    
    # Calculate Detailed Metrics
    returns = net_returns_matrix[:, 0]
    
    if len(returns) == 0:
        print("❌ Backtest returned no data.")
        return
        
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
    
    # Stats from DF
    sortino = stats_df.iloc[0]['sortino']
    total_trades = int(stats_df.iloc[0]['trades'])
    
    # Exposure
    in_market = np.count_nonzero(returns != 0)
    duty_cycle = (in_market / len(returns)) * 100
    avg_duration = (in_market / total_trades) if total_trades > 0 else 0
    
    # Ratings
    rating_return = '⭐⭐⭐⭐⭐' if ann_return > 0.5 else '⭐⭐⭐'
    rating_sortino = '⭐⭐⭐⭐⭐' if sortino > 3.0 else '⭐⭐⭐'
    rating_dd = '⭐⭐⭐⭐⭐' if max_drawdown > -0.10 else '⭐⭐'
    rating_freq = 'High Freq' if total_trades > 500 else 'Med Freq'
    rating_sharpe = 'Excellent' if sharpe > 2.0 else 'Good'
    
    recovery_profile = 'Fast recovery suggested by high Sortino.' if sortino > 2.0 else 'Moderate recovery profile.'
    stability_profile = 'Consistent' if ann_vol < 0.20 else 'Volatile'
    
    edge_type = 'Robust' if sortino > 1.5 else 'Speculative'
    complexity = 'Parsimonious (Simple)' if (len(strat.long_genes) + len(strat.short_genes)) < 4 else 'Complex'
    calmar = abs(ann_return/max_drawdown) if max_drawdown != 0 else 0
    rating_calmar = 'Exceptional' if calmar > 3.0 else 'Acceptable'
    
    # Markdown Construction
    md_content = f"""# 🏆 Prop Firm Analysis: {strat.name}

**Asset Class:** FX/CFD (EUR/USD)  
**Prediction Horizon:** {strat.horizon} Bars  
**Strategy Source:** `{source}`  
**Strategy ID:** `{strat.name}`

---

## 🧬 Genetic Alpha Structure
*The strategy is composed of the following decision logic:*

{GeneTranslator.interpret_strategy_logic(strat_dict)}

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
| **Calmar Ratio** | `{calmar:.2f}` | {rating_calmar} |

### ⏱️ Time Stats
| Metric | Value |
| :--- | :--- |
| **Duty Cycle (In Market)** | `{duty_cycle:.1f}%` |
| **Avg Trade Duration** | `{avg_duration:.1f} bars` |

### 📉 Risk Profile
* **Drawdown Depth:** The strategy experienced a maximum peak-to-valley decline of **{max_drawdown*100:.2f}%**.
* **Recovery:** {recovery_profile}
* **Stability:** The strategy shows a {stability_profile} equity growth trajectory (Vol: {ann_vol*100:.1f}%).

### 💡 Trader's Verdict
> "This strategy demonstrates a **{edge_type}** edge. The use of {len(strat.long_genes) + len(strat.short_genes)} distinct genes suggests a {complexity} approach to market timing. 
> Given the Annualized Return of {ann_return*100:.0f}% against a Max DD of {max_drawdown*100:.1f}%, the **Calmar Ratio is {calmar:.2f}**, which is {rating_calmar} for a funded account."

---
*Generated by Gemini Alpha Factory on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}* 
"""

    print("\n" + "="*80)
    print(md_content)
    print("="*80 + "\n")
    
    # Save Report
    report_filename = f"REPORT_ANALYSIS_{strat.name}.md"
    report_path = os.path.join(config.DIRS['STRATEGIES_DIR'], report_filename)
    with open(report_path, "w") as f:
        f.write(md_content)
    print(f"✅ Report saved to: {report_path}")
    
    # Generate Chart
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(equity_curve)), equity_curve, label=f'{strat.name} (OOS)', color='blue', linewidth=1.5)
    plt.title(f"Strategy Performance: {strat.name}\nSortino: {sortino:.2f} | Profit: ${cum_pnl[-1]:,.0f} | DD: {max_drawdown*100:.2f}%")
    plt.ylabel("Equity ($)")
    plt.xlabel("Bars (Test Set)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(config.DIRS['PLOTS_DIR'], f"analysis_{strat.name}.png")
    plt.savefig(plot_path)
    print(f"📸 Performance chart saved to: {plot_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_strategy.py <strategy_name>")
    else:
        analyze_strategy(sys.argv[1])
