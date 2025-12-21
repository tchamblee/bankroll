import json
import os
import re
import sys
import pandas as pd
import numpy as np
import config
from collections import Counter
from backtest_engine import BacktestEngine
from strategy_genome import Strategy

class MockEngine:
    def __init__(self, df):
        self.bars = df

def get_gene_description(gene):
    if gene.type == 'delta':
        return f"Delta({gene.feature}, {gene.lookback})"
    elif gene.type == 'zscore':
        return f"Z({gene.feature}, {gene.window})"
    elif gene.type == 'relational':
        return f"Rel({gene.feature_left}, {gene.feature_right})"
    elif gene.type == 'static':
        return gene.feature
    elif gene.type == 'consecutive':
        return f"Consecutive({gene.direction})"
    elif gene.type == 'time':
        return f"Time({gene.mode})"
    elif gene.type == 'cross':
        return f"{gene.feature_left} CROSS {gene.direction.upper()} {gene.feature_right}"
    elif gene.type == 'persistence':
        return f"({gene.feature} {gene.operator} {gene.threshold:.2f}) FOR {gene.window} BARS"
    return "Unknown"

import json
import os
import re
import sys
import pandas as pd
import numpy as np
import config
from collections import Counter
from backtest_engine import BacktestEngine
from strategy_genome import Strategy

class MockEngine:
    def __init__(self, df):
        self.bars = df

def get_gene_description(gene):
    if gene.type == 'delta':
        return f"Delta({gene.feature}, {gene.lookback})"
    elif gene.type == 'zscore':
        return f"Z({gene.feature}, {gene.window})"
    elif gene.type == 'relational':
        return f"Rel({gene.feature_left}, {gene.feature_right})"
    elif gene.type == 'static':
        return gene.feature
    elif gene.type == 'consecutive':
        return f"Consecutive({gene.direction})"
    elif gene.type == 'time':
        return f"Time({gene.mode})"
    elif gene.type == 'cross':
        return f"{gene.feature_left} CROSS {gene.direction.upper()} {gene.feature_right}"
    elif gene.type == 'persistence':
        return f"({gene.feature} {gene.operator} {gene.threshold:.2f}) FOR {gene.window} BARS"
    return "Unknown"

def evaluate_batch(backtester, batch):
    """Evaluates a batch on Val, Test, and Full sets, returning a list of result dicts."""
    # 1. Validation Set (60-80%)
    res_val = backtester.evaluate_population(batch, set_type='validation')
    # 2. Test Set (80-100%)
    res_test = backtester.evaluate_population(batch, set_type='test')
    
    # 3. Full Set (Custom manual calc for efficiency/custom range)
    full_signals = backtester.generate_signal_matrix(batch)
    prices = backtester.close_vec
    rets = backtester.returns_vec
    
    sigs_shft = np.roll(full_signals, 1, axis=0); sigs_shft[0,:]=0
    prcs_shft = np.roll(prices, 1, axis=0); prcs_shft[0]=prices[0]
    
    pos_notional = sigs_shft * backtester.standard_lot * prcs_shft
    gross_pnl = pos_notional * rets
    
    lot_change = np.abs(np.diff(full_signals, axis=0, prepend=0))
    costs = lot_change * backtester.standard_lot * prices * backtester.total_cost_pct
    
    full_net_returns = (gross_pnl - costs) / backtester.account_size
    full_rets_pct = np.sum(full_net_returns, axis=0)
    full_trades = np.sum(lot_change, axis=0)

    results = []
    for i, strat in enumerate(batch):
        # Basic filtering: Must have traded
        if full_trades[i] == 0: continue
            
        ret_val = res_val.iloc[i]['total_return']
        ret_test = res_test.iloc[i]['total_return']
        ret_full = full_rets_pct[i]
        
        # Robustness Metric: Min Return across all 3 periods
        # This penalizes strategies that failed in any single period.
        robust_ret = min(ret_val, ret_test, ret_full)
        
        results.append({
            'Strategy': strat,
            'Ret_Val': ret_val,
            'Ret_Test': ret_test,
            'Ret_Full': ret_full,
            'Robust_Ret': robust_ret,
            'Sortino_OOS': res_test.iloc[i]['sortino'],
            'Trades': full_trades[i],
            'Gen': getattr(strat, 'generation_found', '?')
        })
        
    return results

def main():
    print("\n" + "="*120)
    print("🏆 APEX STRATEGY REPORT (ROBUSTNESS CHECK) 🏆")
    print(f"Account: ${config.ACCOUNT_SIZE:,.0f} | Lots: {config.MIN_LOTS}-{config.MAX_LOTS} | Cost: 0.5 bps + Spread")
    print("Sorting by 'Robust Return' = min(Val, Test, Full)")
    print("="*120 + "\n")
    
    if not os.path.exists(config.DIRS['FEATURE_MATRIX']):
        print("❌ Feature Matrix not found.")
        sys.exit(1)
        
    base_df = pd.read_parquet(config.DIRS['FEATURE_MATRIX'])
    backtester = BacktestEngine(base_df, cost_bps=0.5, annualization_factor=181440)
    
    horizons = config.PREDICTION_HORIZONS
    global_gene_counts = Counter()
    
    for h in horizons:
        file_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}.json")
        if not os.path.exists(file_path): continue
            
        print(f"\n--- Horizon: {h} Bars ---")
        try:
            with open(file_path, 'r') as f:
                strategies_data = json.load(f)
            
            strategies = []
            for d in strategies_data:
                try:
                    s = Strategy.from_dict(d)
                    s.generation_found = d.get('generation', '?')
                    strategies.append(s)
                except: pass
            
            if not strategies: continue

            # Batch Processing
            chunk_size = 1000
            all_horizon_results = []
            
            total_chunks = (len(strategies) + chunk_size - 1) // chunk_size
            # print(f"  Evaluating {len(strategies)} strategies in {total_chunks} batches...")
            
            for i in range(0, len(strategies), chunk_size):
                batch = strategies[i:i+chunk_size]
                batch_results = evaluate_batch(backtester, batch)
                all_horizon_results.extend(batch_results)
                backtester.reset_jit_context()
                # sys.stdout.write(f"\r    Batch {i//chunk_size + 1}/{total_chunks} done.")
                # sys.stdout.flush()
                
            # print("\n")
            
            if not all_horizon_results:
                print("  No active strategies found.")
                continue

            # Convert to DataFrame
            df_rows = []
            horizon_genes = []
            
            for res in all_horizon_results:
                strat = res['Strategy']
                strat_genes = [get_gene_description(g) for g in strat.long_genes + strat.short_genes]
                horizon_genes.extend(strat_genes)
                
                df_rows.append({
                    'Gen': res['Gen'],
                    'Name': strat.name,
                    'Ret%(Val)': res['Ret_Val'] * 100,
                    'Ret%(Test)': res['Ret_Test'] * 100,
                    'Ret%(Full)': res['Ret_Full'] * 100,
                    'Robust%': res['Robust_Ret'] * 100,
                    'Sortino(OOS)': res['Sortino_OOS'],
                    'Trades': int(res['Trades']),
                    'Genes': ", ".join(strat_genes[:2]) + ("..." if len(strat_genes)>2 else "")
                })
                
            global_gene_counts.update(horizon_genes)
            
            df = pd.DataFrame(df_rows)
            # Sort by Robust Return (Min of 3)
            df = df.sort_values(by='Robust%', ascending=False)
            
            # Prune to Top 10 as requested
            top_10_df = df.head(10).copy()
            
            # Formatting for Display
            display_df = top_10_df.copy()
            for col in ['Ret%(Val)', 'Ret%(Test)', 'Ret%(Full)', 'Robust%', 'Sortino(OOS)']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
                
            print(display_df.to_string(index=False))
            
            # Save Top 10 to JSON
            top_10_names = set(top_10_df['Name'].values)
            top_10_strategies = []
            for res in all_horizon_results:
                if res['Strategy'].name in top_10_names:
                    s_dict = res['Strategy'].to_dict()
                    # Add computed metrics for reference
                    s_dict['metrics'] = {
                        'robust_return': res['Robust_Ret'],
                        'val_return': res['Ret_Val'],
                        'test_return': res['Ret_Test'],
                        'full_return': res['Ret_Full'],
                        'sortino_oos': res['Sortino_OOS']
                    }
                    top_10_strategies.append(s_dict)
            
            # Sort JSON list to match the DataFrame order
            top_10_strategies.sort(key=lambda x: x['metrics']['robust_return'], reverse=True)
            
            out_path = os.path.join(config.DIRS['STRATEGIES_DIR'], f"apex_strategies_{h}_top10.json")
            with open(out_path, "w") as f:
                json.dump(top_10_strategies, f, indent=4)
            print(f"  💾 Pruned to Top 10. Saved to: {out_path}")
                
        except Exception as e:
            print(f"  Error processing horizon {h}: {e}")
            # import traceback; traceback.print_exc()

    print("\n" + "="*120)
    print("🧬 GLOBAL DOMINANT GENES 🧬")
    if global_gene_counts:
        for i, (g, c) in enumerate(global_gene_counts.most_common(20), 1):
            print(f"{i:2d}. {g:<40} ({c})")

if __name__ == "__main__":
    main()
