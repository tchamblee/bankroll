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
    return "Unknown"

def main():
    print("\n" + "="*120)
    print("🏆 APEX STRATEGY REPORT (REAL-MONEY TRADING MODE) 🏆")
    print(f"Account: ${config.ACCOUNT_SIZE:,.0f} | Lots: {config.MIN_LOTS}-{config.MAX_LOTS} | Cost: 0.5 bps + Spread")
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
                    strategies.append(s)
                except Exception as e:
                    print(f"  Warning: Could not load strategy {d.get('name', 'Unknown')}: {e}")
            
            if not strategies: continue

            # Evaluate on Test Set
            res_df = backtester.evaluate_population(strategies, set_type='test')
            
            # Evaluate on Full History for context
            full_signals = backtester.generate_signal_matrix(strategies)
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

            df_data = []
            horizon_genes = []
            for i, strat in enumerate(strategies):
                row = res_df.iloc[i]
                if row['trades'] == 0 and full_rets_pct[i] == 0: continue
                
                strat_genes = [get_gene_description(g) for g in strat.long_genes + strat.short_genes]
                horizon_genes.extend(strat_genes)
                global_gene_counts.update(strat_genes)
                
                df_data.append({
                    'Name': strat.name,
                    'Sortino(OOS)': row['sortino'],
                    'Ret%(OOS)': row['total_return'] * 100,
                    'Ret%(Full)': full_rets_pct[i] * 100,
                    'Trades(OOS)': int(row['trades']),
                    'Stability': row['stability'],
                    'Status': 'PROFITABLE' if row['total_return'] > 0 else 'LOSS',
                    'Genes': ", ".join(strat_genes[:2]) + ("..." if len(strat_genes)>2 else "")
                })
            
            if df_data:
                df = pd.DataFrame(df_data).sort_values(by='Sortino(OOS)', ascending=False)
                for col in ['Sortino(OOS)', 'Ret%(OOS)', 'Ret%(Full)', 'Stability']:
                    df[col] = df[col].apply(lambda x: f"{x:.2f}")
                print(df.to_string(index=False))
            else:
                print("  No active strategies found.")
            
            print("-" * 120)
            print(f"  Dominant Genes (H{h}): " + ", ".join([f"{g}:{c}" for g,c in Counter(horizon_genes).most_common(5)]))
            
        except Exception as e:
            print(f"  Error processing horizon {h}: {e}")
            import traceback; traceback.print_exc()

    print("\n" + "="*120)
    print("🧬 GLOBAL DOMINANT GENES 🧬")
    if global_gene_counts:
        for i, (g, c) in enumerate(global_gene_counts.most_common(20), 1):
            print(f"{i:2d}. {g:<40} ({c})")

if __name__ == "__main__":
    main()
