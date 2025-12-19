import pandas as pd
import json
from feature_engine import FeatureEngine
from validate_features import triple_barrier_labels

def check_correlations():
    with open("data/survivors.json", "r") as f:
        survivors = json.load(f)
        
    print(f"Loaded {len(survivors)} survivors.")
    
    # Load Data Re-run (Simulating the purge script env) 
    DATA_PATH = "/home/tony/bankroll/data/raw_ticks"
    engine = FeatureEngine(DATA_PATH)
    primary_df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    engine.create_volume_bars(primary_df, volume_threshold=500)
    
    # Load Correlators
    for ticker, suffix in [("RAW_TICKS_TNX*.parquet", "_tnx"), 
                           ("RAW_TICKS_DXY*.parquet", "_dxy"), 
                           ("RAW_TICKS_BUND*.parquet", "_bund"),
                           ("RAW_TICKS_SPY*.parquet", "_spy")]:
        corr_df = engine.load_ticker_data(ticker)
        if corr_df is not None:
            engine.add_correlator_residual(corr_df, suffix=suffix)

    engine.add_features_to_bars(windows=[50, 100, 200, 400]) 
    engine.add_physics_features()
    engine.add_delta_features(lookback=10) 
    engine.add_delta_features(lookback=50) 
    
    df = engine.bars[survivors]
    
    # Focus on Volatility Cluster
    vol_cols = [c for c in survivors if 'volatility' in c]
    print(f"\nVolatility Cluster ({len(vol_cols)}): {vol_cols}")
    
    corr = df[vol_cols].corr(method='spearman')
    print(corr.round(3))

if __name__ == "__main__":
    check_correlations()
