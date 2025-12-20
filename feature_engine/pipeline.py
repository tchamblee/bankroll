import config
from .core import FeatureEngine

def create_full_feature_engine(data_dir=None, volume_threshold=250):
    """
    Initializes the FeatureEngine, loads all data (Primary, Correlators, Crypto, GDELT, Macro),
    and generates the complete feature set.
    
    Returns:
        engine (FeatureEngine): The populated engine with engine.bars containing all features.
    """
    if data_dir is None:
        data_dir = config.DIRS['DATA_RAW_TICKS']
        
    engine = FeatureEngine(data_dir)
    
    # 1. Load Primary (EUR/USD)
    print("Loading Primary Ticker (EURUSD)...")
    primary_df = engine.load_ticker_data("RAW_TICKS_EURUSD*.parquet")
    if primary_df is None:
        print("❌ Failed to load primary data.")
        return None
        
    engine.create_volume_bars(primary_df, volume_threshold=volume_threshold)
    
    # 2. Load Correlators
    correlators = [
        ("RAW_TICKS_TNX*.parquet", "_tnx"), 
        ("RAW_TICKS_DXY*.parquet", "_dxy"), 
        ("RAW_TICKS_BUND*.parquet", "_bund"),
        ("RAW_TICKS_SPY*.parquet", "_spy")
    ]
    
    for ticker, suffix in correlators:
        corr_df = engine.load_ticker_data(ticker)
        if corr_df is not None:
            engine.add_correlator_residual(corr_df, suffix=suffix)

    # DROP REDUNDANT RESIDUALS (Keep Betas, Drop Residuals for TNX/DXY as they are redundant with SPY)
    # They survived beta check but failed residual check.
    if engine.bars is not None:
        drop_residuals = ['residual_tnx', 'residual_dxy']
        engine.bars.drop(columns=[c for c in drop_residuals if c in engine.bars.columns], inplace=True)
            
    # 3. Standard Features
    windows_list = [25, 50, 100, 200, 400, 800, 1600, 3200]
    engine.add_features_to_bars(windows=windows_list)
    
    # 4. Crypto Features
    engine.add_crypto_features("CLEAN_IBIT.parquet")
    
    # 5. GDELT Features
    gdelt_df = engine.load_gdelt_data()
    if gdelt_df is not None:
        engine.add_gdelt_features(gdelt_df)
        
    # 6. Macro Voltage
    engine.add_macro_voltage_features()
    
    # 7. Physics & Microstructure
    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_advanced_physics_features(windows=windows_list)
    
    # 8. Deltas (Flow)
    engine.add_delta_features(lookback=10)
    engine.add_delta_features(lookback=50)
    engine.add_delta_features(lookback=100)
    
    return engine
