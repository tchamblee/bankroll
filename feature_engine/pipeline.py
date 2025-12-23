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
        data_dir = config.DIRS['DATA_CLEAN_TICKS']
        
    engine = FeatureEngine(data_dir)
    
    # 1. Load Primary (EUR/USD)
    # Using the CLEAN pattern
    print("Loading Primary Ticker (EURUSD)...")
    primary_df = engine.load_ticker_data("CLEAN_EURUSD.parquet")
    if primary_df is None:
        print("❌ Failed to load primary data.")
        return None
        
    engine.create_volume_bars(primary_df, volume_threshold=volume_threshold)
    
    # 2. Load Shared Data (Correlators & Macro)
    # We load them once here to avoid reloading in sub-modules
    print("Loading Shared Market Data...")
    
    # Define Loading Specs
    load_specs = {
        'tnx': "CLEAN_TNX.parquet",
        'dxy': "CLEAN_DXY.parquet",
        'bund': "CLEAN_BUND.parquet",
        'us2y': "CLEAN_US2Y.parquet",
        'schatz': "CLEAN_SCHATZ.parquet",
        'es': "CLEAN_ES.parquet",
        'zn': "CLEAN_ZN.parquet",
        '6e': "CLEAN_6E.parquet",
        'ibit': "CLEAN_IBIT.parquet"
    }
    
    data_cache = {}
    for key, pattern in load_specs.items():
        data_cache[key] = engine.load_ticker_data(pattern)

    # 3. Add Correlator Residuals (TNX, DXY, BUND)
    # Using cached data
    if data_cache['tnx'] is not None: engine.add_correlator_residual(data_cache['tnx'], suffix="_tnx")
    if data_cache['dxy'] is not None: engine.add_correlator_residual(data_cache['dxy'], suffix="_dxy")
    if data_cache['bund'] is not None: engine.add_correlator_residual(data_cache['bund'], suffix="_bund")

    # 4. Intermarket Robust Features (ES, ZN, 6E)
    # Pass dictionary of DataFrames
    intermarket_dfs = {
        '_es': data_cache['es'],
        '_zn': data_cache['zn'],
        '_6e': data_cache['6e']
    }
    engine.add_intermarket_features(intermarket_dfs)

    # DROP REDUNDANT RESIDUALS 
    # (Keep Betas, Drop Residuals for TNX/DXY as they are redundant with SPY/Other)
    if engine.bars is not None:
        drop_residuals = ['residual_tnx', 'residual_dxy', 'residual_bund']
        engine.bars.drop(columns=[c for c in drop_residuals if c in engine.bars.columns], inplace=True)
            
    # 5. Standard Features
    windows_list = [25, 50, 100, 200, 400, 800, 1600, 3200]
    engine.add_features_to_bars(windows=windows_list)
    
    # 6. Crypto Features
    engine.add_crypto_features(data_cache['ibit'])
    
    # 7. GDELT Features
    gdelt_df = engine.load_gdelt_data()
    if gdelt_df is not None:
        engine.add_gdelt_features(gdelt_df)
        
    # 8. Macro Voltage
    # Pass DataFrames directly
    engine.add_macro_voltage_features(
        us2y_df=data_cache['us2y'],
        schatz_df=data_cache['schatz'],
        tnx_df=data_cache['tnx'],
        bund_df=data_cache['bund'],
        windows=[50, 100]
    )
    
    # 9. Physics & Microstructure
    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_advanced_physics_features(windows=windows_list)
    
    # 8. Deltas (Flow)
    engine.add_delta_features(lookback=25)
    engine.add_delta_features(lookback=50)
    engine.add_delta_features(lookback=100)
    
    return engine