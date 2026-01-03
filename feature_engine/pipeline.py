import config
from .core import FeatureEngine
from .seasonality import add_seasonality_features
from .fred import add_fred_features_v2 as add_fred_features
from .cot import add_cot_features

def run_pipeline(engine, data_cache=None):
    """
    Applies the standardized feature engineering pipeline to an engine that already has bars.
    
    Args:
        engine (FeatureEngine): Initialized engine with .bars populated.
        data_cache (dict): Dictionary containing external DataFrames:
                           tnx, usdchf, bund, us2y, schatz, es, zn, 6e, ibit, gdelt, tick_nyse, trin_nyse.
    """
    if data_cache is None: data_cache = {}
    
    # Helper to safely get DF
    def get_df(key): return data_cache.get(key)

    # 3. Add Correlator Residuals (TNX, USDCHF, BUND)
    if get_df('tnx') is not None: engine.add_correlator_residual(get_df('tnx'), suffix="_tnx")
    if get_df('usdchf') is not None: engine.add_correlator_residual(get_df('usdchf'), suffix="_usdchf")
    if get_df('bund') is not None: engine.add_correlator_residual(get_df('bund'), suffix="_bund")

    # 4. Intermarket Robust Features (ES, ZN, 6E)
    raw_intermarket = {
        '_es': get_df('es'),
        '_zn': get_df('zn'),
        '_6e': get_df('6e'),
        '_tick_nyse': get_df('tick_nyse'),
        '_trin_nyse': get_df('trin_nyse')
    }
    intermarket_dfs = {k: v for k, v in raw_intermarket.items() if v is not None}
    if intermarket_dfs:
        engine.add_intermarket_features(intermarket_dfs)

    # DROP REDUNDANT RESIDUALS 
    if engine.bars is not None:
        drop_residuals = ['residual_tnx', 'residual_usdchf', 'residual_bund']
        engine.bars.drop(columns=[c for c in drop_residuals if c in engine.bars.columns], inplace=True)
            
    # 5. Standard Features
    windows_list = [25, 50, 100, 200, 400, 800, 1600, 3200]
    engine.add_features_to_bars(windows=windows_list)

    # 5b. Event Decay
    engine.add_event_decay_features(high_windows=[100, 200, 400], shock_windows=[50, 100])
    
    # 6. Crypto Features
    engine.add_crypto_features(get_df('ibit'))
    
    # 7. GDELT Features
    if get_df('gdelt') is not None:
        engine.add_gdelt_features(get_df('gdelt'))
        
    # 8. Macro Voltage
    engine.add_macro_voltage_features(
        us2y_df=get_df('us2y'),
        schatz_df=get_df('schatz'),
        tnx_df=get_df('tnx'),
        bund_df=get_df('bund'),
        windows=[50, 100]
    )
    
    # 9. Physics & Microstructure
    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_advanced_physics_features(windows=windows_list)
    
    # 10. Deltas
    engine.add_delta_features(lookback=25)
    engine.add_delta_features(lookback=50)
    engine.add_delta_features(lookback=100)

    # 11. Seasonality
    if engine.bars is not None:
        engine.bars = add_seasonality_features(engine.bars, lookback_days=20)

    # 12. FRED
    if engine.bars is not None:
        engine.bars = add_fred_features(engine.bars)

    # 13. COT
    if engine.bars is not None:
        engine.bars = add_cot_features(engine.bars)
        
    return engine

def create_full_feature_engine(data_dir=None, volume_threshold=250):
    """
    Initializes the FeatureEngine, loads all data (Primary, Correlators, Crypto, GDELT, Macro),
    and generates the complete feature set.
    """
    if data_dir is None:
        data_dir = config.DIRS['DATA_CLEAN_TICKS']
        
    engine = FeatureEngine(data_dir)
    
    # 1. Load Primary (EUR/USD)
    print("Loading Primary Ticker (EURUSD)...")
    primary_df = engine.load_ticker_data("CLEAN_EURUSD.parquet")
    if primary_df is None:
        print("❌ Failed to load primary data.")
        return None
        
    engine.create_volume_bars(primary_df, volume_threshold=volume_threshold)
    
    # 2. Load Shared Data (Correlators & Macro)
    print("Loading Shared Market Data...")
    load_specs = {
        'tnx': "CLEAN_TNX.parquet",
        'usdchf': "CLEAN_USDCHF.parquet",
        'bund': "CLEAN_BUND.parquet",
        'us2y': "CLEAN_US2Y.parquet",
        'schatz': "CLEAN_SCHATZ.parquet",
        'es': "CLEAN_ES.parquet",
        'zn': "CLEAN_ZN.parquet",
        '6e': "CLEAN_6E.parquet",
        'ibit': "CLEAN_IBIT.parquet",
        'tick_nyse': "CLEAN_TICK_NYSE.parquet",
        'trin_nyse': "CLEAN_TRIN_NYSE.parquet"
    }
    
    data_cache = {}
    for key, pattern in load_specs.items():
        data_cache[key] = engine.load_ticker_data(pattern)
        
    # Load GDELT
    data_cache['gdelt'] = engine.load_gdelt_data()
    
    # Run Standard Pipeline
    run_pipeline(engine, data_cache)
    
    return engine