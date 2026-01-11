import config
from .core import FeatureEngine
from .seasonality import add_seasonality_features
from .calendar import add_calendar_features
from .fred import add_fred_features_v2 as add_fred_features
from .cot import add_cot_features
from .experimental import add_experimental_features
from .physics import add_interaction_features
from .profile import add_market_profile_features
from .dsp import add_dsp_features
from .information_geometry import add_information_geometry_features

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
    # Refactor: Increased window to 200 for stable Beta
    if get_df('tnx') is not None: engine.add_correlator_residual(get_df('tnx'), suffix="_tnx", window=200)
    if get_df('usdchf') is not None: engine.add_correlator_residual(get_df('usdchf'), suffix="_usdchf", window=200)
    if get_df('bund') is not None: engine.add_correlator_residual(get_df('bund'), suffix="_bund", window=200)

    # 4. Intermarket Robust Features (ES, ZN, 6E)
    raw_intermarket = {
        '_es': get_df('es'),
        '_zn': get_df('zn'),
        '_6e': get_df('6e'),
        '_tick_nyse': get_df('tick_nyse'),
        '_trin_nyse': get_df('trin_nyse'),
        # Majors (The Matrix)
        '_gbpusd': get_df('gbpusd'),
        '_usdjpy': get_df('usdjpy'),
        # Spread Components (for explicit spread calc in intermarket.py)
        '_tnx': get_df('tnx'),
        '_bund': get_df('bund'),
        '_btp': get_df('btp'),
        '_us2y': get_df('us2y'),
        '_schatz': get_df('schatz'),
        '_vix': get_df('vix')
    }
    intermarket_dfs = {k: v for k, v in raw_intermarket.items() if v is not None}
    if intermarket_dfs:
        engine.add_intermarket_features(intermarket_dfs)

    # 5. Standard Features
    windows_list = [25, 50, 100, 200, 400, 800, 1600, 3200]
    engine.add_features_to_bars(windows=windows_list)
    
    # 5c. Experimental Features (Choppiness, Vortex, EOM)
    if engine.bars is not None:
        engine.bars = add_experimental_features(engine.bars, windows=[14, 100, 400])

    # 5b. Event Decay
    engine.add_event_decay_features(high_windows=[100, 200, 400], shock_windows=[50, 100])
    
    # 6. Crypto Features
    engine.add_crypto_features(get_df('ibit'))
    
    # 7. GDELT Features
    if get_df('gdelt') is not None:
        engine.add_gdelt_features(get_df('gdelt'))
        
    # 8. Macro Voltage
    # Refactor: Increased windows to [100, 200, 400] for stability
    engine.add_macro_voltage_features(
        us2y_df=get_df('us2y'),
        schatz_df=get_df('schatz'),
        tnx_df=get_df('tnx'),
        bund_df=get_df('bund'),
        windows=[100, 200, 400]
    )
    
    # 9. Physics & Microstructure
    engine.add_physics_features()
    engine.add_microstructure_features()
    engine.add_advanced_physics_features(windows=windows_list)
    
    # 9b. Interaction Features
    if engine.bars is not None:
        engine.bars = add_interaction_features(engine.bars, windows=[100, 200, 400])
        
    # 9c. Market Profile
    if engine.bars is not None:
        engine.bars = add_market_profile_features(engine.bars)

    # 9d. DSP Features (Fisher, Hilbert)
    if engine.bars is not None:
        engine.bars = add_dsp_features(engine.bars, windows=[20, 50, 100])
        
    # 9e. Information Geometry Features (Fisher Info Velocity)
    if engine.bars is not None:
        engine.bars = add_information_geometry_features(engine.bars, windows=[50, 100, 200])
    
    # 10. Deltas
    engine.add_delta_features(lookback=25)
    engine.add_delta_features(lookback=50)
    engine.add_delta_features(lookback=100)

    # 11. Seasonality
    if engine.bars is not None:
        engine.bars = add_seasonality_features(engine.bars, lookback_days=20)

    # 12. Economic Calendar (Event Risk)
    if engine.bars is not None:
        engine.bars = add_calendar_features(engine.bars)

    # 13. FRED
    if engine.bars is not None:
        engine.bars = add_fred_features(engine.bars)

    # 13. COT
    # Refactor: Removed COT features (Weekly data is step-function noise for intraday)
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
    
    # 1. Load Primary ({config.PRIMARY_TICKER})
    print(f"Loading Primary Ticker ({config.PRIMARY_TICKER})...")
    primary_df = engine.load_ticker_data(f"CLEAN_{config.PRIMARY_TICKER}.parquet")
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
        'btp': "CLEAN_BTP.parquet",
        'us2y': "CLEAN_US2Y.parquet",
        'schatz': "CLEAN_SCHATZ.parquet",
        'es': "CLEAN_ES.parquet",
        'zn': "CLEAN_ZN.parquet",
        '6e': "CLEAN_6E.parquet",
        'gbpusd': "CLEAN_GBPUSD.parquet",
        'usdjpy': "CLEAN_USDJPY.parquet",
        'ibit': "CLEAN_IBIT.parquet",
        'tick_nyse': "CLEAN_TICK.parquet",
        'trin_nyse': "CLEAN_TRIN.parquet",
        'vix': "CLEAN_VIX.parquet"
    }
    
    data_cache = {}
    for key, pattern in load_specs.items():
        data_cache[key] = engine.load_ticker_data(pattern)
        
    # Load GDELT
    data_cache['gdelt'] = engine.load_gdelt_data()
    
    # Run Standard Pipeline
    run_pipeline(engine, data_cache)
    
    return engine