import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS = {
    "DATA_DIR": os.path.join(BASE_DIR, "data"),
    "DATA_RAW_TICKS": os.path.join(BASE_DIR, "data", "raw_ticks"),
    "DATA_GDELT": os.path.join(BASE_DIR, "data", "gdelt"),
    
    "PROCESSED_DIR": os.path.join(BASE_DIR, "processed_data"),
    "DATA_CLEAN_TICKS": os.path.join(BASE_DIR, "processed_data", "clean_ticks"),
    "FEATURE_MATRIX": os.path.join(BASE_DIR, "processed_data", "feature_matrix.parquet"),
    
    "OUTPUT_DIR": os.path.join(BASE_DIR, "output"),
    "FEATURES_DIR": os.path.join(BASE_DIR, "output", "features"),
    "STRATEGIES_DIR": os.path.join(BASE_DIR, "output", "strategies"),
    "PLOTS_DIR": os.path.join(BASE_DIR, "output", "plots"),
    
    "LOGS": os.path.join(BASE_DIR, "logs"),
}

# Prediction Horizons (in bars)
# Target: 1h, 2h, 3h (assuming ~2 min per 250-tick bar)
# Added 120 (4h) and 240 (8h) for cost efficiency
PREDICTION_HORIZONS = [60, 120, 240]

IBKR_HOST = "172.18.32.1"
IBKR_PORT = 4001

# --- TRADING CONSTRAINTS ---
ACCOUNT_SIZE = 30000.0  # USD
STANDARD_LOT_SIZE = 100000.0 # Units of EUR/USD
MIN_LOTS = 1
MAX_LOTS = 3
COST_BPS = 0.25
SPREAD_BPS = 0.25
ANNUALIZATION_FACTOR = 114408 # ~454 bars/day * 252 days (Volume Threshold: 1B units)
DEFAULT_STOP_LOSS = 2.0 # ATR Multiplier
DEFAULT_TAKE_PROFIT = 4.0 # ATR Multiplier
DEFAULT_TIME_LIMIT = 120 # Bars (4 hours at 2-min bars)
MIN_TRADES_FOR_METRICS = 10
MIN_TRADES_COEFFICIENT = 1200 # target = max(10, coeff/horizon + 5)

# --- DATA & VALIDATION SETTINGS ---
TRAIN_SPLIT_RATIO = 0.6
VAL_SPLIT_RATIO = 0.8
WFV_FOLDS = 5

# --- EVOLUTIONARY ALGORITHM SETTINGS ---
EVO_BATCH_SIZE = 2000
COMPLEXITY_PENALTY_PER_GENE = 0.01
DOMINANCE_PENALTY_THRESHOLD = 0.4
DOMINANCE_PENALTY_MULTIPLIER = 3.0
ELITE_PERCENTAGE = 0.3
MUTATION_RATE = 0.33
IMMIGRATION_PERCENTAGE = 0.10

# --- TIME FILTERS ---
# London Open (08:00 UTC) to NY Close (17:00 EST -> ~22:00 UTC)
TRADING_START_HOUR = 8
TRADING_END_HOUR = 22

# --- GDELT SETTINGS ---
GDELT_KEYWORDS = {
    'EUR_LOCS': ['Europe', 'Brussels', 'Germany', 'France', 'Italy', 'Spain', 'EUR', 'Euro'],
    'USD_LOCS': ['United States', 'US', 'Washington', 'New York', 'America', 'Fed'],
    'CONFLICT_THEMES': 'ARMEDCONFLICT|CRISISLEX|UNREST',
    'EPU_THEMES': 'EPU',
    'INFLATION_THEMES': 'ECON_INFLATION|TAX_FNCACT',
    'CB_THEMES': 'CENTRAL_BANK',
    'ENERGY_THEMES': 'ENV_OIL|ECON_ENERGY_PRICES',
}