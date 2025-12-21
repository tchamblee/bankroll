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
PREDICTION_HORIZONS = [30, 60, 90, 120, 180, 240, 300]

IBKR_HOST = "172.18.32.1"
IBKR_PORT = 4001

# --- TRADING CONSTRAINTS ---
ACCOUNT_SIZE = 30000.0  # USD
STANDARD_LOT_SIZE = 100000.0 # Units of EUR/USD
MIN_LOTS = 1
MAX_LOTS = 3

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