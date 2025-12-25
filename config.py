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
PREDICTION_HORIZONS = [60, 90, 120, 180]

IBKR_HOST = "172.18.32.1"
IBKR_PORT = 4001

# --- TRADING CONSTRAINTS ---
ACCOUNT_SIZE = 30000.0  # USD
STANDARD_LOT_SIZE = 100000.0 # Units of EUR/USD
MIN_LOTS = 1
MAX_LOTS = 3
COST_BPS = 0.25
SPREAD_BPS = 0.25

# --- BAR DEFINITION ---
# 600M units is approx 250 ticks for EURUSD (High Resolution)
VOLUME_THRESHOLD = 600_000_000 

# Annualization based on Volume Density
# Approx 750 bars/day (for 600M threshold) * 252 days
ANNUALIZATION_FACTOR = 189000 

DEFAULT_STOP_LOSS = 2.0 # ATR Multiplier
DEFAULT_TAKE_PROFIT = 4.0 # ATR Multiplier
DEFAULT_TIME_LIMIT = 120 # Bars (4 hours at 2-min bars)
MIN_TRADES_FOR_METRICS = 50
MIN_TRADES_COEFFICIENT = 3000 # target = max(50, coeff/horizon + 5)
STOP_LOSS_COOLDOWN_BARS = 12 # Bars to wait after SL before re-entry (approx 1 hour)

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
MUTATION_RATE = 0.40
IMMIGRATION_PERCENTAGE = 0.20

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

TARGETS = [
    {
        "name": "EURUSD",
        "symbol": "EUR",
        "secType": "CASH",
        "currency": "USD",
        "exchange": "IDEALPRO",
        "mode": "TICKS_BID_ASK", 
    },
    {
        "name": "TNX",
        "symbol": "TNX",
        "secType": "IND",
        "currency": "USD",
        "exchange": "CBOE",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "BUND",
        "symbol": "GBL",
        "secType": "CONTFUT",
        "currency": "EUR",
        "exchange": "EUREX",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "USDCHF",
        "symbol": "USD",
        "secType": "CASH",
        "currency": "CHF",
        "exchange": "IDEALPRO",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "SPY",
        "symbol": "SPY",
        "secType": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "IBIT",
        "symbol": "IBIT",
        "secType": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "US2Y",
        "symbol": "ZT",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CBOT",
        "mode": "BARS_TRADES_1MIN",
        "rollover_days": 30,
    },
    {
        "name": "SCHATZ",
        "symbol": "GBS",
        "secType": "CONTFUT",
        "currency": "EUR",
        "exchange": "EUREX",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "6E",
        "symbol": "EUR",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CME",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "ES",
        "symbol": "ES",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CME",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "ZN",
        "symbol": "ZN",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CBOT",
        "mode": "BARS_TRADES_1MIN",
    }
]