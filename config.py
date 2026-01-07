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
    "STRATEGY_INBOX": os.path.join(BASE_DIR, "output", "strategies", "found_strategies.json"),
    "PLOTS_DIR": os.path.join(BASE_DIR, "output", "plots"),
    
    "LOGS": os.path.join(BASE_DIR, "logs"),
}

# --- SHARED FILE PATHS & CONSTANTS ---
CANDIDATES_FILE = os.path.join(DIRS['STRATEGIES_DIR'], "candidates.json")
MUTEX_PORTFOLIO_FILE = os.path.join(DIRS['STRATEGIES_DIR'], "mutex_portfolio.json")
PURGE_MARKER_FILE = os.path.join(DIRS['FEATURES_DIR'], "PURGE_COMPLETE")
FRED_DATA_FILE = os.path.join(DIRS['DATA_DIR'], "fred_macro_daily.parquet")
COT_DATA_FILE = os.path.join(DIRS['DATA_DIR'], "cot_weekly.parquet")

# Templates for formatted strings (usage: config.SURVIVORS_FILE_TEMPLATE.format(horizon))
SURVIVORS_FILE_TEMPLATE = os.path.join(DIRS['FEATURES_DIR'], "survivors_{}.json")
APEX_FILE_TEMPLATE = os.path.join(DIRS['STRATEGIES_DIR'], "apex_strategies_{}.json")

PRIMARY_TICKER = "EURUSD"

# Data File Conventions
RAW_DATA_PREFIX_TICKS = "RAW_TICKS"
RAW_DATA_PREFIX_BARS = "RAW_BARS"

# Prediction Horizons (in bars)
PREDICTION_HORIZONS = [60, 90, 120, 180, 240]

IBKR_HOST = os.getenv("IBKR_HOST", "172.18.32.1")
IBKR_PORT = 4001
IBKR_CLIENT_ID_INGEST = 1
IBKR_CLIENT_ID_PAPER = 2

# --- TRADING CONSTRAINTS ---
ACCOUNT_SIZE = 60000.0  # USD
RISK_PER_TRADE_PERCENT = 0.01 # 1% of account per trade (Target Risk)
STANDARD_LOT_SIZE = 100000.0 # Units of EUR/USD
MIN_LOTS = 1
MAX_LOTS = 4
COST_BPS = 0.20
SPREAD_BPS = 0.25
MIN_RETURN_THRESHOLD = 0.001
MIN_SORTINO_THRESHOLD = 1.5
SLIPPAGE_ATR_FACTOR = 0.1

# --- MATH CONSTANTS ---
EPSILON = 1e-9

# --- BAR DEFINITION ---
# 600M units is approx 250 ticks for PRIMARY_TICKER (High Resolution)
VOLUME_THRESHOLD = 600_000_000 
AVG_BAR_MINS = 1.5 # Average duration of a volume bar in minutes

# Annualization based on Volume Density
# Approx 1342 bars/day (from Data Analysis 2026-01-03) * 252 days
ANNUALIZATION_FACTOR = 338363

ATR_FALLBACK_BPS = 10.0 # 0.1% of price
MIN_ATR_BPS = 1.5 # 0.015% of price (~1.8 pips floor for EURUSD)

DEFAULT_STOP_LOSS = 2.0 # ATR Multiplier
DEFAULT_TAKE_PROFIT = 4.0 # ATR Multiplier
STOP_LOSS_OPTIONS = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
TAKE_PROFIT_OPTIONS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
# Weighted towards Market Orders (0.0), but allowing Limit Orders up to 1.0 ATR
LIMIT_DIST_OPTIONS = [0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]

DEFAULT_TIME_LIMIT = 120 # Bars (4 hours at 2-min bars)
MIN_TRADES_FOR_METRICS = 50
MIN_TRADES_COEFFICIENT = 3000 # target = max(50, coeff/horizon + 5)
STOP_LOSS_COOLDOWN_BARS = 12 # Bars to wait after SL before re-entry (approx 1 hour)
MIN_COMMISSION = 2.0
COMMISSION_THRESHOLD = 1e-6

# --- SIMULATION & EVALUATION ---
VOLATILITY_NORMALIZATION_WINDOW = 1000
ATR_WINDOW = 50
STABILITY_PENALTY_THRESHOLD = 0.6
STABILITY_PENALTY_FACTOR = 0.6

# --- DATA & VALIDATION SETTINGS ---
TRAIN_START_DATE = "2025-07-07"
TRAIN_SPLIT_RATIO = 0.6
VAL_SPLIT_RATIO = 0.8
WFV_FOLDS = 5
CPCV_N_FOLDS = 6
CPCV_N_TEST_FOLDS = 2
CPCV_MIN_TRADES_SLICE = 3

# --- EVOLUTIONARY ALGORITHM SETTINGS ---
EVO_BATCH_SIZE = 2000
GENE_COUNT_MIN = 2
GENE_COUNT_MAX = 4
COMPLEXITY_PENALTY_PER_GENE = 0.02
DOMINANCE_PENALTY_THRESHOLD = 0.40
DOMINANCE_PENALTY_MULTIPLIER = 1.0
ELITE_PERCENTAGE = 0.3
MUTATION_RATE = 0.40
IMMIGRATION_PERCENTAGE = 0.20

# --- TIME FILTERS ---
# London Open (08:00 UTC) to NY Close (17:00 EST -> ~22:00 UTC)
TRADING_START_HOUR = 8
TRADING_END_HOUR = 22

# --- GDELT SETTINGS ---
PANIC_SCORE_THRESHOLD = -2.0
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
        "name": PRIMARY_TICKER,
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
        "name": "GBPUSD",
        "symbol": "GBP",
        "secType": "CASH",
        "currency": "USD",
        "exchange": "IDEALPRO",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "USDJPY",
        "symbol": "USD",
        "secType": "CASH",
        "currency": "JPY",
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
    },
    {
        "name": "TICK_NYSE",
        "symbol": "TICK-NYSE",
        "secType": "IND",
        "currency": "USD",
        "exchange": "NYSE",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "TRIN_NYSE",
        "symbol": "TRIN-NYSE",
        "secType": "IND",
        "currency": "USD",
        "exchange": "NYSE",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "VIX",
        "symbol": "VIX",
        "secType": "IND",
        "currency": "USD",
        "exchange": "CBOE",
        "mode": "BARS_TRADES_1MIN",
    }
]

# --- CONTROL FLAGS ---
ENABLE_SOUND = False
STOP_REQUESTED = False
