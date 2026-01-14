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
COT_STALENESS_DAYS = 14  # Max age for COT data before considered stale (weekly data = 2 reports)

# Templates for formatted strings (usage: config.SURVIVORS_FILE_TEMPLATE.format(horizon))
SURVIVORS_FILE_TEMPLATE = os.path.join(DIRS['FEATURES_DIR'], "survivors_{}.json")
APEX_FILE_TEMPLATE = os.path.join(DIRS['STRATEGIES_DIR'], "apex_strategies_{}.json")

PRIMARY_TICKER = "ES"

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
STANDARD_LOT_SIZE = 50.0 # ES point value ($50/point)
MIN_LOTS = 1
MAX_LOTS = 10
COST_BPS = 0.02  # ES commission ~$2.50/contract at ~$5000/contract
SPREAD_BPS = 0.05  # ES spread ~0.25 points
MIN_RETURN_THRESHOLD = 0.001
MIN_SORTINO_THRESHOLD = 0.9
MIN_SORTINO_FLOOR = 0.3  # Minimum per-slice floor (no catastrophic regime failures)
MIN_HOF_SORTINO = 0.5  # Early gate for HOF entry (defense-in-depth before final 0.9 filter)
MIN_TEST_SORTINO = 1.5  # Minimum test Sortino for final acceptance (OOS quality gate)
MIN_VAL_SORTINO = 1.5  # Minimum validation Sortino (require OOS consistency across both val and test)
MIN_CPCV_THRESHOLD = 1.75  # Minimum CPCV score for robustness (filters overfit strategies)
SLIPPAGE_ATR_FACTOR = 0.1

# --- MATH CONSTANTS ---
EPSILON = 1e-9

# --- BAR DEFINITION ---
# ES trades ~2M contracts/day; target ~1.5-2 min bars
VOLUME_THRESHOLD = 5000  # Contracts per bar (ES uses contract volume)
AVG_BAR_MINS = 2.0 # Average duration of a volume bar in minutes

# Annualization based on Volume Density
# PLACEHOLDER for ES - recalculate after data collection:
#   df = pd.read_parquet('processed_data/feature_matrix.parquet')
#   trading_days = (df['time_start'].max() - df['time_start'].min()).days * (252/365)
#   ANNUALIZATION_FACTOR = int(len(df) / trading_days * 252)
# Estimate: ~100 bars/hour * 23 hours * 252 days = ~580,000
ANNUALIZATION_FACTOR = 35444  # ~141 bars/day with 5000-contract volume bars

ATR_FALLBACK_BPS = 50.0 # ES typical daily range ~50 points = 1%
MIN_ATR_BPS = 5.0 # ~25 points floor for ES

DEFAULT_STOP_LOSS = 2.0 # ATR Multiplier
DEFAULT_TAKE_PROFIT = 4.0 # ATR Multiplier
STOP_LOSS_OPTIONS = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
TAKE_PROFIT_OPTIONS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]

# Volatility-Scaled Barriers: Tighten SL when vol spikes during a trade
# If current_atr / entry_atr > VOL_SCALE_THRESHOLD, multiply SL by VOL_SCALE_TIGHTEN
VOL_SCALE_THRESHOLD = 1.5  # Vol must spike 50%+ to trigger tightening
VOL_SCALE_TIGHTEN = 0.8    # Tighten SL by 20% when triggered
# Weighted towards Market Orders (0.0), but allowing Limit Orders up to 0.3 ATR
# 0.0 appears 15 times (~75% chance). Small limits (0.05-0.2) are rare "mutations".
LIMIT_DIST_OPTIONS = [0.0] * 15 + [0.05, 0.1, 0.15, 0.2, 0.3]

DEFAULT_TIME_LIMIT = 120 # Bars (4 hours at 2-min bars)
MIN_TRADES_FOR_METRICS = 50  # Increased from 30 for statistical significance
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
TRAIN_START_DATE = "2024-08-01"
TRAIN_SPLIT_RATIO = 0.6
VAL_SPLIT_RATIO = 0.8

# Scaled trade minimums based on slice size (relative to train)
# Train=60%, Val=20%, Test=20% → Val/Test need 1/3 the trades of Train
# BUT: enforce hard floors for statistical validity (30+ trades for significance)
_TRAIN_SIZE = TRAIN_SPLIT_RATIO
_VAL_SIZE = VAL_SPLIT_RATIO - TRAIN_SPLIT_RATIO
_TEST_SIZE = 1.0 - VAL_SPLIT_RATIO
MIN_TRADES_TRAIN = MIN_TRADES_FOR_METRICS
MIN_TRADES_VAL = max(20, int(MIN_TRADES_FOR_METRICS * _VAL_SIZE / _TRAIN_SIZE))
MIN_TRADES_TEST = max(75, int(MIN_TRADES_FOR_METRICS * _TEST_SIZE / _TRAIN_SIZE))  # Critical OOS gate (75 for 18mo data)
MAX_TRAIN_TEST_DECAY = 0.50  # Reject if test_return < 50% of train_return (>50% decay = overfit)

WFV_FOLDS = 5
CPCV_N_FOLDS = 6
CPCV_N_TEST_FOLDS = 2
CPCV_MIN_TRADES_SLICE = 10  # Increased from 5 for per-fold validity

# --- EVOLUTIONARY ALGORITHM SETTINGS ---
EVO_BATCH_SIZE = 2000
GENE_COUNT_MIN = 1  # Allow single-gene strategies (less overfitting risk)
GENE_COUNT_MAX = 4
COMPLEXITY_PENALTY_PER_GENE = 0.10  # 10% penalty per gene to favor simpler strategies
DOMINANCE_PENALTY_THRESHOLD = 0.40
DOMINANCE_PENALTY_MULTIPLIER = 1.0
ELITE_PERCENTAGE = 0.3
MUTATION_RATE = 0.40
IMMIGRATION_PERCENTAGE = 0.20

# --- OPTIMIZER SETTINGS ---
OPTIMIZE_SL_OPTIONS = [1.0, 1.5, 2.0, 2.5, 3.0]
OPTIMIZE_TP_OPTIONS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
OPTIMIZE_JITTER_PCT = 0.05  # 5% parameter jitter for robustness testing
OPTIMIZE_MIN_IMPROVEMENT = 0.05  # 5% relative improvement required to beat parent
OPTIMIZE_USE_WALK_FORWARD = False  # Use walk-forward validation instead of train/val split
OPTIMIZE_STOPS_MIN_SORTINO = 1.0  # Minimum sortino threshold for stop optimization

# --- TIME FILTERS ---
# ES trades nearly 24h (Sun 5pm - Fri 4pm CT)
TRADING_START_HOUR = 0
TRADING_END_HOUR = 23

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
    # Primary - ES Futures
    {
        "name": PRIMARY_TICKER,
        "symbol": "ES",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CME",
        "mode": "BARS_TRADES_1MIN",
    },
    # US Treasury Rates - directly affect equity valuations
    {
        "name": "TNX",
        "symbol": "TNX",
        "secType": "IND",
        "currency": "USD",
        "exchange": "CBOE",
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
        "name": "ZN",
        "symbol": "ZN",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CBOT",
        "mode": "BARS_TRADES_1MIN",
    },
    # Volatility - critical for ES
    {
        "name": "VIX",
        "symbol": "VIX",
        "secType": "IND",
        "currency": "USD",
        "exchange": "CBOE",
        "mode": "BARS_TRADES_1MIN",
    },
    # Market Breadth - directly ES-relevant
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
    # Risk Sentiment Proxy
    {
        "name": "IBIT",
        "symbol": "IBIT",
        "secType": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "mode": "BARS_TRADES_1MIN",
    },
    # Index Correlators - divergence signals
    {
        "name": "NQ",
        "symbol": "NQ",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CME",
        "mode": "BARS_TRADES_1MIN",
    },
    {
        "name": "RTY",
        "symbol": "RTY",
        "secType": "CONTFUT",
        "currency": "USD",
        "exchange": "CME",
        "mode": "BARS_TRADES_1MIN",
    },
]

# --- CONTROL FLAGS ---
ENABLE_SOUND = True
STOP_REQUESTED = False
