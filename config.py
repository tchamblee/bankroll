import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS = {
    "DATA_DIR": os.path.join(BASE_DIR, "data"),
    "DATA_RAW_TICKS": os.path.join(BASE_DIR, "data", "raw_ticks"),
    "DATA_CLEAN_TICKS": os.path.join(BASE_DIR, "data", "clean_ticks"),
    "LOGS": os.path.join(BASE_DIR, "logs"),
}

# Prediction Horizons (in bars)
# Target: 1h, 2h, 3h (assuming ~2 min per 250-tick bar)
PREDICTION_HORIZONS = [30, 60, 90]

IBKR_HOST = "172.18.32.1"
IBKR_PORT = 4001
