import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIRS = {
    "DATA_RAW_TICKS": os.path.join(BASE_DIR, "data", "raw_ticks"),
    "LOGS": os.path.join(BASE_DIR, "logs"),
}

IBKR_HOST = "172.18.32.1"
IBKR_PORT = 4001
