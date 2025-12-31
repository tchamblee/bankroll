import os
import sys
import logging
import config as cfg
from utils import setup_logging

# IBKR Config
DATA_DIR = cfg.DIRS["DATA_RAW_TICKS"]
CHUNK_SIZE = 1000
FLUSH_INTERVAL_SEC = 60 # Force flush every minute even if chunk not full
RECONNECT_DELAY = 15
DATA_TIMEOUT = 120  

# GDELT Config
GDELT_POLL_INTERVAL_SEC = 30           
GDELT_LAG_MINUTES = 5            

# GDELT Directories (Reuse if needed, but we delegate to ingest_gdelt package usually)
GDELT_ROOT = cfg.DIRS.get("DATA_GDELT", os.path.join(os.getcwd(), "data", "gdelt"))

# Logging
logger = setup_logging("ingest_live", "ingest_live.log")