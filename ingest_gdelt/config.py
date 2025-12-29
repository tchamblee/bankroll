import sys
import os
import logging
import config as cfg
from utils import setup_logging

# Handle Windows console encoding if needed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# GDELT endpoints
GDELT_GKG_DAILY_BASE = "http://data.gdeltproject.org/gkg"
GDELT_V2_BASE = "http://data.gdeltproject.org/gdeltv2"

HTTP_TIMEOUT = 20                # seconds per request
POLL_INTERVAL_SEC = 30           # main loop sleep
GDELT_LAG_MINUTES = 5            # don't try to ingest last 5 minutes
INTRADAY_WINDOW_DAYS = 2         # look back this many days for intraday gaps
GKG_BACKFILL_DAYS = 7            # how many completed days of GKG to keep up to date

# Paths
GDELT_ROOT = cfg.DIRS.get("DATA_GDELT", os.path.join(os.getcwd(), "data", "gdelt"))

GKG_DAILY_DIR = os.path.join(GDELT_ROOT, "gkg_daily")
V2_EVENTS_DIR = os.path.join(GDELT_ROOT, "v2_events")
V2_MENTIONS_DIR = os.path.join(GDELT_ROOT, "v2_mentions")
V2_GKG_DIR = os.path.join(GDELT_ROOT, "v2_gkg")

os.makedirs(GKG_DAILY_DIR, exist_ok=True)
os.makedirs(V2_EVENTS_DIR, exist_ok=True)
os.makedirs(V2_MENTIONS_DIR, exist_ok=True)
os.makedirs(V2_GKG_DIR, exist_ok=True)

# Logging
logger = setup_logging("ingest_gdelt", "ingest_gdelt.log")