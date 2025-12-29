import os
import sys
import logging
import asyncio
import config as cfg
from utils import setup_logging

# --- BACKFILL CONFIGURATION ---
DAYS_TO_BACKFILL = 180  # Full year
TEST_PROBE = False      # SET TO TRUE FOR A QUICK 2-DAY TEST
CONCURRENT_SYMBOLS = 1  # Number of symbols to fetch in parallel
USE_RTH = False         # Include data outside Regular Trading Hours

# --- LOGGING ---
logger = setup_logging("Backfill", "backfill.log")

# Limit concurrent symbol backfills to stay within IB limits
symbol_semaphore = asyncio.Semaphore(CONCURRENT_SYMBOLS)

# --- CONTROL FLAGS ---
STOP_REQUESTED = False
