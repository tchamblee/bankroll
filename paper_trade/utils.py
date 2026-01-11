"""
Paper Trade Module - Utilities and Constants
"""
import logging
import os
import subprocess

import config as cfg
from utils import setup_logging

# --------------------------------------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------------------------------------

CLIENT_ID = cfg.IBKR_CLIENT_ID_PAPER
WINDOW_SIZE = 4000
WARMUP_DAYS = 7
LIVE_STATE_FILE = os.path.join(cfg.DIRS['OUTPUT_DIR'], "live_state.json")
LIVE_BARS_FILE = os.path.join(cfg.DIRS['PROCESSED_DIR'], "live_bars.parquet")

# --------------------------------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------------------------------

logger = setup_logging("PaperTrade", "paper_trade.log")

def configure_library_logging():
    """Attach ib_insync and asyncio loggers to our file handler."""
    try:
        file_handler = [h for h in logger.handlers if isinstance(h, logging.FileHandler)][0]
        logging.getLogger("ib_insync").addHandler(file_handler)
        logging.getLogger("asyncio").addHandler(file_handler)
        logging.getLogger("ib_insync").setLevel(logging.INFO)
        logging.getLogger("asyncio").setLevel(logging.WARNING)
    except Exception as e:
        logger.warning(f"Failed to attach file handler to libraries: {e}")

# --------------------------------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------------------------------

def play_sound(sound_file):
    """Plays a sound file using mpg123 in a non-blocking subprocess if enabled."""
    if not cfg.ENABLE_SOUND:
        return

    if os.path.exists(sound_file):
        try:
            subprocess.Popen(
                ['mpg123', '-q', sound_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except Exception as e:
            logger.error(f"Failed to play sound {sound_file}: {e}")
