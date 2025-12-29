import datetime as dt
import time
import logging
import config as cfg

from .config import (
    logger,
    GDELT_ROOT,
    GKG_DAILY_DIR,
    V2_EVENTS_DIR,
    V2_MENTIONS_DIR,
    GDELT_LAG_MINUTES,
    POLL_INTERVAL_SEC
)
from .gkg_daily import ensure_gkg_recent
from .v2_events import ensure_v2_events_until
from .v2_mentions import ensure_v2_mentions_until
from .v2_gkg import ensure_v2_gkg_until

# FinBERT streaming integration (requires finbert_streaming.py in the project root or path)
try:
    import finbert_streaming
except ImportError:
    finbert_streaming = None
    logger.warning("finbert_streaming module not found. Sentiment updates will be skipped.")

def run_cycle():
    """Run a single ingestion cycle."""
    try:
        now_utc = dt.datetime.utcnow()
        cutoff = now_utc - dt.timedelta(minutes=GDELT_LAG_MINUTES)

        logger.info("=" * 80)
        logger.info("Tick @ %s (cutoff=%s)", now_utc.isoformat(), cutoff.isoformat())

        # 1) Ensure recent GKG 1.0 daily files exist (completed days only)
        ensure_gkg_recent()

        # 2) Ensure V2 events are ingested up to cutoff
        new_event_batches = ensure_v2_events_until(cutoff)

        # 3) Ensure V2 mentions are ingested up to cutoff
        ensure_v2_mentions_until(cutoff)
        
        # 4) Ensure V2 GKG is ingested up to cutoff (NEW: Intraday EPU/Themes)
        ensure_v2_gkg_until(cutoff)

        # 5) Incremental FinBERT sentiment updates for newly ingested event batches
        if new_event_batches and finbert_streaming:
            try:
                finbert_streaming.update_finbert_for_batches(
                    new_event_batches,
                    max_new_articles=300,  # tune if you want more/less HTTP + compute
                )
            except Exception as e:
                logger.error("🔥 FinBERT streaming update failed: %s", e, exc_info=True)
    except Exception as e:
        logger.error("🔥 Unexpected error in run_cycle: %s", e, exc_info=True)

def main_loop():
    logger.info("ALT_DATA_ROOT: %s", GDELT_ROOT)
    logger.info("GKG_DAILY_DIR: %s", GKG_DAILY_DIR)
    logger.info("V2_EVENTS_DIR: %s", V2_EVENTS_DIR)
    logger.info("V2_MENTIONS_DIR: %s", V2_MENTIONS_DIR)

    while True:
        if cfg.STOP_REQUESTED:
            logger.info("🛑 GDELT ingestion stopped by flag.")
            break
            
        try:
            run_cycle()
        except KeyboardInterrupt:
            logger.info("🛑 GDELT ingestion stopped by user.")
            break
        except Exception as e:
            logger.error("🔥 Unexpected error in main_loop: %s", e, exc_info=True)

        logger.info("Sleeping for %d seconds...", POLL_INTERVAL_SEC)
        time.sleep(POLL_INTERVAL_SEC)