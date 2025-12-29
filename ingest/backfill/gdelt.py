import datetime as dt
import os
import requests
from ingest_gdelt.gkg_daily import download_gkg_daily
from ingest_gdelt.v2_gkg import download_v2_gkg_for_ts
from ingest_gdelt.config import GKG_DAILY_DIR, V2_GKG_DIR
from .config import logger
import backfill.config as bf_cfg

def download_gdelt_v2_day(target_date: dt.date):
    """
    Downloads all 15-minute GKG 2.0 files for the target date.
    Iterates from 00:00 to 23:45.
    """
    start_dt = dt.datetime.combine(target_date, dt.datetime.min.time())
    
    # 96 intervals per day
    for i in range(96):
        if bf_cfg.STOP_REQUESTED:
            logger.info("🛑 Stop requested. Aborting GDELT backfill.")
            return

        ts = start_dt + dt.timedelta(minutes=15 * i)
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(V2_GKG_DIR, f"gdelt_v2_gkg_{ts_str}.parquet")
        
        # logger.debug(f"Checking for existence: {filename}")
        if os.path.exists(filename):
            # logger.debug(f"Skipping {ts_str}, already exists.")
            continue
            
        try:
            df = download_v2_gkg_for_ts(ts_str)
            if df is not None and not df.is_empty():
                df.write_parquet(filename)
                # logger.info(f"   [GDELT V2] Saved {ts_str}")
        except Exception as e:
            logger.warning(f"   [GDELT V2] Error {ts_str}: {e}")

def download_gdelt_gkg(target_date: dt.date):
    """Downloads and processes the Daily GKG 1.0 file for the target date."""
    ymd = target_date.strftime("%Y%m%d")
    filename = os.path.join(GKG_DAILY_DIR, f"gdelt_gkg_{ymd}.parquet")
    
    if os.path.exists(filename):
        logger.info(f"   [GDELT] {ymd} exists. Skipping.")
        return

    try:
        df = download_gkg_daily(target_date)
        if df is not None and not df.is_empty():
            df.write_parquet(filename)
            logger.info(f"   [GDELT] Saved {filename} ({df.height} rows)")
        else:
            logger.warning(f"   [GDELT] No data for {ymd} or download failed.")
    except Exception as e:
        logger.error(f"   [GDELT] Error processing {ymd}: {e}")
