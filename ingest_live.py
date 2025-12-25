import sys
import asyncio
import os
import logging
import io
import time
import zipfile
import datetime as dt
from datetime import datetime, timezone
from typing import Optional, List, Set

import pandas as pd
import polars as pl
import requests
from ib_insync import *
import nest_asyncio
import config as cfg

# Try importing FinBERT streaming (optional)
try:
    import finbert_streaming
except ImportError:
    finbert_streaming = None

nest_asyncio.apply()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

# IBKR Config
DATA_DIR = cfg.DIRS["DATA_RAW_TICKS"]
CHUNK_SIZE = 1000
RECONNECT_DELAY = 15
DATA_TIMEOUT = 120  

# GDELT Config
GDELT_GKG_DAILY_BASE = "http://data.gdeltproject.org/gkg"
GDELT_V2_BASE = "http://data.gdeltproject.org/gdeltv2"
HTTP_TIMEOUT = 20                
GDELT_POLL_INTERVAL_SEC = 30           
GDELT_LAG_MINUTES = 5            
INTRADAY_WINDOW_DAYS = 2         
GKG_BACKFILL_DAYS = 7            

# GDELT Directories
# Use cfg.DIRS["DATA_GDELT"] as base if available, else fallback
GDELT_ROOT = cfg.DIRS.get("DATA_GDELT", os.path.join(os.getcwd(), "data", "gdelt"))
GKG_DAILY_DIR = os.path.join(GDELT_ROOT, "gkg_raw")
V2_EVENTS_DIR = os.path.join(GDELT_ROOT, "v2_events")
V2_MENTIONS_DIR = os.path.join(GDELT_ROOT, "v2_mentions")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GKG_DAILY_DIR, exist_ok=True)
os.makedirs(V2_EVENTS_DIR, exist_ok=True)
os.makedirs(V2_MENTIONS_DIR, exist_ok=True)

# Logging
LOG_DIR = cfg.DIRS.get("LOGS", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, # Changed to INFO to reduce noise, use DEBUG for specific modules if needed
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ingest_live.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# GDELT UTILITIES
# --------------------------------------------------------------------------------------

def http_get(url: str) -> Optional[bytes]:
    """HTTP GET with basic error handling."""
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code == 404:
            # logger.debug("HTTP 404: %s", url) 
            return None
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        logger.warning("HTTP error for %s: %s", url, e)
        return None

def unzip_single_file(zip_bytes: bytes) -> Optional[io.BytesIO]:
    """Unzip a GDELT zip (single CSV) into a BytesIO."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            if not names:
                return None
            with zf.open(names[0]) as f:
                data = f.read()
        return io.BytesIO(data)
    except zipfile.BadZipFile as e:
        logger.warning("Bad zip file: %s", e)
        return None

def quarter_hour_floor(dt_obj: dt.datetime) -> dt.datetime:
    minute = (dt_obj.minute // 15) * 15
    return dt_obj.replace(minute=minute, second=0, microsecond=0)

def quarter_hour_range(start_dt: dt.datetime, end_dt: dt.datetime) -> List[dt.datetime]:
    cur = quarter_hour_floor(start_dt)
    end_dt = quarter_hour_floor(end_dt)
    out = []
    while cur <= end_dt:
        out.append(cur)
        cur += dt.timedelta(minutes=15)
    return out

def list_processed_ts(dir_path: str, prefix: str) -> Set[str]:
    result = set()
    if not os.path.exists(dir_path):
        return result
    for fname in os.listdir(dir_path):
        if not fname.startswith(prefix) or not fname.endswith(".parquet"):
            continue
        base = fname.replace(".parquet", "")
        parts = base.split("_")
        ts = parts[-1]
        if len(ts) == 14 and ts.isdigit():
            result.add(ts)
    return result

# --------------------------------------------------------------------------------------
# GKG 1.0 DAILY SUMMARY
# --------------------------------------------------------------------------------------

def download_gkg_daily(date_: dt.date) -> Optional[pl.DataFrame]:
    ymd = date_.strftime("%Y%m%d")
    url = f"{GDELT_GKG_DAILY_BASE}/{ymd}.gkg.csv.zip"
    
    raw_bytes = http_get(url)
    if raw_bytes is None: return None
    csv_buffer = unzip_single_file(raw_bytes)
    if csv_buffer is None: return None

    try:
        df = pl.read_csv(csv_buffer, separator="\t", has_header=True, ignore_errors=True)
    except Exception as e:
        logger.warning(f"[GKG {ymd}] Failed to read CSV: {e}")
        return None

    # Rename & Clean
    rename_map = {
        "DATE": "date_yyyymmdd", "NUMARTS": "num_articles", "COUNTS": "Counts",
        "THEMES": "V1Themes", "LOCATIONS": "V1Locations", "PERSONS": "V1Persons",
        "ORGANIZATIONS": "V1Organizations", "TONE": "ToneRaw",
        "CAMEOEVENTIDS": "CameoEventIds", "SOURCES": "SourceCommonName",
        "SOURCEURLS": "DocumentIdentifier",
    }
    # Only rename cols that exist
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(actual_rename)

    if "date_yyyymmdd" in df.columns:
        df = df.with_columns(
            pl.col("date_yyyymmdd").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias("date_utc")
        )
    
    if "ToneRaw" in df.columns:
        df = df.with_columns(pl.col("ToneRaw").cast(pl.Utf8).str.split(",").alias("_tone_list"))
        df = df.with_columns(
            pl.col("_tone_list").list.get(0).cast(pl.Float64).alias("tone_mean"),
            pl.col("_tone_list").list.get(1).cast(pl.Float64).alias("tone_positive"),
            pl.col("_tone_list").list.get(2).cast(pl.Float64).alias("tone_negative"),
            pl.col("_tone_list").list.get(3).cast(pl.Float64).alias("tone_polarity"),
        ).drop("_tone_list")

    return df

def ensure_gkg_recent():
    today = dt.datetime.now(timezone.utc).date()
    start_day = today - dt.timedelta(days=1)
    
    for i in range(GKG_BACKFILL_DAYS):
        day = start_day - dt.timedelta(days=i)
        ymd = day.strftime("%Y%m%d")
        out_path = os.path.join(GKG_DAILY_DIR, f"gdelt_gkg_{ymd}.parquet")
        if os.path.exists(out_path): continue
        
        df = download_gkg_daily(day)
        if df is not None and not df.is_empty():
            df.write_parquet(out_path)
            logger.info(f"[GKG] Downloaded {ymd}")

# --------------------------------------------------------------------------------------
# GDELT 2.0 EVENTS
# --------------------------------------------------------------------------------------

def download_v2_events_for_ts(ts_str: str) -> Optional[pl.DataFrame]:
    url = f"{GDELT_V2_BASE}/{ts_str}.export.CSV.zip"
    raw_bytes = http_get(url)
    if raw_bytes is None: return None
    buf = unzip_single_file(raw_bytes)
    if buf is None: return None

    try:
        df = pl.read_csv(buf, separator="\t", has_header=False, ignore_errors=True, infer_schema_length=0)
    except: return None
    
    if df.width < 61: return df

    cols = df.columns
    rename_map = {
        cols[0]: "GlobalEventID", cols[1]: "SQLDATE", cols[26]: "EventCode",
        cols[31]: "NumMentions", cols[33]: "NumArticles", cols[34]: "AvgTone",
        cols[53]: "ActionGeo_CountryCode", cols[60]: "SOURCEURL"
    }
    df = df.rename(rename_map)
    df = df.with_columns(pl.lit(ts_str).alias("batch_ts"))
    return df

def ensure_v2_events_until(cutoff: dt.datetime) -> List[str]:
    processed_ts = list_processed_ts(V2_EVENTS_DIR, "gdelt_v2_events")
    ts_list = quarter_hour_range(cutoff - dt.timedelta(days=INTRADAY_WINDOW_DAYS), cutoff)
    new_batches = []
    
    for ts in ts_list:
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        if ts_str in processed_ts: continue
        
        df = download_v2_events_for_ts(ts_str)
        if df is not None and not df.is_empty():
            out_path = os.path.join(V2_EVENTS_DIR, f"gdelt_v2_events_{ts_str}.parquet")
            df.write_parquet(out_path)
            logger.info(f"[EVT] Downloaded {ts_str}")
            new_batches.append(ts_str)
            
    return new_batches

# --------------------------------------------------------------------------------------
# GDELT 2.0 MENTIONS
# --------------------------------------------------------------------------------------

def download_v2_mentions_for_ts(ts_str: str) -> Optional[pl.DataFrame]:
    url = f"{GDELT_V2_BASE}/{ts_str}.mentions.CSV.zip"
    raw_bytes = http_get(url)
    if raw_bytes is None: return None
    buf = unzip_single_file(raw_bytes)
    if buf is None: return None

    try:
        df = pl.read_csv(buf, separator="\t", has_header=False, ignore_errors=True, infer_schema_length=0)
    except: return None

    if df.width < 14: return df
    
    cols = df.columns
    rename_map = {
        cols[0]: "GlobalEventID", cols[1]: "EventTimeDate", cols[2]: "MentionTimeDate",
        cols[4]: "MentionSourceName", cols[7]: "MentionDocTone", cols[13]: "Confidence"
    }
    df = df.rename(rename_map)
    df = df.with_columns(pl.lit(ts_str).alias("batch_ts"))
    return df

def ensure_v2_mentions_until(cutoff: dt.datetime):
    processed_ts = list_processed_ts(V2_MENTIONS_DIR, "gdelt_v2_mentions")
    ts_list = quarter_hour_range(cutoff - dt.timedelta(days=INTRADAY_WINDOW_DAYS), cutoff)
    
    for ts in ts_list:
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        if ts_str in processed_ts: continue
        
        df = download_v2_mentions_for_ts(ts_str)
        if df is not None and not df.is_empty():
            out_path = os.path.join(V2_MENTIONS_DIR, f"gdelt_v2_mentions_{ts_str}.parquet")
            df.write_parquet(out_path)
            logger.info(f"[MNT] Downloaded {ts_str}")

def on_error(reqId, errorCode, errorString, contract):
    logger.error(f"IB Error {errorCode}: {errorString} (ReqId: {reqId})")

# --------------------------------------------------------------------------------------
# IBKR INGESTION
# --------------------------------------------------------------------------------------

last_data_time = datetime.now(timezone.utc)

def save_chunk(df: pd.DataFrame, filename: str) -> bool:
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename):
            existing_df = pd.read_parquet(filename)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_parquet(filename, index=False)
        else:
            df.to_parquet(filename, index=False)
        return True
    except Exception as e:
        logger.error(f"Write Error {filename}: {e}")
        return False

async def ingest_stream(ib: IB, stop_event: asyncio.Event):
    global last_data_time

    logger.info("🔌 Setting up Market Data Subscriptions...")
    contract_map = {}
    l1_cols = ["ts_event", "pricebid", "priceask", "sizebid", "sizeask", "last_price", "last_size", "volume"]
    buffers = { t["name"]: [] for t in cfg.TARGETS }

    for t_conf in cfg.TARGETS:
        contract = None
        # Simplified Contract Setup
        if t_conf["secType"] == "CASH":
            contract = Forex(t_conf["symbol"] + t_conf["currency"], exchange=t_conf["exchange"])
        elif t_conf["secType"] == "IND":
            contract = Index(t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
        elif t_conf["secType"] == "FUT":
            contract = Future(symbol=t_conf["symbol"], lastTradeDateOrContractMonth=t_conf["lastTradeDate"], exchange=t_conf["exchange"], currency=t_conf["currency"])
        elif t_conf["secType"] == "CONTFUT":
            contract = ContFuture(symbol=t_conf["symbol"], exchange=t_conf["exchange"], currency=t_conf["currency"])
        elif t_conf["secType"] == "STK":
            contract = Stock(t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
        else:
            continue

        qual = await ib.qualifyContractsAsync(contract)
        if not qual:
            logger.error(f"Could not qualify {t_conf['name']}")
            continue
        
        contract = qual[0]
        if t_conf["secType"] == "CONTFUT":
            specific_contract = Contract(conId=contract.conId)
            await ib.qualifyContractsAsync(specific_contract)
            contract = specific_contract

        contract_map[contract.conId] = t_conf

        if t_conf["mode"] == "TICKS_BID_ASK":
            ib.reqMktData(contract, "", False, False)
            ib.reqTickByTickData(contract, "BidAsk", numberOfTicks=0, ignoreSize=False)
            if t_conf["secType"] != "CASH":
                ib.reqTickByTickData(contract, "AllLast", numberOfTicks=0, ignoreSize=False)
            logger.info(f"   ✅ Subscribed Ticks: {contract.localSymbol}")
        
        elif t_conf["mode"] == "BARS_TRADES_1MIN":
            ib.reqMktData(contract, "", False, False)
            logger.info(f"   ✅ Subscribed Updates: {contract.localSymbol}")

    logger.info("🔴 STREAM ACTIVE. Waiting for events...")

    last_data_time = datetime.now(timezone.utc)
    loop = asyncio.get_running_loop()
    
    def on_new_tick(tickers):
        global last_data_time
        updated = False
        for ticker in tickers:
            c_id = ticker.contract.conId
            if c_id not in contract_map: continue
            conf = contract_map[c_id]
            name = conf["name"]
            
            if conf["mode"] == "TICKS_BID_ASK":
                if ticker.tickByTicks:
                    for tick in ticker.tickByTicks:
                        ts = tick.time
                        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                        
                        if isinstance(tick, TickByTickBidAsk):
                            buffers[name].append([ts, tick.bidPrice, tick.askPrice, tick.bidSize, tick.askSize, float("nan"), float("nan"), float("nan")])
                            updated = True
                        elif isinstance(tick, (TickByTickAllLast, TickByTickLast)):
                            buffers[name].append([ts, float("nan"), float("nan"), float("nan"), float("nan"), tick.price, tick.size, float("nan")])
                            updated = True

            elif conf["mode"] == "BARS_TRADES_1MIN":
                ts = ticker.time
                if not ts: ts = datetime.now(timezone.utc)
                elif ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                last_p = ticker.last if ticker.last and not pd.isna(ticker.last) else ticker.close
                if last_p and not pd.isna(last_p):
                     buffers[name].append([ts, float("nan"), float("nan"), float("nan"), float("nan"), last_p, float("nan"), float("nan")])
                     updated = True

        if updated:
            last_data_time = datetime.now(timezone.utc)

    ib.pendingTickersEvent += on_new_tick

    # Main Loop with Stop Check
    while not stop_event.is_set():
        await asyncio.sleep(1) 
        now_utc = datetime.now(timezone.utc)
        if (now_utc - last_data_time).total_seconds() > DATA_TIMEOUT:
            logger.warning(f"⚠️ NO IBKR DATA received for {DATA_TIMEOUT}s.")

        for name, buffer in buffers.items():
            if len(buffer) >= CHUNK_SIZE:
                chunk_data = list(buffer) 
                buffers[name] = [] 
                chunk_data.sort(key=lambda x: x[0])
                df = pd.DataFrame(chunk_data, columns=l1_cols)
                if not df.empty:
                    df["day"] = df["ts_event"].apply(lambda x: x.strftime("%Y%m%d"))
                    for day_str, group in df.groupby("day"):
                        fn = os.path.join(DATA_DIR, f"RAW_TICKS_{name}_{day_str}.parquet")
                        save_cols = [c for c in l1_cols] 
                        await loop.run_in_executor(None, save_chunk, group[save_cols], fn)
                        print(f"\r💾 Saved {name} Chunk to {fn} ({len(group)} rows)...", end="")

# --------------------------------------------------------------------------------------
# GDELT ASYNC TASK
# --------------------------------------------------------------------------------------

def run_gdelt_cycle_sync():
    """Blocking GDELT cycle to be run in executor."""
    try:
        now_utc = dt.datetime.now(timezone.utc)
        cutoff = now_utc - dt.timedelta(minutes=GDELT_LAG_MINUTES)
        
        # logger.info(f"--- GDELT Sync Cycle @ {now_utc.strftime('%H:%M:%S')} ---")
        ensure_gkg_recent()
        new_event_batches = ensure_v2_events_until(cutoff)
        ensure_v2_mentions_until(cutoff)
        
        if new_event_batches and finbert_streaming:
            try:
                finbert_streaming.update_finbert_for_batches(new_event_batches, max_new_articles=300)
            except Exception as e:
                logger.error(f"FinBERT Error: {e}")
                
    except Exception as e:
        logger.error(f"GDELT Cycle Error: {e}", exc_info=True)

async def gdelt_monitor(stop_event: asyncio.Event):
    """Async wrapper for GDELT monitoring."""
    logger.info("🌍 GDELT Monitor Started")
    loop = asyncio.get_running_loop()
    
    while not stop_event.is_set():
        # Check stop before running long sync task
        if stop_event.is_set(): break
        
        await loop.run_in_executor(None, run_gdelt_cycle_sync)
        
        # Sleep in small chunks to be responsive to stop_event
        for _ in range(GDELT_POLL_INTERVAL_SEC):
            if stop_event.is_set(): break
            await asyncio.sleep(1)

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------

async def main_loop():
    logger.info(f"📂 IBKR Data Lake: {DATA_DIR}")
    logger.info(f"📂 GDELT Data Lake: {GDELT_ROOT}")
    
    stop_event = asyncio.Event()
    
    # Start GDELT Monitor as separate task
    gdelt_task = asyncio.create_task(gdelt_monitor(stop_event))

    ib = IB()
    ib.errorEvent += on_error
    
    try:
        logger.info("🔄 Connecting to IBKR...")
        await asyncio.wait_for(ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=0), timeout=10)
        ib.reqMarketDataType(3) 
        
        # Run IBKR Ingestion
        await ingest_stream(ib, stop_event)
        
    except (OSError, ConnectionError) as e:
        logger.error(f"⚠️ Connection Drop: {e}")
    except asyncio.CancelledError:
        logger.info("🛑 Main Loop Cancelled")
    except Exception as e:
        logger.error(f"🔥 Error: {e}", exc_info=True)
    finally:
        logger.info("🛑 Shutting down...")
        stop_event.set() # Signal GDELT to stop
        
        try: 
            ib.disconnect()
            logger.info("🔌 IBKR Disconnected")
        except: pass
        
        # Wait for GDELT task to finish gracefully
        if not gdelt_task.done():
            logger.info("⏳ Waiting for GDELT task to finish...")
            try:
                await asyncio.wait_for(gdelt_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("⚠️ GDELT task timed out, cancelling...")
                gdelt_task.cancel()
                try: await gdelt_task
                except: pass

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        # Pass here, the finally block in main_loop won't run if run() is interrupted directly
        # But asyncio.run handles signal handlers in newer python...
        # Ideally we catch it inside the coroutine or let asyncio.run finish
        logger.info("👋 Keyboard Interrupt received. Exiting.")