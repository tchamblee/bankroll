import asyncio
import os
import sys
import logging
import requests
import zipfile
import io
import pandas as pd
import polars as pl
from datetime import datetime, timedelta, timezone
from ib_insync import *
import nest_asyncio
import config as cfg

nest_asyncio.apply()

# --- WINDOWS CONSOLE FIX ---
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# --- BACKFILL CONFIGURATION ---
DAYS_TO_BACKFILL = 180  # Full year
TEST_PROBE = False      # SET TO TRUE FOR A QUICK 2-DAY TEST
CONCURRENT_SYMBOLS = 1  # Number of symbols to fetch in parallel
USE_RTH = False         # Include data outside Regular Trading Hours

# --- LOGGING ---
os.makedirs(cfg.DIRS["LOGS"], exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(cfg.DIRS["LOGS"], "backfill.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Backfill")

# Limit concurrent symbol backfills to stay within IB limits
symbol_semaphore = asyncio.Semaphore(CONCURRENT_SYMBOLS)

# --- GDELT CONFIG ---
GDELT_GKG_BASE = "http://data.gdeltproject.org/gkg"
GDELT_V2_BASE = "http://data.gdeltproject.org/gdeltv2"
os.makedirs(cfg.DIRS["DATA_GDELT"], exist_ok=True)
V2_GKG_DIR = os.path.join(cfg.DIRS["DATA_GDELT"], "v2_gkg")
os.makedirs(V2_GKG_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# GDELT FUNCTIONS
# --------------------------------------------------------------------------------------

def download_gdelt_v2_day(target_date: datetime):
    """
    Downloads all 15-minute GKG 2.0 files for the target date.
    Iterates from 00:00 to 23:45.
    """
    # Generate 15-min timestamps for the day
    start_dt = datetime.combine(target_date, datetime.min.time())
    
    # 96 intervals per day
    for i in range(96):
        ts = start_dt + timedelta(minutes=15 * i)
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(V2_GKG_DIR, f"gdelt_v2_gkg_{ts_str}.parquet")
        
        if os.path.exists(filename):
            continue
            
        url = f"{GDELT_V2_BASE}/{ts_str}.gkg.csv.zip"
        
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                # Silent skip for missing files (common in older dates or glitches)
                continue
            resp.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                if not zf.namelist(): continue
                with zf.open(zf.namelist()[0]) as csv_file:
                    # GKG 2.0 Parsing
                    # Use quote_char=None and encoding='utf8-lossy' as per fix
                    df = pl.read_csv(
                        csv_file.read(), 
                        separator="\t", 
                        has_header=False, 
                        ignore_errors=True, 
                        infer_schema_length=0,
                        quote_char=None,
                        encoding="utf8-lossy"
                    )
                    
                    if df.width < 27: continue
                    
                    cols = df.columns
                    rename_map = {
                        cols[0]: "GKGRECORDID",
                        cols[1]: "DATE",
                        cols[3]: "SourceCommonName",
                        cols[4]: "DocumentIdentifier",
                        cols[5]: "Counts",
                        cols[7]: "Themes",
                        cols[9]: "Locations",
                        cols[15]: "V2Tone",
                    }
                    
                    df = df.rename({k:v for k,v in rename_map.items() if k in df.columns})
                    keep = ["GKGRECORDID", "DATE", "SourceCommonName", "DocumentIdentifier", "Counts", "Themes", "Locations", "V2Tone"]
                    existing = [c for c in keep if c in df.columns]
                    df = df.select(existing)
                    
                    # Parse Tone
                    if "V2Tone" in df.columns:
                        df = df.with_columns(
                            pl.col("V2Tone").cast(pl.Utf8).str.split(",").alias("_tone_list")
                        )
                        df = df.with_columns(
                            pl.col("_tone_list").list.get(0).cast(pl.Float64).alias("tone_mean"),
                            pl.col("_tone_list").list.get(3).cast(pl.Float64).alias("tone_polarity"),
                            pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y%m%d%H%M%S", strict=False).alias("date_utc"),
                            pl.lit(ts_str).alias("batch_ts")
                        ).drop("_tone_list")
                    
                    df.write_parquet(filename)
                    # logger.info(f"   [GDELT V2] Saved {ts_str}")

        except Exception as e:
            logger.warning(f"   [GDELT V2] Error {ts_str}: {e}")

def download_gdelt_gkg(target_date: datetime):
    """Downloads and processes the Daily GKG 1.0 file for the target date."""
    date_str = target_date.strftime("%Y%m%d")
    filename = os.path.join(cfg.DIRS["DATA_GDELT"], f"GDELT_GKG_{date_str}.parquet")
    
    if os.path.exists(filename):
        logger.info(f"   [GDELT] {date_str} exists. Skipping.")
        return

    url = f"{GDELT_GKG_BASE}/{date_str}.gkg.csv.zip"
    logger.info(f"   [GDELT] Fetching {url}...")

    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code == 404:
            logger.warning(f"   [GDELT] 404 Not Found for {date_str} (might be too recent or missing)")
            return
        resp.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            if not zf.namelist():
                return
            with zf.open(zf.namelist()[0]) as csv_file:
                # Use Polars for fast CSV parsing
                df = pl.read_csv(csv_file.read(), separator="\t", has_header=True, ignore_errors=True, infer_schema_length=0)
                
                # Keep relevant columns for volume/themes
                # GKG 1.0 Cols: DATE, NUMARTS, COUNTS, THEMES, LOCATIONS, PERSONS, ORGANIZATIONS, TONE...
                # We want mainly NUMARTS, THEMES, TONE, COUNTS
                
                # Filter/Rename
                # Note: Schema checking is skipped for brevity, assuming standard GKG 1.0 layout
                if "DATE" in df.columns:
                    df = df.rename({"DATE": "date_str", "NUMARTS": "num_arts", "TONE": "tone_raw"})
                    
                    # Select subset to save space
                    cols_to_keep = [c for c in ["date_str", "num_arts", "tone_raw", "THEMES", "LOCATIONS", "COUNTS"] if c in df.columns]
                    df = df.select(cols_to_keep)
                    
                    df.write_parquet(filename)
                    logger.info(f"   [GDELT] Saved {filename} ({df.height} rows)")
                else:
                    logger.warning(f"   [GDELT] Unexpected columns in {date_str}")

    except Exception as e:
        logger.error(f"   [GDELT] Error processing {date_str}: {e}")

# --------------------------------------------------------------------------------------
# IBKR FUNCTIONS
# --------------------------------------------------------------------------------------

FUTURES_CHAINS = {} # symbol -> list of contract details

async def get_futures_chain(ib: IB, symbol: str, exchange: str, currency: str):
    if symbol in FUTURES_CHAINS:
        return FUTURES_CHAINS[symbol]
    
    logger.info(f"   [CHAIN] Fetching futures chain for {symbol}...")
    contract = Contract(secType='FUT', symbol=symbol, exchange=exchange, currency=currency, includeExpired=True)
    try:
        details = await ib.reqContractDetailsAsync(contract)
        # Sort by expiry
        sorted_details = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)
        FUTURES_CHAINS[symbol] = sorted_details
        return sorted_details
    except Exception as e:
        logger.error(f"   [CHAIN] Error fetching chain for {symbol}: {e}")
        return []

def find_contract_for_date(chain, target_date, days_offset=10):
    # target_date is a datetime.date or datetime
    # We want to switch to the next contract BEFORE the current one expires/becomes illiquid.
    # Rollover usually happens ~1 week before expiry.
    # Adding a buffer (e.g., 10 days) forces selection of the next contract 
    # if the current one is within 10 days of expiry.
    
    adjusted_date = target_date + timedelta(days=days_offset)
    t_str = adjusted_date.strftime("%Y%m%d")
    
    for d in chain:
        # Expiry is usually YYYYMMDD
        expiry = d.contract.lastTradeDateOrContractMonth.split()[0] 
        if expiry >= t_str:
            return d.contract
    return None

def save_chunk(df: pd.DataFrame, filename: str):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_parquet(filename, index=False)
        logger.info(f"Saved {filename} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Write Error {filename}: {e}")

async def backfill_ticks(ib: IB, contract: Contract, name: str, start_dt: datetime, end_dt: datetime):
    """Specific logic for dense tick-by-tick data (FX/Futures)"""
    date_str = end_dt.strftime("%Y%m%d")
    filename = os.path.join(cfg.DIRS["DATA_RAW_TICKS"], f"{cfg.RAW_DATA_PREFIX_TICKS}_{name}_{date_str}.parquet")

    existing_max_ts = None

    if os.path.exists(filename):
        # SMART SKIP: Check if file looks complete (ends near end of day)
        try:
            # Read just timestamp column to check coverage
            existing_df = pd.read_parquet(filename, columns=["ts_event"])
            if not existing_df.empty:
                last_ts = existing_df["ts_event"].max()
                if last_ts.tzinfo is None: last_ts = last_ts.replace(tzinfo=timezone.utc)
                
                existing_max_ts = last_ts

                # If data goes up to within 4 hours of end_dt, consider it complete.
                if (end_dt - last_ts).total_seconds() < 14400: 
                    logger.info(f"   [SKIP] {name} {date_str} exists and looks complete (Last: {last_ts}).")
                    return
                else:
                    logger.warning(f"   [REPAIR] {name} {date_str} ends early ({last_ts}). Fetching missing tail...")
            else:
                logger.warning(f"   [REPAIR] {name} {date_str} exists but is empty. Re-fetching...")
        except Exception as e:
             logger.warning(f"   [REPAIR] {name} {date_str} exists but could not be read ({e}). Re-fetching...")

    logger.info(f"   [TICK] Fetching {name} for {date_str}...")
    current_end = end_dt
    total_saved = 0
    buffer = []
    
    # Determine whatToShow based on contract
    what_to_show = "BID_ASK"
    if contract.secType in ["FUT", "CONTFUT", "STK"]:
        what_to_show = "BID_ASK" # We want bid/ask if possible

    retries = 0
    MAX_RETRIES = 5
    
    # IBKR PACING: 60 requests per 10 minutes (approx 1 req every 10s).
    # We use 11s to be safe. 
    PACING_SLEEP = 11.0 

    while True:
        # STOP CONDITION: If we've reached the point where we have data
        if existing_max_ts and current_end <= existing_max_ts:
            logger.info(f"      [MERGE] Reached existing data at {current_end}. Stopping fetch.")
            break

        try:
            # Throttle BEFORE request to ensure spacing
            if len(buffer) > 0: 
                await asyncio.sleep(PACING_SLEEP)

            ticks = await asyncio.wait_for(
                ib.reqHistoricalTicksAsync(
                    contract, startDateTime=None, endDateTime=current_end,
                    numberOfTicks=1000, whatToShow=what_to_show, useRth=USE_RTH, ignoreSize=False
                ),
                timeout=300.0
            )
            
            # Reset retries on success
            retries = 0

            if not ticks:
                break
                
            valid_ticks = [t for t in ticks if t.time >= start_dt]
            if not valid_ticks:
                break
                
            buffer.extend(valid_ticks)
            current_end = ticks[0].time - timedelta(microseconds=1)
            
            # Flush every 10k ticks
            if len(buffer) >= 10000:
                data = []
                for t in buffer:
                    if isinstance(t, HistoricalTickBidAsk):
                        data.append({
                            "ts_event": t.time, "pricebid": t.priceBid, "priceask": t.priceAsk,
                            "sizebid": t.sizeBid, "sizeask": t.sizeAsk,
                            "last_price": float("nan"), "last_size": float("nan"), "volume": float("nan")
                        })
                    elif isinstance(t, (HistoricalTickLast, HistoricalTick)):
                        data.append({
                            "ts_event": t.time, "pricebid": float("nan"), "priceask": float("nan"),
                            "sizebid": float("nan"), "sizeask": float("nan"),
                            "last_price": t.price, "last_size": t.size, "volume": float("nan")
                        })
                
                if data:
                    df = pd.DataFrame(data).drop_duplicates(subset=["ts_event", "pricebid", "priceask", "last_price"])
                    
                    # LOAD & MERGE to keep file consistent during long fetches
                    if os.path.exists(filename):
                        existing_df = pd.read_parquet(filename)
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset=["ts_event", "pricebid", "priceask", "last_price"])
                        combined_df = combined_df.sort_values(by="ts_event")
                        combined_df.to_parquet(filename, index=False)
                    else:
                        df = df.sort_values(by="ts_event")
                        df.to_parquet(filename, index=False)
                        
                    total_saved += len(data)
                    logger.info(f"      Saved {len(data)} ticks (Total: {total_saved}). Cursor: {current_end}")
                
                buffer = []

        except (asyncio.TimeoutError, Exception) as e:
            retries += 1
            if retries > MAX_RETRIES:
                logger.error(f"      MAX RETRIES EXCEEDED fetching ticks for {name} @ {current_end}. Error: {e}")
                break
            
            is_pacing_error = "162" in str(e) or "cancelled" in str(e).lower()
            
            if is_pacing_error:
                wait_time = 600 
                logger.warning(f"      PACING VIOLATION detected for {name}. Sleeping {wait_time}s to reset IBKR limits...")
            else:
                wait_time = 5 * retries
                logger.warning(f"      Retry {retries}/{MAX_RETRIES} fetching ticks for {name} @ {current_end} after error: {e}. Waiting {wait_time}s...")
            
            await asyncio.sleep(wait_time)
            
    # Final Flush
    if buffer:
        data = []
        for t in buffer:
            if isinstance(t, HistoricalTickBidAsk):
                data.append({
                    "ts_event": t.time, "pricebid": t.priceBid, "priceask": t.priceAsk,
                    "sizebid": t.sizeBid, "sizeask": t.sizeAsk,
                    "last_price": float("nan"), "last_size": float("nan"), "volume": float("nan")
                })
            elif isinstance(t, (HistoricalTickLast, HistoricalTick)):
                data.append({
                    "ts_event": t.time, "pricebid": float("nan"), "priceask": float("nan"),
                    "sizebid": float("nan"), "sizeask": float("nan"),
                    "last_price": t.price, "last_size": t.size, "volume": float("nan")
                })
        
        if data:
            df = pd.DataFrame(data).drop_duplicates(subset=["ts_event", "pricebid", "priceask", "last_price"])
            if os.path.exists(filename):
                existing_df = pd.read_parquet(filename)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["ts_event", "pricebid", "priceask", "last_price"])
                combined_df = combined_df.sort_values(by="ts_event")
                combined_df.to_parquet(filename, index=False)
            else:
                df = df.sort_values(by="ts_event")
                df.to_parquet(filename, index=False)
            logger.info(f"      Final flush {len(data)} ticks. Total for day: {total_saved + len(data)}")

async def backfill_bars(ib: IB, contract: Contract, name: str, end_dt: datetime, duration="86400 S"):
    date_str = end_dt.strftime("%Y%m%d")
    filename = os.path.join(cfg.DIRS["DATA_RAW_TICKS"], f"{cfg.RAW_DATA_PREFIX_BARS}_{name}_{date_str}.parquet")

    existing_len = 0
    if os.path.exists(filename):
        try:
            existing_len = len(pd.read_parquet(filename))
        except: pass

    logger.info(f"   [BARS] Fetching {name} for {date_str}...")
    
    req_end = end_dt
    if end_dt.date() >= datetime.now(timezone.utc).date():
        req_end = ''

    # Correct whatToShow for Asset Type
    what_to_show = 'TRADES'
    if contract.secType == 'CASH':
        what_to_show = 'MIDPOINT'

    MAX_RETRIES = 3
    for retry in range(MAX_RETRIES + 1):
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract, endDateTime=req_end, durationStr=duration,
                barSizeSetting='1 min', whatToShow=what_to_show, useRTH=USE_RTH, formatDate=2,
                timeout=300.0
            )
            
            if bars:
                data = [{
                    "ts_event": b.date.replace(tzinfo=timezone.utc) if b.date.tzinfo is None else b.date,
                    "pricebid": float("nan"), "priceask": float("nan"), "sizebid": float("nan"), "sizeask": float("nan"),
                    "last_price": b.close, "last_size": float("nan"), "volume": b.volume
                } for b in bars]
                
                if existing_len > 0:
                    if len(data) > existing_len:
                         logger.warning(f"      [REPAIR] {name} {date_str}: Replaced {existing_len} rows with {len(data)} rows.")
                         df = pd.DataFrame(data).drop_duplicates(subset=["ts_event"])
                         df = df.sort_values(by="ts_event").reset_index(drop=True)
                         save_chunk(df, filename)
                    elif len(data) == existing_len:
                         logger.info(f"      [VERIFY OK] {name} {date_str}: Count matches ({existing_len}). Skipping write.")
                         return
                    else:
                         logger.warning(f"      [IGNORE] {name} {date_str}: Fetched fewer rows ({len(data)}) than existing ({existing_len}). Keeping existing.")
                         return
                else:
                    # New file
                    df = pd.DataFrame(data).drop_duplicates(subset=["ts_event"])
                    df = df.sort_values(by="ts_event").reset_index(drop=True)
                    save_chunk(df, filename)
            break # Success

        except (asyncio.TimeoutError, Exception) as e:
            if retry < MAX_RETRIES:
                wait_time = 5 * (retry + 1)
                logger.warning(f"      Retry {retry+1}/{MAX_RETRIES} fetching bars for {name} {date_str} after error: {e}. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"      FINAL FAILURE fetching bars for {name} {date_str}: {e}")

async def process_symbol_for_day(ib: IB, contract: Contract, t_conf: dict, target_date: datetime):
    async with symbol_semaphore:
        name = t_conf["name"]
        use_contract = contract

        # DYNAMIC RESOLUTION FOR CONTINUOUS FUTURES
        if t_conf["secType"] == "CONTFUT":
            chain = await get_futures_chain(ib, t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
            rollover_days = t_conf.get("rollover_days", 10)
            resolved = find_contract_for_date(chain, target_date, rollover_days)
            if resolved:
                 resolved.includeExpired = True
                 use_contract = resolved
                 # logger.info(f"      Resolved {name} -> {use_contract.localSymbol} (Expiry: {use_contract.lastTradeDateOrContractMonth})")
            else:
                 logger.warning(f"      Could not resolve {name} for {target_date}")
                 return
        
        start_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)

        if t_conf["mode"] == "TICKS_BID_ASK":
            await backfill_ticks(ib, use_contract, name, start_dt, end_dt)
        else:
            await backfill_bars(ib, use_contract, name, end_dt)

async def main():
    ib = IB()
    try:
        await ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=103)
        logger.info(f"CONNECTED. Test Probe: {TEST_PROBE}")
        
        # 1. Pre-qualify contracts
        logger.info("Qualifying contracts...")
        qualified_contracts = []
        for t_conf in cfg.TARGETS:
            contract = None
            if t_conf["secType"] == "CASH": 
                contract = Forex(t_conf["symbol"] + t_conf["currency"], exchange=t_conf["exchange"])
            elif t_conf["secType"] == "IND": 
                contract = Index(t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
            elif t_conf["secType"] == "CONTFUT":
                contract = ContFuture(t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
            elif t_conf["secType"] == "STK": 
                contract = Stock(t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
            
            qual = await ib.qualifyContractsAsync(contract)
            if not qual:
                logger.error(f"Could not qualify {t_conf['name']}")
                continue
            
            # FIX: Resolve ContFuture to specific underlying contract
            final_contract = qual[0]
            # (Removed incorrect ContFuture resolution block to allow continuous data fetching)

            qualified_contracts.append((t_conf, final_contract))
            logger.info(f"   Verified: {t_conf['name']}")

        # 2. Iterate Days (Recent -> Oldest)
        days = 2 if TEST_PROBE else DAYS_TO_BACKFILL
        logger.info(f"Starting backfill for {days} days (skipping today)...")

        for i in range(1, days + 1):
            target_date = datetime.now(timezone.utc).date() - timedelta(days=i)
            
            # Skip Weekends
            if target_date.weekday() >= 5:
                logger.info(f"Skipping Weekend: {target_date}")
                continue

            logger.info(f"Processing Date: {target_date}")
            
            # --- 2a. Fetch GDELT Data ---
            # We run this synchronously or in executor because it's HTTP, not IBKR async
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, download_gdelt_gkg, target_date)
            await loop.run_in_executor(None, download_gdelt_v2_day, target_date)

            # --- 2b. Fetch IBKR Data ---
            tasks = []
            for t_conf, contract in qualified_contracts:
                tasks.append(process_symbol_for_day(ib, contract, t_conf, target_date))
            
            await asyncio.gather(*tasks)
            
            await asyncio.sleep(0.5)

    except Exception as e:
        logger.error(f"Fatal Error: {e}")
    finally:
        ib.disconnect()
        logger.info("DONE.")

if __name__ == "__main__":
    asyncio.run(main())
