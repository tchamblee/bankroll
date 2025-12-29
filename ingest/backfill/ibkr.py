import asyncio
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from ib_insync import IB, Contract, HistoricalTickBidAsk, HistoricalTickLast, HistoricalTick
import config as cfg
import backfill.config as bf_cfg
from utils import save_chunk
from .config import logger, USE_RTH, symbol_semaphore

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
    adjusted_date = target_date + timedelta(days=days_offset)
    t_str = adjusted_date.strftime("%Y%m%d")
    
    for d in chain:
        expiry = d.contract.lastTradeDateOrContractMonth.split()[0] 
        if expiry >= t_str:
            return d.contract
    return None

async def backfill_ticks(ib: IB, contract: Contract, name: str, start_dt: datetime, end_dt: datetime):
    """Specific logic for dense tick-by-tick data (FX/Futures)"""
    date_str = end_dt.strftime("%Y%m%d")
    filename = os.path.join(cfg.DIRS["DATA_RAW_TICKS"], f"{cfg.RAW_DATA_PREFIX_TICKS}_{name}_{date_str}.parquet")

    existing_max_ts = None

    if os.path.exists(filename):
        try:
            existing_df = pd.read_parquet(filename, columns=["ts_event"])
            if not existing_df.empty:
                last_ts = existing_df["ts_event"].max()
                if last_ts.tzinfo is None: last_ts = last_ts.replace(tzinfo=timezone.utc)
                existing_max_ts = last_ts
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
    
    what_to_show = "BID_ASK"
    if contract.secType in ["FUT", "CONTFUT", "STK"]:
        what_to_show = "BID_ASK" 

    retries = 0
    MAX_RETRIES = 5
    PACING_SLEEP = 11.0 

    while True:
        if bf_cfg.STOP_REQUESTED:
            logger.info("🛑 Stop requested. Aborting tick fetch.")
            break

        if existing_max_ts and current_end <= existing_max_ts:
            logger.info(f"      [MERGE] Reached existing data at {current_end}. Stopping fetch.")
            break

        try:
            if len(buffer) > 0: 
                await asyncio.sleep(PACING_SLEEP)

            ticks = await asyncio.wait_for(
                ib.reqHistoricalTicksAsync(
                    contract, startDateTime=None, endDateTime=current_end,
                    numberOfTicks=1000, whatToShow=what_to_show, useRth=USE_RTH, ignoreSize=False
                ),
                timeout=300.0
            )
            
            retries = 0

            if not ticks:
                break
                
            valid_ticks = [t for t in ticks if t.time >= start_dt]
            if not valid_ticks:
                break
                
            buffer.extend(valid_ticks)
            current_end = ticks[0].time - timedelta(microseconds=1)
            
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

    what_to_show = 'TRADES'
    use_rth_override = USE_RTH
    
    if contract.secType == 'CASH':
        what_to_show = 'MIDPOINT'
        
    if "TICK" in name or "TRIN" in name:
        use_rth_override = True
        logger.info(f"      [INTERNAL] Fetching Market Breadth: {name} (Forcing RTH)")

    MAX_RETRIES = 3
    for retry in range(MAX_RETRIES + 1):
        try:
            bars = await ib.reqHistoricalDataAsync(
                contract, endDateTime=req_end, durationStr=duration,
                barSizeSetting='1 min', whatToShow=what_to_show, useRTH=use_rth_override, formatDate=2,
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
                    df = pd.DataFrame(data).drop_duplicates(subset=["ts_event"])
                    df = df.sort_values(by="ts_event").reset_index(drop=True)
                    save_chunk(df, filename)
            break 

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

        if t_conf["secType"] == "CONTFUT":
            chain = await get_futures_chain(ib, t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
            rollover_days = t_conf.get("rollover_days", 10)
            resolved = find_contract_for_date(chain, target_date, rollover_days)
            if resolved:
                 resolved.includeExpired = True
                 use_contract = resolved
            else:
                 logger.warning(f"      Could not resolve {name} for {target_date}")
                 return
        
        start_dt = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        end_dt = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)

        if t_conf["mode"] == "TICKS_BID_ASK":
            await backfill_ticks(ib, use_contract, name, start_dt, end_dt)
        else:
            await backfill_bars(ib, use_contract, name, end_dt)