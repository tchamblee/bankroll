import asyncio
import datetime as dt
from datetime import datetime, timezone
import os
import logging
from pathlib import Path
import pandas as pd
from ib_insync import IB, Forex, Index, Future, ContFuture, Stock, Contract, TickByTickBidAsk, TickByTickAllLast
import config as cfg
from utils import save_chunk
from .config import logger, DATA_DIR, CHUNK_SIZE, DATA_TIMEOUT, RECONNECT_DELAY, FLUSH_INTERVAL_SEC

class IBKRStreamer:
    def __init__(self):
        self.ib = IB()
        self.ib.errorEvent += self.on_error
        self.last_data_time = datetime.now(timezone.utc)
        self.buffers = {}
        self.contract_map = {}
        self.name_to_mode = {}
        self.l1_cols = ["ts_event", "pricebid", "priceask", "sizebid", "sizeask", "last_price", "last_size", "volume"]

    def on_error(self, reqId, errorCode, errorString, contract):
        logger.error(f"IB Error {errorCode}: {errorString} (ReqId: {reqId})")

    async def connect_and_stream(self, stop_event: asyncio.Event):
        try:
            logger.info("🔄 Connecting to IBKR...")
            await asyncio.wait_for(self.ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=0), timeout=15)
            self.ib.reqMarketDataType(3) 
            logger.info("✅ Connected. Starting Stream...")
            await self.ingest_stream(stop_event)
        except (OSError, ConnectionError, asyncio.TimeoutError) as e:
            logger.warning(f"⚠️ Connection Drop/Timeout: {e}")
        except asyncio.CancelledError:
            logger.info("🛑 IBKR Task Cancelled")
            raise
        except Exception as e:
            logger.error(f"🔥 IBKR Error: {e}", exc_info=True)
        finally:
            try: 
                self.ib.disconnect()
                logger.info("🔌 IBKR Disconnected")
            except: pass

    async def ingest_stream(self, stop_event: asyncio.Event):
        logger.info("🔌 Setting up Market Data Subscriptions...")
        self.contract_map = {}
        self.name_to_mode = {t["name"]: t["mode"] for t in cfg.TARGETS}
        self.buffers = {t["name"]: [] for t in cfg.TARGETS}

        for t_conf in cfg.TARGETS:
            contract = None
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

            qual = await self.ib.qualifyContractsAsync(contract)
            if not qual:
                logger.error(f"Could not qualify {t_conf['name']}")
                continue
            
            contract = qual[0]
            if t_conf["secType"] == "CONTFUT":
                specific_contract = Contract(conId=contract.conId)
                await self.ib.qualifyContractsAsync(specific_contract)
                contract = specific_contract

            self.contract_map[contract.conId] = t_conf

            if t_conf["mode"] == "TICKS_BID_ASK":
                self.ib.reqMktData(contract, "", False, False)
                self.ib.reqTickByTickData(contract, "BidAsk", numberOfTicks=0, ignoreSize=False)
                if t_conf["secType"] != "CASH":
                    self.ib.reqTickByTickData(contract, "AllLast", numberOfTicks=0, ignoreSize=False)
                logger.info(f"   ✅ Subscribed Ticks: {contract.localSymbol}")
            
            elif t_conf["mode"] == "BARS_TRADES_1MIN":
                self.ib.reqMktData(contract, "", False, False)
                if "TICK" in t_conf["name"] or "TRIN" in t_conf["name"]:
                    logger.info(f"   📊 Monitoring Market Breadth: {contract.localSymbol}")
                else:
                    logger.info(f"   ✅ Subscribed Updates: {contract.localSymbol}")

        logger.info("🔴 STREAM ACTIVE. Waiting for events...")

        self.last_data_time = datetime.now(timezone.utc)
        loop = asyncio.get_running_loop()
        
        # Hook up the callback
        self.ib.pendingTickersEvent += self.on_new_tick

        last_warning_time = datetime.min.replace(tzinfo=timezone.utc)
        last_flush_check = datetime.now(timezone.utc)
        
        while not stop_event.is_set():
            await asyncio.sleep(1) 
            now_utc = datetime.now(timezone.utc)
            
            # Heartbeat (Touch file)
            Path("logs/ingest_heartbeat").touch()
            
            delta_s = (now_utc - self.last_data_time).total_seconds()
            
            if delta_s > DATA_TIMEOUT:
                if delta_s > (DATA_TIMEOUT * 5): # 10 minutes of silence -> Kill it
                    raise ConnectionError(f"💀 Dead Connection: No data for {delta_s}s. Forcing Reconnect.")
                    
                if (now_utc - last_warning_time).total_seconds() > 300:
                    logger.warning(f"⚠️ NO IBKR DATA received for {delta_s:.0f}s (Market likely closed or idle).")
                    last_warning_time = now_utc

            should_flush_time = (now_utc - last_flush_check).total_seconds() > FLUSH_INTERVAL_SEC

            for name, buffer in self.buffers.items():
                if len(buffer) >= CHUNK_SIZE or (should_flush_time and len(buffer) > 0):
                    chunk_data = list(buffer) 
                    self.buffers[name] = [] 
                    chunk_data.sort(key=lambda x: x[0])
                    df = pd.DataFrame(chunk_data, columns=self.l1_cols)
                    if not df.empty:
                        df["day"] = df["ts_event"].apply(lambda x: x.strftime("%Y%m%d"))
                        for day_str, group in df.groupby("day"):
                            prefix = cfg.RAW_DATA_PREFIX_TICKS if "TICKS" in self.name_to_mode[name] else cfg.RAW_DATA_PREFIX_BARS
                            fn = os.path.join(DATA_DIR, f"{prefix}_{name}_{day_str}.parquet")
                            save_cols = [c for c in self.l1_cols] 
                            # Use global save_chunk
                            await loop.run_in_executor(None, save_chunk, group[save_cols], fn)
                            print(f"\r💾 Saved {name} Chunk to {fn} ({len(group)} rows)...", end="")

            if should_flush_time:
                last_flush_check = now_utc

        self.ib.pendingTickersEvent -= self.on_new_tick

    def on_new_tick(self, tickers):
        updated = False
        for ticker in tickers:
            c_id = ticker.contract.conId
            if c_id not in self.contract_map: continue
            conf = self.contract_map[c_id]
            name = conf["name"]
            
            if conf["mode"] == "TICKS_BID_ASK":
                if ticker.tickByTicks:
                    for tick in ticker.tickByTicks:
                        ts = tick.time
                        if ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                        
                        if isinstance(tick, TickByTickBidAsk):
                            self.buffers[name].append([ts, tick.bidPrice, tick.askPrice, tick.bidSize, tick.askSize, float("nan"), float("nan"), float("nan")])
                            updated = True
                        elif isinstance(tick, (TickByTickAllLast,)):
                            self.buffers[name].append([ts, float("nan"), float("nan"), float("nan"), float("nan"), tick.price, tick.size, float("nan")])
                            updated = True

            elif conf["mode"] == "BARS_TRADES_1MIN":
                ts = ticker.time
                if not ts: ts = datetime.now(timezone.utc)
                elif ts.tzinfo is None: ts = ts.replace(tzinfo=timezone.utc)
                last_p = ticker.last if ticker.last and not pd.isna(ticker.last) else ticker.close
                if last_p and not pd.isna(last_p):
                     self.buffers[name].append([ts, float("nan"), float("nan"), float("nan"), float("nan"), last_p, float("nan"), float("nan")])
                     updated = True

        if updated:
            self.last_data_time = datetime.now(timezone.utc)