import asyncio
import logging
import argparse
import sys
import nest_asyncio
from datetime import datetime, timedelta, timezone
from ib_insync import IB, Forex, Index, ContFuture, Stock
import config as cfg
from ingest_fred import ingest_fred_data
from ingest_cot import process_cot_data
from .config import logger, TEST_PROBE, DAYS_TO_BACKFILL
from .gdelt import download_gdelt_gkg, download_gdelt_v2_day
from .ibkr import process_symbol_for_day

nest_asyncio.apply()

async def main(symbols=None, days=None):
    ib = IB()
    try:
        await ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=103)
        logger.info(f"CONNECTED. Test Probe: {TEST_PROBE}")
        
        # 0. Fetch FRED & COT Macro Data (Skip if targeting specific symbols to save time)
        if not symbols:
            logger.info("Fetching FRED & COT Macro Data...")
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, ingest_fred_data, 365*5)
                await loop.run_in_executor(None, process_cot_data)
            except Exception as e:
                logger.error(f"Macro Ingest Failed: {e}")
        
        # 1. Pre-qualify contracts
        logger.info("Qualifying contracts...")
        qualified_contracts = []
        
        targets_to_process = cfg.TARGETS
        if symbols:
            targets_to_process = [t for t in cfg.TARGETS if t['name'] in symbols]
            if not targets_to_process:
                logger.error(f"No targets found matching symbols: {symbols}")
                return

        for t_conf in targets_to_process:
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
            
            final_contract = qual[0]
            qualified_contracts.append((t_conf, final_contract))
            logger.info(f"   Verified: {t_conf['name']}")

        # 2. Iterate Days (Recent -> Oldest)
        # Use provided days, or TEST_PROBE, or Default
        duration = days if days is not None else (2 if TEST_PROBE else DAYS_TO_BACKFILL)
        
        logger.info(f"Starting backfill for {duration} days (skipping today)...")

        today = datetime.now().date()
        for i in range(1, duration + 1):
            target_date = today - timedelta(days=i)
            
            # Skip Weekends
            if target_date.weekday() >= 5:
                logger.info(f"Skipping Weekend: {target_date}")
                continue

            logger.info(f"Processing Date: {target_date}")
            
            # --- 2a. Fetch GDELT Data (Skip if targeting specific symbols) ---
            if not symbols:
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
    parser = argparse.ArgumentParser(description="Backfill Data Pipeline")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backfill (e.g. VIX EURUSD)")
    parser.add_argument("--days", type=int, help="Number of days to backfill (overrides config)")
    args = parser.parse_args()
    
    asyncio.run(main(symbols=args.symbols, days=args.days))
