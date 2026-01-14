import asyncio
import logging
import sys
import nest_asyncio
from datetime import datetime, timedelta, timezone
from ib_insync import IB
import config as cfg
from ingest_fred import ingest_fred_data
from ingest_cot import process_cot_data
from ingest_treasury_auctions import ingest_treasury_auctions
from .config import logger, TEST_PROBE, DAYS_TO_BACKFILL
from .gdelt import download_gdelt_gkg, download_gdelt_v2_day
from .ibkr import process_symbol_for_day
from ingest.ibkr_utils import build_and_qualify

nest_asyncio.apply()

async def main(symbols=None, days=None, fast=False):
    ib = IB()
    try:
        await ib.connectAsync(cfg.IBKR_HOST, cfg.IBKR_PORT, clientId=103)
        logger.info(f"CONNECTED. Test Probe: {TEST_PROBE}{' (FAST MODE)' if fast else ''}")
        
        # 0. Fetch FRED, COT & Treasury Auction Data (Skip if targeting specific symbols to save time)
        if not symbols:
            logger.info("Fetching FRED, COT & Treasury Auction Data...")
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, ingest_fred_data, 365*5)
                await loop.run_in_executor(None, process_cot_data)
                await loop.run_in_executor(None, ingest_treasury_auctions, 5)  # Last 5 years
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
            contract, _ = await build_and_qualify(ib, t_conf, logger)
            if contract is None:
                continue
            qualified_contracts.append((t_conf, contract))
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
                tasks.append(process_symbol_for_day(ib, contract, t_conf, target_date, fast=fast))

            await asyncio.gather(*tasks)
            
            await asyncio.sleep(0.5)

    except Exception as e:
        logger.error(f"Fatal Error: {e}", exc_info=True)
        raise
    finally:
        ib.disconnect()
        logger.info("DONE.")
