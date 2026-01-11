import asyncio
import logging
import nest_asyncio
import config as cfg
from ingest_fred import ingest_fred_data
from ingest_cot import process_cot_data
from ingest_treasury_auctions import ingest_treasury_auctions
from .config import logger, DATA_DIR, GDELT_ROOT, RECONNECT_DELAY
from .ibkr import IBKRStreamer
from .gdelt_worker import gdelt_monitor

# Apply nest_asyncio
nest_asyncio.apply()

async def main_loop():
    logger.info(f"📂 IBKR Data Lake: {DATA_DIR}")
    logger.info(f"📂 GDELT Data Lake: {GDELT_ROOT}")
    
    stop_event = asyncio.Event()
    
    # Start GDELT Monitor as separate task (Lives across IBKR reconnects)
    gdelt_task = asyncio.create_task(gdelt_monitor(stop_event))

    # Initial FRED, COT & Treasury Auction Fetch
    logger.info("Fetching FRED, COT & Treasury Auction Data...")
    try:
        loop = asyncio.get_running_loop()
        # Run these in executor to avoid blocking the loop start
        await loop.run_in_executor(None, ingest_fred_data, 365*2)
        await loop.run_in_executor(None, process_cot_data)
        await loop.run_in_executor(None, ingest_treasury_auctions, 2)  # Last 2 years
    except Exception as e:
        logger.error(f"Macro/Fundamental Ingest Failed: {e}")

    ib_worker = IBKRStreamer()

    while not stop_event.is_set() and not cfg.STOP_REQUESTED:
        try:
            # Run IBKR Ingestion (Blocks until disconnect/error)
            await ib_worker.connect_and_stream(stop_event)
            
        except asyncio.CancelledError:
            logger.info("🛑 Main Loop Cancelled")
            stop_event.set()
        except Exception as e:
            logger.error(f"🔥 Error in Main Loop: {e}", exc_info=True)
        
        if not stop_event.is_set() and not cfg.STOP_REQUESTED:
            logger.info(f"⏳ Reconnecting in {RECONNECT_DELAY}s...")
            await asyncio.sleep(RECONNECT_DELAY)

    # Cleanup GDELT
    if not gdelt_task.done():
        logger.info("Waiting for GDELT task to finish...")
        gdelt_task.cancel()
        try:
            await gdelt_task
        except asyncio.CancelledError:
            pass  # Expected when we cancel the task
        except Exception as e:
            logger.warning(f"Error during GDELT cleanup: {e}")
