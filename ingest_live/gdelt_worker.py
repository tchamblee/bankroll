import asyncio
import logging
import config as cfg
from ingest_gdelt.pipeline import run_cycle
from .config import logger, GDELT_POLL_INTERVAL_SEC

async def gdelt_monitor(stop_event: asyncio.Event):
    """Async wrapper for GDELT monitoring."""
    logger.info("🌍 GDELT Monitor Started")
    loop = asyncio.get_running_loop()
    
    while not stop_event.is_set() and not cfg.STOP_REQUESTED:
        # Check stop before running long sync task
        if stop_event.is_set() or cfg.STOP_REQUESTED: break
        
        # Run the GDELT cycle (blocking) in a separate thread
        try:
            await loop.run_in_executor(None, run_cycle)
        except Exception as e:
            logger.error(f"GDELT Cycle Error: {e}", exc_info=True)
        
        # Sleep in small chunks to be responsive to stop_event
        for _ in range(GDELT_POLL_INTERVAL_SEC):
            if stop_event.is_set(): break
            await asyncio.sleep(1)
