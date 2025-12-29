#!/usr/bin/env python3
"""
Entry point for Live Ingestion (IBKR + GDELT + Macro).
Logic has been refactored into the `ingest_live` package.
"""
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import logging
import config as cfg

# Handle Windows console encoding if needed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from ingest_live.pipeline import main_loop

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        cfg.STOP_REQUESTED = True
        logging.getLogger("ingest_live").info("👋 Keyboard Interrupt received. Exiting.")
        # We can't do much else here since main_loop is already cancelled by asyncio.run logic
        # But setting the flag helps if threads are checking it.
