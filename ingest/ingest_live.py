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
    exit_code = 0
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        cfg.STOP_REQUESTED = True
        logging.getLogger("ingest_live").info("Keyboard Interrupt received. Exiting.")
        exit_code = 130
    except Exception as e:
        logging.getLogger("ingest_live").error(f"Live ingestion failed: {e}")
        exit_code = 1

    sys.exit(exit_code)
