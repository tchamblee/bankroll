#!/usr/bin/env python3
"""
Entry point for Backfill Operations.
Logic has been refactored into the `backfill` package.
"""
import sys
import asyncio
import logging

# Handle Windows console encoding if needed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from backfill.pipeline import main
import backfill.config as bf_cfg

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        bf_cfg.STOP_REQUESTED = True
        logging.getLogger("Backfill").info("👋 Keyboard Interrupt received. Exiting.")
        # Give tasks a moment to see the flag if they are polling
        # But since we are in main exception handler, main loop is likely dying.
        # We rely on the threads checking this flag.