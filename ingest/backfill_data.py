#!/usr/bin/env python3
"""
Entry point for Backfill Operations.
Logic has been refactored into the `backfill` package.
"""
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import logging
import argparse

# Handle Windows console encoding if needed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from backfill.pipeline import main
import backfill.config as bf_cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Data Pipeline Wrapper")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backfill (e.g. VIX EURUSD)")
    parser.add_argument("--days", type=int, help="Number of days to backfill (overrides config)")
    args = parser.parse_args()

    try:
        asyncio.run(main(symbols=args.symbols, days=args.days))
    except KeyboardInterrupt:
        bf_cfg.STOP_REQUESTED = True
        logging.getLogger("Backfill").info("👋 Keyboard Interrupt received. Exiting.")
        # Give tasks a moment to see the flag if they are polling
        # But since we are in main exception handler, main loop is likely dying.
        # We rely on the threads checking this flag.