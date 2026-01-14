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
import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Data Pipeline Wrapper")
    parser.add_argument("--symbols", nargs="+", help=f"Specific symbols to backfill (e.g. VIX {config.PRIMARY_TICKER})")
    parser.add_argument("--days", type=int, help="Number of days to backfill (overrides config)")
    parser.add_argument("--fast", action="store_true", help="Skip IBKR verification for existing files")
    args = parser.parse_args()

    exit_code = 0
    try:
        asyncio.run(main(symbols=args.symbols, days=args.days, fast=args.fast))
    except KeyboardInterrupt:
        bf_cfg.STOP_REQUESTED = True
        logging.getLogger("Backfill").info("Keyboard Interrupt received. Exiting.")
        exit_code = 130
    except Exception as e:
        logging.getLogger("Backfill").error(f"Backfill failed: {e}")
        exit_code = 1

    sys.exit(exit_code)