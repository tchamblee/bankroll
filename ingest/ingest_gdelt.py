#!/usr/bin/env python3
"""
Entry point for GDELT ingestion pipeline.
Logic has been refactored into the `ingest_gdelt` package.
"""
import sys
import os
# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingest_gdelt.pipeline import main_loop

if __name__ == "__main__":
    main_loop()