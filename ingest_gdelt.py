#!/usr/bin/env python3
"""
Entry point for GDELT ingestion pipeline.
Logic has been refactored into the `ingest_gdelt` package.
"""
from ingest_gdelt.pipeline import main_loop

if __name__ == "__main__":
    main_loop()