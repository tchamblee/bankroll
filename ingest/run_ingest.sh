#!/bin/bash
cd "$(dirname "$0")"
while true; do
    echo "[$(date)] Starting Ingest..."
    python3 ingest_live.py
    echo "[$(date)] Ingest crashed. Restarting in 10s..."
    sleep 10
done
