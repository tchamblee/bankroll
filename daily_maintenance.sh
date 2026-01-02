#!/bin/bash

# Configuration
LOG_FILE="logs/daily_maintenance.log"
SUPERVISOR_SCRIPT="./run_supervisor.sh"
CACHE_FILE="processed_data/live_bars.parquet"

# Ensure we are in the project root
cd "$(dirname "$0")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "🛑 STARTING DAILY MAINTENANCE"

# 1. Stop the Supervisor
log "1. Stopping Supervisor..."
pkill -f "run_supervisor.sh"
pkill -f "python ingest/ingest_live.py"
pkill -f "python paper_trade.py"
sleep 5

# Double check kill
if pgrep -f "paper_trade.py" > /dev/null; then
    log "⚠️ Forced kill required..."
    pkill -9 -f "paper_trade.py"
fi

# 2. Backfill Gaps (Ensure Data Lake is Complete)
log "2. Backfilling Data Lake Gaps..."
# This script scans local files, finds time gaps, and fetches from IBKR
# It ensures the Lake is the "Source of Truth"
python3 -m ingest.backfill.pipeline >> "$LOG_FILE" 2>&1

# 3. Clean Data Lake (Optimization)
log "3. Cleaning/Deduplicating Data Lake..."
python3 clean_data_lake.py >> "$LOG_FILE" 2>&1

# 4. Purge Live Cache
log "4. Purging Live Cache ($CACHE_FILE)..."
if [ -f "$CACHE_FILE" ]; then
    rm "$CACHE_FILE"
    log "   ✅ Cache deleted. System will rebuild from Lake on startup."
else
    log "   ℹ️ No cache file found (clean start)."
fi

# 5. Restart Supervisor
log "5. Restarting Supervisor..."
nohup $SUPERVISOR_SCRIPT > /dev/null 2>&1 &

log "✅ MAINTENANCE COMPLETE. Supervisor is running in background."
