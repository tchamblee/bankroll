#!/bin/bash

# Configuration
LOG_DIR="logs"
mkdir -p $LOG_DIR
INGEST_LOG="$LOG_DIR/ingest_live.log"
PAPER_LOG="$LOG_DIR/paper_trade.log"

INGEST_HB="$LOG_DIR/ingest_heartbeat"
PAPER_HB="$LOG_DIR/paper_trade_heartbeat"

# Timeouts (seconds)
HEARTBEAT_TIMEOUT=600 # 10 minutes

# Python Detection
PYTHON_EXEC=".venv/bin/python"
if [ ! -f "$PYTHON_EXEC" ]; then PYTHON_EXEC="python"; fi

kill_and_clean() {
    echo "🧹 Cleaning up processes..."
    pkill -f "python ingest/ingest_live.py"
    pkill -f "python paper_trade.py"
    rm -f $INGEST_HB $PAPER_HB
}

trap kill_and_clean EXIT

start_ingest() {
    echo "🚀 Starting Ingest..."
    # Redirect stdout/stderr to .out to avoid double logging in .log
    $PYTHON_EXEC ingest/ingest_live.py > "$LOG_DIR/ingest_live.out" 2>&1 &
    INGEST_PID=$!
    echo "   PID: $INGEST_PID"
    touch $INGEST_HB
}

start_paper() {
    echo "🚀 Starting Paper Trade..."
    # Redirect stdout/stderr to .out to avoid double logging in .log
    $PYTHON_EXEC paper_trade.py > "$LOG_DIR/paper_trade.out" 2>&1 &
    PAPER_PID=$!
    echo "   PID: $PAPER_PID"
    touch $PAPER_HB
}

check_heartbeat() {
    local file=$1
    local name=$2
    
    if [ ! -f "$file" ]; then
        echo "⚠️  $name heartbeat file missing. Assuming starting..."
        touch "$file"
        return 0
    fi
    
    current_time=$(date +%s)
    last_mod=$(date -r "$file" +%s)
    diff=$((current_time - last_mod))
    
    if [ $diff -gt $HEARTBEAT_TIMEOUT ]; then
        echo "💀 $name STALLED (Last heartbeat: ${diff}s ago). Restarting..."
        return 1
    fi
    return 0
}

# Initial Start
kill_and_clean
start_ingest
start_paper

# Stream logs to console
# We tail the LOG files (populated by Python FileHandler)
tail -f $INGEST_LOG $PAPER_LOG &
TAIL_PID=$!
trap "kill $TAIL_PID; kill_and_clean" EXIT

while true; do
    sleep 30
    
    # Check Ingest
    if ! kill -0 $INGEST_PID 2>/dev/null; then
        echo "❌ Ingest process died. Restarting..."
        start_ingest
    elif ! check_heartbeat $INGEST_HB "Ingest"; then
        kill -9 $INGEST_PID
        start_ingest
    fi
    
    # Check Paper
    if ! kill -0 $PAPER_PID 2>/dev/null; then
        echo "❌ Paper process died. Restarting..."
        start_paper
    elif ! check_heartbeat $PAPER_HB "Paper"; then
        kill -9 $PAPER_PID
        start_paper
    fi
done
