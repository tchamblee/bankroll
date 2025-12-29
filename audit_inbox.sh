#!/bin/bash
# audit_inbox.sh
# Lists inbox strategies.

echo "========================================"
echo "      INBOX STRATEGY AUDIT TOOL         "
echo "========================================"
echo ""

# 1. List current strategies
echo "[1/1] Listing Current Inbox Strategies..."
python manage_candidates.py inbox
if [ $? -ne 0 ]; then
    echo "Error listing strategies."
    exit 1
fi

echo ""
echo "========================================"
echo "✅ Audit Complete."
echo "========================================"