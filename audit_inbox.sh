#!/bin/bash
# audit_inbox.sh
# Lists inbox strategies and generates a comprehensive audit report.

echo "========================================"
echo "      INBOX STRATEGY AUDIT TOOL         "
echo "========================================"
echo ""

# 1. List current strategies
echo "[1/2] Listing Current Inbox Strategies..."
python manage_candidates.py inbox
if [ $? -ne 0 ]; then
    echo "Error listing strategies."
    exit 1
fi
echo ""

# 2. Genetic Analysis
echo "[2/3] Running Genetic Analysis..."
python analyze_inbox_genes.py
if [ $? -ne 0 ]; then
    echo "Error analyzing genes."
fi
echo ""

# 3. Generate Report
echo "[3/3] Running Deep Audit & Generating Report..."
# Note: generate_prop_report.py has been modified to run the inbox audit by default.
# We skip refresh here because step 1 (manage_candidates.py) already refreshed them.
python generate_prop_report.py --skip-refresh
if [ $? -ne 0 ]; then
    echo "Error generating report."
    exit 1
fi

echo ""
echo "========================================"
echo "✅ Audit Complete."
echo "📄 Report: output/strategies/REPORT_INBOX_AUDIT.md"
echo "========================================"
