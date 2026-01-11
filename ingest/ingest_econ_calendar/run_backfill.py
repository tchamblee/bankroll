#!/usr/bin/env python3
"""
ForexFactory Economic Calendar Backfill

Uses PowerShell to scrape via Chrome CDP (works from WSL).

Prerequisites:
    Chrome running with:
    Start-Process "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" -ArgumentList "--remote-debugging-port=9222"

Usage:
    python run_backfill.py --days 180
    python run_backfill.py --week jul14.2025
"""

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ingest.ingest_econ_calendar.parser import parse_event_row

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "econ_calendar"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
EVENTS_FILE = PROCESSED_DIR / "econ_events.parquet"
PS_SCRIPT = Path(__file__).parent / "scrape_week.ps1"


def scrape_week_via_powershell(week_str: str) -> list:
    """Scrape a week using PowerShell script."""
    ps_path = str(PS_SCRIPT).replace("/", "\\")
    wsl_path = subprocess.run(
        ["wslpath", "-w", str(PS_SCRIPT)],
        capture_output=True, text=True
    ).stdout.strip()

    cmd = [
        "powershell.exe", "-ExecutionPolicy", "Bypass",
        "-File", wsl_path,
        "-Week", week_str
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        return []

    # Parse JSON output (last line should be the JSON)
    lines = result.stdout.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("["):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    print(f"  Failed to parse output")
    return []


def filter_high_impact_usd_eur(events: list) -> list:
    """Filter to only high-impact USD and EUR events."""
    return [
        e for e in events
        if e.get('impact') == 'High'
        and e.get('currency') in ('USD', 'EUR')
    ]


def generate_week_urls(days_back: int) -> list:
    """Generate list of week strings for backfill."""
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    current = start_date - timedelta(days=start_date.weekday())

    weeks = []
    while current <= end_date:
        week_str = current.strftime("%b%d.%Y").lower()
        weeks.append((current, week_str))
        current += timedelta(days=7)

    return weeks


def save_events(events: list, week_str: str):
    """Save events to parquet."""
    if not events:
        return

    # Parse events
    parsed = []
    for raw in events:
        event = parse_event_row(raw)
        if event:
            event['week'] = week_str
            parsed.append(event)

    # Filter to high-impact USD/EUR
    parsed = filter_high_impact_usd_eur(parsed)
    if not parsed:
        return

    new_df = pd.DataFrame(parsed)
    new_df['scraped_at'] = datetime.now(pytz.UTC)

    # Load existing and merge
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if EVENTS_FILE.exists():
        existing_df = pd.read_parquet(EVENTS_FILE)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['week', 'currency', 'event_name'],
            keep='last'
        )
    else:
        combined = new_df

    combined = combined.sort_values('week').reset_index(drop=True)
    combined.to_parquet(EVENTS_FILE, index=False)

    # Raw backup
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = RAW_DIR / f"week_{week_str.replace('.', '_')}.parquet"
    new_df.to_parquet(raw_file, index=False)

    return len(parsed)


def run_backfill(days: int):
    """Run full backfill."""
    print(f"\n{'='*60}")
    print(f"ECONOMIC CALENDAR BACKFILL")
    print(f"{'='*60}")
    print(f"Days to backfill: {days}")
    print(f"{'='*60}\n")

    weeks = generate_week_urls(days)
    print(f"Will scrape {len(weeks)} weeks\n")

    total_events = 0
    for i, (week_date, week_str) in enumerate(weeks):
        print(f"[{i+1}/{len(weeks)}] Week of {week_date} ({week_str})")

        try:
            events = scrape_week_via_powershell(week_str)
            if events:
                count = save_events(events, week_str)
                if count:
                    print(f"  Saved {count} high-impact USD/EUR events")
                    total_events += count
                else:
                    print(f"  No high-impact USD/EUR events")
            else:
                print(f"  No events found")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_events} total events saved")
    print(f"{'='*60}")


def run_single_week(week_str: str):
    """Scrape a single week."""
    print(f"Scraping week: {week_str}")
    events = scrape_week_via_powershell(week_str)

    if events:
        count = save_events(events, week_str)
        print(f"Saved {count or 0} high-impact USD/EUR events")

        # Show them
        parsed = [parse_event_row(e) for e in events]
        high_impact = filter_high_impact_usd_eur([e for e in parsed if e])
        for e in high_impact:
            surprise = ""
            if e.get('actual') is not None and e.get('forecast') is not None:
                s = e['actual'] - e['forecast']
                surprise = f" (surprise: {s:+.2f})"
            print(f"  {e['currency']} | {e['event_name']} | A:{e.get('actual')} F:{e.get('forecast')}{surprise}")


def main():
    parser = argparse.ArgumentParser(description="Backfill ForexFactory economic calendar")
    parser.add_argument("--days", type=int, help="Number of days to backfill")
    parser.add_argument("--week", type=str, help="Specific week (e.g., jul14.2025)")

    args = parser.parse_args()

    if args.days:
        run_backfill(args.days)
    elif args.week:
        run_single_week(args.week)
    else:
        print("Usage:")
        print("  python run_backfill.py --days 180")
        print("  python run_backfill.py --week jul14.2025")
        print("\nPrerequisite: Chrome with --remote-debugging-port=9222")


if __name__ == "__main__":
    main()
