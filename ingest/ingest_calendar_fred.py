import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Constants
FRED_API_KEY = os.getenv("FRED_API_KEY")
OUTPUT_FILE = "resources/economic_calendar.csv"

# Series to Fetch
# (SeriesID, EventName, TimeEST)
EVENTS = [
    ("CPIAUCSL", "CPI", "08:30"),
    ("PAYEMS", "NFP", "08:30"),
    # FOMC is harder via FRED. We'll skip for now or add manually if found.
]

def get_vintage_dates(series_id, start_date="2020-01-01"):
    url = "https://api.stlouisfed.org/fred/series/vintagedates"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "start_date": start_date
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("vintage_dates", [])
    except Exception as e:
        print(f"Error fetching {series_id}: {e}")
        return []

def ingest_calendar():
    if not FRED_API_KEY:
        print("❌ FRED_API_KEY not found.")
        return

    all_events = []
    
    print(f"Fetching calendar events from FRED (CPI, NFP)...")
    
    for series_id, event_name, time_str in EVENTS:
        dates = get_vintage_dates(series_id)
        print(f"  {event_name} ({series_id}): Found {len(dates)} releases.")
        
        for d in dates:
            # d is 'YYYY-MM-DD'
            # Construct naive datetime
            dt_str = f"{d} {time_str}:00"
            
            # Localize to EST (New York) then convert to UTC
            # NFP/CPI are strictly NY time.
            try:
                # Naive -> NY
                ny_tz = pytz.timezone('US/Eastern')
                naive = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
                local_dt = ny_tz.localize(naive)
                
                # NY -> UTC
                utc_dt = local_dt.astimezone(pytz.utc)
                
                all_events.append({
                    "timestamp_utc": utc_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": event_name,
                    "impact": "High"
                })
            except Exception as e:
                print(f"Error parsing date {d}: {e}")

    # Convert to DataFrame
    if not all_events:
        print("No events found.")
        return

    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp_utc").drop_duplicates()
    
    # Filter for reasonable range (e.g. 2024-2026) to keep file small?
    # Actually, keep all history is fine, it's small text.
    
    # Save
    os.makedirs("resources", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved {len(df)} events to {OUTPUT_FILE}")
    print(df.tail())

if __name__ == "__main__":
    ingest_calendar()
