"""
Parser for ForexFactory economic calendar values.

Handles various formats:
- Percentages: "0.3%", "-0.2%"
- Suffixed numbers: "-16.3K", "2.48T", "793B", "450M"
- Plain numbers: "47.9", "50.0"
- Bond auctions: "4.18|2.6" (yield|bid-to-cover)
- Empty/invalid: "", "Tentative", etc.
"""

import re
from typing import Optional, Tuple
from datetime import datetime, date, timedelta
import pytz


# Multipliers for suffixed numbers
SUFFIXES = {
    'K': 1_000,
    'M': 1_000_000,
    'B': 1_000_000_000,
    'T': 1_000_000_000_000,
}

# Impact level mapping
IMPACT_MAP = {
    'icon--ff-impact-red': 'High',
    'icon--ff-impact-ora': 'Medium',
    'icon--ff-impact-yel': 'Low',
    'icon--ff-impact-gra': 'Holiday',
}


def parse_value(s: str) -> Optional[float]:
    """
    Parse an economic value string into a float.

    Args:
        s: Raw string from ForexFactory (e.g., "0.3%", "-16.3K", "47.9")

    Returns:
        Parsed float value, or None if unparseable

    Examples:
        >>> parse_value("0.3%")
        0.3
        >>> parse_value("-16.3K")
        -16300.0
        >>> parse_value("2.48T")
        2480000000000.0
        >>> parse_value("47.9")
        47.9
        >>> parse_value("")
        None
    """
    if not s or not isinstance(s, str):
        return None

    s = s.strip()
    if not s or s.lower() in ('', 'tentative', 'n/a', '-'):
        return None

    # Handle bond auction format "yield|bid-to-cover" - take first value (yield)
    if '|' in s:
        s = s.split('|')[0].strip()

    # Remove percentage sign but remember it was a percentage
    is_percent = '%' in s
    s = s.replace('%', '').strip()

    # Check for suffix (K, M, B, T)
    multiplier = 1.0
    if s and s[-1].upper() in SUFFIXES:
        multiplier = SUFFIXES[s[-1].upper()]
        s = s[:-1]

    # Try to parse the numeric part
    try:
        value = float(s) * multiplier
        return value
    except (ValueError, TypeError):
        return None


def parse_impact(impact_class: str) -> str:
    """
    Parse impact level from ForexFactory CSS class.

    Args:
        impact_class: CSS class string (e.g., "icon icon--ff-impact-red")

    Returns:
        Impact level: "High", "Medium", "Low", "Holiday", or "Unknown"
    """
    if not impact_class:
        return "Unknown"

    for css_class, level in IMPACT_MAP.items():
        if css_class in impact_class:
            return level

    return "Unknown"


def normalize_event_key(event_name: str) -> str:
    """
    Create a normalized key for event matching.

    Args:
        event_name: Raw event name (e.g., "Core CPI m/m")

    Returns:
        Normalized key (e.g., "core_cpi_m_m")
    """
    if not event_name:
        return ""

    # Lowercase, replace spaces and special chars with underscore
    key = event_name.lower()
    key = re.sub(r'[^a-z0-9]+', '_', key)
    key = key.strip('_')

    return key


def parse_ff_datetime(date_str: str, time_str: str, ff_timezone: str = "America/New_York") -> Optional[datetime]:
    """
    Parse ForexFactory date and time strings into UTC datetime.

    Args:
        date_str: Date string (e.g., "Jan 13")
        time_str: Time string (e.g., "6:30am", "Tentative", "All Day")
        ff_timezone: ForexFactory display timezone (default: US Eastern)

    Returns:
        UTC datetime, or None if unparseable
    """
    if not date_str:
        return None

    # Handle special time values
    if not time_str or time_str.lower() in ('tentative', 'all day', ''):
        time_str = "12:00pm"  # Default to midday

    # Parse the date - need to infer year
    # ForexFactory shows "Jan 13" without year
    today = datetime.now()

    try:
        # Try to parse month and day
        date_str = date_str.strip()

        # Handle formats like "Jan 13" or "Jan13"
        match = re.match(r'([A-Za-z]+)\s*(\d+)', date_str)
        if not match:
            return None

        month_str, day_str = match.groups()

        # Parse month
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month = month_map.get(month_str.lower()[:3])
        if not month:
            return None

        day = int(day_str)

        # Infer year - assume current year, but handle Dec/Jan boundary
        year = today.year
        if month == 12 and today.month == 1:
            year -= 1
        elif month == 1 and today.month == 12:
            year += 1

        # Parse time
        time_str = time_str.strip().lower()
        time_match = re.match(r'(\d+):(\d+)(am|pm)', time_str)
        if time_match:
            hour, minute, ampm = time_match.groups()
            hour = int(hour)
            minute = int(minute)
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
        else:
            hour, minute = 12, 0  # Default to noon

        # Create datetime in FF timezone
        tz = pytz.timezone(ff_timezone)
        local_dt = tz.localize(datetime(year, month, day, hour, minute))

        # Convert to UTC
        utc_dt = local_dt.astimezone(pytz.UTC)

        return utc_dt

    except Exception as e:
        return None


def parse_event_row(row_data: dict) -> Optional[dict]:
    """
    Parse a single event row from scraped data.

    Args:
        row_data: Dict with keys: currency, event, actual, forecast, previous, impact, date, time

    Returns:
        Parsed event dict, or None if invalid
    """
    event_name = row_data.get('event', '').strip()
    if not event_name:
        return None

    currency = row_data.get('currency', '').strip().upper()
    if not currency:
        return None

    # Parse values
    actual = parse_value(row_data.get('actual', ''))
    forecast = parse_value(row_data.get('forecast', ''))
    previous = parse_value(row_data.get('previous', ''))

    # Parse impact
    impact = parse_impact(row_data.get('impact', ''))

    # Parse datetime
    timestamp = parse_ff_datetime(
        row_data.get('date', ''),
        row_data.get('time', '')
    )

    # Create normalized key
    event_key = normalize_event_key(event_name)

    return {
        'timestamp_utc': timestamp,
        'currency': currency,
        'event_name': event_name,
        'event_key': event_key,
        'impact': impact,
        'actual': actual,
        'forecast': forecast,
        'previous': previous,
    }


def calculate_surprise(actual: Optional[float], forecast: Optional[float]) -> Optional[float]:
    """
    Calculate raw surprise value.

    Args:
        actual: Actual release value
        forecast: Consensus forecast

    Returns:
        Surprise (actual - forecast), or None if either is missing
    """
    if actual is None or forecast is None:
        return None
    return actual - forecast


def generate_week_urls(start_date: date, end_date: date) -> list:
    """
    Generate ForexFactory calendar URLs for a date range.

    Args:
        start_date: Start of range
        end_date: End of range

    Returns:
        List of URLs like "https://www.forexfactory.com/calendar?week=jul14.2025"
    """
    urls = []

    # Start from Monday of start_date's week
    current = start_date - timedelta(days=start_date.weekday())

    while current <= end_date:
        # ForexFactory uses format: ?week=jul14.2025 (lowercase month)
        week_str = current.strftime("%b%d.%Y").lower()
        url = f"https://www.forexfactory.com/calendar?week={week_str}"
        urls.append((current, url))
        current += timedelta(days=7)

    return urls


# ----- TESTS -----

def _test_parser():
    """Run basic parser tests."""
    print("Testing parse_value()...")

    # Percentages
    assert parse_value("0.3%") == 0.3, "Failed: 0.3%"
    assert parse_value("-0.2%") == -0.2, "Failed: -0.2%"

    # Suffixes
    assert parse_value("-16.3K") == -16300.0, "Failed: -16.3K"
    assert parse_value("2.48T") == 2480000000000.0, "Failed: 2.48T"
    assert parse_value("793B") == 793000000000.0, "Failed: 793B"
    assert parse_value("450M") == 450000000.0, "Failed: 450M"

    # Plain numbers
    assert parse_value("47.9") == 47.9, "Failed: 47.9"
    assert parse_value("50.0") == 50.0, "Failed: 50.0"

    # Bond auction format
    assert parse_value("4.18|2.6") == 4.18, "Failed: 4.18|2.6"

    # Empty/invalid
    assert parse_value("") is None, "Failed: empty"
    assert parse_value("Tentative") is None, "Failed: Tentative"
    assert parse_value(None) is None, "Failed: None"

    print("Testing normalize_event_key()...")
    assert normalize_event_key("Core CPI m/m") == "core_cpi_m_m"
    assert normalize_event_key("NFP") == "nfp"
    assert normalize_event_key("10-y Bond Auction") == "10_y_bond_auction"

    print("Testing parse_impact()...")
    assert parse_impact("icon icon--ff-impact-red") == "High"
    assert parse_impact("icon icon--ff-impact-ora") == "Medium"
    assert parse_impact("icon icon--ff-impact-yel") == "Low"

    print("Testing calculate_surprise()...")
    assert abs(calculate_surprise(47.9, 48.3) - (-0.4)) < 0.001, "Failed: 47.9 - 48.3"
    assert abs(calculate_surprise(0.3, 0.2) - 0.1) < 0.001, "Failed: 0.3 - 0.2"
    assert calculate_surprise(None, 0.2) is None

    print("All parser tests passed!")


if __name__ == "__main__":
    _test_parser()
