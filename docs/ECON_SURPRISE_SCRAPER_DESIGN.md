# Economic Surprise Data Scraper - Design Document

## Overview
Scrape economic calendar data from ForexFactory to calculate surprise values for high-impact events (NFP, CPI, etc.) and integrate into the feature matrix.

**Goal:** Generate `surprise_z = (actual - forecast) / historical_std` for EUR/USD-relevant events.

---

## Data Source Analysis

### ForexFactory Calendar
- **URL:** `https://www.forexfactory.com/calendar`
- **Data Available:**
  - Event name, currency, date/time, impact level
  - Forecast, Previous, Actual values
  - Impact classification (High/Medium/Low)

### Data Access Methods

| Method | Pros | Cons |
|--------|------|------|
| **JSON Feed** | Clean, structured, no parsing | No "actual" values (future events only) |
| **Page Scraping** | Has actual values | HTML parsing, more fragile |
| **Hybrid** | Best of both | More complex |

**Recommendation:** Hybrid approach
1. JSON feed for upcoming event schedule
2. Page scraping for historical actuals
3. Local cache to store processed data

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECONOMIC SURPRISE PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Scraper    │───>│    Parser    │───>│    Cache     │       │
│  │ (Chrome MCP) │    │              │    │  (Parquet)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         v                   v                   v                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  ForexFactory│    │ Normalize    │    │ Historical   │       │
│  │  Calendar    │    │ Values       │    │ Std Calc     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                 │                │
│                                                 v                │
│                                          ┌──────────────┐       │
│                                          │ Surprise Z   │       │
│                                          │ Calculation  │       │
│                                          └──────────────┘       │
│                                                 │                │
│                                                 v                │
│                                          ┌──────────────┐       │
│                                          │ Feature      │       │
│                                          │ Integration  │       │
│                                          └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Scraper Module (`ingest/ingest_econ_calendar/scraper.py`)

**Responsibilities:**
- Navigate ForexFactory calendar using Chrome DevTools MCP
- Handle historical week pagination
- Extract event data from page DOM

**Methods:**
```python
class ForexFactoryScraper:
    def __init__(self, chrome_mcp):
        self.mcp = chrome_mcp

    def scrape_week(self, week_str: str) -> List[Dict]:
        """Scrape a specific week (e.g., 'jan4.2026')"""
        pass

    def scrape_historical(self, start_date: date, end_date: date) -> pd.DataFrame:
        """Scrape multiple weeks of historical data"""
        pass

    def get_json_feed(self) -> List[Dict]:
        """Get this week's upcoming events from JSON feed"""
        pass
```

**DOM Selectors:**
```javascript
// Event rows
document.querySelectorAll('tr.calendar__row')

// Fields within row
row.querySelector('.calendar__currency')      // USD, EUR, etc.
row.querySelector('.calendar__event-title')   // "CPI m/m"
row.querySelector('.calendar__actual')        // "0.3%"
row.querySelector('.calendar__forecast')      // "0.2%"
row.querySelector('.calendar__previous')      // "0.1%"
row.querySelector('.calendar__impact span')   // Impact icon class
row.querySelector('.calendar__time')          // "6:30am"
```

### 2. Parser Module (`ingest/ingest_econ_calendar/parser.py`)

**Responsibilities:**
- Parse numeric values from strings ("0.3%", "-16.3K", "47.9")
- Handle different formats (%, K, M, B, T suffixes)
- Normalize to comparable units

**Value Parsing:**
```python
def parse_value(s: str) -> Optional[float]:
    """
    Parse economic values:
    - "0.3%" -> 0.3
    - "-16.3K" -> -16300
    - "47.9" -> 47.9
    - "2.48T" -> 2480000000000
    """
    pass
```

### 3. Data Store (`data/econ_calendar/`)

**Schema:**
```
econ_calendar/
├── raw/
│   └── ff_calendar_YYYYMMDD.parquet   # Raw scraped data by scrape date
├── processed/
│   └── econ_events.parquet            # Consolidated, parsed events
└── cache/
    └── historical_stats.parquet        # Rolling std by event type
```

**Event Schema (Parquet):**
| Column | Type | Description |
|--------|------|-------------|
| timestamp_utc | datetime64[ns, UTC] | Event release time |
| currency | str | USD, EUR, GBP, etc. |
| event_name | str | "CPI m/m", "NFP", etc. |
| event_key | str | Normalized key (lowercase, no spaces) |
| impact | str | High, Medium, Low |
| actual | float | Actual release value |
| forecast | float | Consensus forecast |
| previous | float | Previous period value |
| surprise | float | actual - forecast |
| surprise_z | float | surprise / rolling_std |
| scraped_at | datetime64[ns, UTC] | When data was scraped |

### 4. Surprise Calculator (`feature_engine/econ_surprise.py`)

**Responsibilities:**
- Calculate rolling historical std for each event type
- Compute surprise_z scores
- Handle missing/null values
- Apply decay functions for feature integration

**Key Formulas:**
```python
# Surprise (raw)
surprise = actual - forecast

# Surprise Z-Score (normalized)
# Use 2-year rolling std of surprises for that event type
rolling_std = df.groupby('event_key')['surprise'].transform(
    lambda x: x.rolling(24, min_periods=6).std()  # ~2 years of monthly data
)
surprise_z = surprise / rolling_std

# Decay function for feature integration
# Half-life of 2 hours (80 bars at 1.5 min/bar)
decay_factor = 0.5 ** (bars_since_event / 80)
surprise_z_decayed = surprise_z * decay_factor
```

### 5. Feature Integration (`feature_engine/econ_surprise.py`)

**New Features:**
| Feature | Description |
|---------|-------------|
| `econ_surprise_usd` | Latest USD high-impact surprise z-score |
| `econ_surprise_eur` | Latest EUR high-impact surprise z-score |
| `econ_surprise_diff` | USD - EUR relative surprise |
| `surprise_decay_2h` | Surprise with 2-hour half-life decay |
| `surprise_decay_4h` | Surprise with 4-hour half-life decay |
| `nfp_surprise_z` | Most recent NFP surprise (monthly) |
| `cpi_surprise_z` | Most recent CPI surprise (monthly) |

---

## Implementation Phases

### Phase 1: Historical Backfill (One-time)
**Goal:** Populate 180 days (~26 weeks) of historical events

**Backfill Strategy:**
ForexFactory allows navigation to historical weeks via URL parameter:
- `https://www.forexfactory.com/calendar?week=jul14.2025`
- Each week page contains all events with actual/forecast/previous values
- DOM structure is identical to current week (confirmed via testing)

**Backfill Script Logic:**
```python
def generate_week_urls(start_date: date, end_date: date) -> List[str]:
    """Generate ForexFactory week URLs for date range."""
    # FF uses format: ?week=jul14.2025 (start of week)
    weeks = []
    current = start_date - timedelta(days=start_date.weekday())  # Monday
    while current <= end_date:
        week_str = current.strftime("%b%d.%Y").lower()  # jul14.2025
        weeks.append(f"https://www.forexfactory.com/calendar?week={week_str}")
        current += timedelta(days=7)
    return weeks

def backfill_historical(start_date: date, end_date: date):
    """Scrape historical weeks with rate limiting."""
    urls = generate_week_urls(start_date, end_date)
    all_events = []

    for i, url in enumerate(urls):
        print(f"Scraping week {i+1}/{len(urls)}: {url}")
        events = scrape_week(url)
        all_events.extend(events)

        # Rate limit: 3 seconds between requests
        time.sleep(3)

        # Checkpoint every 10 weeks
        if i % 10 == 0:
            save_checkpoint(all_events, i)

    return pd.DataFrame(all_events)
```

**Estimated Scraping Time:**
- 26 weeks × 3 seconds = ~78 seconds of request time
- Plus page load/parse time: ~2-3 minutes total
- With safety margin: 5-10 minutes for full backfill

**Checkpoint/Resume:**
- Save progress every 10 weeks to `data/econ_calendar/backfill_checkpoint.json`
- On failure, resume from last checkpoint
- Final output: `data/econ_calendar/processed/econ_events.parquet`

### Phase 2: Core Pipeline
**Goal:** Working scraper + parser + storage

**Files to Create:**
- `ingest/ingest_econ_calendar/__init__.py`
- `ingest/ingest_econ_calendar/scraper.py`
- `ingest/ingest_econ_calendar/parser.py`
- `ingest/ingest_econ_calendar/run_ingest.py`

### Phase 3: Feature Integration
**Goal:** Surprise features in feature matrix

**Files to Modify:**
- `feature_engine/econ_surprise.py` (new)
- `feature_engine/pipeline.py` (add call to econ_surprise)
- `generate_features.py` (load econ data)

### Phase 4: Live Updates
**Goal:** Automated daily/weekly updates

**Options:**
1. Manual: Run scraper after major releases
2. Cron: Daily scrape of last week's data
3. Event-driven: Scrape 30 mins after known high-impact events

---

## Edge Cases & Handling

### 1. Missing Actual Values
- Event scheduled but not yet released
- **Handling:** Store null, skip surprise calculation

### 2. Revised Values
- Initial release later revised (common for NFP)
- **Handling:** Store revision as separate row, use first release for surprise

### 3. Tentative Times
- Some events have "Tentative" instead of specific time
- **Handling:** Use midday estimate, flag as tentative

### 4. Value Parsing Failures
- Unusual formats or text instead of numbers
- **Handling:** Log warning, store null, exclude from std calculation

### 5. Holiday/No Data Days
- Market closed, no events
- **Handling:** Forward-fill last surprise value with decay

---

## Rate Limiting & Reliability

### ForexFactory Considerations
- No explicit API rate limits documented
- Use 2-3 second delay between page loads
- Rotate user agent if needed
- Use real Chrome (not headless) to avoid detection

### Error Recovery
- Store progress checkpoint during backfill
- Resume from last successful week on failure
- Log all scraping errors with context

---

## Testing Strategy

### Unit Tests
- Value parser (various formats)
- Surprise calculation
- Decay function

### Integration Tests
- Scrape single week, verify structure
- Parse known historical event, verify values
- End-to-end: scrape -> parse -> feature

### Validation
- Compare scraped values to known releases (BLS, etc.)
- Verify surprise_z distribution is ~N(0,1)

---

## File Structure

```
bankroll/
├── ingest/
│   └── ingest_econ_calendar/
│       ├── __init__.py
│       ├── scraper.py          # Chrome MCP scraping
│       ├── parser.py           # Value parsing
│       ├── run_ingest.py       # CLI entry point
│       └── backfill.py         # Historical backfill
├── data/
│   └── econ_calendar/
│       ├── raw/
│       ├── processed/
│       └── cache/
├── feature_engine/
│   └── econ_surprise.py        # Feature calculation
└── resources/
    └── economic_calendar.csv   # Existing (may deprecate)
```

---

## Dependencies

- Chrome DevTools MCP (already configured)
- pandas, numpy (existing)
- pytz (existing)

No new external dependencies required.

---

## Success Criteria

1. **Data Coverage:** 2+ years of historical high-impact USD/EUR events
2. **Accuracy:** Parsed values match official releases (spot check)
3. **Feature Quality:** surprise_z has reasonable distribution
4. **Integration:** Features appear in feature matrix after generate_features.py
5. **Reliability:** Scraper runs without manual intervention

---

## Design Decisions (Finalized)

1. **Scope of Events:** All high-impact events (USD and EUR focus)
   - NFP, CPI, Core CPI, PPI, Retail Sales, FOMC, ECB, GDP, etc.
   - Filter by `impact == "High"` and `currency in ["USD", "EUR"]`

2. **Update Frequency:** Daily at market close (~22:00 UTC)
   - High-impact events release during trading hours
   - Actuals available same day
   - Captures all events within 24 hours

3. **Historical Depth:** 180 days (aligns with IBKR tick data limit)
   - ~26 weeks of data
   - Sufficient for rolling std calculation (need ~6 data points per event type)

4. **Timezone Handling:** ForexFactory displays in user's local time
   - Scraper will extract times and convert to UTC
   - Store all timestamps as UTC in parquet

---

## Next Steps (Implementation Checklist)

### Setup
- [ ] Create directory structure (`ingest/ingest_econ_calendar/`, `data/econ_calendar/`)
- [ ] Create `__init__.py` files

### Phase 2: Core Pipeline
- [ ] Implement `parser.py` - Value parsing (%, K, M, B, T suffixes)
- [ ] Write parser unit tests
- [ ] Implement `scraper.py` - Chrome MCP integration
- [ ] Test single week scrape

### Phase 1: Backfill (depends on Phase 2)
- [ ] Implement `backfill.py` - Historical scraping with checkpoints
- [ ] Run backfill (180 days, ~26 weeks)
- [ ] Verify data quality (spot check against known releases)
- [ ] Calculate historical rolling std per event type

### Phase 3: Feature Integration
- [ ] Create `feature_engine/econ_surprise.py`
- [ ] Implement surprise_z calculation
- [ ] Implement decay functions
- [ ] Add to `feature_engine/pipeline.py`
- [ ] Test with `generate_features.py --force`
- [ ] Verify new features in matrix

### Phase 4: Live Updates
- [ ] Create `run_ingest.py` CLI entry point
- [ ] Add to `daily_maintenance.sh`
- [ ] Test daily update flow

### Validation
- [ ] Compare scraped NFP values to BLS official releases
- [ ] Verify surprise_z distribution is approximately N(0,1)
- [ ] Check feature correlation with EUR/USD moves around events
