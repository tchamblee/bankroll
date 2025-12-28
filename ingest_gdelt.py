#!/usr/bin/env python3
import sys
import os
import io
import time
import zipfile
import logging
import datetime as dt
from typing import Optional, List, Set

import requests
import polars as pl
import config as cfg

# FinBERT streaming integration (requires finbert_streaming.py in the same project)
import finbert_streaming

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------

# Handle Windows console encoding if needed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# GDELT endpoints
GDELT_GKG_DAILY_BASE = "http://data.gdeltproject.org/gkg"
GDELT_V2_BASE = "http://data.gdeltproject.org/gdeltv2"

HTTP_TIMEOUT = 20                # seconds per request
POLL_INTERVAL_SEC = 30           # main loop sleep
GDELT_LAG_MINUTES = 5            # don't try to ingest last 5 minutes
INTRADAY_WINDOW_DAYS = 2         # look back this many days for intraday gaps
GKG_BACKFILL_DAYS = 7            # how many completed days of GKG to keep up to date

# Paths
DATA_LAKE_ROOT = cfg.DIRS.get("DATA_LAKE", os.path.join(os.getcwd(), "data_lake"))
ALT_DATA_ROOT = cfg.DIRS.get("ALT_DATA_ROOT", os.path.join(DATA_LAKE_ROOT, "altdata"))

GKG_DAILY_DIR = os.path.join(ALT_DATA_ROOT, "gdelt_gkg_raw")
V2_EVENTS_DIR = os.path.join(ALT_DATA_ROOT, "gdelt_v2", "events")
V2_MENTIONS_DIR = os.path.join(ALT_DATA_ROOT, "gdelt_v2", "mentions")
V2_GKG_DIR = os.path.join(ALT_DATA_ROOT, "gdelt_v2", "gkg")

os.makedirs(GKG_DAILY_DIR, exist_ok=True)
os.makedirs(V2_EVENTS_DIR, exist_ok=True)
os.makedirs(V2_MENTIONS_DIR, exist_ok=True)
os.makedirs(V2_GKG_DIR, exist_ok=True)

# Logging
LOG_DIR = cfg.DIRS.get("LOGS", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "ingest_gdelt.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------------------

def http_get(url: str) -> Optional[bytes]:
    """HTTP GET with basic error handling."""
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code == 404:
            logger.info("HTTP 404: %s", url)
            return None
        resp.raise_for_status()
        return resp.content
    except requests.RequestException as e:
        logger.warning("HTTP error for %s: %s", url, e)
        return None


def unzip_single_file(zip_bytes: bytes) -> Optional[io.BytesIO]:
    """Unzip a GDELT zip (single CSV) into a BytesIO."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            if not names:
                logger.warning("Zip archive has no files.")
                return None
            with zf.open(names[0]) as f:
                data = f.read()
        return io.BytesIO(data)
    except zipfile.BadZipFile as e:
        logger.warning("Bad zip file: %s", e)
        return None


def quarter_hour_floor(dt_obj: dt.datetime) -> dt.datetime:
    """Floor a datetime to the nearest quarter hour."""
    minute = (dt_obj.minute // 15) * 15
    return dt_obj.replace(minute=minute, second=0, microsecond=0)


def quarter_hour_range(start_dt: dt.datetime, end_dt: dt.datetime) -> List[dt.datetime]:
    """Generate quarter-hour timestamps between start and end (inclusive)."""
    cur = quarter_hour_floor(start_dt)
    end_dt = quarter_hour_floor(end_dt)
    if cur > end_dt:
        return []
    out: List[dt.datetime] = []
    while cur <= end_dt:
        out.append(cur)
        cur += dt.timedelta(minutes=15)
    return out


def list_processed_ts(dir_path: str, prefix: str) -> Set[str]:
    """
    Scan a directory and return set of timestamp strings (YYYYMMDDHHMMSS)
    extracted from filenames like: prefix_YYYYMMDDHHMMSS.parquet
    """
    result: Set[str] = set()
    if not os.path.exists(dir_path):
        return result

    for fname in os.listdir(dir_path):
        if not fname.startswith(prefix) or not fname.endswith(".parquet"):
            continue
        # e.g. gdelt_v2_events_20251210121500.parquet
        base = fname[:-8]  # strip '.parquet'
        parts = base.split("_")
        ts = parts[-1]
        if len(ts) == 14 and ts.isdigit():
            result.add(ts)
    return result


# --------------------------------------------------------------------------------------
# GKG 1.0 DAILY SUMMARY
# --------------------------------------------------------------------------------------

def download_gkg_daily(date_: dt.date) -> Optional[pl.DataFrame]:
    """
    Download GDELT GKG 1.0 daily summary for a given date.
    Columns (tab-separated, header row):
        DATE, NUMARTS, COUNTS, THEMES, LOCATIONS,
        PERSONS, ORGANIZATIONS, TONE, CAMEOEVENTIDS, SOURCES, SOURCEURLS
    """
    ymd = date_.strftime("%Y%m%d")
    url = f"{GDELT_GKG_DAILY_BASE}/{ymd}.gkg.csv.zip"
    logger.info("[GKG] Fetch daily GKG for %s -> %s", ymd, url)

    raw_bytes = http_get(url)
    if raw_bytes is None:
        return None

    csv_buffer = unzip_single_file(raw_bytes)
    if csv_buffer is None:
        return None

    try:
        df = pl.read_csv(
            csv_buffer,
            separator="\t",
            has_header=True,
            ignore_errors=True,
        )
    except Exception as e:
        logger.warning("[GKG %s] Failed to read CSV: %s", ymd, e)
        return None

    logger.info("[GKG %s] Raw shape: %d rows x %d cols", ymd, df.height, df.width)
    logger.info("[GKG %s] Raw columns: %s", ymd, df.columns)

    expected_cols = {
        "DATE",
        "NUMARTS",
        "COUNTS",
        "THEMES",
        "LOCATIONS",
        "PERSONS",
        "ORGANIZATIONS",
        "TONE",
        "CAMEOEVENTIDS",
        "SOURCES",
        "SOURCEURLS",
    }
    if not expected_cols.issubset(set(df.columns)):
        logger.warning(
            "[GKG %s] Missing expected columns. Got: %s",
            ymd,
            df.columns,
        )
        # Write raw schema anyway for debugging if you want later
        return df

    df = df.rename(
        {
            "DATE": "date_yyyymmdd",
            "NUMARTS": "num_articles",
            "COUNTS": "Counts",
            "THEMES": "V1Themes",
            "LOCATIONS": "V1Locations",
            "PERSONS": "V1Persons",
            "ORGANIZATIONS": "V1Organizations",
            "TONE": "ToneRaw",
            "CAMEOEVENTIDS": "CameoEventIds",
            "SOURCES": "SourceCommonName",
            "SOURCEURLS": "DocumentIdentifier",
        }
    )

    # DATE -> proper date_utc
    df = df.with_columns(
        pl.col("date_yyyymmdd")
        .cast(pl.Utf8)
        .str.strptime(pl.Date, format="%Y%m%d", strict=False)
        .alias("date_utc")
    )

    # Parse TONE: "tone,pos,neg,polarity,activity,wordcount"
    df = df.with_columns(
        pl.col("ToneRaw").cast(pl.Utf8).str.split(",").alias("_tone_list")
    )
    df = df.with_columns(
        pl.col("_tone_list").list.get(0).cast(pl.Float64).alias("tone_mean"),
        pl.col("_tone_list").list.get(1).cast(pl.Float64).alias("tone_positive"),
        pl.col("_tone_list").list.get(2).cast(pl.Float64).alias("tone_negative"),
        pl.col("_tone_list").list.get(3).cast(pl.Float64).alias("tone_polarity"),
        pl.col("_tone_list").list.get(4).cast(pl.Float64).alias("tone_activity"),
        pl.col("_tone_list").list.get(5).cast(pl.Float64).alias("tone_word_count"),
    ).drop("_tone_list")

    # Sanity log: small sample
    sample_cols = [
        "date_utc",
        "num_articles",
        "V1Themes",
        "V1Locations",
        "tone_mean",
        "tone_polarity",
        "SourceCommonName",
        "DocumentIdentifier",
    ]
    existing_sample_cols = [c for c in sample_cols if c in df.columns]

    logger.info("[GKG %s] Sample rows:\n%s", ymd, df.select(existing_sample_cols).head(3))

    return df


def ensure_gkg_recent():
    """
    Make sure last GKG_BACKFILL_DAYS *completed* days exist locally.

    We skip 'today' because the daily GKG file is only complete after the day ends.
    """
    today = dt.datetime.utcnow().date()
    start_day = today - dt.timedelta(days=1)  # yesterday is the most recent completed day

    logger.info(
        "[GKG] Ensuring daily GKG exists for last %d completed days, starting from %s",
        GKG_BACKFILL_DAYS,
        start_day,
    )

    for i in range(GKG_BACKFILL_DAYS):
        day = start_day - dt.timedelta(days=i)
        ymd = day.strftime("%Y%m%d")
        out_path = os.path.join(GKG_DAILY_DIR, f"gdelt_gkg_{ymd}.parquet")

        if os.path.exists(out_path):
            continue

        df = download_gkg_daily(day)
        if df is None or df.is_empty():
            logger.info("[GKG %s] No data ingested.", ymd)
            continue

        df.write_parquet(out_path)
        logger.info("[GKG %s] Wrote %d rows -> %s", ymd, df.height, out_path)


# --------------------------------------------------------------------------------------
# GDELT 2.0 EVENTS
# --------------------------------------------------------------------------------------

def download_v2_events_for_ts(ts_str: str) -> Optional[pl.DataFrame]:
    """
    Download a single GDELT 2.0 Event file for a quarter-hour timestamp.

    ts_str: 'YYYYMMDDHHMMSS' (GDELT naming convention)
    """
    url = f"{GDELT_V2_BASE}/{ts_str}.export.CSV.zip"
    logger.info("[EVT %s] Fetch events -> %s", ts_str, url)

    raw_bytes = http_get(url)
    if raw_bytes is None:
        return None

    buf = unzip_single_file(raw_bytes)
    if buf is None:
        return None

    try:
        df = pl.read_csv(
            buf,
            separator="\t",
            has_header=False,
            ignore_errors=True,
            infer_schema_length=0,
        )
    except Exception as e:
        logger.warning("[EVT %s] Failed to parse CSV: %s", ts_str, e)
        return None

    logger.info("[EVT %s] Raw shape: %d rows x %d cols", ts_str, df.height, df.width)

    # Bail out if structure is weird
    if df.width < 61:
        logger.warning(
            "[EVT %s] Expected >=61 columns, got %d. Keeping raw schema.",
            ts_str,
            df.width,
        )
        return df

    cols = df.columns  # column_0, column_1, ...

    rename_map = {
        cols[0]: "GlobalEventID",
        cols[1]: "SQLDATE",
        cols[2]: "MonthYear",
        cols[3]: "Year",
        cols[4]: "FractionDate",
        cols[26]: "EventCode",
        cols[28]: "EventRootCode",
        cols[31]: "NumMentions",
        cols[32]: "NumSources",
        cols[33]: "NumArticles",
        cols[34]: "AvgTone",
        cols[35]: "Actor1Geo_Type",
        cols[36]: "Actor1Geo_FullName",
        cols[37]: "Actor1Geo_CountryCode",
        cols[40]: "Actor1Geo_Lat",
        cols[41]: "Actor1Geo_Long",
        cols[42]: "Actor1Geo_FeatureID",
        cols[43]: "Actor2Geo_Type",
        cols[44]: "Actor2Geo_FullName",
        cols[45]: "Actor2Geo_CountryCode",
        cols[48]: "Actor2Geo_Lat",
        cols[49]: "Actor2Geo_Long",
        cols[50]: "Actor2Geo_FeatureID",
        cols[51]: "ActionGeo_Type",
        cols[52]: "ActionGeo_FullName",
        cols[53]: "ActionGeo_CountryCode",
        cols[56]: "ActionGeo_Lat",
        cols[57]: "ActionGeo_Long",
        cols[58]: "ActionGeo_FeatureID",
        cols[59]: "DATEADDED",
        cols[60]: "SOURCEURL",
    }

    df = df.rename(rename_map)

    # Parse dates
    df = df.with_columns(
        [
            pl.col("SQLDATE")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .alias("sql_date"),
            pl.col("DATEADDED")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S", strict=False)
            .alias("date_added_utc"),
            pl.lit(ts_str).alias("batch_ts"),
        ]
    )

    # Sanity sample
    sample_cols = [
        "GlobalEventID",
        "sql_date",
        "date_added_utc",
        "EventCode",
        "EventRootCode",
        "ActionGeo_CountryCode",
        "AvgTone",
        "NumMentions",
        "NumSources",
        "NumArticles",
        "SOURCEURL",
    ]
    existing_sample_cols = [c for c in sample_cols if c in df.columns]

    logger.info(
        "[EVT %s] Sample rows:\n%s",
        ts_str,
        df.select(existing_sample_cols).head(3),
    )

    return df


def ensure_v2_events_until(cutoff: dt.datetime) -> List[str]:
    """
    Ensure GDELT 2.0 event files exist for quarter-hour timestamps within
    [cutoff - INTRADAY_WINDOW_DAYS, cutoff], skipping already ingested ones.

    Returns a list of ts_str (YYYYMMDDHHMMSS) for which new parquet files
    were written in this call.
    """
    end_dt = cutoff
    start_dt = end_dt - dt.timedelta(days=INTRADAY_WINDOW_DAYS)

    processed_ts = list_processed_ts(V2_EVENTS_DIR, "gdelt_v2_events")

    ts_list = quarter_hour_range(start_dt, end_dt)
    if not ts_list:
        return []

    new_batches: List[str] = []

    for ts in ts_list:
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        if ts_str in processed_ts:
            continue

        df = download_v2_events_for_ts(ts_str)
        if df is None or df.is_empty():
            logger.info("[EVT %s] No data ingested.", ts_str)
            continue

        out_path = os.path.join(V2_EVENTS_DIR, f"gdelt_v2_events_{ts_str}.parquet")
        df.write_parquet(out_path)
        logger.info("[EVT %s] Wrote %d rows -> %s", ts_str, df.height, out_path)
        new_batches.append(ts_str)

    return new_batches


# --------------------------------------------------------------------------------------
# GDELT 2.0 MENTIONS
# --------------------------------------------------------------------------------------

def download_v2_mentions_for_ts(ts_str: str) -> Optional[pl.DataFrame]:
    """
    Download a single GDELT 2.0 Mentions file for a quarter-hour timestamp.

    Columns (no header, tab-separated):
        0 GlobalEventID
        1 EventTimeDate (YYYYMMDDHHMMSS)
        2 MentionTimeDate (YYYYMMDDHHMMSS)
        3 MentionType
        4 MentionSourceName
        5 MentionIdentifier
        6 MentionDocLen
        7 MentionDocTone
        8 SentenceID
        9 Actor1CharOffset
       10 Actor2CharOffset
       11 ActionCharOffset
       12 InRawText
       13 Confidence
    """
    url = f"{GDELT_V2_BASE}/{ts_str}.mentions.CSV.zip"
    logger.info("[MNT %s] Fetch mentions -> %s", ts_str, url)

    raw_bytes = http_get(url)
    if raw_bytes is None:
        return None

    buf = unzip_single_file(raw_bytes)
    if buf is None:
        return None

    try:
        df = pl.read_csv(
            buf,
            separator="\t",
            has_header=False,
            ignore_errors=True,
            infer_schema_length=0,
        )
    except Exception as e:
        logger.warning("[MNT %s] Failed to parse CSV: %s", ts_str, e)
        return None

    logger.info("[MNT %s] Raw shape: %d rows x %d cols", ts_str, df.height, df.width)

    if df.width < 14:
        logger.warning(
            "[MNT %s] Expected >=14 columns, got %d. Keeping raw schema.",
            ts_str,
            df.width,
        )
        return df

    cols = df.columns
    rename_map = {
        cols[0]: "GlobalEventID",
        cols[1]: "EventTimeDate",
        cols[2]: "MentionTimeDate",
        cols[3]: "MentionType",
        cols[4]: "MentionSourceName",
        cols[5]: "MentionIdentifier",
        cols[6]: "MentionDocLen",
        cols[7]: "MentionDocTone",
        cols[8]: "SentenceID",
        cols[9]: "Actor1CharOffset",
        cols[10]: "Actor2CharOffset",
        cols[11]: "ActionCharOffset",
        cols[12]: "InRawText",
        cols[13]: "Confidence",
    }
    df = df.rename(rename_map)

    df = df.with_columns(
        [
            pl.col("EventTimeDate")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S", strict=False)
            .alias("event_time_utc"),
            pl.col("MentionTimeDate")
            .cast(pl.Utf8)
            .str.strptime(pl.Datetime, format="%Y%m%d%H%M%S", strict=False)
            .alias("mention_time_utc"),
            pl.col("MentionDocTone")
            .cast(pl.Float64, strict=False)
            .alias("mention_doc_tone_float"),
            pl.lit(ts_str).alias("batch_ts"),
        ]
    )

    sample_cols = [
        "GlobalEventID",
        "event_time_utc",
        "mention_time_utc",
        "MentionSourceName",
        "MentionIdentifier",
        "mention_doc_tone_float",
        "Confidence",
    ]
    existing_sample_cols = [c for c in sample_cols if c in df.columns]

    logger.info(
        "[MNT %s] Sample rows:\n%s",
        ts_str,
        df.select(existing_sample_cols).head(3),
    )

    return df


def ensure_v2_mentions_until(cutoff: dt.datetime):
    """Ensure GDELT 2.0 mentions exist for recent quarter-hour timestamps."""
    end_dt = cutoff
    start_dt = end_dt - dt.timedelta(days=INTRADAY_WINDOW_DAYS)

    processed_ts = list_processed_ts(V2_MENTIONS_DIR, "gdelt_v2_mentions")

    ts_list = quarter_hour_range(start_dt, end_dt)
    if not ts_list:
        return

    for ts in ts_list:
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        if ts_str in processed_ts:
            continue

        df = download_v2_mentions_for_ts(ts_str)
        if df is None or df.is_empty():
            logger.info("[MNT %s] No data ingested.", ts_str)
            continue

        out_path = os.path.join(
            V2_MENTIONS_DIR, f"gdelt_v2_mentions_{ts_str}.parquet"
        )
        df.write_parquet(out_path)
        logger.info("[MNT %s] Wrote %d rows -> %s", ts_str, df.height, out_path)


# --------------------------------------------------------------------------------------
# GDELT 2.0 GKG (15-min)
# --------------------------------------------------------------------------------------

def download_v2_gkg_for_ts(ts_str: str) -> Optional[pl.DataFrame]:
    """
    Download a single GDELT 2.0 GKG file for a quarter-hour timestamp.
    
    Columns (tab-separated):
    GKGRECORDID, DATE, SourceCollectionIdentifier, SourceCommonName, 
    DocumentIdentifier, Counts, V2Counts, Themes, V2Themes, Locations, 
    V2Locations, Persons, V2Persons, Organizations, V2Organizations, 
    V2Tone, Dates, GCAM, SharingImage, RelatedImages, SocialImageEmbeds, 
    SocialVideoEmbeds, Quotations, AllNames, Amounts, TranslationInfo, Extras
    """
    url = f"{GDELT_V2_BASE}/{ts_str}.gkg.csv.zip"
    logger.info("[GKG2 %s] Fetch -> %s", ts_str, url)

    raw_bytes = http_get(url)
    if raw_bytes is None:
        return None

    buf = unzip_single_file(raw_bytes)
    if buf is None:
        return None

    try:
        # GKG 2.0 has many columns, variable length sometimes. 
        # reading with infer_schema_length=0 treats everything as string (safe)
        df = pl.read_csv(
            buf,
            separator="\t",
            has_header=False,
            ignore_errors=True,
            infer_schema_length=0,
            quote_char=None,  # GDELT GKG contains unescaped quotes, disable quote parsing
            encoding="utf8-lossy"
        )
    except Exception as e:
        logger.warning("[GKG2 %s] Failed to parse CSV: %s", ts_str, e)
        return None

    if df.width < 27:
        logger.warning("[GKG2 %s] Expected >=27 cols, got %d.", ts_str, df.width)
        return df

    cols = df.columns
    rename_map = {
        cols[0]: "GKGRECORDID",
        cols[1]: "DATE", # YYYYMMDDHHMMSS
        cols[3]: "SourceCommonName",
        cols[4]: "DocumentIdentifier",
        cols[5]: "Counts",
        cols[7]: "Themes", # V1 Themes
        cols[9]: "Locations", # V1 Locations
        cols[15]: "V2Tone", # Tone, Pos, Neg, Polarity, AR, WordCount
        cols[23]: "AllNames",
        cols[24]: "Amounts",
    }
    # Only rename what we really need for now to save memory/processing
    # We might need to select only these columns to keep it light
    
    df = df.rename({k:v for k,v in rename_map.items() if k in df.columns})
    
    # Keep only essential columns to reduce parquet size
    keep_cols = ["GKGRECORDID", "DATE", "SourceCommonName", "DocumentIdentifier", 
                 "Counts", "Themes", "Locations", "V2Tone"]
    
    existing_keep = [c for c in keep_cols if c in df.columns]
    df = df.select(existing_keep)
    
    # Parse Tone
    df = df.with_columns(
        pl.col("V2Tone").cast(pl.Utf8).str.split(",").alias("_tone_list")
    )
    df = df.with_columns(
        pl.col("_tone_list").list.get(0).cast(pl.Float64).alias("tone_mean"),
        pl.col("_tone_list").list.get(3).cast(pl.Float64).alias("tone_polarity"),
        pl.col("DATE").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y%m%d%H%M%S", strict=False).alias("date_utc"),
        pl.lit(ts_str).alias("batch_ts")
    ).drop("_tone_list")

    return df

def ensure_v2_gkg_until(cutoff: dt.datetime):
    """Ensure GDELT 2.0 GKG files exist for recent quarter-hour timestamps."""
    end_dt = cutoff
    start_dt = end_dt - dt.timedelta(days=INTRADAY_WINDOW_DAYS)

    processed_ts = list_processed_ts(V2_GKG_DIR, "gdelt_v2_gkg")

    ts_list = quarter_hour_range(start_dt, end_dt)
    if not ts_list:
        return

    for ts in ts_list:
        ts_str = ts.strftime("%Y%m%d%H%M%S")
        if ts_str in processed_ts:
            continue

        df = download_v2_gkg_for_ts(ts_str)
        if df is None or df.is_empty():
            logger.info("[GKG2 %s] No data ingested.", ts_str)
            continue

        out_path = os.path.join(V2_GKG_DIR, f"gdelt_v2_gkg_{ts_str}.parquet")
        df.write_parquet(out_path)
        logger.info("[GKG2 %s] Wrote %d rows -> %s", ts_str, df.height, out_path)

# --------------------------------------------------------------------------------------
# MAIN LOOP
# --------------------------------------------------------------------------------------

def main_loop():
    logger.info("ALT_DATA_ROOT: %s", ALT_DATA_ROOT)
    logger.info("GKG_DAILY_DIR: %s", GKG_DAILY_DIR)
    logger.info("V2_EVENTS_DIR: %s", V2_EVENTS_DIR)
    logger.info("V2_MENTIONS_DIR: %s", V2_MENTIONS_DIR)

    while True:
        try:
            now_utc = dt.datetime.utcnow()
            cutoff = now_utc - dt.timedelta(minutes=GDELT_LAG_MINUTES)

            logger.info("=" * 80)
            logger.info("Tick @ %s (cutoff=%s)", now_utc.isoformat(), cutoff.isoformat())

            # 1) Ensure recent GKG 1.0 daily files exist (completed days only)
            ensure_gkg_recent()

            # 2) Ensure V2 events are ingested up to cutoff
            new_event_batches = ensure_v2_events_until(cutoff)

            # 3) Ensure V2 mentions are ingested up to cutoff
            ensure_v2_mentions_until(cutoff)
            
            # 4) Ensure V2 GKG is ingested up to cutoff (NEW: Intraday EPU/Themes)
            ensure_v2_gkg_until(cutoff)

            # 5) Incremental FinBERT sentiment updates for newly ingested event batches
            if new_event_batches:
                try:
                    finbert_streaming.update_finbert_for_batches(
                        new_event_batches,
                        max_new_articles=300,  # tune if you want more/less HTTP + compute
                    )
                except Exception as e:
                    logger.error("🔥 FinBERT streaming update failed: %s", e, exc_info=True)

        except KeyboardInterrupt:
            logger.info("🛑 GDELT ingestion stopped by user.")
            break
        except Exception as e:
            logger.error("🔥 Unexpected error in main_loop: %s", e, exc_info=True)

        logger.info("Sleeping for %d seconds...", POLL_INTERVAL_SEC)
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main_loop()
