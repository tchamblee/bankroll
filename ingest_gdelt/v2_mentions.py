import datetime as dt
import os
import polars as pl
from typing import Optional

from .config import logger, GDELT_V2_BASE, INTRADAY_WINDOW_DAYS, V2_MENTIONS_DIR
from .utils import http_get, unzip_single_file, quarter_hour_range, list_processed_ts

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
    logger.debug("[MNT %s] Fetch mentions -> %s", ts_str, url)

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

    # logger.info("[MNT %s] Raw shape: %d rows x %d cols", ts_str, df.height, df.width)

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
        logger.debug("[MNT %s] Wrote %d rows -> %s", ts_str, df.height, out_path)

