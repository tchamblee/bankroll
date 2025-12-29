import datetime as dt
import os
import polars as pl
from typing import Optional, List

from .config import logger, GDELT_V2_BASE, INTRADAY_WINDOW_DAYS, V2_EVENTS_DIR
from .utils import http_get, unzip_single_file, quarter_hour_range, list_processed_ts

def download_v2_events_for_ts(ts_str: str) -> Optional[pl.DataFrame]:
    """
    Download a single GDELT 2.0 Event file for a quarter-hour timestamp.

    ts_str: 'YYYYMMDDHHMMSS' (GDELT naming convention)
    """
    url = f"{GDELT_V2_BASE}/{ts_str}.export.CSV.zip"
    logger.debug("[EVT %s] Fetch events -> %s", ts_str, url)

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

    # logger.info("[EVT %s] Raw shape: %d rows x %d cols", ts_str, df.height, df.width)

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
        logger.debug("[EVT %s] Wrote %d rows -> %s", ts_str, df.height, out_path)
        new_batches.append(ts_str)

    return new_batches
