import datetime as dt
import os
import polars as pl
from typing import Optional

from .config import logger, GDELT_GKG_DAILY_BASE, GKG_BACKFILL_DAYS, GKG_DAILY_DIR
from .utils import http_get, unzip_single_file

def download_gkg_daily(date_: dt.date) -> Optional[pl.DataFrame]:
    """
    Download GDELT GKG 1.0 daily summary for a given date.
    Columns (tab-separated, header row):
        DATE, NUMARTS, COUNTS, THEMES, LOCATIONS,
        PERSONS, ORGANIZATIONS, TONE, CAMEOEVENTIDS, SOURCES, SOURCEURLS
    """
    ymd = date_.strftime("%Y%m%d")
    url = f"{GDELT_GKG_DAILY_BASE}/{ymd}.gkg.csv.zip"
    logger.debug("[GKG] Fetch daily GKG for %s -> %s", ymd, url)

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

    # logger.info("[GKG %s] Raw shape: %d rows x %d cols", ymd, df.height, df.width)
    # logger.info("[GKG %s] Raw columns: %s", ymd, df.columns)

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

    return df


def ensure_gkg_recent():
    """
    Make sure last GKG_BACKFILL_DAYS *completed* days exist locally.

    We skip 'today' because the daily GKG file is only complete after the day ends.
    """
    today = dt.datetime.utcnow().date()
    start_day = today - dt.timedelta(days=1)  # yesterday is the most recent completed day

    logger.debug(
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
            logger.debug("[GKG %s] No data ingested.", ymd)
            continue

        df.write_parquet(out_path)
        logger.debug("[GKG %s] Wrote %d rows -> %s", ymd, df.height, out_path)
