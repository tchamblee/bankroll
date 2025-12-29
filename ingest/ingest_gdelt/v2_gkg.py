import datetime as dt
import os
import polars as pl
from typing import Optional

from .config import logger, GDELT_V2_BASE, INTRADAY_WINDOW_DAYS, V2_GKG_DIR
from .utils import http_get, unzip_single_file, quarter_hour_range, list_processed_ts

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
    logger.debug("[GKG2 %s] Fetch -> %s", ts_str, url)

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
        logger.debug("[GKG2 %s] Wrote %d rows -> %s", ts_str, df.height, out_path)
