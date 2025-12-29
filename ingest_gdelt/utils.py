import io
import os
import requests
import zipfile
import datetime as dt
from typing import Optional, List, Set
from .config import logger, HTTP_TIMEOUT

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
