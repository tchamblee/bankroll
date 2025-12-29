import os
import pandas as pd
import logging

logger = logging.getLogger("utils.io")

def save_chunk(df: pd.DataFrame, filename: str) -> bool:
    """
    Appends or writes a DataFrame to a parquet file.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename):
            existing_df = pd.read_parquet(filename)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Optional: Deduplicate? 
            # In live ingestion, we might want to dedup. In backfill we definitely do.
            # But making it generic might be risky if columns differ.
            # Assuming schema match.
            combined_df.to_parquet(filename, index=False)
        else:
            df.to_parquet(filename, index=False)
        return True
    except Exception as e:
        logger.error(f"Write Error {filename}: {e}")
        return False
