import os
import sys
import logging
import pandas as pd
import config
from utils import setup_logging

def get_logger(name, log_file):
    """
    Wrapper for setup_logging.
    """
    return setup_logging(name, log_file)

def save_data(df, filename, logger):
    """
    Saves DataFrame to Parquet in the configured data directory.
    """
    output_path = os.path.join(config.DIRS['DATA_DIR'], filename)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved data to {output_path} ({len(df)} rows)")
