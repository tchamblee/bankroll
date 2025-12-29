import os
import sys
import logging
import config as cfg

def setup_logging(name: str, log_file: str, level=logging.INFO):
    log_dir = cfg.DIRS.get("LOGS", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        
    return logger
