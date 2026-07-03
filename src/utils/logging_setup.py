"""Per-run file logging.

Every pipeline run writes a timestamped log file under the configured logs
directory in addition to the console, so runs are reproducible from the record
and no numbers ever need to be hand-copied out of a terminal.
"""

import logging
from datetime import datetime
from pathlib import Path


def get_run_logger(name: str, logs_dir: Path) -> logging.Logger:
    """Return a logger that writes to both stdout and logs_dir/<name>_<timestamp>.log."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # avoid duplicate handlers on re-invocation
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    logger.info("Logging to %s", log_file)
    return logger
