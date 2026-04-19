"""
src/utils/logger.py
───────────────────
Centralised logging configuration for TriFusion.

All modules should use get_logger(__name__) instead of
logging.getLogger() directly to ensure consistent formatting
and to honour the LOG_LEVEL environment variable.
"""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Log level is read from the LOG_LEVEL environment variable
    (default: INFO). Valid values: DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level     = getattr(logging, level_str, logging.INFO)

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False   # prevent duplicate output from root logger

    return logger
