"""Utilities to handle logging in the whole project."""

import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

formatter = logging.Formatter("%(message)s")
handler = RichHandler(console=Console(), rich_tracebacks=True)
handler.setFormatter(formatter)
handler.setLevel(LEVEL)


def _is_interactive():
    import __main__ as main

    return not hasattr(main, "__file__")


def get_logger(file_path: str, level: str | None = None) -> logging.Logger:
    if level is None:
        level = LEVEL
    name = Path(file_path).stem
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        logger.addHandler(handler)
        if level == "DEBUG":
            # Ensure the log directory exists
            log_dir = Path(__file__).parent.parent / "log"
            log_dir.mkdir(exist_ok=True)

            # Create and add the file handler
            file_handler = logging.FileHandler(log_dir / f"{datetime.now(UTC).isoformat()}.log")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
    return logger
