"""Contracts for application logging configuration."""

import logging
from pathlib import Path


def configure_logging(level: str, log_directory: Path) -> logging.Logger:
    """Configure and return the project logger."""
    raise NotImplementedError
