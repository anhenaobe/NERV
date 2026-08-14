"""Bounded atomic-file publication helpers."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger(__name__)

_IS_WINDOWS: Final = os.name == "nt"
_TRANSIENT_WINDOWS_REPLACE_ERRORS: Final = frozenset({5, 32})
_RETRY_DELAYS_SECONDS: Final = (0.05, 0.10, 0.25, 0.50, 1.00)


def _is_transient_windows_replace_error(error: OSError) -> bool:
    """Return whether Windows reported transient access or sharing contention."""
    return _IS_WINDOWS and getattr(error, "winerror", None) in (
        _TRANSIENT_WINDOWS_REPLACE_ERRORS
    )


def _atomic_replace_with_retry(source: Path, destination: Path) -> None:
    """Atomically replace the destination with bounded Windows-only retries.

    The caller must place the source on the destination filesystem. Only
    Windows access-denied (5) and sharing-violation (32) errors are retried.
    The original source is reused unchanged for every attempt.
    """
    retry_count = 0
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if (
                not _is_transient_windows_replace_error(error)
                or retry_count >= len(_RETRY_DELAYS_SECONDS)
            ):
                raise
            delay_seconds = _RETRY_DELAYS_SECONDS[retry_count]
            retry_count += 1
            LOGGER.warning(
                "Atomic replace was transiently denied with WinError %s; "
                "retry %d/%d in %.2f seconds: %s -> %s",
                getattr(error, "winerror", None),
                retry_count,
                len(_RETRY_DELAYS_SECONDS),
                delay_seconds,
                source,
                destination,
                extra={
                    "atomic_replace_retry": True,
                    "atomic_replace_retry_count": retry_count,
                    "atomic_replace_winerror": getattr(error, "winerror", None),
                },
            )
            time.sleep(delay_seconds)
