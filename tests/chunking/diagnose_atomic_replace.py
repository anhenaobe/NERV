"""Short Windows diagnostic for repeated same-filesystem atomic replacement."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Final
from uuid import uuid4

RETRY_DELAYS_SECONDS: Final = (0.05, 0.10, 0.25, 0.50, 1.00)
TRANSIENT_WINDOWS_ERRORS: Final = frozenset({5, 32})


@dataclass
class LocationResult:
    """Observed replacement behavior for one directory."""

    location: str
    iteration_count: int
    completed_replacements: int = 0
    first_failure_iteration: int | None = None
    first_winerror: int | None = None
    first_failure_source: str | None = None
    destination_path: str | None = None
    destination_existed_at_first_failure: bool | None = None
    first_attempt_failures: int = 0
    transient_retries: int = 0
    maximum_successful_retry_count: int = 0
    terminal_failures: int = 0


class _RetryCountingHandler(logging.Handler):
    """Count retry warnings emitted by the production helper."""

    def __init__(self) -> None:
        super().__init__()
        self.retry_count = 0
        self.first_winerror: int | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(record, "atomic_replace_retry", False):
            self.retry_count += 1
            if self.first_winerror is None:
                self.first_winerror = getattr(
                    record,
                    "atomic_replace_winerror",
                    None,
                )


def _raw_replace_with_retry(
    source: Path,
    destination: Path,
) -> tuple[int, int | None]:
    """Replace directly and return retry count plus the first WinError."""
    retry_count = 0
    first_winerror: int | None = None
    while True:
        try:
            os.replace(source, destination)
            return retry_count, first_winerror
        except OSError as error:
            winerror = getattr(error, "winerror", None)
            if first_winerror is None:
                first_winerror = winerror
            if (
                os.name != "nt"
                or winerror not in TRANSIENT_WINDOWS_ERRORS
                or retry_count >= len(RETRY_DELAYS_SECONDS)
            ):
                raise
            time.sleep(RETRY_DELAYS_SECONDS[retry_count])
            retry_count += 1


def _replace_with_project_helper(
    source: Path,
    destination: Path,
) -> tuple[int, int | None]:
    """Use the project helper and report its logged retry behavior."""
    from nerv.chunking.atomic_io import _atomic_replace_with_retry

    logger = logging.getLogger("nerv.chunking.atomic_io")
    handler = _RetryCountingHandler()
    logger.addHandler(handler)
    try:
        _atomic_replace_with_retry(source, destination)
    finally:
        logger.removeHandler(handler)
    return handler.retry_count, handler.first_winerror


def _exercise_location(
    directory: Path,
    *,
    iterations: int,
    use_project_helper: bool,
) -> LocationResult:
    """Run bounded tiny replacements in one directory."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (
        f".nerv_atomic_replace_diagnostic.{uuid4().hex}.json"
    )

    result = LocationResult(
        location=str(directory.resolve()),
        iteration_count=iterations,
        destination_path=str(destination.resolve()),
    )
    replace = (
        _replace_with_project_helper
        if use_project_helper
        else _raw_replace_with_retry
    )
    try:
        for iteration in range(iterations):
            temporary_path: Path | None = None
            destination_existed = destination.exists()
            try:
                with NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="\n",
                    prefix=".nerv_atomic_replace.",
                    suffix=".tmp",
                    dir=directory,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    json.dump({"iteration": iteration}, temporary)
                    temporary.write("\n")
                    temporary.flush()
                    os.fsync(temporary.fileno())

                retry_count, first_winerror = replace(temporary_path, destination)
                result.completed_replacements += 1
                if retry_count:
                    if result.first_failure_iteration is None:
                        result.first_failure_iteration = iteration
                        result.first_winerror = first_winerror
                        result.first_failure_source = str(temporary_path.resolve())
                        result.destination_existed_at_first_failure = (
                            destination_existed
                        )
                    result.first_attempt_failures += 1
                    result.transient_retries += retry_count
                    result.maximum_successful_retry_count = max(
                        result.maximum_successful_retry_count,
                        retry_count,
                    )
            except OSError as error:
                if result.first_failure_iteration is None:
                    result.first_failure_iteration = iteration
                    result.first_winerror = getattr(error, "winerror", None)
                    result.first_failure_source = (
                        str(temporary_path.resolve())
                        if temporary_path is not None
                        else None
                    )
                    result.destination_existed_at_first_failure = destination_existed
                result.terminal_failures += 1
                break
            finally:
                if temporary_path is not None:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except OSError:
                        logging.getLogger(__name__).warning(
                            "Could not remove diagnostic temporary %s.",
                            temporary_path,
                            exc_info=True,
                        )
    finally:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).warning(
                "Could not remove diagnostic destination %s.",
                destination,
                exc_info=True,
            )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Tiny replacements to attempt per location.",
    )
    parser.add_argument(
        "--mode",
        choices=("raw", "helper"),
        default="raw",
        help="Use direct os.replace retries or the project helper.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the compact diagnostic result.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the three-location Windows diagnostic."""
    args = _parse_args()
    if os.name != "nt":
        raise SystemExit("This diagnostic is Windows-specific.")
    if args.iterations <= 0:
        raise ValueError("--iterations must be greater than zero.")

    root = Path(__file__).resolve().parents[2]
    fixed_locations = [
        root / "outputs/resultados",
        Path(r"C:\nerv_work"),
    ]
    results: list[LocationResult] = []
    with TemporaryDirectory(prefix="nerv-atomic-replace-") as fresh_temp:
        locations = [*fixed_locations, Path(fresh_temp)]
        for location in locations:
            results.append(
                _exercise_location(
                    location,
                    iterations=args.iterations,
                    use_project_helper=args.mode == "helper",
                )
            )

    payload = {
        "mode": args.mode,
        "platform": os.name,
        "temp_root": tempfile.gettempdir(),
        "retry_delays_seconds": list(RETRY_DELAYS_SECONDS),
        "results": [asdict(result) for result in results],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(f"{rendered}\n", encoding="utf-8")
    return 1 if any(result.terminal_failures for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
