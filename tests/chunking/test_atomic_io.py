"""Atomic replace retry and failure-preservation tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import nerv.chunking.atomic_io as atomic_io
import nerv.chunking.pipeline as pipeline
import nerv.chunking.real_corpus as real_corpus
from nerv.chunking.real_corpus import parse_args, validate_command

from .test_real_corpus_commands import _write_source


def _windows_permission_error(winerror: int) -> PermissionError:
    error = PermissionError(13, f"simulated WinError {winerror}")
    error.winerror = winerror
    return error


def test_atomic_replace_succeeds_on_first_attempt(tmp_path: Path) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new\n", encoding="utf-8")
    destination.write_text("old\n", encoding="utf-8")

    atomic_io._atomic_replace_with_retry(source, destination)

    assert destination.read_text(encoding="utf-8") == "new\n"
    assert not source.exists()


def test_atomic_replace_retries_once_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new\n", encoding="utf-8")
    original_replace = atomic_io.os.replace
    attempts = 0
    delays: list[float] = []

    def flaky_replace(source_path: Path, destination_path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _windows_permission_error(5)
        original_replace(source_path, destination_path)

    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", True)
    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)

    atomic_io._atomic_replace_with_retry(source, destination)

    assert attempts == 2
    assert delays == [0.05]
    assert destination.read_text(encoding="utf-8") == "new\n"


def test_atomic_replace_retries_multiple_times_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new\n", encoding="utf-8")
    original_replace = atomic_io.os.replace
    attempts = 0
    delays: list[float] = []

    def flaky_replace(source_path: Path, destination_path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            raise _windows_permission_error(32)
        original_replace(source_path, destination_path)

    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", True)
    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)

    atomic_io._atomic_replace_with_retry(source, destination)

    assert attempts == 4
    assert delays == [0.05, 0.10, 0.25]
    assert destination.read_text(encoding="utf-8") == "new\n"


def test_atomic_replace_propagates_after_retry_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new\n", encoding="utf-8")
    destination.write_text("old\n", encoding="utf-8")
    attempts = 0
    delays: list[float] = []
    terminal_error = _windows_permission_error(5)

    def blocked_replace(_: Path, __: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise terminal_error

    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", True)
    monkeypatch.setattr(atomic_io.os, "replace", blocked_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)

    with pytest.raises(PermissionError) as captured:
        atomic_io._atomic_replace_with_retry(source, destination)

    assert captured.value is terminal_error
    assert attempts == 6
    assert delays == [0.05, 0.10, 0.25, 0.50, 1.00]
    assert source.read_text(encoding="utf-8") == "new\n"
    assert destination.read_text(encoding="utf-8") == "old\n"


def test_atomic_replace_propagates_unrelated_oserror_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new\n", encoding="utf-8")
    attempts = 0
    delays: list[float] = []
    unrelated_error = OSError(22, "invalid argument")

    def invalid_replace(_: Path, __: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise unrelated_error

    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", True)
    monkeypatch.setattr(atomic_io.os, "replace", invalid_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)

    with pytest.raises(OSError) as captured:
        atomic_io._atomic_replace_with_retry(source, destination)

    assert captured.value is unrelated_error
    assert attempts == 1
    assert delays == []


def test_atomic_replace_does_not_retry_on_non_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.tmp"
    destination = tmp_path / "destination.json"
    source.write_text("new\n", encoding="utf-8")
    delays: list[float] = []
    access_error = _windows_permission_error(5)

    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", False)
    monkeypatch.setattr(
        atomic_io.os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(access_error),
    )
    monkeypatch.setattr(atomic_io.time, "sleep", delays.append)

    with pytest.raises(PermissionError) as captured:
        atomic_io._atomic_replace_with_retry(source, destination)

    assert captured.value is access_error
    assert delays == []


def test_json_writer_preserves_complete_temp_on_terminal_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "metrics.json"
    destination.write_text('{"status": "old"}\n', encoding="utf-8")
    terminal_error = _windows_permission_error(5)

    def blocked_replace(_: Path, __: Path) -> None:
        raise terminal_error

    monkeypatch.setattr(real_corpus, "_atomic_replace_with_retry", blocked_replace)

    with pytest.raises(PermissionError) as captured:
        real_corpus._write_json_atomic(destination, {"status": "new"})

    assert captured.value is terminal_error
    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "old"}
    preserved = list(tmp_path.glob(".metrics.json.*.tmp"))
    assert len(preserved) == 1
    assert json.loads(preserved[0].read_text(encoding="utf-8")) == {"status": "new"}


def test_json_writer_replaces_existing_destination_without_contention(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "metrics.json"
    destination.write_text('{"status": "old"}\n', encoding="utf-8")

    real_corpus._write_json_atomic(destination, {"status": "new"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "new"}
    assert not list(tmp_path.glob(".metrics.json.*.tmp"))


def test_cross_directory_publication_uses_adjacent_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "work" / "chunks.tmp"
    destination = tmp_path / "published" / "chunks.jsonl"
    source.parent.mkdir()
    destination.parent.mkdir()
    source.write_text("complete\n", encoding="utf-8")
    observed: list[tuple[Path, Path]] = []
    actual_replace = atomic_io._atomic_replace_with_retry

    def observing_replace(source_path: Path, destination_path: Path) -> None:
        observed.append((source_path, destination_path))
        actual_replace(source_path, destination_path)

    monkeypatch.setattr(pipeline, "_atomic_replace_with_retry", observing_replace)

    pipeline._publish_completed_file(source, destination)

    assert destination.read_text(encoding="utf-8") == "complete\n"
    assert not source.exists()
    assert len(observed) == 1
    assert observed[0][0].parent == destination.parent
    assert observed[0][1] == destination


def test_secondary_metric_failure_does_not_mask_primary_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    metrics = tmp_path / "metrics.json"
    _write_source(source)
    chunks.write_text("{}\n", encoding="utf-8")
    args = parse_args(
        [
            "validate",
            "--input",
            str(source),
            "--chunks",
            str(chunks),
            "--metrics-output",
            str(metrics),
        ]
    )
    actual_write = real_corpus._write_json_atomic
    write_count = 0

    class PrimaryValidationError(RuntimeError):
        pass

    class SecondaryPersistenceError(OSError):
        pass

    def fail_validation(_: Path) -> tuple[dict[str, tuple[object, ...]], list[str]]:
        raise PrimaryValidationError("primary validation failure")

    def fail_secondary_write(path: Path, value: object) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            actual_write(path, value)
            return
        raise SecondaryPersistenceError("secondary metrics failure")

    monkeypatch.setattr(real_corpus, "_load_source_contract", fail_validation)
    monkeypatch.setattr(real_corpus, "_write_json_atomic", fail_secondary_write)

    with caplog.at_level(logging.ERROR, logger="nerv.chunking.real_corpus"):
        result = validate_command(args)

    assert result == 1
    assert write_count == 2
    primary_records = [
        record
        for record in caplog.records
        if "Validation setup failed" in record.getMessage()
    ]
    secondary_records = [
        record
        for record in caplog.records
        if "primary failure remains authoritative" in record.getMessage()
    ]
    assert len(primary_records) == 1
    assert isinstance(primary_records[0].exc_info[1], PrimaryValidationError)
    assert len(secondary_records) == 1
    assert isinstance(
        secondary_records[0].exc_info[1],
        SecondaryPersistenceError,
    )
