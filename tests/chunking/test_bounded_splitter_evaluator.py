"""Tests for checkpoint, timeout, and resume behavior of the evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from nerv.chunking import atomic_io
from tests.chunking import evaluate_bounded_splitter as evaluator


def _document(
    doc_id: str,
    text: str = "A sentence. Another sentence.",
) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "fuente": f"{doc_id}.pdf",
        "formato": "pdf",
        "fenomeno": 1,
        "texto": text,
    }


def _args(tmp_path: Path, manifest: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=tmp_path / "documents.jsonl",
        manifest=manifest,
        output=tmp_path / "metrics.json",
        whole_timeout_seconds=900.0,
        reuse_synthetic_metrics=None,
    )


def test_selected_documents_reject_duplicate_selected_ids(tmp_path: Path) -> None:
    """A selected source identity cannot silently use its last occurrence."""
    source = tmp_path / "documents.jsonl"
    source.write_text(
        json.dumps(_document("DOC-1"))
        + "\n"
        + json.dumps(_document("DOC-1", "Changed text."))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"lines 1 and 2"):
        list(evaluator._selected_documents(source, {"DOC-1"}))


def test_atomic_write_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short OneDrive lock does not discard the next checkpoint."""
    destination = tmp_path / "metrics.json"
    original_replace = atomic_io.os.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            error = PermissionError("transient test lock")
            error.winerror = 5  # type: ignore[attr-defined]
            raise error
        original_replace(source, target)

    monkeypatch.setattr(atomic_io, "_IS_WINDOWS", True)
    monkeypatch.setattr(atomic_io.os, "replace", flaky_replace)
    monkeypatch.setattr(atomic_io.time, "sleep", lambda _delay: None)
    evaluator._atomic_write_json(destination, {"status": "in_progress"})

    assert attempts == 3
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "status": "in_progress"
    }


def test_whole_timeout_kills_worker_and_removes_temporary_files(
    tmp_path: Path,
) -> None:
    """Timeout is terminal for the baseline worker, not the parent evaluator."""
    result = evaluator._run_whole_baseline(
        "A sentence. " * 100,
        language="en",
        timeout_seconds=0.001,
        work_directory=tmp_path,
    )

    assert result["status"] == "timed_out"
    assert not list(tmp_path.glob(".bounded-splitter-baseline-*"))


def test_summary_distinguishes_finalized_success_and_findings() -> None:
    """Machine-readable aggregates never turn absent work into success."""
    payload = {
        "real_pdf_comparisons": [
            {
                "doc_id": "DOC-1",
                "status": "completed_with_findings",
                "whole_document": {"status": "error"},
                "bounded": {"status": "error"},
                "comparison": {"exact_sentence_list_match": None},
                "failures": ["bounded failed"],
            }
        ]
    }

    summary = evaluator._summarize(payload)

    assert summary["finalized_document_count"] == 1
    assert summary["successful_document_count"] == 0
    assert summary["finding_document_ids"] == ["DOC-1"]
    assert summary["baseline_error_doc_ids"] == ["DOC-1"]
    assert summary["bounded_error_doc_ids"] == ["DOC-1"]
    assert summary["all_completed_bounded_reconstructions_match"] is False


def test_resume_fingerprint_changes_with_selected_text(
    tmp_path: Path,
) -> None:
    """In-place source edits make an existing checkpoint incompatible."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    args = _args(tmp_path, manifest)
    first = {"DOC-1": _document("DOC-1", "First text.")}
    second = {"DOC-1": _document("DOC-1", "Changed text.")}

    first_configuration = evaluator._configuration(args, ["DOC-1"], first)
    second_configuration = evaluator._configuration(args, ["DOC-1"], second)

    assert first_configuration != second_configuration
    assert (
        first_configuration["selected_documents"][0]["text_sha256"]
        != second_configuration["selected_documents"][0]["text_sha256"]
    )


def test_resume_marks_stale_run_abandoned(tmp_path: Path) -> None:
    """A hard-killed run gains a truthful terminal status on recovery."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    args = _args(tmp_path, manifest)
    documents = {"DOC-1": _document("DOC-1")}
    payload = evaluator._load_or_initialize(
        args,
        documents,
        ["DOC-1"],
        {"DOC-1": "small"},
    )
    payload["status"] = "interrupted"
    payload["evaluation_runs"] = [
        {
            "started_at": "2026-08-05T00:00:00+00:00",
            "finished_at": None,
            "status": "in_progress",
            "duration_seconds": None,
        }
    ]
    evaluator._atomic_write_json(args.output, payload)

    resumed = evaluator._load_or_initialize(
        args,
        documents,
        ["DOC-1"],
        {"DOC-1": "small"},
    )

    assert resumed["resume_count"] == 1
    assert resumed["evaluation_runs"][0]["status"] == "abandoned_on_resume"
    assert resumed["evaluation_runs"][0]["finished_at"] is not None
