"""Checkpointed comparison of whole-document and bounded PDF segmentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, sleep
from typing import Any, cast

from nerv.chunking import sentence_splitter as sentence_splitter_module
from nerv.chunking.language_detector import detect_language
from nerv.chunking.sentence_splitter import (
    DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
    DEFAULT_TARGET_BLOCK_CHARACTERS,
    _normalize_for_integrity,
    _split_sentences_bounded_with_stats,
    _split_sentences_whole_document,
)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = Path("outputs/resultados/documentos.jsonl")
DEFAULT_MANIFEST_PATH = HERE / "fixtures" / "bounded_splitter_pdf_manifest.json"
DEFAULT_OUTPUT_PATH = HERE / "results" / "bounded_splitter_metrics.json"
DEFAULT_WHOLE_TIMEOUT_SECONDS = 15 * 60.0
SCHEMA_VERSION = 2
_SPLITTER_LOGGER_NAME = "nerv.chunking.sentence_splitter"
_TERMINAL_DOCUMENT_STATUSES = frozenset({"completed", "completed_with_findings"})
_TERMINAL_METRICS_STATUSES = frozenset({"completed", "completed_with_findings"})


class _WarningCollector(logging.Handler):
    """Collect warning messages without document content."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--whole-timeout-seconds",
        type=float,
        default=DEFAULT_WHOLE_TIMEOUT_SECONDS,
        help="Per-document timeout for the isolated whole-document baseline.",
    )
    parser.add_argument(
        "--reuse-synthetic-metrics",
        type=Path,
        help="Reuse synthetic_comparisons from an existing metrics artifact.",
    )
    worker = parser.add_argument_group(argparse.SUPPRESS)
    worker.add_argument("--whole-worker", action="store_true", help=argparse.SUPPRESS)
    worker.add_argument("--worker-text", type=Path, help=argparse.SUPPRESS)
    worker.add_argument("--worker-language", help=argparse.SUPPRESS)
    worker.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.whole_timeout_seconds <= 0:
        parser.error("--whole-timeout-seconds must be greater than zero.")
    return args


def _sentence_list_digest(sentences: Sequence[str]) -> str:
    """Hash a sentence list with unambiguous length-delimited UTF-8 items."""
    digest = hashlib.sha256()
    for sentence in sentences:
        encoded = sentence.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(distribution_name: str) -> str:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return "not-installed-as-distribution"


def _normalized_reconstruction_matches(text: str, sentences: Sequence[str]) -> bool:
    return _normalize_for_integrity(" ".join(sentences)) == (
        _normalize_for_integrity(text)
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one evaluator-owned checkpoint atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        for attempt in range(50):
            try:
                temporary_path.replace(path)
                break
            except PermissionError:
                if attempt == 49:
                    raise
                sleep(0.1)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _whole_worker(text_path: Path, language: str, output_path: Path) -> int:
    """Run one interruptible whole-document baseline in an isolated process."""
    text = text_path.read_text(encoding="utf-8")
    collector = _WarningCollector()
    splitter_logger = logging.getLogger(_SPLITTER_LOGGER_NAME)
    splitter_logger.addHandler(collector)
    try:
        started = perf_counter()
        sentences = _split_sentences_whole_document(text, language=language)
        duration = perf_counter() - started
    finally:
        splitter_logger.removeHandler(collector)
        collector.close()
    result: dict[str, Any] = {
        "status": "completed",
        "duration_seconds": duration,
        "sentence_count": len(sentences),
        "normalized_reconstruction_match": _normalized_reconstruction_matches(
            text,
            sentences,
        ),
        "sentence_list_digest": _sentence_list_digest(sentences),
        "warnings": collector.messages,
    }
    _atomic_write_json(output_path, result)
    return 0


def _run_whole_baseline(
    text: str,
    *,
    language: str,
    timeout_seconds: float,
    work_directory: Path,
) -> dict[str, Any]:
    """Run and contain one baseline so a timeout cannot retain the evaluator."""
    text_path: Path | None = None
    result_path: Path | None = None
    process: subprocess.Popen[str] | None = None
    started = perf_counter()
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=work_directory,
            prefix=".bounded-splitter-baseline-text.",
            suffix=".txt",
            delete=False,
        ) as text_file:
            text_path = Path(text_file.name)
            text_file.write(text)
        with NamedTemporaryFile(
            dir=work_directory,
            prefix=".bounded-splitter-baseline-result.",
            suffix=".json",
            delete=False,
        ) as result_file:
            result_path = Path(result_file.name)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--whole-worker",
            "--worker-text",
            str(text_path),
            "--worker-language",
            language,
            "--worker-output",
            str(result_path),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return {
                "status": "timed_out",
                "duration_seconds": perf_counter() - started,
                "timeout_seconds": timeout_seconds,
                "sentence_count": None,
                "normalized_reconstruction_match": None,
                "sentence_list_digest": None,
                "error": None,
            }
        except BaseException:
            process.kill()
            process.communicate()
            raise

        if process.returncode != 0:
            return {
                "status": "error",
                "duration_seconds": perf_counter() - started,
                "timeout_seconds": timeout_seconds,
                "sentence_count": None,
                "normalized_reconstruction_match": None,
                "sentence_list_digest": None,
                "error": (
                    f"Worker exited with status {process.returncode}. "
                    f"stdout={stdout[-1000:]!r} stderr={stderr[-1000:]!r}"
                ),
            }
        result = cast(
            dict[str, Any],
            json.loads(result_path.read_text(encoding="utf-8")),
        )
        result["timeout_seconds"] = timeout_seconds
        result["error"] = None
        return result
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if text_path is not None:
            text_path.unlink(missing_ok=True)
        if result_path is not None:
            result_path.unlink(missing_ok=True)


def _run_bounded(text: str, *, language: str) -> dict[str, Any]:
    """Run bounded segmentation and retain compact correctness evidence."""
    collector = _WarningCollector()
    splitter_logger = logging.getLogger(_SPLITTER_LOGGER_NAME)
    splitter_logger.addHandler(collector)
    try:
        result = _split_sentences_bounded_with_stats(text, language=language)
    finally:
        splitter_logger.removeHandler(collector)
        collector.close()
    return {
        "status": "completed",
        "duration_seconds": result.stats.duration_seconds,
        "sentence_count": len(result.sentences),
        "normalized_reconstruction_match": _normalized_reconstruction_matches(
            text,
            result.sentences,
        ),
        "sentence_list_digest": _sentence_list_digest(result.sentences),
        "block_count": result.stats.block_count,
        "maximum_block_size": result.stats.largest_block_characters,
        "maximum_carry_size": result.stats.maximum_carry_characters,
        "oversized_carry_warning_count": (
            result.stats.oversized_carry_warning_count
        ),
        "warnings": collector.messages,
        "error": None,
    }


def _load_manifest(path: Path) -> tuple[list[str], dict[str, str]]:
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema_version") != 2:
        raise ValueError("Manifest schema_version must be 2.")
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Manifest documents must be a non-empty list.")
    doc_ids: list[str] = []
    size_classes: dict[str, str] = {}
    for item in documents:
        if not isinstance(item, dict):
            raise TypeError("Every manifest document must be an object.")
        doc_id = item.get("doc_id")
        size_class = item.get("size_class")
        if not isinstance(doc_id, str) or not isinstance(size_class, str):
            raise TypeError("Manifest doc_id and size_class must be strings.")
        if doc_id in size_classes:
            raise ValueError(f"Duplicate manifest doc_id: {doc_id}.")
        doc_ids.append(doc_id)
        size_classes[doc_id] = size_class
    expected_classes = ["small", "small", "medium", "medium", "large", "extreme"]
    actual_classes = [size_classes[doc_id] for doc_id in doc_ids]
    if actual_classes != expected_classes:
        raise ValueError(
            "Manifest must contain small, small, medium, medium, large, "
            "and extreme documents in that order."
        )
    return doc_ids, size_classes


def _selected_documents(
    input_path: Path,
    selected_ids: set[str],
) -> Iterator[dict[str, Any]]:
    found_lines: dict[str, int] = {}
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            document = cast(dict[str, Any], json.loads(line))
            doc_id = document.get("doc_id")
            if doc_id not in selected_ids:
                continue
            if doc_id in found_lines:
                raise ValueError(
                    f"Selected document {doc_id!r} appears more than once: "
                    f"lines {found_lines[doc_id]} and {line_number}."
                )
            document_format = document.get("formato")
            if not isinstance(document_format, str):
                raise TypeError(f"Selected document {doc_id!r} has invalid format.")
            if document_format.casefold() != "pdf":
                raise ValueError(f"Selected document {doc_id!r} is not a PDF.")
            text = document.get("texto")
            if not isinstance(text, str):
                raise TypeError(
                    f"Selected document {doc_id!r} has non-string text "
                    f"at line {line_number}."
                )
            found_lines[str(doc_id)] = line_number
            yield document
    missing = selected_ids - found_lines.keys()
    if missing:
        raise ValueError(f"Selected documents not found: {sorted(missing)!r}.")


def _load_synthetic_comparisons(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    comparisons = payload.get("synthetic_comparisons")
    if not isinstance(comparisons, list):
        raise ValueError("Reusable metrics lack synthetic_comparisons.")
    return comparisons


def _configuration(
    args: argparse.Namespace,
    doc_ids: Sequence[str],
    documents: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    splitter_file = sentence_splitter_module.__file__
    if splitter_file is None:
        raise RuntimeError("The sentence splitter module has no filesystem path.")
    splitter_path = Path(splitter_file).resolve()
    return {
        "target_block_characters": DEFAULT_TARGET_BLOCK_CHARACTERS,
        "maximum_block_characters": DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
        "whole_document_timeout_seconds": args.whole_timeout_seconds,
        "input_path": str(args.input.resolve()),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _file_digest(args.manifest),
        "evaluator_sha256": _file_digest(Path(__file__).resolve()),
        "sentence_splitter_path": str(splitter_path),
        "sentence_splitter_sha256": _file_digest(splitter_path),
        "pysbd_version": _package_version("pysbd"),
        "selected_doc_ids": list(doc_ids),
        "selected_documents": [
            {
                "doc_id": doc_id,
                "character_count": len(str(documents[doc_id]["texto"])),
                "text_sha256": hashlib.sha256(
                    str(documents[doc_id]["texto"]).encode("utf-8")
                ).hexdigest(),
            }
            for doc_id in doc_ids
        ],
    }


def _new_document_result(
    document: dict[str, Any],
    *,
    size_class: str,
) -> dict[str, Any]:
    text = str(document["texto"])
    detection = detect_language(text)
    return {
        "doc_id": str(document["doc_id"]),
        "format": str(document["formato"]),
        "size_class": size_class,
        "character_count": len(text),
        "line_count": text.count("\n") + 1,
        "detected_language": detection.language,
        "status": "pending",
        "current_stage": None,
        "whole_document": {"status": "pending"},
        "bounded": {"status": "pending"},
        "comparison": {
            "exact_sentence_list_match": None,
            "normalized_reconstruction_match": None,
            "accepted_differences": [],
            "comparison_unavailable_reason": None,
        },
        "failures": [],
    }


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    documents = payload["real_pdf_comparisons"]
    completed_baselines = [
        item["doc_id"]
        for item in documents
        if item["whole_document"].get("status") == "completed"
    ]
    timed_out_baselines = [
        item["doc_id"]
        for item in documents
        if item["whole_document"].get("status") == "timed_out"
    ]
    completed_bounded = [
        item for item in documents if item["bounded"].get("status") == "completed"
    ]
    finalized_documents = [
        item for item in documents if item.get("status") in _TERMINAL_DOCUMENT_STATUSES
    ]
    successful_documents = [
        item
        for item in finalized_documents
        if item.get("status") == "completed" and not item["failures"]
    ]
    baseline_error_doc_ids = [
        item["doc_id"]
        for item in documents
        if item["whole_document"].get("status") == "error"
    ]
    bounded_error_doc_ids = [
        item["doc_id"]
        for item in documents
        if item["bounded"].get("status") == "error"
    ]
    finding_doc_ids = [item["doc_id"] for item in documents if item["failures"]]
    return {
        "selected_document_count": len(documents),
        "finalized_document_count": len(finalized_documents),
        "successful_document_count": len(successful_documents),
        "successful_document_ids": [item["doc_id"] for item in successful_documents],
        "finding_document_ids": finding_doc_ids,
        "baseline_error_doc_ids": baseline_error_doc_ids,
        "bounded_error_doc_ids": bounded_error_doc_ids,
        "completed_baseline_doc_ids": completed_baselines,
        "timed_out_baseline_doc_ids": timed_out_baselines,
        "completed_bounded_doc_ids": [item["doc_id"] for item in completed_bounded],
        "all_completed_bounded_reconstructions_match": (
            bool(completed_bounded)
            and all(
                item["bounded"]["normalized_reconstruction_match"]
                for item in completed_bounded
            )
        ),
        "exact_match_doc_ids": [
            item["doc_id"]
            for item in documents
            if item["comparison"].get("exact_sentence_list_match") is True
        ],
        "bounded_fastest_doc_id": (
            min(
                completed_bounded,
                key=lambda item: item["bounded"]["duration_seconds"],
            )["doc_id"]
            if completed_bounded
            else None
        ),
        "bounded_slowest_doc_id": (
            max(
                completed_bounded,
                key=lambda item: item["bounded"]["duration_seconds"],
            )["doc_id"]
            if completed_bounded
            else None
        ),
    }


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    payload["checkpoint_sequence"] = int(payload.get("checkpoint_sequence", 0)) + 1
    payload["updated_at"] = datetime.now(UTC).isoformat()
    payload["summary"] = _summarize(payload)
    _atomic_write_json(path, payload)


def _load_or_initialize(
    args: argparse.Namespace,
    documents: dict[str, dict[str, Any]],
    doc_ids: Sequence[str],
    size_classes: dict[str, str],
) -> dict[str, Any]:
    configuration = _configuration(args, doc_ids, documents)
    if args.output.exists():
        payload = cast(
            dict[str, Any],
            json.loads(args.output.read_text(encoding="utf-8")),
        )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Existing metrics are not a resumable schema v2 file.")
        if payload.get("status") in _TERMINAL_METRICS_STATUSES:
            raise FileExistsError(
                f"Completed metrics will not be overwritten: {args.output}"
            )
        if payload.get("configuration") != configuration:
            raise ValueError("Existing checkpoint configuration is incompatible.")
        checkpoint_doc_ids = [
            item.get("doc_id") for item in payload.get("real_pdf_comparisons", [])
        ]
        if checkpoint_doc_ids != list(doc_ids):
            raise ValueError(
                "Existing checkpoint document IDs/order do not match the manifest."
            )
        recovery_time = datetime.now(UTC).isoformat()
        for run in payload.get("evaluation_runs", []):
            if run.get("status") == "in_progress":
                run["status"] = "abandoned_on_resume"
                run["finished_at"] = recovery_time
                run["recovery_note"] = (
                    "The evaluator resumed after the prior process ended without "
                    "recording a terminal run status."
                )
        payload["resume_count"] = int(payload.get("resume_count", 0)) + 1
        payload["status"] = "in_progress"
        payload["current_doc_id"] = None
        payload["current_stage"] = "resume"
        for item in payload["real_pdf_comparisons"]:
            if item["whole_document"].get("status") == "running":
                item["whole_document"] = {"status": "pending"}
            if item["bounded"].get("status") == "running":
                item["bounded"] = {"status": "pending"}
            if item.get("status") == "in_progress":
                item["status"] = "pending"
                item["current_stage"] = None
            if (
                item["whole_document"].get("status")
                in {"completed", "timed_out", "error"}
                and item["bounded"].get("status") in {"completed", "error"}
            ):
                _compare_results(item)
        return payload

    now = datetime.now(UTC).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "generated_at": now,
        "updated_at": now,
        "resume_count": 0,
        "checkpoint_sequence": 0,
        "current_doc_id": None,
        "current_stage": "initialize",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "configuration": configuration,
        "synthetic_comparisons": _load_synthetic_comparisons(
            args.reuse_synthetic_metrics
        ),
        "real_pdf_comparisons": [
            _new_document_result(
                documents[doc_id],
                size_class=size_classes[doc_id],
            )
            for doc_id in doc_ids
        ],
        "summary": {},
        "evaluation_runs": [],
    }


def _append_unique(messages: list[str], message: str) -> None:
    if message not in messages:
        messages.append(message)


def _compare_results(item: dict[str, Any]) -> None:
    baseline = item["whole_document"]
    bounded = item["bounded"]
    comparison = item["comparison"]
    comparison["accepted_differences"] = []
    comparison["comparison_unavailable_reason"] = None
    comparison["speedup_ratio"] = None
    comparison["normalized_reconstruction_match"] = bounded.get(
        "normalized_reconstruction_match"
    )
    if baseline.get("status") == "completed" and bounded.get("status") == "completed":
        bounded_duration = bounded["duration_seconds"]
        comparison["speedup_ratio"] = (
            baseline["duration_seconds"] / bounded_duration
            if bounded_duration
            else None
        )
        comparison["exact_sentence_list_match"] = (
            baseline["sentence_count"] == bounded["sentence_count"]
            and baseline["sentence_list_digest"] == bounded["sentence_list_digest"]
        )
        if not comparison["exact_sentence_list_match"]:
            _append_unique(
                item["failures"],
                "Whole-document and bounded sentence lists differ; the "
                "difference was not accepted automatically.",
            )
        elif (
            not baseline["normalized_reconstruction_match"]
            and not bounded["normalized_reconstruction_match"]
        ):
            comparison["accepted_differences"].append(
                "Both methods produced the same sentence-list digest, but "
                "PySBD's output did not reconstruct the OCR source after "
                "whitespace normalization; bounded execution introduced no "
                "additional difference from the baseline."
            )
        elif (
            baseline["normalized_reconstruction_match"]
            and not bounded["normalized_reconstruction_match"]
        ):
            _append_unique(
                item["failures"],
                "Bounded segmentation failed normalized reconstruction while "
                "the whole-document baseline preserved it.",
            )
    elif baseline.get("status") == "timed_out":
        comparison["comparison_unavailable_reason"] = (
            "Whole-document baseline timed out; exact comparison is unavailable."
        )
    elif baseline.get("status") == "error":
        comparison["comparison_unavailable_reason"] = (
            "Whole-document baseline failed; exact comparison is unavailable."
        )
    if (
        bounded.get("status") == "completed"
        and not bounded.get("normalized_reconstruction_match")
        and baseline.get("status") != "completed"
    ):
        _append_unique(
            item["failures"],
            "Bounded segmentation failed normalized reconstruction and no "
            "completed baseline is available to classify the difference.",
        )


def _evaluate_documents(
    args: argparse.Namespace,
    payload: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    result_by_id = {
        item["doc_id"]: item for item in payload["real_pdf_comparisons"]
    }
    for doc_id in payload["configuration"]["selected_doc_ids"]:
        item = result_by_id[doc_id]
        baseline_status = item["whole_document"].get("status")
        bounded_status = item["bounded"].get("status")
        if (
            item.get("status") in _TERMINAL_DOCUMENT_STATUSES
            and baseline_status in {"completed", "timed_out", "error"}
            and bounded_status in {"completed", "error"}
        ):
            continue

        text = str(documents[doc_id]["texto"])
        language = str(item["detected_language"])
        item["status"] = "in_progress"
        payload["current_doc_id"] = doc_id

        if baseline_status not in {"completed", "timed_out", "error"}:
            payload["current_stage"] = "whole_document"
            item["current_stage"] = "whole_document"
            item["whole_document"] = {
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
            }
            _checkpoint(args.output, payload)
            item["whole_document"] = _run_whole_baseline(
                text,
                language=language,
                timeout_seconds=args.whole_timeout_seconds,
                work_directory=args.output.parent,
            )
            _checkpoint(args.output, payload)

        if item["bounded"].get("status") not in {"completed", "error"}:
            payload["current_stage"] = "bounded"
            item["current_stage"] = "bounded"
            item["bounded"] = {
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
            }
            _checkpoint(args.output, payload)
            try:
                item["bounded"] = _run_bounded(text, language=language)
            except Exception as error:  # noqa: BLE001 - preserve evaluator evidence
                item["bounded"] = {
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                item["failures"].append(item["bounded"]["error"])
            _checkpoint(args.output, payload)

        _compare_results(item)
        item["status"] = (
            "completed_with_findings"
            if item["failures"]
            or item["whole_document"].get("status") == "error"
            or item["bounded"].get("status") == "error"
            else "completed"
        )
        item["current_stage"] = None
        payload["current_stage"] = "document_completed"
        _checkpoint(args.output, payload)


def _run_parent(args: argparse.Namespace) -> int:
    doc_ids, size_classes = _load_manifest(args.manifest)
    documents = {
        str(document["doc_id"]): document
        for document in _selected_documents(args.input, set(doc_ids))
    }
    payload = _load_or_initialize(
        args,
        documents,
        doc_ids,
        size_classes,
    )
    run_started = perf_counter()
    run_record: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "status": "in_progress",
        "duration_seconds": None,
    }
    payload["evaluation_runs"].append(run_record)
    _checkpoint(args.output, payload)
    try:
        _evaluate_documents(args, payload, documents)
    except KeyboardInterrupt:
        run_record["finished_at"] = datetime.now(UTC).isoformat()
        run_record["status"] = "interrupted"
        run_record["duration_seconds"] = perf_counter() - run_started
        payload["status"] = "interrupted"
        _checkpoint(args.output, payload)
        print(f"Evaluation interrupted; checkpoint preserved at {args.output}")
        return 130
    except Exception as error:  # noqa: BLE001 - preserve checkpoint on any error
        run_record["finished_at"] = datetime.now(UTC).isoformat()
        run_record["status"] = "error"
        run_record["duration_seconds"] = perf_counter() - run_started
        run_record["error"] = f"{type(error).__name__}: {error}"
        payload["status"] = "error"
        _checkpoint(args.output, payload)
        raise

    run_record["finished_at"] = datetime.now(UTC).isoformat()
    run_record["status"] = "completed"
    run_record["duration_seconds"] = perf_counter() - run_started
    payload["status"] = (
        "completed_with_findings"
        if any(item["failures"] for item in payload["real_pdf_comparisons"])
        or any(
            item["whole_document"].get("status") == "error"
            or item["bounded"].get("status") == "error"
            for item in payload["real_pdf_comparisons"]
        )
        else "completed"
    )
    payload["current_doc_id"] = None
    payload["current_stage"] = None
    _checkpoint(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Metrics written to {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.whole_worker:
        if (
            args.worker_text is None
            or args.worker_language is None
            or args.worker_output is None
        ):
            raise ValueError("Whole worker arguments are incomplete.")
        return _whole_worker(
            args.worker_text,
            args.worker_language,
            args.worker_output,
        )
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
