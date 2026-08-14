"""Retain the superseded monolithic real-corpus evaluator for reference.

Direct execution is intentionally disabled. Use
``python -m nerv.chunking.real_corpus`` so production, validation, and
benchmarking remain independent and recoverable.
"""

# ruff: noqa: E501

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from nerv.chunking import TokenCounter, split_sentences
from nerv.chunking.configuration import load_encoder_config
from nerv.chunking.language_detector import LanguageDetectionResult, detect_language
from nerv.chunking.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/resultados/documentos.jsonl"
OUTPUT = ROOT / "outputs/resultados/chunks.jsonl"
RUN_CONFIG = ROOT / "outputs/resultados/chunking_config.json"
METRICS = ROOT / "outputs/resultados/real_corpus_chunking_metrics.json"
REVIEW = ROOT / "outputs/resultados/real_corpus_chunking_review.md"
REPORT = ROOT / "docs/chunking/real_corpus_chunking_report.md"
SYNTHETIC_METRICS = ROOT / "tests/chunking/results/frozen_pipeline_metrics.json"
REPETITIONS = 5
PREVIEW_CHARACTERS = 500
MAX_RECORDED_FAILURES = 100
REQUIRED_DOCUMENT_FIELDS = {"doc_id", "fuente", "formato", "fenomeno", "texto"}
REQUIRED_CHUNK_FIELDS = {
    "doc_id",
    "chunk_id",
    "fuente",
    "formato",
    "fenomeno",
    "posicion",
    "num_tokens",
    "texto",
}
TABULAR_FORMATS = {"csv", "tsv", "xls", "xlsx"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="strict", newline="") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} in {path} is not an object.")
            if not all(isinstance(key, str) for key in value):
                raise ValueError(f"Line {line_number} in {path} has a non-string key.")
            yield line_number, value


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "maximum": None,
            "average": None,
            "median": None,
            "standard_deviation": None,
            "percentile_25": None,
            "percentile_75": None,
            "percentile_90": None,
            "percentile_95": None,
            "percentile_99": None,
        }
    return {
        "count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "average": statistics.fmean(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
        "percentile_25": _percentile(values, 0.25),
        "percentile_75": _percentile(values, 0.75),
        "percentile_90": _percentile(values, 0.90),
        "percentile_95": _percentile(values, 0.95),
        "percentile_99": _percentile(values, 0.99),
    }


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("nerv", "transformers", "tokenizers", "sentencepiece", "pysbd"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _available_ram() -> tuple[int | None, str | None]:
    if platform.system() != "Windows":
        return None, "Available RAM measurement is implemented only for Windows."
    status = _MemoryStatus()
    status.length = ctypes.sizeof(status)
    try:
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError) as error:
        return None, f"Windows memory query failed: {error}"
    if not success:
        return None, "Windows GlobalMemoryStatusEx returned failure."
    return int(status.available_physical), None


def _preview(text: str, limit: int = PREVIEW_CHARACTERS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _overlap_sentences(
    left_text: str,
    right_text: str,
    *,
    language: str,
) -> list[str]:
    left = split_sentences(left_text, language=language)
    right = split_sentences(right_text, language=language)
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return left[-size:]
    return []


def _tabular_overlap(left_text: str, right_text: str) -> list[str]:
    """Return a bounded complete-text overlap for row-preserving chunks."""
    maximum = min(len(left_text), len(right_text), 16_384)
    for size in range(maximum, 0, -1):
        left_start = len(left_text) - size
        left_boundary = left_start == 0 or left_text[left_start - 1].isspace()
        right_boundary = size == len(right_text) or right_text[size].isspace()
        if left_boundary and right_boundary and left_text.endswith(right_text[:size]):
            return [right_text[:size]]
    return []


def _document_units(document: dict[str, Any], *, language: str) -> list[str]:
    """Mirror production segmentation for narrative and tabular documents."""
    text = str(document["texto"])
    document_format = str(document["formato"]).casefold()
    if document_format in TABULAR_FORMATS:
        return [line.strip() for line in text.splitlines() if line.strip()]
    return split_sentences(text, language=language)


def _failure(
    failures: list[dict[str, Any]],
    failed_documents: set[str],
    *,
    doc_id: str,
    category: str,
    detail: str,
) -> None:
    failed_documents.add(doc_id)
    if len(failures) < MAX_RECORDED_FAILURES:
        failures.append({"doc_id": doc_id, "category": category, "detail": detail})


def _candidate(
    document: dict[str, Any],
    detection: LanguageDetectionResult,
    sentences: list[str],
    chunks: list[dict[str, Any]],
    *,
    reason: str,
    selected_chunk: dict[str, Any] | None = None,
    overlap_tokens: int | None = None,
) -> dict[str, Any]:
    chunk = selected_chunk or (chunks[0] if chunks else {})
    return {
        "reason": reason,
        "doc_id": document["doc_id"],
        "fuente": document["fuente"],
        "detected_language": detection.language,
        "confidence": detection.confidence,
        "margin": detection.margin,
        "scores": detection.scores,
        "original_text_preview": _preview(str(document["texto"])),
        "sentence_preview": [_preview(sentence, 240) for sentence in sentences[:3]],
        "chunk_id": chunk.get("chunk_id"),
        "chunk_preview": _preview(str(chunk.get("texto", ""))),
        "num_tokens": chunk.get("num_tokens"),
        "actual_overlap_tokens": overlap_tokens,
    }


def _validate_input(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    label_values: Counter[str] = Counter()
    field_type_failures: Counter[str] = Counter()
    doc_ids: set[str] = set()
    failures: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="strict", newline="") as source:
        for line_number, line in enumerate(source, 1):
            counts["line_count"] += 1
            if not line.strip():
                counts["empty_line_count"] += 1
                continue
            try:
                value: object = json.loads(line)
            except json.JSONDecodeError as error:
                counts["malformed_line_count"] += 1
                if len(failures) < MAX_RECORDED_FAILURES:
                    failures.append(
                        {
                            "line": line_number,
                            "category": "malformed_json",
                            "detail": error.msg,
                        }
                    )
                continue
            if not isinstance(value, dict):
                counts["non_object_line_count"] += 1
                continue
            counts["valid_json_object_count"] += 1
            missing = sorted(REQUIRED_DOCUMENT_FIELDS - value.keys())
            if missing:
                counts["missing_required_fields_count"] += 1
                if len(failures) < MAX_RECORDED_FAILURES:
                    failures.append(
                        {
                            "line": line_number,
                            "category": "missing_fields",
                            "fields": missing,
                        }
                    )
            for field in ("doc_id", "fuente", "formato", "texto"):
                field_value = value.get(field)
                if not isinstance(field_value, str) or not field_value.strip():
                    field_type_failures[field] += 1
            phenomenon = value.get("fenomeno")
            if isinstance(phenomenon, bool) or not isinstance(phenomenon, int):
                field_type_failures["fenomeno"] += 1
            text = value.get("texto")
            if not isinstance(text, str) or not text.strip():
                counts["empty_or_invalid_text_count"] += 1
            doc_id = value.get("doc_id")
            if isinstance(doc_id, str):
                if doc_id in doc_ids:
                    counts["duplicate_doc_id_count"] += 1
                doc_ids.add(doc_id)
            language = value.get("idioma")
            if isinstance(language, str):
                label_values[language] += 1
    defaults = (
        "empty_line_count",
        "malformed_line_count",
        "non_object_line_count",
        "missing_required_fields_count",
        "empty_or_invalid_text_count",
        "duplicate_doc_id_count",
    )
    for key in defaults:
        counts[key] += 0
    return {
        "source_path": path.relative_to(ROOT).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256_before": _sha256(path),
        "counts": dict(counts),
        "field_type_failures": dict(field_type_failures),
        "trusted_label_field": "idioma",
        "trusted_label_values": dict(sorted(label_values.items())),
        "validation_failures": failures,
    }


def _run_benchmark(
    counter: TokenCounter,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[float, list[float], list[str]]:
    logging.getLogger("nerv.chunking.language_detector").setLevel(logging.ERROR)
    logging.getLogger("nerv.chunking.sentence_splitter").setLevel(logging.ERROR)
    started = time.perf_counter()
    run_pipeline(
        SOURCE,
        OUTPUT,
        token_counter=counter,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        config_path=RUN_CONFIG,
    )
    warm_up = time.perf_counter() - started
    reference = _sha256(OUTPUT)
    durations: list[float] = []
    hashes: list[str] = []
    for repetition in range(1, REPETITIONS + 1):
        started = time.perf_counter()
        run_pipeline(
            SOURCE,
            OUTPUT,
            token_counter=counter,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            config_path=RUN_CONFIG,
        )
        duration = time.perf_counter() - started
        output_hash = _sha256(OUTPUT)
        durations.append(duration)
        hashes.append(output_hash)
        if output_hash != reference:
            raise RuntimeError(
                f"Output changed during measured repetition {repetition}."
            )
    return warm_up, durations, [reference, *hashes]


def _analyze(
    counter: TokenCounter,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document_count = 0
    chunk_count = 0
    sentence_count = 0
    empty_sentence_count = 0
    oversized_sentence_count = 0
    duplicated_sentence_count = 0
    language_distribution: Counter[str] = Counter()
    ambiguous_count = 0
    confidence_values: list[float] = []
    margin_values: list[float] = []
    confidence_by_language: defaultdict[str, list[float]] = defaultdict(list)
    margin_by_language: defaultdict[str, list[float]] = defaultdict(list)
    confusion: defaultdict[str, Counter[str]] = defaultdict(Counter)
    trusted_counts: Counter[str] = Counter()
    trusted_correct: Counter[str] = Counter()
    language_failure_cases: list[dict[str, Any]] = []
    detection_cases: list[dict[str, Any]] = []
    sentences_per_document: list[int] = []
    sentence_characters: list[int] = []
    sentence_tokens: list[int] = []
    chunks_per_document: list[int] = []
    chunk_tokens: list[int] = []
    actual_overlap_tokens: list[int] = []
    unique_chunk_ids: set[str] = set()
    failures: list[dict[str, Any]] = []
    failed_documents: set[str] = set()
    zero_chunk_documents: list[str] = []
    no_sentence_documents: list[str] = []
    one_sentence_documents = 0
    content_preservation_failures: list[str] = []
    suspicious_long_sentences: list[dict[str, Any]] = []
    high_sentence_documents: list[dict[str, Any]] = []
    most_chunked_documents: list[dict[str, Any]] = []
    shortest_chunks: list[dict[str, Any]] = []
    longest_normal_chunks: list[dict[str, Any]] = []
    longest_oversized_chunks: list[dict[str, Any]] = []
    review_by_reason: dict[str, dict[str, Any]] = {}

    chunk_iterator = iter(_jsonl(OUTPUT))
    pending = next(chunk_iterator, None)

    for _, document in _jsonl(SOURCE):
        document_count += 1
        doc_id = str(document["doc_id"])
        detection = detect_language(str(document["texto"]))
        language = detection.language
        language_distribution[language] += 1
        confidence_values.append(detection.confidence)
        margin_values.append(detection.margin)
        confidence_by_language[language].append(detection.confidence)
        margin_by_language[language].append(detection.margin)
        ambiguous_count += int(detection.is_ambiguous)
        detection_cases.append(
            {
                "doc_id": doc_id,
                "fuente": document["fuente"],
                "predicted_language": language,
                "confidence": detection.confidence,
                "margin": detection.margin,
                "scores": detection.scores,
                "is_ambiguous": detection.is_ambiguous,
                "used_default_fallback": detection.is_ambiguous,
                "text_preview": _preview(str(document["texto"]), 240),
            }
        )
        trusted_label = document.get("idioma")
        if trusted_label in {"es", "en", "pt"}:
            label = str(trusted_label)
            trusted_counts[label] += 1
            confusion[label][language] += 1
            if label == language:
                trusted_correct[label] += 1
            elif len(language_failure_cases) < MAX_RECORDED_FAILURES:
                language_failure_cases.append(
                    {
                        "doc_id": doc_id,
                        "fuente": document["fuente"],
                        "expected": label,
                        "predicted": language,
                        "confidence": detection.confidence,
                        "margin": detection.margin,
                        "text_preview": _preview(str(document["texto"]), 240),
                    }
                )

        try:
            sentences = _document_units(document, language=language)
        except Exception as error:
            _failure(
                failures,
                failed_documents,
                doc_id=doc_id,
                category="splitter_exception",
                detail=f"{type(error).__name__}: {error}",
            )
            sentences = []
        sentence_count += len(sentences)
        sentences_per_document.append(len(sentences))
        empty_sentence_count += sum(not sentence.strip() for sentence in sentences)
        if not sentences:
            no_sentence_documents.append(doc_id)
        if len(sentences) == 1:
            one_sentence_documents += 1
        normalized_source = " ".join(str(document["texto"]).split())
        normalized_sentences = " ".join(" ".join(sentences).split())
        if normalized_source != normalized_sentences:
            content_preservation_failures.append(doc_id)
            _failure(
                failures,
                failed_documents,
                doc_id=doc_id,
                category="splitter_content_preservation",
                detail="Whitespace-normalized sentence text differs from source text.",
            )

        maximum_sentence_tokens = 0
        maximum_sentence_characters = 0
        sentence_token_counts = counter.count_many(
            sentences,
            include_document_prefix=True,
        )
        for sentence, tokens in zip(sentences, sentence_token_counts, strict=True):
            characters = len(sentence)
            sentence_characters.append(characters)
            sentence_tokens.append(tokens)
            maximum_sentence_tokens = max(maximum_sentence_tokens, tokens)
            maximum_sentence_characters = max(maximum_sentence_characters, characters)
            oversized_sentence_count += int(tokens > max_tokens)

        document_chunks: list[dict[str, Any]] = []
        while pending is not None and str(pending[1].get("doc_id")) == doc_id:
            _, chunk = pending
            document_chunks.append(chunk)
            pending = next(chunk_iterator, None)
        chunks_per_document.append(len(document_chunks))
        chunk_count += len(document_chunks)
        if not document_chunks:
            zero_chunk_documents.append(doc_id)
            _failure(
                failures,
                failed_documents,
                doc_id=doc_id,
                category="missing_output_document",
                detail="The source document produced no chunks.",
            )

        valid_chunk_texts = [
            str(chunk["texto"])
            for chunk in document_chunks
            if isinstance(chunk.get("texto"), str) and str(chunk["texto"]).strip()
        ]
        valid_chunk_counts = iter(
            counter.count_many(
                valid_chunk_texts,
                include_document_prefix=True,
            )
        )
        for position, chunk in enumerate(document_chunks):
            missing = sorted(REQUIRED_CHUNK_FIELDS - chunk.keys())
            if missing:
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="missing_chunk_fields",
                    detail=", ".join(missing),
                )
            expected_id = f"{doc_id}-chunk-{position:04d}"
            chunk_id = str(chunk.get("chunk_id"))
            if chunk_id != expected_id:
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="invalid_chunk_id",
                    detail=f"Expected {expected_id!r}, received {chunk_id!r}.",
                )
            if chunk_id in unique_chunk_ids:
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="duplicate_chunk_id",
                    detail=chunk_id,
                )
            unique_chunk_ids.add(chunk_id)
            if chunk.get("posicion") != position:
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="invalid_position",
                    detail=f"Expected {position}, received {chunk.get('posicion')!r}.",
                )
            for field in ("doc_id", "fuente", "formato", "fenomeno"):
                if chunk.get(field) != document.get(field):
                    _failure(
                        failures,
                        failed_documents,
                        doc_id=doc_id,
                        category="metadata_mismatch",
                        detail=f"Field {field!r} differs from the source document.",
                    )
            chunk_text = chunk.get("texto")
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="empty_chunk",
                    detail=chunk_id,
                )
                continue
            if chunk_text.startswith(counter.document_prefix):
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="stored_document_prefix",
                    detail=chunk_id,
                )
            actual_tokens = next(valid_chunk_counts)
            if chunk.get("num_tokens") != actual_tokens:
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="token_count_mismatch",
                    detail=(
                        f"{chunk_id} stores {chunk.get('num_tokens')!r}; "
                        f"tokenizer returned {actual_tokens}."
                    ),
                )
            chunk_tokens.append(actual_tokens)
            summary = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "num_tokens": actual_tokens,
                "text_preview": _preview(chunk_text, 240),
            }
            shortest_chunks.append(summary)
            if actual_tokens <= max_tokens:
                longest_normal_chunks.append(summary)
            else:
                longest_oversized_chunks.append(summary)

        chunk_blob = "\n".join(str(chunk.get("texto", "")) for chunk in document_chunks)
        for sentence in sentences:
            if sentence not in chunk_blob:
                _failure(
                    failures,
                    failed_documents,
                    doc_id=doc_id,
                    category="lost_or_cut_sentence",
                    detail=_preview(sentence, 240),
                )

        largest_overlap = 0
        for left, right in zip(document_chunks, document_chunks[1:], strict=False):
            if str(document["formato"]).casefold() in TABULAR_FORMATS:
                overlap = _tabular_overlap(
                    str(left["texto"]),
                    str(right["texto"]),
                )
            else:
                overlap = _overlap_sentences(
                    str(left["texto"]),
                    str(right["texto"]),
                    language=language,
                )
            duplicated_sentence_count += len(overlap)
            overlap_text = " ".join(overlap)
            tokens = (
                counter.count(overlap_text, include_document_prefix=True)
                if overlap_text
                else 0
            )
            actual_overlap_tokens.append(tokens)
            largest_overlap = max(largest_overlap, tokens)

        high_sentence_documents.append(
            {
                "doc_id": doc_id,
                "fuente": document["fuente"],
                "sentence_count": len(sentences),
                "maximum_sentence_tokens": maximum_sentence_tokens,
            }
        )
        most_chunked_documents.append(
            {
                "doc_id": doc_id,
                "fuente": document["fuente"],
                "chunk_count": len(document_chunks),
            }
        )
        if maximum_sentence_tokens > max_tokens or maximum_sentence_characters > 5_000:
            suspicious_long_sentences.append(
                {
                    "doc_id": doc_id,
                    "fuente": document["fuente"],
                    "maximum_sentence_tokens": maximum_sentence_tokens,
                    "maximum_sentence_characters": maximum_sentence_characters,
                }
            )

        if (
            language in {"es", "en", "pt"}
            and f"representative_{language}" not in review_by_reason
        ):
            if not detection.is_ambiguous:
                review_by_reason[f"representative_{language}"] = _candidate(
                    document,
                    detection,
                    sentences,
                    document_chunks,
                    reason=f"Representative {language} document",
                )
        ambiguous_key = "ambiguous_language"
        existing_ambiguous = review_by_reason.get(ambiguous_key)
        if detection.is_ambiguous and (
            existing_ambiguous is None
            or detection.confidence < float(existing_ambiguous["confidence"])
        ):
            review_by_reason[ambiguous_key] = _candidate(
                document,
                detection,
                sentences,
                document_chunks,
                reason="Lowest-confidence ambiguous language detection",
            )
        many_key = "many_chunks"
        existing_many = review_by_reason.get(many_key)
        if existing_many is None or len(document_chunks) > int(
            existing_many["chunk_count"]
        ):
            candidate = _candidate(
                document,
                detection,
                sentences,
                document_chunks,
                reason="Document producing the most chunks",
            )
            candidate["chunk_count"] = len(document_chunks)
            review_by_reason[many_key] = candidate
        if document_chunks:
            shortest = min(document_chunks, key=lambda item: int(item["num_tokens"]))
            short_key = "very_short_chunk"
            existing_short = review_by_reason.get(short_key)
            if existing_short is None or int(shortest["num_tokens"]) < int(
                existing_short["num_tokens"]
            ):
                review_by_reason[short_key] = _candidate(
                    document,
                    detection,
                    sentences,
                    document_chunks,
                    reason="Shortest generated chunk",
                    selected_chunk=shortest,
                )
            normal = [
                item
                for item in document_chunks
                if int(item["num_tokens"]) <= max_tokens
            ]
            if normal:
                utilized = max(normal, key=lambda item: int(item["num_tokens"]))
                utilized_key = "high_utilization"
                existing_utilized = review_by_reason.get(utilized_key)
                if existing_utilized is None or int(utilized["num_tokens"]) > int(
                    existing_utilized["num_tokens"]
                ):
                    review_by_reason[utilized_key] = _candidate(
                        document,
                        detection,
                        sentences,
                        document_chunks,
                        reason="Highest-utilization normal chunk",
                        selected_chunk=utilized,
                    )
            oversized = [
                item for item in document_chunks if int(item["num_tokens"]) > max_tokens
            ]
            if oversized:
                largest = max(oversized, key=lambda item: int(item["num_tokens"]))
                oversized_key = "oversized_sentence"
                existing_oversized = review_by_reason.get(oversized_key)
                if existing_oversized is None or int(largest["num_tokens"]) > int(
                    existing_oversized["num_tokens"]
                ):
                    review_by_reason[oversized_key] = _candidate(
                        document,
                        detection,
                        sentences,
                        document_chunks,
                        reason="Largest complete oversized sentence chunk",
                        selected_chunk=largest,
                    )
        if largest_overlap:
            overlap_key = "overlap_example"
            existing_overlap = review_by_reason.get(overlap_key)
            if existing_overlap is None or largest_overlap > int(
                existing_overlap["actual_overlap_tokens"]
            ):
                review_by_reason[overlap_key] = _candidate(
                    document,
                    detection,
                    sentences,
                    document_chunks,
                    reason="Largest observed whole-sentence overlap",
                    overlap_tokens=largest_overlap,
                )

    if pending is not None:
        _failure(
            failures,
            failed_documents,
            doc_id=str(pending[1].get("doc_id")),
            category="orphan_output_chunk",
            detail="Output contains a chunk that does not align with source order.",
        )

    detection_cases.sort(
        key=lambda item: (float(item["confidence"]), str(item["doc_id"]))
    )
    low_margin = sorted(
        detection_cases,
        key=lambda item: (float(item["margin"]), str(item["doc_id"])),
    )
    trusted_total = sum(trusted_counts.values())
    total_tokens = sum(chunk_tokens)
    overlap_total = sum(actual_overlap_tokens)
    chunk_metrics = {
        "document_count": document_count,
        "chunk_count": chunk_count,
        "chunks_per_document": _distribution(chunks_per_document),
        "tokens_per_chunk": _distribution(chunk_tokens),
        "chunk_limit_utilization_rate": (
            statistics.fmean(chunk_tokens) / max_tokens if chunk_tokens else None
        ),
        "chunks_below_25_percent_utilization": sum(
            tokens < max_tokens * 0.25 for tokens in chunk_tokens
        ),
        "chunks_below_50_percent_utilization": sum(
            tokens < max_tokens * 0.50 for tokens in chunk_tokens
        ),
        "chunks_above_90_percent_utilization": sum(
            max_tokens * 0.90 < tokens <= max_tokens for tokens in chunk_tokens
        ),
        "chunks_exceeding_limit_count": sum(
            tokens > max_tokens for tokens in chunk_tokens
        ),
        "oversized_sentence_chunk_count": sum(
            tokens > max_tokens for tokens in chunk_tokens
        ),
        "empty_chunk_count": sum(tokens <= 0 for tokens in chunk_tokens),
        "documents_producing_zero_chunks": zero_chunk_documents,
        "documents_producing_most_chunks": sorted(
            most_chunked_documents,
            key=lambda item: (-int(item["chunk_count"]), str(item["doc_id"])),
        )[:20],
        "shortest_chunks": sorted(
            shortest_chunks,
            key=lambda item: (int(item["num_tokens"]), str(item["chunk_id"])),
        )[:20],
        "longest_normal_chunks": sorted(
            longest_normal_chunks,
            key=lambda item: (-int(item["num_tokens"]), str(item["chunk_id"])),
        )[:20],
        "longest_oversized_chunks": sorted(
            longest_oversized_chunks,
            key=lambda item: (-int(item["num_tokens"]), str(item["chunk_id"])),
        )[:20],
    }
    analysis = {
        "language_detection": {
            "detected_document_count_by_language": dict(
                sorted(language_distribution.items())
            ),
            "detected_percentage_by_language": {
                language: count / document_count
                for language, count in sorted(language_distribution.items())
            },
            "ambiguous_detection_count": ambiguous_count,
            "ambiguous_detection_rate": ambiguous_count / document_count,
            "default_language_fallback_count": ambiguous_count,
            "average_confidence": statistics.fmean(confidence_values),
            "average_margin": statistics.fmean(margin_values),
            "average_confidence_by_detected_language": {
                language: statistics.fmean(values)
                for language, values in sorted(confidence_by_language.items())
            },
            "average_margin_by_detected_language": {
                language: statistics.fmean(values)
                for language, values in sorted(margin_by_language.items())
            },
            "lowest_confidence_documents": detection_cases[:20],
            "smallest_margin_documents": low_margin[:20],
            "trusted_label_field": "idioma",
            "trusted_label_count": trusted_total,
            "excluded_untrusted_label_count": document_count - trusted_total,
            "accuracy_on_trusted_labels": (
                sum(trusted_correct.values()) / trusted_total if trusted_total else None
            ),
            "accuracy_by_trusted_language": {
                language: trusted_correct[language] / count
                for language, count in sorted(trusted_counts.items())
            },
            "confusion_matrix": {
                language: dict(sorted(row.items()))
                for language, row in sorted(confusion.items())
            },
            "failure_cases": language_failure_cases,
            "accuracy_limitation": (
                "Accuracy excludes idioma='desconocido' because it is not a trusted "
                "language class."
            ),
        },
        "sentences": {
            "document_count": document_count,
            "total_sentences": sentence_count,
            "sentences_per_document": _distribution(sentences_per_document),
            "empty_sentence_count": empty_sentence_count,
            "sentence_length_characters": _distribution(sentence_characters),
            "sentence_length_real_tokens_including_document_prefix": _distribution(
                sentence_tokens
            ),
            "oversized_sentence_count": oversized_sentence_count,
            "documents_with_no_sentences": no_sentence_documents,
            "documents_with_one_sentence_count": one_sentence_documents,
            "documents_with_high_sentence_count": sorted(
                high_sentence_documents,
                key=lambda item: (-int(item["sentence_count"]), str(item["doc_id"])),
            )[:20],
            "documents_with_suspiciously_long_sentences": sorted(
                suspicious_long_sentences,
                key=lambda item: (
                    -int(item["maximum_sentence_tokens"]),
                    str(item["doc_id"]),
                ),
            )[:50],
            "content_preservation_failure_count": len(content_preservation_failures),
            "splitter_failure_count": sum(
                failure["category"] == "splitter_exception" for failure in failures
            ),
            "boundary_quality_limitation": (
                "The real corpus has no human-verified sentence boundaries; precision, "
                "recall, and F1 are not claimed."
            ),
        },
        "chunks": chunk_metrics,
        "overlap": {
            "configured_overlap_tokens": overlap_tokens,
            "adjacent_chunk_pair_count": len(actual_overlap_tokens),
            "actual_overlap_tokens": _distribution(actual_overlap_tokens),
            "duplicated_sentence_count": duplicated_sentence_count,
            "duplicated_token_estimate": overlap_total,
            "duplicated_content_rate": overlap_total / total_tokens
            if total_tokens
            else 0.0,
        },
        "integrity": {
            "cut_sentence_count": sum(
                failure["category"] == "lost_or_cut_sentence" for failure in failures
            ),
            "lost_sentence_count": sum(
                failure["category"] == "lost_or_cut_sentence" for failure in failures
            ),
            "note": (
                "A sentence absent as a complete substring is conservatively counted in "
                "both cut and lost categories; complete oversized sentences are excluded."
            ),
            "validation_failure_count": len(failures),
            "failed_document_count": len(failed_documents),
            "failures": failures,
            "failed_documents": sorted(failed_documents),
        },
        "suspicious_cases": {
            "long_sentences": sorted(
                suspicious_long_sentences,
                key=lambda item: (
                    -int(item["maximum_sentence_tokens"]),
                    str(item["doc_id"]),
                ),
            )[:50],
            "high_sentence_documents": sorted(
                high_sentence_documents,
                key=lambda item: (-int(item["sentence_count"]), str(item["doc_id"])),
            )[:20],
        },
    }
    return analysis, list(review_by_reason.values())


def _write_review(samples: list[dict[str, Any]]) -> None:
    lines = [
        "# Real Corpus Chunking Review Samples",
        "",
        "All previews are truncated. This file does not reproduce complete documents.",
        "",
    ]
    for sample in samples:
        lines.extend(
            (
                f"## {sample['reason']}",
                "",
                f"- `doc_id`: `{sample['doc_id']}`",
                f"- `fuente`: `{sample['fuente']}`",
                f"- detected language: `{sample['detected_language']}`",
                f"- confidence: `{float(sample['confidence']):.6f}`",
                f"- margin: `{float(sample['margin']):.6f}`",
                f"- chunk: `{sample['chunk_id']}`",
                f"- tokens: `{sample['num_tokens']}`",
                f"- actual overlap tokens: `{sample['actual_overlap_tokens']}`",
                "",
                "Original text preview:",
                "",
                f"> {sample['original_text_preview']}",
                "",
                "Sentence previews:",
                "",
            )
        )
        sentence_previews = sample["sentence_preview"]
        if isinstance(sentence_previews, list):
            lines.extend(f"- {preview}" for preview in sentence_previews)
        lines.extend(("", "Chunk preview:", "", f"> {sample['chunk_preview']}", ""))
    REVIEW.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _format_number(value: object, digits: int = 4) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_report(metrics: dict[str, Any]) -> None:
    input_validation = metrics["input_validation"]
    language = metrics["language_detection"]
    sentences = metrics["sentence_metrics"]
    chunks = metrics["chunk_metrics"]
    overlap = metrics["overlap_metrics"]
    performance = metrics["performance"]
    storage = metrics["theoretical_embedding_storage"]
    integrity = metrics["validation"]
    synthetic = metrics["synthetic_baseline"]
    assert isinstance(input_validation, dict)
    assert isinstance(language, dict)
    assert isinstance(sentences, dict)
    assert isinstance(chunks, dict)
    assert isinstance(overlap, dict)
    assert isinstance(performance, dict)
    assert isinstance(storage, dict)
    assert isinstance(integrity, dict)
    assert isinstance(synthetic, dict)
    input_counts = input_validation["counts"]
    token_distribution = chunks["tokens_per_chunk"]
    sentence_distribution = sentences["sentences_per_document"]
    overlap_distribution = overlap["actual_overlap_tokens"]
    assert isinstance(input_counts, dict)
    assert isinstance(token_distribution, dict)
    assert isinstance(sentence_distribution, dict)
    assert isinstance(overlap_distribution, dict)
    readiness = metrics["embedding_readiness"]
    assert isinstance(readiness, dict)
    lines = [
        "# Real Corpus Chunking Report",
        "",
        "## Purpose and measured scope",
        "",
        "This report records structural and performance facts measured on the real "
        "CODEFEST JSONL. It does not measure retrieval quality and does not include "
        "encoder inference, embeddings, FAISS, retrieval, or agents.",
        "",
        "## Corpus source and frozen configuration",
        "",
        "- Source: `outputs/resultados/documentos.jsonl`",
        "- Output: `outputs/resultados/chunks.jsonl`",
        "- Encoder tokenizer: `intfloat/multilingual-e5-small`",
        "- Normal limit: 256 tokens including `passage: `",
        "- Whole-sentence overlap target: 32 tokens",
        "- Embedding contract: 384 dimensions, L2 normalization, `IndexFlatIP`",
        "",
        "## Input validation",
        "",
        f"The corpus contains {input_counts['line_count']} lines and "
        f"{input_counts['valid_json_object_count']} valid JSON objects. Empty lines: "
        f"{input_counts['empty_line_count']}; malformed lines: "
        f"{input_counts['malformed_line_count']}; missing-field documents: "
        f"{input_counts['missing_required_fields_count']}; invalid or empty text: "
        f"{input_counts['empty_or_invalid_text_count']}.",
        "",
        "## Language distribution and ambiguity",
        "",
        f"Detected distribution: `{language['detected_document_count_by_language']}`. "
        f"Ambiguous detections: {language['ambiguous_detection_count']} "
        f"({_format_number(language['ambiguous_detection_rate'] * 100, 2)}%). "
        f"Accuracy on trusted `idioma` labels was "
        f"{_format_number(language['accuracy_on_trusted_labels'] * 100, 2)}%; "
        "documents labeled `desconocido` were excluded.",
        "",
        "Low-confidence and smallest-margin cases are recorded in the JSON metrics and "
        "truncated examples are available in the review artifact.",
        "",
        "## Sentence statistics",
        "",
        f"Total sentences: {sentences['total_sentences']}. Sentences per document: "
        f"minimum {_format_number(sentence_distribution['minimum'])}, median "
        f"{_format_number(sentence_distribution['median'])}, average "
        f"{_format_number(sentence_distribution['average'])}, p95 "
        f"{_format_number(sentence_distribution['percentile_95'])}, maximum "
        f"{_format_number(sentence_distribution['maximum'])}. Oversized sentences: "
        f"{sentences['oversized_sentence_count']}.",
        "",
        "No boundary precision, recall, or F1 is claimed because the real corpus has no "
        "human-verified sentence-boundary annotations. The synthetic controlled baseline "
        "remains separate.",
        "",
        "## Chunk statistics",
        "",
        f"Generated chunks: {chunks['chunk_count']}. Tokens per chunk: minimum "
        f"{_format_number(token_distribution['minimum'])}, median "
        f"{_format_number(token_distribution['median'])}, average "
        f"{_format_number(token_distribution['average'])}, p95 "
        f"{_format_number(token_distribution['percentile_95'])}, maximum "
        f"{_format_number(token_distribution['maximum'])}. Normal-limit utilization: "
        f"{_format_number(chunks['chunk_limit_utilization_rate'] * 100, 2)}%. Empty chunks: "
        f"{chunks['empty_chunk_count']}.",
        "",
        "## Overlap and oversized behavior",
        "",
        f"Adjacent chunk pairs: {overlap['adjacent_chunk_pair_count']}. Actual overlap "
        f"tokens had minimum {_format_number(overlap_distribution['minimum'])}, median "
        f"{_format_number(overlap_distribution['median'])}, average "
        f"{_format_number(overlap_distribution['average'])}, and maximum "
        f"{_format_number(overlap_distribution['maximum'])}. Complete oversized sentence "
        f"chunks: {chunks['oversized_sentence_chunk_count']}. These are valid exceptions, "
        "not cut sentences.",
        "",
        "## Throughput, memory, and output size",
        "",
        f"After one warm-up, five steady-state runs averaged "
        f"{_format_number(performance['average_duration_seconds'])} seconds (minimum "
        f"{_format_number(performance['minimum_duration_seconds'])}, median "
        f"{_format_number(performance['median_duration_seconds'])}, maximum "
        f"{_format_number(performance['maximum_duration_seconds'])}). Average throughput: "
        f"{_format_number(performance['documents_per_second'])} documents/s, "
        f"{_format_number(performance['sentences_per_second'])} sentences/s, and "
        f"{_format_number(performance['chunks_per_second'])} chunks/s. Peak process memory "
        "was not measured because no suitable dependency was installed; available system "
        "RAM is recorded when the Windows API succeeds.",
        "",
        f"Input size: {metrics['input_size_bytes']} bytes. Output size: "
        f"{metrics['output_size_bytes']} bytes. Output/input ratio: "
        f"{_format_number(metrics['output_to_input_size_ratio'])}.",
        "",
        "## Theoretical embedding storage",
        "",
        f"Each raw float32 vector requires {storage['bytes_per_embedding']} bytes. The "
        f"actual chunk set would require {storage['actual_corpus']['bytes']} raw bytes "
        f"({_format_number(storage['actual_corpus']['mib'])} MiB). This excludes FAISS, "
        "metadata, model files, temporary arrays, Python objects, and OS overhead.",
        "",
        "## Validation failures and suspicious documents",
        "",
        f"Validation failures: {integrity['validation_failure_count']}; failed documents: "
        f"{integrity['failed_document_count']}; cut sentences: "
        f"{integrity['cut_sentence_count']}; lost sentences: "
        f"{integrity['lost_sentence_count']}. Suspicious structural cases are listed in "
        "the JSON metrics and selected previews appear in the review artifact.",
        "",
        "## Comparison with the synthetic baseline",
        "",
        f"The preserved synthetic baseline contains {synthetic['document_count']} documents "
        f"and {synthetic['chunk_count']} chunks. It is useful for controlled invariants; "
        "the real-corpus measurements above describe scale and structure. Neither artifact "
        "measures retrieval quality.",
        "",
        "## Limitations and recommendations",
        "",
        "- Review low-confidence and `desconocido` language cases manually.",
        "- Review the longest sentences and every oversized chunk before embedding.",
        "- Pin the tokenizer revision before a final production rebuild.",
        "- Evaluate retrieval quality only after aligned embeddings and metadata exist.",
        "- Revisit the encoder, 256-token limit, 32-token overlap, Portuguese fallback, "
        "splitter, or oversized policy only if real retrieval or manual-review evidence "
        "shows a concrete failure.",
        "",
        "## Embedding teammate readiness",
        "",
        f"Status: **{readiness['status']}**",
        "",
        str(readiness["reason"]),
        "",
        "The embedding teammate must preserve JSONL order, prepend `passage: ` exactly "
        "once, use 384-dimensional vectors from the exact E5 model, apply L2 "
        "normalization, and maintain row alignment with metadata. Query encoding must use "
        "`query: `. The future baseline index is `IndexFlatIP`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _legacy_main() -> None:
    """Run, validate, measure, and document the real-corpus chunking pipeline."""
    config = load_encoder_config()
    input_validation = _validate_input(SOURCE)
    source_hash_before = str(input_validation["sha256_before"])
    cold_started = time.perf_counter()
    counter = TokenCounter(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
        local_files_only=True,
    )
    tokenizer_class = counter.tokenizer_class
    cold_load_seconds = time.perf_counter() - cold_started
    warm_up_seconds, durations, output_hashes = _run_benchmark(
        counter,
        max_tokens=config["chunk_max_tokens"],
        overlap_tokens=config["overlap_tokens"],
    )
    analysis, review_samples = _analyze(
        counter,
        max_tokens=config["chunk_max_tokens"],
        overlap_tokens=config["overlap_tokens"],
    )
    source_hash_after = _sha256(SOURCE)
    if source_hash_after != source_hash_before:
        raise RuntimeError("The source corpus changed during evaluation.")
    available_ram, ram_limitation = _available_ram()
    chunk_metrics = analysis["chunks"]
    sentence_metrics = analysis["sentences"]
    integrity = analysis["integrity"]
    assert isinstance(chunk_metrics, dict)
    assert isinstance(sentence_metrics, dict)
    assert isinstance(integrity, dict)
    document_count = int(chunk_metrics["document_count"])
    chunk_count = int(chunk_metrics["chunk_count"])
    sentence_count = int(sentence_metrics["total_sentences"])
    average_duration = statistics.fmean(durations)
    input_size = SOURCE.stat().st_size
    output_size = OUTPUT.stat().st_size
    bytes_per_embedding = config["embedding_dimension"] * 4
    actual_bytes = chunk_count * bytes_per_embedding
    synthetic = json.loads(SYNTHETIC_METRICS.read_text(encoding="utf-8"))
    critical_failure_count = int(integrity["validation_failure_count"])
    warning_count = (
        int(chunk_metrics["oversized_sentence_chunk_count"])
        + int(analysis["language_detection"]["ambiguous_detection_count"])
        + len(sentence_metrics["documents_with_suspiciously_long_sentences"])
    )
    if critical_failure_count:
        readiness_status = "NOT READY"
        readiness_reason = (
            "Output or sentence-integrity validation failures must be resolved before "
            "embedding."
        )
    elif warning_count:
        readiness_status = "READY WITH WARNINGS"
        readiness_reason = (
            "The output contract and integrity checks passed, but ambiguous language and "
            "structural edge cases require manual review."
        )
    else:
        readiness_status = "READY"
        readiness_reason = (
            "All measured output, token, metadata, and integrity checks passed."
        )
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_mode": "real_corpus_chunking",
        "corpus_type": "real_codefest_jsonl",
        "source_path": SOURCE.relative_to(ROOT).as_posix(),
        "output_path": OUTPUT.relative_to(ROOT).as_posix(),
        "frozen_configuration": config,
        "tokenizer_observation": {
            "class": tokenizer_class,
            "revision": counter.revision,
            "local_files_only": True,
            "full_encoder_loaded": False,
        },
        "environment": {
            "python": sys.version,
            "operating_system": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "available_ram_bytes": available_ram,
            "available_ram_limitation": ram_limitation,
            "peak_process_memory_bytes": None,
            "peak_process_memory_limitation": (
                "psutil is unavailable and tracemalloc excludes native tokenizer memory; "
                "a misleading partial value was not recorded."
            ),
            "packages": _package_versions(),
        },
        "input_validation": {
            **input_validation,
            "sha256_after": source_hash_after,
            "source_unchanged": source_hash_after == source_hash_before,
        },
        "language_detection": analysis["language_detection"],
        "sentence_metrics": sentence_metrics,
        "chunk_metrics": chunk_metrics,
        "overlap_metrics": analysis["overlap"],
        "performance": {
            "tokenizer_cold_load_seconds_without_download": cold_load_seconds,
            "warm_up_duration_seconds": warm_up_seconds,
            "steady_state_repetitions": REPETITIONS,
            "steady_state_durations_seconds": durations,
            "minimum_duration_seconds": min(durations),
            "maximum_duration_seconds": max(durations),
            "average_duration_seconds": average_duration,
            "median_duration_seconds": statistics.median(durations),
            "standard_deviation_seconds": statistics.pstdev(durations),
            "documents_per_second": document_count / average_duration,
            "sentences_per_second": sentence_count / average_duration,
            "chunks_per_second": chunk_count / average_duration,
            "megabytes_per_second": (input_size / 1_000_000) / average_duration,
            "same_tokenizer_instance_reused": True,
            "benchmark_logging_level": "ERROR",
            "download_time_included": False,
        },
        "input_size_bytes": input_size,
        "output_size_bytes": output_size,
        "output_to_input_size_ratio": output_size / input_size,
        "theoretical_embedding_storage": {
            "embedding_dimension": config["embedding_dimension"],
            "dtype": "float32",
            "float32_bytes": 4,
            "bytes_per_embedding": bytes_per_embedding,
            "actual_corpus": {
                "chunk_count": chunk_count,
                "bytes": actual_bytes,
                "kib": actual_bytes / 1024,
                "mib": actual_bytes / (1024**2),
                "gib": actual_bytes / (1024**3),
            },
            "estimates": {
                label: {
                    "chunks": count,
                    "bytes": count * bytes_per_embedding,
                    "kib": count * bytes_per_embedding / 1024,
                    "mib": count * bytes_per_embedding / (1024**2),
                    "gib": count * bytes_per_embedding / (1024**3),
                }
                for label, count in (
                    ("1k", 1_000),
                    ("10k", 10_000),
                    ("100k", 100_000),
                    ("1m", 1_000_000),
                )
            },
            "exclusions": [
                "FAISS overhead",
                "metadata.jsonl",
                "model files",
                "temporary arrays",
                "Python object overhead",
                "operating system overhead",
            ],
        },
        "determinism": {
            "result": len(set(output_hashes)) == 1,
            "executions_compared": len(output_hashes),
            "output_sha256": output_hashes[0],
            "all_hashes": output_hashes,
        },
        "validation": integrity,
        "failed_documents": integrity["failed_documents"],
        "suspicious_cases": analysis["suspicious_cases"],
        "synthetic_baseline": {
            "path": SYNTHETIC_METRICS.relative_to(ROOT).as_posix(),
            "document_count": synthetic["document_count"],
            "sentence_count": synthetic["sentence_count"],
            "chunk_count": synthetic["chunk_count"],
            "cut_sentence_count": synthetic["cut_sentence_count"],
            "lost_sentence_count": synthetic["lost_sentence_count"],
            "note": "Preserved controlled baseline; not overwritten by this evaluation.",
        },
        "embedding_readiness": {
            "status": readiness_status,
            "reason": readiness_reason,
            "checklist": {
                "chunks_jsonl_exists": OUTPUT.is_file(),
                "canonical_encoder_configuration_exists": (
                    ROOT / "config/encoder_config.json"
                ).is_file(),
                "exact_model_recorded": True,
                "document_prefix_recorded": True,
                "tokenizer_behavior_verified": critical_failure_count == 0,
                "embedding_dimension_recorded": True,
                "normalization_recorded": True,
                "future_faiss_index_recorded": True,
                "stable_chunk_order_verified": len(set(output_hashes)) == 1,
                "metadata_contract_verified": critical_failure_count == 0,
            },
        },
        "limitations": [
            "Real-corpus sentence boundaries have no human-verified gold labels.",
            "Language accuracy excludes idioma='desconocido'.",
            "Portuguese uses the Spanish PySBD backend.",
            "Tokenizer revision is not pinned to a commit hash.",
            "Peak process memory was not measured without a suitable dependency.",
            "No retrieval-quality claim is possible before embeddings and FAISS exist.",
        ],
        "scope_confirmation": {
            "source_corpus_modified": False,
            "embeddings_generated": False,
            "faiss_index_generated": False,
            "retrieval_implemented": False,
            "agents_implemented": False,
        },
    }
    METRICS.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_review(review_samples)
    _write_report(metrics)
    print(
        json.dumps(
            {
                "documents": document_count,
                "sentences": sentence_count,
                "chunks": chunk_count,
                "failures": critical_failure_count,
                "readiness": readiness_status,
                "durations_seconds": durations,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    """Reject the superseded monolithic path with actionable commands."""
    raise SystemExit(
        "The monolithic real-corpus evaluator is retired. Use one independent "
        "command: 'python -m nerv.chunking.real_corpus produce', "
        "'python -m nerv.chunking.real_corpus validate', or "
        "'python -m nerv.chunking.real_corpus benchmark'."
    )


if __name__ == "__main__":
    main()
