"""Localize bounded-splitter differences without publishing document content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter, sleep
from typing import Any, cast

from nerv.chunking.sentence_splitter import (
    DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
    DEFAULT_TARGET_BLOCK_CHARACTERS,
    _BoundedSegmentationTrace,
    _get_segmenter,
    _iter_text_blocks,
    _normalize_for_integrity,
    _split_sentences_bounded_with_stats,
    _split_sentences_whole_document,
)

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("outputs/resultados/documentos.jsonl")
DEFAULT_OUTPUT = HERE / "results" / "bounded_splitter_diagnostics_v1.json"
DEFAULT_TIMEOUT_SECONDS = 15 * 60.0
DEFAULT_EXCERPT_CHARACTERS = 320
TARGET_DOC_IDS = ("DOC-83e80067468b", "DOC-2d7c54534744")
_TERMINAL_STATUSES = frozenset({"completed", "completed_with_findings"})


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--whole-timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--excerpt-characters",
        type=int,
        default=DEFAULT_EXCERPT_CHARACTERS,
    )
    parser.add_argument(
        "--refine-existing",
        action="store_true",
        help="Refine local classifications without rerunning segmentation.",
    )
    worker = parser.add_argument_group(argparse.SUPPRESS)
    worker.add_argument("--whole-worker", action="store_true")
    worker.add_argument("--worker-text", type=Path)
    worker.add_argument("--worker-language")
    worker.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.whole_timeout_seconds <= 0:
        parser.error("--whole-timeout-seconds must be greater than zero.")
    if args.excerpt_characters <= 0:
        parser.error("--excerpt-characters must be greater than zero.")
    return args


def _digest_sentences(sentences: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for sentence in sentences:
        encoded = sentence.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    before = limit // 2
    after = limit - before
    return f"{value[:before]}...[truncated]...{value[-after:]}"


def _sanitize_excerpt(value: str, limit: int) -> str:
    """Preserve OCR punctuation/spacing shape without publishing corpus text."""
    clipped = _clip(value, limit)
    return "".join(
        "L" if character.isalpha() else "D" if character.isdigit() else character
        for character in clipped
    )


def _sentence_evidence(sentence: str, source: str, limit: int) -> dict[str, Any]:
    first = source.find(sentence)
    offset = first if first >= 0 and first == source.rfind(sentence) else None
    return {
        "sanitized_text_shape": _sanitize_excerpt(sentence, limit),
        "sanitized_normalized_shape": _sanitize_excerpt(
            _normalize_for_integrity(sentence), limit
        ),
        "length": len(sentence),
        "source_offset": offset,
    }


def compare_sentence_sequences(
    baseline: Sequence[str],
    bounded: Sequence[str],
    *,
    source: str,
    excerpt_characters: int,
) -> dict[str, Any]:
    """Return compact evidence for the first sequential sentence difference."""
    index = next(
        (
            position
            for position, (baseline_item, bounded_item) in enumerate(
                zip(baseline, bounded, strict=False)
            )
            if baseline_item != bounded_item
        ),
        min(len(baseline), len(bounded)),
    )
    exact_match = len(baseline) == len(bounded) and index == len(baseline)
    return {
        "exact_match": exact_match,
        "first_differing_sentence_index": None if exact_match else index,
        "matching_before": [
            _sentence_evidence(item, source, excerpt_characters)
            for item in baseline[max(0, index - 5) : index]
        ],
        "baseline_after": [
            _sentence_evidence(item, source, excerpt_characters)
            for item in baseline[index : index + 5]
        ],
        "bounded_after": [
            _sentence_evidence(item, source, excerpt_characters)
            for item in bounded[index : index + 5]
        ],
    }


def _first_linear_difference(left: str, right: str) -> int | None:
    for offset, (left_character, right_character) in enumerate(
        zip(left, right, strict=False)
    ):
        if left_character != right_character:
            return offset
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _normalized_offset_to_source_offset(text: str, target: int) -> int | None:
    """Map one normalized position to source in a single pass and constant space."""
    normalized_offset = 0
    pending_space = False
    emitted_any = False
    for source_offset, character in enumerate(text):
        if character.isspace():
            if emitted_any:
                pending_space = True
            continue
        if pending_space:
            if normalized_offset == target:
                return source_offset
            normalized_offset += 1
            pending_space = False
        if normalized_offset == target:
            return source_offset
        normalized_offset += 1
        emitted_any = True
    return len(text) if normalized_offset == target else None


def _change_kind(original: str, reconstructed: str, offset: int) -> str:
    lookahead = 128
    original_character = original[offset : offset + 1]
    reconstructed_character = reconstructed[offset : offset + 1]
    if reconstructed_character and original_character in reconstructed[
        offset + 1 : offset + lookahead
    ]:
        return "added"
    if original_character and reconstructed_character in original[
        offset + 1 : offset + lookahead
    ]:
        return "missing"
    if not original_character:
        return "added"
    if not reconstructed_character:
        return "missing"
    return "changed"


def _nearest_trace(
    traces: Sequence[_BoundedSegmentationTrace], source_offset: int | None
) -> _BoundedSegmentationTrace | None:
    if source_offset is None or not traces:
        return None
    containing = next(
        (
            trace
            for trace in traces
            if trace.source_start_offset <= source_offset < trace.source_end_offset
        ),
        None,
    )
    if containing is not None:
        return containing
    return min(
        traces,
        key=lambda trace: min(
            abs(source_offset - trace.source_start_offset),
            abs(source_offset - trace.source_end_offset),
        ),
    )


def diagnose_reconstruction(
    original_text: str,
    sentences: Sequence[str],
    *,
    traces: Sequence[_BoundedSegmentationTrace],
    excerpt_characters: int,
) -> dict[str, Any]:
    """Locate a normalized reconstruction difference with a linear scan."""
    original = _normalize_for_integrity(original_text)
    reconstructed = _normalize_for_integrity(" ".join(sentences))
    offset = _first_linear_difference(original, reconstructed)
    if offset is None:
        return {
            "matches": True,
            "first_differing_normalized_offset": None,
            "source_offset": None,
            "change_kind": None,
            "original_context": None,
            "reconstructed_context": None,
            "nearest_block": None,
        }
    radius = max(1, excerpt_characters // 2)
    source_offset = _normalized_offset_to_source_offset(original_text, offset)
    trace = _nearest_trace(traces, source_offset)
    nearest_block: dict[str, Any] | None = None
    if trace is not None:
        boundary_distance = min(
            abs(cast(int, source_offset) - trace.source_start_offset),
            abs(trace.source_end_offset - cast(int, source_offset)),
        )
        nearest_block = {
            "block_number": trace.block_number,
            "source_start_offset": trace.source_start_offset,
            "source_end_offset": trace.source_end_offset,
            "distance_to_boundary": boundary_distance,
            "near_boundary": boundary_distance <= 256,
            "carry_before_characters": trace.carry_before_characters,
            "carry_after_characters": trace.carry_after_characters,
            "carry_before_normalized_digest": (
                trace.carry_before_normalized_digest
            ),
            "carry_after_normalized_digest": trace.carry_after_normalized_digest,
        }
    return {
        "matches": False,
        "first_differing_normalized_offset": offset,
        "source_offset": source_offset,
        "change_kind": _change_kind(original, reconstructed, offset),
        "original_context_shape": _sanitize_excerpt(
            original[max(0, offset - radius) : offset + radius],
            excerpt_characters,
        ),
        "reconstructed_context_shape": _sanitize_excerpt(
            reconstructed[max(0, offset - radius) : offset + radius],
            excerpt_characters,
        ),
        "nearest_block": nearest_block,
    }


def _trace_for_source_offset(
    traces: Sequence[_BoundedSegmentationTrace], source_offset: int | None
) -> dict[str, Any] | None:
    trace = _nearest_trace(traces, source_offset)
    if trace is None or source_offset is None:
        return None
    distance = min(
        abs(source_offset - trace.source_start_offset),
        abs(trace.source_end_offset - source_offset),
    )
    payload = asdict(trace)
    payload["distance_to_boundary"] = distance
    payload["near_boundary"] = distance <= 256
    return payload


def _capture_affected_block(
    text: str,
    *,
    language: str,
    block_number: int,
    source_offset: int,
    excerpt_characters: int,
    target_block_characters: int = DEFAULT_TARGET_BLOCK_CHARACTERS,
    maximum_block_characters: int = DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
) -> dict[str, Any]:
    """Replay one block and retain only bounded excerpts from its PySBD output."""
    normalized_text = text.strip()
    segmenter = _get_segmenter(language)
    carry = ""
    source_start = 0
    blocks = list(
        _iter_text_blocks(
            normalized_text,
            target_block_characters=target_block_characters,
            maximum_block_characters=maximum_block_characters,
        )
    )
    for current_number, block in enumerate(blocks, start=1):
        combined = f"{carry}{block}"
        raw_segments = [item for item in segmenter.segment(combined) if item.strip()]
        if current_number == block_number:
            local_offset = len(carry) + max(0, source_offset - source_start)
            cumulative = 0
            affected_index = 0
            for item_index, item in enumerate(raw_segments):
                cumulative += len(item)
                if cumulative >= local_offset:
                    affected_index = item_index
                    break
            return {
                "block_number": current_number,
                "source_start_offset": source_start,
                "source_end_offset": source_start + len(block),
                "carry_before_segmentation": {
                    "length": len(carry),
                    "sanitized_excerpt_shape": _sanitize_excerpt(
                        carry, excerpt_characters
                    ),
                },
                "combined_length": len(combined),
                "affected_pysbd_item_index": affected_index,
                "pysbd_output_near_affected_item": [
                    {
                        "index": item_index,
                        "length": len(item),
                        "sanitized_excerpt_shape": _sanitize_excerpt(
                            item, excerpt_characters
                        ),
                    }
                    for item_index, item in enumerate(raw_segments)
                    if max(0, affected_index - 5)
                    <= item_index
                    < affected_index + 6
                ],
                "carry_after_segmentation": {
                    "length": (
                        0
                        if current_number == len(blocks) or not raw_segments
                        else len(raw_segments[-1])
                    ),
                    "sanitized_excerpt_shape": (
                        ""
                        if current_number == len(blocks) or not raw_segments
                        else _sanitize_excerpt(
                            raw_segments[-1], excerpt_characters
                        )
                    ),
                },
            }
        carry = raw_segments[-1] if raw_segments else combined
        source_start += len(block)
    raise ValueError(f"Block number {block_number} is outside the document.")


def _local_window_comparisons(
    text: str,
    *,
    language: str,
    source_offset: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for radius in (1_600, 5_000):
        start = max(0, source_offset - radius)
        end = min(len(text), source_offset + radius)
        window = text[start:end]
        whole = _split_sentences_whole_document(window, language=language)
        bounded = _split_sentences_bounded_with_stats(
            window,
            language=language,
            target_block_characters=256,
            maximum_block_characters=512,
        ).sentences
        original_normalized = _normalize_for_integrity(window)
        whole_reconstructed = _normalize_for_integrity(" ".join(whole))
        bounded_reconstructed = _normalize_for_integrity(" ".join(bounded))
        whole_difference = _first_linear_difference(
            original_normalized,
            whole_reconstructed,
        )
        bounded_difference = _first_linear_difference(
            original_normalized,
            bounded_reconstructed,
        )

        def difference_digest(reconstructed: str, offset: int | None) -> str | None:
            if offset is None:
                return None
            context = reconstructed[max(0, offset - 128) : offset + 128]
            return hashlib.sha256(context.encode("utf-8")).hexdigest()

        results.append(
            {
                "radius_characters": radius,
                "source_start_offset": start,
                "source_end_offset": end,
                "whole_sentence_count": len(whole),
                "bounded_sentence_count": len(bounded),
                "exact_sentence_list_match": whole == bounded,
                "whole_reconstruction_match": (
                    whole_reconstructed == original_normalized
                ),
                "bounded_reconstruction_match": (
                    bounded_reconstructed == original_normalized
                ),
                "whole_first_reconstruction_difference": whole_difference,
                "bounded_first_reconstruction_difference": bounded_difference,
                "whole_reconstruction_difference_digest": difference_digest(
                    whole_reconstructed,
                    whole_difference,
                ),
                "bounded_reconstruction_difference_digest": difference_digest(
                    bounded_reconstructed,
                    bounded_difference,
                ),
            }
        )
    return results


def _classify_sentence_difference(
    comparison: dict[str, Any], local_windows: Sequence[dict[str, Any]]
) -> str:
    if comparison["exact_match"]:
        return "unresolved"
    if local_windows and all(
        item["exact_sentence_list_match"] for item in local_windows
    ):
        return "block_context_difference"
    return "unresolved"


def _classify_reconstruction(
    local_windows: Sequence[dict[str, Any]],
) -> str:
    if any(
        not item["whole_reconstruction_match"]
        and not item["bounded_reconstruction_match"]
        and item["whole_first_reconstruction_difference"]
        == item["bounded_first_reconstruction_difference"]
        and item["whole_reconstruction_difference_digest"]
        == item["bounded_reconstruction_difference_digest"]
        and item["whole_reconstruction_difference_digest"] is not None
        for item in local_windows
    ):
        return "inherited_pysbd_transformation"
    return "unresolved"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
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
            temporary = Path(destination.name)
            json.dump(payload, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        for attempt in range(50):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 49:
                    raise
                sleep(0.1)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_documents(path: Path) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = cast(dict[str, Any], json.loads(line))
            doc_id = item.get("doc_id")
            if doc_id not in TARGET_DOC_IDS:
                continue
            if doc_id in documents:
                raise ValueError(
                    f"Diagnostic document {doc_id!r} is duplicated at line "
                    f"{line_number}."
                )
            if not isinstance(item.get("texto"), str):
                raise TypeError(f"Diagnostic document {doc_id!r} has invalid text.")
            documents[cast(str, doc_id)] = item
    missing = set(TARGET_DOC_IDS) - documents.keys()
    if missing:
        raise ValueError(f"Diagnostic documents not found: {sorted(missing)!r}.")
    return documents


def _whole_worker(text_path: Path, language: str, output_path: Path) -> int:
    text = text_path.read_text(encoding="utf-8")
    started = perf_counter()
    sentences = _split_sentences_whole_document(text, language=language)
    payload = {
        "status": "completed",
        "duration_seconds": perf_counter() - started,
        "sentences": sentences,
    }
    _atomic_write_json(output_path, payload)
    return 0


def _run_whole(
    text: str,
    *,
    language: str,
    timeout_seconds: float,
    work_directory: Path,
) -> tuple[dict[str, Any], list[str] | None]:
    text_path: Path | None = None
    result_path: Path | None = None
    process: subprocess.Popen[str] | None = None
    started = perf_counter()
    del work_directory
    try:
        temporary_directory_context = TemporaryDirectory(
            prefix="nerv-bounded-diagnostic-"
        )
        temporary_directory = Path(temporary_directory_context.name)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=temporary_directory,
            prefix=".bounded-diagnostic-text.",
            suffix=".txt",
            delete=False,
        ) as text_file:
            text_path = Path(text_file.name)
            text_file.write(text)
        with NamedTemporaryFile(
            dir=temporary_directory,
            prefix=".bounded-diagnostic-result.",
            suffix=".json",
            delete=False,
        ) as result_file:
            result_path = Path(result_file.name)
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--whole-worker",
                "--worker-text",
                str(text_path),
                "--worker-language",
                language,
                "--worker-output",
                str(result_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return (
                {
                    "status": "timed_out",
                    "duration_seconds": perf_counter() - started,
                    "timeout_seconds": timeout_seconds,
                    "sentence_count": None,
                    "sentence_list_digest": None,
                    "normalized_reconstruction_match": None,
                },
                None,
            )
        if process.returncode != 0:
            return (
                {
                    "status": "error",
                    "duration_seconds": perf_counter() - started,
                    "sentence_count": None,
                    "error": (
                        f"Worker exit {process.returncode}; "
                        f"stdout={stdout[-500:]!r}; stderr={stderr[-500:]!r}"
                    ),
                },
                None,
            )
        result = cast(
            dict[str, Any], json.loads(result_path.read_text(encoding="utf-8"))
        )
        sentences = cast(list[str], result.pop("sentences"))
        result.update(
            {
                "timeout_seconds": timeout_seconds,
                "sentence_count": len(sentences),
                "sentence_list_digest": _digest_sentences(sentences),
                "normalized_reconstruction_match": (
                    _normalize_for_integrity(" ".join(sentences))
                    == _normalize_for_integrity(text)
                ),
            }
        )
        return result, sentences
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if text_path is not None:
            text_path.unlink(missing_ok=True)
        if result_path is not None:
            result_path.unlink(missing_ok=True)
        if "temporary_directory_context" in locals():
            temporary_directory_context.cleanup()


def _new_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostics: {args.output}."
        )
    return {
        "schema_version": 1,
        "status": "in_progress",
        "generated_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "current_doc_id": None,
        "current_stage": None,
        "configuration": {
            "input_path": str(args.input.resolve()),
            "output_path": str(args.output.resolve()),
            "whole_document_timeout_seconds": args.whole_timeout_seconds,
            "excerpt_characters": args.excerpt_characters,
            "target_block_characters": DEFAULT_TARGET_BLOCK_CHARACTERS,
            "maximum_block_characters": DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
            "selected_doc_ids": list(TARGET_DOC_IDS),
        },
        "documents": [],
        "checkpoint_sequence": 0,
    }


def _checkpoint(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now(UTC).isoformat()
    payload["checkpoint_sequence"] += 1
    _atomic_write_json(args.output, payload)


def _evaluate_document(
    args: argparse.Namespace,
    payload: dict[str, Any],
    document: dict[str, Any],
) -> None:
    doc_id = cast(str, document["doc_id"])
    text = cast(str, document["texto"])
    language = "en"
    item: dict[str, Any] = {
        "doc_id": doc_id,
        "character_count": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "status": "in_progress",
        "classification": [],
        "fixed": False,
    }
    payload["documents"].append(item)
    payload["current_doc_id"] = doc_id
    payload["current_stage"] = "whole_document"
    _checkpoint(args, payload)

    whole_result, whole_sentences = _run_whole(
        text,
        language=language,
        timeout_seconds=args.whole_timeout_seconds,
        work_directory=args.output.parent,
    )
    item["whole_document"] = whole_result
    payload["current_stage"] = "bounded"
    _checkpoint(args, payload)

    started = perf_counter()
    bounded = _split_sentences_bounded_with_stats(
        text,
        language=language,
        trace=True,
    )
    bounded_result = {
        "status": "completed",
        "duration_seconds": perf_counter() - started,
        "sentence_count": len(bounded.sentences),
        "sentence_list_digest": _digest_sentences(bounded.sentences),
        "normalized_reconstruction_match": (
            _normalize_for_integrity(" ".join(bounded.sentences))
            == _normalize_for_integrity(text)
        ),
        "block_count": bounded.stats.block_count,
        "maximum_block_size": bounded.stats.largest_block_characters,
        "maximum_carry_size": bounded.stats.maximum_carry_characters,
        "oversized_carry_warning_count": (
            bounded.stats.oversized_carry_warning_count
        ),
    }
    item["bounded"] = bounded_result
    payload["current_stage"] = "diagnostics"
    _checkpoint(args, payload)

    reconstruction = diagnose_reconstruction(
        text,
        bounded.sentences,
        traces=bounded.traces,
        excerpt_characters=args.excerpt_characters,
    )
    item["bounded_reconstruction_diagnostic"] = reconstruction
    reconstruction_source_offset = cast(int | None, reconstruction["source_offset"])
    divergence_source_offset: int | None = None

    sentence_comparison: dict[str, Any] | None = None
    if whole_sentences is not None:
        sentence_comparison = compare_sentence_sequences(
            whole_sentences,
            bounded.sentences,
            source=text,
            excerpt_characters=args.excerpt_characters,
        )
        item["first_divergence"] = sentence_comparison
        if not sentence_comparison["exact_match"]:
            first_bounded = sentence_comparison["bounded_after"]
            first_baseline = sentence_comparison["baseline_after"]
            candidates = first_bounded + first_baseline
            sentence_offset = next(
                (
                    cast(int, candidate["source_offset"])
                    for candidate in candidates
                    if candidate["source_offset"] is not None
                ),
                None,
            )
            if sentence_offset is not None:
                divergence_source_offset = sentence_offset
    else:
        item["first_divergence"] = {
            "status": "unavailable",
            "reason": "Whole-document baseline did not complete.",
        }

    if divergence_source_offset is not None:
        trace_context = _trace_for_source_offset(
            bounded.traces, divergence_source_offset
        )
        item["divergence_block_and_carry_context"] = trace_context
        if trace_context is not None:
            item["divergence_affected_block_pysbd"] = _capture_affected_block(
                text,
                language=language,
                block_number=cast(int, trace_context["block_number"]),
                source_offset=divergence_source_offset,
                excerpt_characters=args.excerpt_characters,
            )
        divergence_windows = _local_window_comparisons(
            text,
            language=language,
            source_offset=divergence_source_offset,
        )
    else:
        divergence_windows = []
    item["divergence_local_windows"] = divergence_windows

    if reconstruction_source_offset is not None:
        reconstruction_trace = _trace_for_source_offset(
            bounded.traces,
            reconstruction_source_offset,
        )
        item["reconstruction_block_and_carry_context"] = reconstruction_trace
        if reconstruction_trace is not None:
            item["reconstruction_affected_block_pysbd"] = (
                _capture_affected_block(
                    text,
                    language=language,
                    block_number=cast(int, reconstruction_trace["block_number"]),
                    source_offset=reconstruction_source_offset,
                    excerpt_characters=args.excerpt_characters,
                )
            )
        reconstruction_windows = _local_window_comparisons(
            text,
            language=language,
            source_offset=reconstruction_source_offset,
        )
    else:
        reconstruction_windows = []
    item["reconstruction_local_windows"] = reconstruction_windows

    if sentence_comparison is not None and not sentence_comparison["exact_match"]:
        item["first_divergence_classification"] = _classify_sentence_difference(
            sentence_comparison,
            divergence_windows,
        )
        item["classification"].append(item["first_divergence_classification"])
    if not reconstruction["matches"]:
        item["reconstruction_classification"] = _classify_reconstruction(
            reconstruction_windows
        )
        item["classification"].append(item["reconstruction_classification"])
    item["classification"] = list(dict.fromkeys(item["classification"]))
    item["status"] = "completed_with_findings"
    item["remaining_limitation"] = (
        "No algorithm correction was applied because the diagnostics did not "
        "demonstrate block construction or carry corruption."
    )
    payload["current_stage"] = None
    _checkpoint(args, payload)


def _refine_existing(args: argparse.Namespace) -> int:
    if not args.output.exists():
        raise FileNotFoundError(f"Diagnostics do not exist: {args.output}.")
    payload = cast(
        dict[str, Any], json.loads(args.output.read_text(encoding="utf-8"))
    )
    if payload.get("status") not in _TERMINAL_STATUSES:
        raise ValueError("Only terminal diagnostics can be refined safely.")
    documents = _load_documents(args.input)
    for item in payload["documents"]:
        doc_id = cast(str, item["doc_id"])
        text = cast(str, documents[doc_id]["texto"])
        expected_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if item.get("text_sha256") != expected_digest:
            raise ValueError(f"Source text changed for {doc_id}; refusing refinement.")
        divergence = item.get("first_divergence", {})
        divergence_candidates = divergence.get("bounded_after", []) + divergence.get(
            "baseline_after", []
        )
        divergence_offset = next(
            (
                cast(int, candidate["source_offset"])
                for candidate in divergence_candidates
                if candidate.get("source_offset") is not None
            ),
            None,
        )
        reconstruction_offset = item["bounded_reconstruction_diagnostic"].get(
            "source_offset"
        )
        divergence_windows = (
            _local_window_comparisons(
                text,
                language="en",
                source_offset=divergence_offset,
            )
            if divergence_offset is not None
            else []
        )
        reconstruction_windows = (
            _local_window_comparisons(
                text,
                language="en",
                source_offset=reconstruction_offset,
            )
            if reconstruction_offset is not None
            else []
        )
        item["divergence_local_windows"] = divergence_windows
        item["reconstruction_local_windows"] = reconstruction_windows

        traced = _split_sentences_bounded_with_stats(
            text,
            language="en",
            trace=True,
        )
        if divergence_offset is not None:
            divergence_trace = _trace_for_source_offset(
                traced.traces,
                divergence_offset,
            )
            item["divergence_block_and_carry_context"] = divergence_trace
            if divergence_trace is not None:
                item["divergence_affected_block_pysbd"] = (
                    _capture_affected_block(
                        text,
                        language="en",
                        block_number=cast(
                            int, divergence_trace["block_number"]
                        ),
                        source_offset=divergence_offset,
                        excerpt_characters=args.excerpt_characters,
                    )
                )
        if reconstruction_offset is not None:
            reconstruction_trace = _trace_for_source_offset(
                traced.traces,
                reconstruction_offset,
            )
            item["reconstruction_block_and_carry_context"] = reconstruction_trace
            if reconstruction_trace is not None:
                item["reconstruction_affected_block_pysbd"] = (
                    _capture_affected_block(
                        text,
                        language="en",
                        block_number=cast(
                            int, reconstruction_trace["block_number"]
                        ),
                        source_offset=reconstruction_offset,
                        excerpt_characters=args.excerpt_characters,
                    )
                )
        if divergence.get("exact_match") is False:
            item["first_divergence_classification"] = (
                _classify_sentence_difference(divergence, divergence_windows)
            )
        if item["bounded_reconstruction_diagnostic"].get("matches") is False:
            item["reconstruction_classification"] = _classify_reconstruction(
                reconstruction_windows
            )
        item["classification"] = list(
            dict.fromkeys(
                value
                for value in (
                    item.get("first_divergence_classification"),
                    item.get("reconstruction_classification"),
                )
                if value is not None
            )
        )
        for obsolete_key in (
            "block_and_carry_context",
            "affected_block_pysbd",
            "controlled_local_windows",
        ):
            item.pop(obsolete_key, None)

    def sanitize_existing(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for child_key, child_value in value.items():
                if child_key in {
                    "text",
                    "normalized",
                    "original_context",
                    "reconstructed_context",
                    "excerpt",
                } and isinstance(child_value, str):
                    replacement_key = {
                        "text": "sanitized_text_shape",
                        "normalized": "sanitized_normalized_shape",
                        "original_context": "original_context_shape",
                        "reconstructed_context": "reconstructed_context_shape",
                        "excerpt": "sanitized_excerpt_shape",
                    }[child_key]
                    sanitized[replacement_key] = _sanitize_excerpt(
                        child_value,
                        args.excerpt_characters,
                    )
                else:
                    sanitized[child_key] = sanitize_existing(
                        child_value, child_key
                    )
            return sanitized
        if isinstance(value, list):
            return [sanitize_existing(item, key) for item in value]
        return value

    payload = cast(dict[str, Any], sanitize_existing(payload))
    payload["refined_at"] = datetime.now(UTC).isoformat()
    payload["checkpoint_sequence"] += 1
    payload["updated_at"] = datetime.now(UTC).isoformat()
    _atomic_write_json(args.output, payload)
    print(f"Refined diagnostics written to {args.output}")
    return 0


def _run(args: argparse.Namespace) -> int:
    documents = _load_documents(args.input)
    payload = _new_payload(args)
    _checkpoint(args, payload)
    try:
        for doc_id in TARGET_DOC_IDS:
            _evaluate_document(args, payload, documents[doc_id])
    except BaseException as error:
        payload["status"] = "error"
        payload["error"] = f"{type(error).__name__}: {error}"
        _checkpoint(args, payload)
        raise
    payload["status"] = "completed_with_findings"
    payload["current_doc_id"] = None
    payload["current_stage"] = None
    _checkpoint(args, payload)
    print(f"Diagnostics written to {args.output}")
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
    if args.refine_existing:
        return _refine_existing(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
