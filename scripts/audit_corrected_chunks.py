"""Post-run Stage-1 audit for a human-generated corrected chunks artifact."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from nerv.evaluation.contamination import detectar_chunks_contaminados

PASS = "CORRECTED_CHUNKS_PASS"
FAIL = "CORRECTED_CHUNKS_FAIL"
NEEDS_REVIEW = "CORRECTED_CHUNKS_NEEDS_HUMAN_REVIEW"

_TERMINAL_END = re.compile(r"[.!?…][\"'”’»\)\]\}]*$")
_JSON_FIELD = re.compile(
    r"(?:^|\s)(?:title|date|authors?\[\d+\]|abstract|keywords?\[\d+\]|"
    r"doi|issue|body_paragraphs\[\d+\]|lists\[\d+\]|alerta_meta\.[^: ]+):",
    re.IGNORECASE,
)
_STRUCTURED_FORMATS = frozenset({"csv", "xlsx", "xls", "tsv", "pbf", "imagen"})
_CONNECTORS = frozenset(
    {
        "a", "al", "and", "con", "da", "das", "de", "del", "do", "dos",
        "e", "el", "en", "for", "in", "la", "las", "los", "o", "of",
        "on", "or", "para", "por", "que", "the", "to", "um", "uma", "y",
    }
)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="strict") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            yield value


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _first_alpha(text: str) -> str:
    return next((character for character in text if character.isalpha()), "")


def _word_overlap(left: str, right: str) -> int:
    left_words = left.split()
    right_words = right.split()
    for size in range(min(len(left_words), len(right_words), 80), 0, -1):
        if left_words[-size:] == right_words[:size]:
            return size
    return 0


def _looks_like_prose(text: str) -> bool:
    words = text.split()
    if len(words) < 8:
        return False
    alpha_words = sum(any(character.isalpha() for character in word) for word in words)
    return alpha_words / len(words) >= 0.60


def _looks_structural(record: dict[str, Any], text: str) -> bool:
    document_format = str(record.get("formato", "")).casefold()
    if document_format in _STRUCTURED_FORMATS:
        return True
    if record.get("hard_split") is True:
        return True
    if document_format == "json":
        return len(_JSON_FIELD.findall(text)) >= 2
    words = text.split()
    if len(words) <= 12:
        return True
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    if alpha_words and len(words) <= 40:
        upper_words = sum(word.strip(".,:;()[]").isupper() for word in alpha_words)
        return upper_words / len(alpha_words) >= 0.70
    return False


def _continuation_like(left: str, right_new_content: str) -> bool:
    if not _looks_like_prose(left) or not right_new_content:
        return False
    words = left.split()
    last_word = words[-1].casefold().strip(".,;:!?…()[]{}\"'”’»-")
    return (
        _first_alpha(right_new_content).islower()
        or last_word in _CONNECTORS
        or left.rstrip().endswith((",", "-", "–"))
    )


def _load_validation_metrics(path: Path) -> tuple[dict[str, Any], list[str]]:
    metrics: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("chunk validation metrics must be one JSON object.")
    required_zero = (
        "duplicate_chunk_id_count",
        "position_error_count",
        "document_order_error_count",
        "metadata_error_count",
        "token_count_mismatch_count",
        "encoder_hard_limit_exceeded_count",
        "hard_split_metadata_error_count",
        "empty_chunk_count",
        "unknown_document_count",
        "missing_output_document_count",
    )
    failures = [name for name in required_zero if metrics.get(name) != 0]
    if metrics.get("completion_status") != "completed":
        failures.append("validation_completion_status")
    if metrics.get("checked_document_count") != 1760:
        failures.append("validation_document_count")
    maximum = metrics.get("maximum_num_tokens")
    if not isinstance(maximum, int) or maximum > 510:
        failures.append("maximum_num_tokens")
    return metrics, failures


def _audit_chunks(
    chunks_path: Path,
    queries: list[dict[str, Any]],
) -> dict[str, Any]:
    confirmed_boundaries: list[str] = []
    confirmed_chunks: set[str] = set()
    ambiguous: list[str] = []
    contamination_count = 0
    formats: Counter[str] = Counter()
    batch: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    chunk_count = 0

    def flush_contamination() -> None:
        nonlocal contamination_count
        if not batch:
            return
        contamination_count += len(detectar_chunks_contaminados(batch, queries))
        batch.clear()

    for record in _iter_jsonl(chunks_path):
        chunk_count += 1
        formats[str(record.get("formato", "")).casefold()] += 1
        batch.append(record)
        if len(batch) >= 1000:
            flush_contamination()

        if previous is not None and previous.get("doc_id") == record.get("doc_id"):
            left = _normalize(str(previous.get("texto", "")))
            right = _normalize(str(record.get("texto", "")))
            overlap = _word_overlap(left, right)
            right_new = " ".join(right.split()[overlap:])
            previous_id = str(previous.get("chunk_id", ""))
            if (
                not _TERMINAL_END.search(left)
                and _continuation_like(left, right_new)
                and not _looks_structural(previous, left)
            ):
                confirmed_boundaries.append(previous_id)
                confirmed_chunks.update((previous_id, str(record.get("chunk_id", ""))))
            elif (
                not _TERMINAL_END.search(left)
                and _looks_like_prose(left)
                and not _looks_structural(previous, left)
            ):
                ambiguous.append(previous_id)
        previous = record
    flush_contamination()
    return {
        "chunk_count": chunk_count,
        "formats": dict(sorted(formats.items())),
        "confirmed_prose_continuation_boundaries": len(confirmed_boundaries),
        "confirmed_incomplete_prose_chunks": len(confirmed_chunks),
        "ambiguous_prose_cases": len(set(ambiguous)),
        "contamination_count": contamination_count,
        "failure_examples": confirmed_boundaries[:100],
        "ambiguous_examples": sorted(set(ambiguous))[:100],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--chunk-validation", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    document_count = sum(1 for _ in _iter_jsonl(args.documents))
    queries = list(_iter_jsonl(args.queries))
    validation, failures = _load_validation_metrics(args.chunk_validation)
    if document_count != 1760:
        failures.append("documents_source_set")
    linguistic = _audit_chunks(args.chunks, queries)
    if linguistic["confirmed_prose_continuation_boundaries"] != 0:
        failures.append("confirmed_prose_continuation_boundaries")
    if linguistic["confirmed_incomplete_prose_chunks"] != 0:
        failures.append("confirmed_incomplete_prose_chunks")
    if linguistic["contamination_count"] != 0:
        failures.append("contamination")

    if failures:
        status = FAIL
    elif linguistic["ambiguous_prose_cases"]:
        status = NEEDS_REVIEW
    else:
        status = PASS
    report = {
        "status": status,
        "documents_source_set": document_count,
        "validation": validation,
        "linguistic": linguistic,
        "failures": sorted(set(failures)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(status)
    return 0 if status == PASS else 2 if status == NEEDS_REVIEW else 1


if __name__ == "__main__":
    raise SystemExit(main())
