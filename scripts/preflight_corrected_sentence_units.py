"""Preflight accepted documents for oversized reconstructed sentence units.

This diagnostic stops after source-unit preparation, segmentation,
reconciliation, and exact token counting of plausible over-limit units. It
does not pack chunks or invoke an encoder, embedding index, or retrieval path.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from nerv.chunking.language_detector import detect_language
from nerv.chunking.pipeline import (
    _STRICT_PROSE_FORMATS,
    _TABULAR_FORMATS,
    _looks_structural_oversized_unit,
    read_documents,
    validate_document,
)
from nerv.chunking.sentence_splitter import _split_sentences_bounded_with_stats
from nerv.chunking.token_counter import TokenCounter

DEFAULT_DOCUMENTS = Path(
    "outputs/runs/20260816-142614-clean-upstream/artifacts/documentos.jsonl"
)
DEFAULT_OUTPUT = Path(
    "outputs/validation/corrected_sentence_unit_preflight.json"
)
DEFAULT_CONFIG = Path("config/encoder_config.json")
EXPECTED_DOCUMENTS = 1760
DIAGNOSTIC_EDGE_CHARACTERS = 120


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-documents", type=int, default=EXPECTED_DOCUMENTS)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        decoded = json.load(stream)
    if not isinstance(decoded, dict):
        raise ValueError(f"encoder config must be a JSON object: {path}")
    return decoded


def _source_units(document: dict[str, object]) -> list[str]:
    text = document["texto"]
    document_format = document["formato"]
    if not isinstance(text, str) or not isinstance(document_format, str):
        raise TypeError("validated document text and format must be strings")
    if document_format.casefold() in _TABULAR_FORMATS:
        return [line.strip() for line in text.splitlines() if line.strip()]
    language = detect_language(text).language
    return _split_sentences_bounded_with_stats(
        text,
        language=language,
        document_format=document_format,
    ).sentences


def _conservative_normalized_bytes(text: str, *, prefix: str) -> int:
    """Return a cheap, deliberately conservative E5 candidate measure.

    XLM-R normalization is NFKC-family normalization and its subword pieces
    consume non-empty normalized input. UTF-8 bytes therefore provide a safe
    upper envelope for content pieces; a four-byte margin covers the leading
    SentencePiece metaspace marker. Exact acceptance never uses this estimate.
    """
    normalized = unicodedata.normalize("NFKC", f"{prefix}{text}")
    return len(normalized.encode("utf-8")) + 4


def _plausibly_over_limit(text: str, *, prefix: str, limit: int) -> bool:
    return _conservative_normalized_bytes(text, prefix=prefix) > limit


def _classification(
    text: str,
    *,
    document_format: str,
) -> tuple[str, str]:
    normalized_format = document_format.casefold()
    if normalized_format not in _STRICT_PROSE_FORMATS:
        return "STRUCTURAL_MULTI_UNIT", f"{normalized_format}_structural_record"
    if _looks_structural_oversized_unit(text):
        return "STRUCTURAL_MULTI_UNIT", "recognized_structural_prose_span"
    return "AMBIGUOUS", "unclassified_oversized_reconstructed_prose"


def _edge(text: str, *, prefix: bool) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= DIAGNOSTIC_EDGE_CHARACTERS:
        return normalized
    if prefix:
        return normalized[:DIAGNOSTIC_EDGE_CHARACTERS]
    return normalized[-DIAGNOSTIC_EDGE_CHARACTERS:]


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def run_preflight(
    *,
    documents_path: Path,
    output_path: Path,
    config_path: Path,
    expected_documents: int,
    local_files_only: bool,
    progress_every: int,
) -> dict[str, object]:
    """Scan accepted source units and write compact oversized diagnostics."""
    if expected_documents <= 0:
        raise ValueError("expected_documents must be greater than zero")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    config = _load_config(config_path)
    model_name = str(config["encoder_model_name"])
    revision_value = config.get("tokenizer_revision")
    revision = str(revision_value) if revision_value is not None else None
    prefix = str(config["document_prefix"])
    encoder_limit = int(config["encoder_max_input_tokens"])
    counter = TokenCounter(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        add_special_tokens=bool(config["add_special_tokens"]),
        document_prefix=prefix,
    )
    special_overhead = counter.encoder_special_token_overhead
    safe_limit = encoder_limit - special_overhead
    if safe_limit != 510:
        raise ValueError(
            f"frozen effective content limit must be 510, got {safe_limit}"
        )

    documents_scanned = 0
    units_examined = 0
    exact_tokenizer_candidates = 0
    maximum_exact_tokens = 0
    classifications: Counter[str] = Counter()
    candidates: list[dict[str, object]] = []

    for line_number, document in read_documents(documents_path):
        validate_document(document, line_number=line_number)
        documents_scanned += 1
        document_format = str(document["formato"])
        units = _source_units(document)
        units_examined += len(units)
        plausible = [
            (unit_index, unit)
            for unit_index, unit in enumerate(units)
            if _plausibly_over_limit(unit, prefix=prefix, limit=safe_limit)
        ]
        exact_tokenizer_candidates += len(plausible)
        counts = counter.count_many(
            [unit for _, unit in plausible],
            include_document_prefix=True,
            batch_size=256,
        )
        for (unit_index, unit), token_count in zip(plausible, counts, strict=True):
            maximum_exact_tokens = max(maximum_exact_tokens, token_count)
            if token_count <= safe_limit:
                continue
            classification, boundary_pattern = _classification(
                unit,
                document_format=document_format,
            )
            classifications[classification] += 1
            candidates.append(
                {
                    "doc_id": document["doc_id"],
                    "source": document["fuente"],
                    "unit_index": unit_index,
                    "token_count": token_count,
                    "classification": classification,
                    "boundary_pattern": boundary_pattern,
                    "short_prefix": _edge(unit, prefix=True),
                    "short_suffix": _edge(unit, prefix=False),
                }
            )
        if progress_every and documents_scanned % progress_every == 0:
            print(
                f"preflight documents={documents_scanned} units={units_examined} "
                f"over_limit={len(candidates)}",
                file=sys.stderr,
                flush=True,
            )

    if documents_scanned != expected_documents:
        raise ValueError(
            f"expected {expected_documents} accepted documents, scanned "
            f"{documents_scanned}"
        )

    structural_count = classifications["STRUCTURAL_MULTI_UNIT"]
    ambiguous_count = classifications["AMBIGUOUS"]
    genuine_count = classifications["GENUINE_SINGLE_SENTENCE"]
    overmerge_count = classifications["RECONCILIATION_OVERMERGE"]
    artifact_count = classifications["EXTRACTION_ARTIFACT"]
    prose_count = genuine_count + ambiguous_count + overmerge_count + artifact_count
    payload: dict[str, object] = {
        "documents_scanned": documents_scanned,
        "units_examined": units_examined,
        "exact_tokenizer_candidates": exact_tokenizer_candidates,
        "units_over_510": len(candidates),
        "prose_candidates_over_510": prose_count,
        "structural_candidates_over_510": structural_count,
        "ambiguous_candidates": ambiguous_count,
        "maximum_exact_tokens": maximum_exact_tokens,
        "classification_counts": dict(sorted(classifications.items())),
        "candidates": candidates,
    }
    _write_json_atomic(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_preflight(
        documents_path=args.documents,
        output_path=args.output,
        config_path=args.config,
        expected_documents=args.expected_documents,
        local_files_only=args.local_files_only,
        progress_every=args.progress_every,
    )
    summary = {key: value for key, value in payload.items() if key != "candidates"}
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
