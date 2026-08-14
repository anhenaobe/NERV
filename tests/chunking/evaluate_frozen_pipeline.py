"""Generate reproducible evidence for the frozen tokenizer chunking pipeline."""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from nerv.chunking import TokenCounter, split_sentences
from nerv.chunking.language_detector import detect_language
from nerv.chunking.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/chunking/fixtures/codefest_fictional_documents.jsonl"
RESULTS = ROOT / "tests/chunking/results"
CHUNKS = RESULTS / "frozen_pipeline_chunks.jsonl"
CONFIG = RESULTS / "frozen_pipeline_config.json"
METRICS = RESULTS / "frozen_pipeline_metrics.json"
SAMPLES = RESULTS / "frozen_pipeline_sample.md"
MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_TOKENS = 256
OVERLAP_TOKENS = 32
REPETITIONS = 5


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _percentile(values: list[int], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _language_from_id(doc_id: str) -> str:
    return doc_id.split("-")[1].lower()


def _overlap_sentences(left: str, right: str) -> list[str]:
    left_sentences = split_sentences(left)
    right_sentences = split_sentences(right)
    maximum = min(len(left_sentences), len(right_sentences))
    for size in range(maximum, 0, -1):
        if left_sentences[-size:] == right_sentences[:size]:
            return left_sentences[-size:]
    return []


def _package_versions() -> dict[str, str | None]:
    packages = ("nerv", "transformers", "tokenizers", "sentencepiece", "pysbd")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _write_samples(
    documents: list[dict[str, object]],
    grouped: dict[str, list[dict[str, object]]],
) -> None:
    selected_ids = ("SYN-ES-001", "SYN-EN-010", "SYN-ES-010", "SYN-PT-010")
    by_id = {str(document["doc_id"]): document for document in documents}
    lines = [
        "# Frozen pipeline samples",
        "",
        "These examples are synthetic and do not represent the CODEFEST corpus.",
        "The `passage: ` prefix is counted by the tokenizer but is not stored "
        "in chunks.",
        "",
    ]
    for doc_id in selected_ids:
        document = by_id[doc_id]
        source_text = str(document["texto"])
        detection = detect_language(source_text)
        lines.extend(
            (
                f"## {doc_id}",
                "",
                f"Detected language: `{detection.language}` "
                f"(ambiguous: `{str(detection.is_ambiguous).lower()}`)",
                "",
                "Preserved metadata:",
                "",
                "```json",
                json.dumps(
                    {
                        key: document[key]
                        for key in ("doc_id", "fuente", "formato", "fenomeno")
                    },
                    ensure_ascii=False,
                ),
                "```",
                "",
                "Original document:",
                "",
                source_text,
                "",
                "Extracted sentences:",
                "",
            )
        )
        lines.extend(f"{index}. {sentence}" for index, sentence in enumerate(
            split_sentences(source_text), start=1
        ))
        lines.append("")
        for record in grouped[doc_id]:
            lines.extend(
                (
                    f"### {record['chunk_id']} ({record['num_tokens']} tokens)",
                    "",
                    str(record["texto"]),
                    "",
                )
            )
        chunks = grouped[doc_id]
        if len(chunks) > 1:
            overlap = _overlap_sentences(
                str(chunks[0]["texto"]), str(chunks[1]["texto"])
            )
            lines.extend(("Overlap between the first two chunks:", ""))
            lines.extend(f"- {sentence}" for sentence in overlap)
            lines.append("")
    SAMPLES.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    """Run one warm-up plus five measured repetitions and publish evidence."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    cold_started = time.perf_counter()
    counter = TokenCounter(
        MODEL_NAME,
        document_prefix="passage: ",
        add_special_tokens=False,
        local_files_only=True,
    )
    _ = counter.tokenizer_class
    cold_load_duration = time.perf_counter() - cold_started

    run_pipeline(
        FIXTURE,
        CHUNKS,
        token_counter=counter,
        max_tokens=MAX_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
        config_path=CONFIG,
    )
    durations: list[float] = []
    reference = CHUNKS.read_bytes()
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        run_pipeline(
            FIXTURE,
            CHUNKS,
            token_counter=counter,
            max_tokens=MAX_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
            config_path=CONFIG,
        )
        durations.append(time.perf_counter() - started)
        if CHUNKS.read_bytes() != reference:
            raise RuntimeError("Pipeline output changed between repetitions.")

    documents = _read_jsonl(FIXTURE)
    records = _read_jsonl(CHUNKS)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["doc_id"])].append(record)

    sentence_count = 0
    lost_sentences: list[dict[str, str]] = []
    cut_sentences: list[dict[str, str]] = []
    detection_rows: list[dict[str, object]] = []
    language_documents: Counter[str] = Counter()
    language_chunks: Counter[str] = Counter()
    for document in documents:
        doc_id = str(document["doc_id"])
        language = _language_from_id(doc_id)
        language_documents[language] += 1
        language_chunks[language] += len(grouped[doc_id])
        detection = detect_language(str(document["texto"]))
        detection_rows.append(
            {
                "doc_id": doc_id,
                "expected": language,
                "selected": detection.language,
                "ambiguous": detection.is_ambiguous,
                "confidence": detection.confidence,
                "margin": detection.margin,
            }
        )
        source_sentences = split_sentences(str(document["texto"]))
        sentence_count += len(source_sentences)
        chunk_texts = [str(record["texto"]) for record in grouped[doc_id]]
        for sentence in source_sentences:
            if not any(sentence in text for text in chunk_texts):
                lost_sentences.append({"doc_id": doc_id, "sentence": sentence})
            partial_hits = [sentence in text for text in chunk_texts]
            if not any(partial_hits):
                continue
            if not any(text == sentence or sentence in text for text in chunk_texts):
                cut_sentences.append({"doc_id": doc_id, "sentence": sentence})

    actual_overlaps: list[int] = []
    for chunks in grouped.values():
        for left, right in zip(chunks, chunks[1:], strict=False):
            overlap = " ".join(
                _overlap_sentences(str(left["texto"]), str(right["texto"]))
            )
            actual_overlaps.append(
                counter.count(overlap, include_document_prefix=True) if overlap else 0
            )

    token_counts = [int(record["num_tokens"]) for record in records]
    oversized = [count for count in token_counts if count > MAX_TOKENS]
    duplicated_tokens = sum(actual_overlaps)
    total_chunk_tokens = sum(token_counts)
    input_bytes = FIXTURE.stat().st_size
    output_bytes = CHUNKS.stat().st_size
    bytes_per_embedding = 384 * 4
    baseline_path = RESULTS / "chunking_metrics.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    detector_metrics = {
        key: value
        for key, value in baseline["detector"].items()
        if key != "confidence_note"
    }
    splitter_metrics = baseline["splitter"]
    metrics: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "evaluation_mode": "real_tokenizer_synthetic_pipeline",
        "corpus_type": "synthetic",
        "encoder_model_name": MODEL_NAME,
        "tokenizer_class": counter.tokenizer_class,
        "tokenizer_revision": counter.revision,
        "embedding_dimension": 384,
        "document_prefix": "passage: ",
        "add_special_tokens": False,
        "chunk_max_tokens": MAX_TOKENS,
        "overlap_target_tokens": OVERLAP_TOKENS,
        "document_count": len(documents),
        "document_count_by_language": dict(sorted(language_documents.items())),
        "sentence_count": sentence_count,
        "chunk_count": len(records),
        "chunks_per_document": len(records) / len(documents),
        "minimum_tokens_per_chunk": min(token_counts),
        "maximum_tokens_per_chunk": max(token_counts),
        "average_tokens_per_chunk": statistics.fmean(token_counts),
        "median_tokens_per_chunk": statistics.median(token_counts),
        "token_count_standard_deviation": statistics.pstdev(token_counts),
        "percentile_25": _percentile(token_counts, 0.25),
        "percentile_75": _percentile(token_counts, 0.75),
        "percentile_90": _percentile(token_counts, 0.90),
        "chunk_limit_utilization_rate": statistics.fmean(token_counts) / MAX_TOKENS,
        "chunks_exceeding_limit_count": len(oversized),
        "oversized_sentence_chunk_count": len(oversized),
        "cut_sentence_count": len(cut_sentences),
        "lost_sentence_count": len(lost_sentences),
        "empty_chunk_count": sum(not str(record["texto"]) for record in records),
        "configured_overlap_tokens": OVERLAP_TOKENS,
        "average_actual_overlap_tokens": (
            statistics.fmean(actual_overlaps) if actual_overlaps else 0.0
        ),
        "minimum_actual_overlap_tokens": min(actual_overlaps, default=0),
        "maximum_actual_overlap_tokens": max(actual_overlaps, default=0),
        "duplicated_token_estimate": duplicated_tokens,
        "duplicated_content_rate": (
            duplicated_tokens / total_chunk_tokens if total_chunk_tokens else 0.0
        ),
        "deterministic_output": True,
        "processing_duration_seconds": statistics.fmean(durations),
        "documents_per_second": len(documents) / statistics.fmean(durations),
        "sentences_per_second": sentence_count / statistics.fmean(durations),
        "chunks_per_second": len(records) / statistics.fmean(durations),
        "input_size_bytes": input_bytes,
        "output_size_bytes": output_bytes,
        "output_to_input_size_ratio": output_bytes / input_bytes,
        "dataset": {
            "kind": "synthetic",
            "document_count": len(documents),
            "languages": dict(sorted(language_documents.items())),
            "input_bytes": input_bytes,
        },
        "configuration": {
            "encoder_model_name": MODEL_NAME,
            "embedding_dimension": 384,
            "chunk_max_tokens": MAX_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "document_prefix": "passage: ",
            "query_prefix": "query: ",
            "add_special_tokens": False,
            "normalize_embeddings": True,
            "normalization_method": "l2",
            "future_similarity_metric": "inner_product",
            "future_faiss_index": "IndexFlatIP",
            "oversized_sentence_policy": "preserve_complete_sentence",
            "tokenizer_class": counter.tokenizer_class,
            "tokenizer_revision": counter.revision,
        },
        "pipeline": {
            "sentence_count": sentence_count,
            "chunk_count": len(records),
            "chunks_by_language": dict(sorted(language_chunks.items())),
            "empty_chunk_count": sum(not str(record["texto"]) for record in records),
            "cut_sentence_count": len(cut_sentences),
            "lost_sentence_count": len(lost_sentences),
            "oversized_sentence_chunk_count": len(oversized),
            "token_count": {
                "minimum": min(token_counts),
                "maximum": max(token_counts),
                "mean": statistics.fmean(token_counts),
                "median": statistics.median(token_counts),
                "p95": _percentile(token_counts, 0.95),
                "p99": _percentile(token_counts, 0.99),
            },
            "actual_overlap_tokens": {
                "pair_count": len(actual_overlaps),
                "minimum": min(actual_overlaps, default=0),
                "maximum": max(actual_overlaps, default=0),
                "mean": statistics.fmean(actual_overlaps) if actual_overlaps else 0.0,
                "values": actual_overlaps,
            },
            "duplicated_token_estimate": duplicated_tokens,
            "duplicated_content_rate": (
                duplicated_tokens / total_chunk_tokens if total_chunk_tokens else 0.0
            ),
            "deterministic_across_measured_repetitions": True,
            "input_output_size_ratio": output_bytes / input_bytes,
            "output_bytes": output_bytes,
        },
        "detector": {
            "ambiguous_count": sum(bool(row["ambiguous"]) for row in detection_rows),
            "selected_language_accuracy": sum(
                row["expected"] == row["selected"] for row in detection_rows
            )
            / len(detection_rows),
            "cases": detection_rows,
        },
        "performance": {
            "warm_up_runs": 1,
            "tokenizer_cold_load_seconds_without_download": cold_load_duration,
            "measured_repetitions": REPETITIONS,
            "durations_seconds": durations,
            "mean_seconds": statistics.fmean(durations),
            "median_seconds": statistics.median(durations),
            "min_seconds": min(durations),
            "max_seconds": max(durations),
            "standard_deviation_seconds": statistics.pstdev(durations),
            "mean_documents_per_second": len(documents) / statistics.fmean(durations),
            "mean_chunks_per_second": len(records) / statistics.fmean(durations),
        },
        "embedding_storage_theoretical": {
            "dtype": "float32",
            "bytes_per_embedding": bytes_per_embedding,
            "current_chunk_count_bytes": len(records) * bytes_per_embedding,
            "1k_chunks_bytes": 1_000 * bytes_per_embedding,
            "10k_chunks_bytes": 10_000 * bytes_per_embedding,
            "100k_chunks_bytes": 100_000 * bytes_per_embedding,
            "1m_chunks_bytes": 1_000_000 * bytes_per_embedding,
            "scope": "raw vectors only; excludes index and metadata overhead",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "available_ram_bytes": None,
            "available_ram_note": "Not queried without adding a platform dependency.",
            "packages": _package_versions(),
        },
        "baseline_detector_splitter": {
            "source": "chunking_metrics.json",
            "detector": detector_metrics,
            "splitter": splitter_metrics,
        },
        "language_detector_metrics": detector_metrics,
        "sentence_splitter_metrics": splitter_metrics,
        "failures": {
            "lost_sentences": lost_sentences,
            "cut_sentences": cut_sentences,
        },
        "pipeline_validation_failures": len(lost_sentences) + len(cut_sentences),
        "failed_cases": [*lost_sentences, *cut_sentences],
        "limitations": [
            "The 30 documents are synthetic and do not represent the CODEFEST corpus.",
            "No encoder weights, embeddings, FAISS index, retrieval, or agents "
            "were run.",
            "Theoretical storage excludes index structures and metadata.",
            "Portuguese currently uses the Spanish PySBD rules.",
            "Tokenizer cache revision is not pinned to a commit hash.",
        ],
    }
    METRICS.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_samples(documents, grouped)
    print(json.dumps(metrics["pipeline"], ensure_ascii=False, indent=2))
    print(json.dumps(metrics["performance"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
