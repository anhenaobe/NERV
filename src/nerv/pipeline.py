"""Minimal identity-safe end-to-end orchestration for NERV.

This module owns sequencing, artifact reuse, timing, atomic run metadata, and
failure reporting.  Domain algorithms remain in their existing subsystem
modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from nerv.chunking.atomic_io import _atomic_replace_with_retry
from nerv.chunking.configuration import (
    DEFAULT_ENCODER_CONFIG_PATH,
    EncoderConfig,
    load_encoder_config,
    resolve_encoder_capacity,
)
from nerv.chunking.pipeline import (
    PipelineSummary,
    read_documents,
    validate_document,
)
from nerv.chunking.pipeline import run_pipeline as run_chunking_pipeline
from nerv.chunking.real_corpus import validate_command as validate_chunk_artifact
from nerv.chunking.token_counter import TokenCounter
from nerv.embeddings.artifacts import (
    EmbeddingManifest,
    _release_memmap,
    generate_embedding_artifact,
    load_embedding_manifest,
    ordered_chunk_identity,
    sha256_file,
    validate_embedding_artifact,
)
from nerv.embeddings.encoder import load_encoder
from nerv.evaluation.validator import validate_results
from nerv.ingestion.lector_corpus import (  # type: ignore[attr-defined]
    BASURA_SILENCIOSA,
    DIRECTORIOS_IGNORADOS,
    EXTENSIONES_SOPORTADAS,
    MODO_DOC_ID_RAPIDO,
    OCR_IDIOMAS,
    guardar_jsonl,
    procesar_corpus,
    pytesseract,
    resolver_pendientes_ocr,
)
from nerv.retrieval.queries import QueryRecord, load_queries
from nerv.retrieval.query_encoder import encode_query
from nerv.retrieval.retriever import retrieve
from nerv.vector_database.build_index import build_index_from_artifact
from nerv.vector_database.load_index import load_index
from nerv.vector_database.save_index import index_manifest_path, save_index

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
STAGES: Final = (
    "ingestion",
    "chunking",
    "embeddings",
    "faiss",
    "retrieval",
    "validation",
)
CANDIDATE_TOP_K: Final = 50
FRAGMENT_TOP_K: Final = 10
DOCUMENT_TOP_K: Final = 3
MAX_WORDS_PER_FRAGMENT: Final = 250
EXPECTED_FULL_DOCUMENT_COUNT: Final = 1761
EXPECTED_FULL_DOCUMENT_SHA256: Final = (
    "FE4EB08FC22781B228EAB70E8FA5AF663EF34DF6D6FDAB32D2A5A2457472A982"
)
EXPECTED_FULL_CHUNK_COUNT: Final = 336245
EXPECTED_FULL_CHUNK_SHA256: Final = (
    "D69A943E8069092D7F68AFB31EC2F5872EACE2743B20ABFFCF6BC4A0BDD835F3"
)
REPRODUCIBILITY_CLASSIFICATIONS: Final = (
    "BYTE_IDENTICAL",
    "SEMANTICALLY_EQUIVALENT",
    "STRUCTURALLY_EQUIVALENT",
    "DIFFERENT",
)
LOGGER = logging.getLogger("nerv.pipeline")

STAGE_CALLABLES: Final = {
    "ingestion": (
        "nerv.ingestion.lector_corpus",
        "procesar_corpus + guardar_jsonl",
        "corpus directory",
        "documentos.jsonl + ingestion statistics",
    ),
    "chunking": (
        "nerv.chunking.pipeline",
        "run_pipeline",
        "documentos.jsonl + TokenCounter + semantic config",
        "chunks.jsonl + chunking_config.json + PipelineSummary",
    ),
    "embeddings": (
        "nerv.embeddings.artifacts",
        "generate_embedding_artifact / validate_embedding_artifact",
        "chunks.jsonl + real encoder + TokenCounter",
        "embeddings.npy + bound embedding manifest",
    ),
    "faiss": (
        "nerv.vector_database",
        "build_index_from_artifact + save_index / load_index",
        "validated embedding artifact + ordered chunk metadata",
        "IndexFlatIP + identity sidecar",
    ),
    "retrieval": (
        "nerv.retrieval",
        "load_queries + encode_query + retrieve + ranking callables",
        "queries.jsonl + verified IndexFlatIP + ordered metadata",
        "resultados.jsonl",
    ),
    "validation": (
        "nerv.evaluation.validator",
        "validate_results",
        "result records q001-q050",
        "valid result contract or fail-closed exception",
    ),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    """Atomically publish one small JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _atomic_replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    """Atomically publish a UTF-8 plain-text report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        _atomic_replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _configure_file_logging(path: Path) -> None:
    """Write detailed orchestration evidence to one run-local log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in list(LOGGER.handlers):
        handler.close()
        LOGGER.removeHandler(handler)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    LOGGER.addHandler(handler)
    root = logging.getLogger()
    for root_handler in list(root.handlers):
        root_handler.close()
        root.removeHandler(root_handler)
    root_handler = logging.FileHandler(path, encoding="utf-8")
    root_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root.addHandler(root_handler)
    root.setLevel(logging.INFO)


def _runtime_device_info(device: str) -> dict[str, object]:
    """Capture Torch/CUDA facts without changing device selection."""
    try:
        import torch
    except ImportError:
        return {
            "torch_version": None,
            "torch_cuda": None,
            "cuda_available": False,
            "gpu": None,
            "gpu_total_memory": None,
        }
    cuda_available = bool(torch.cuda.is_available())
    gpu = torch.cuda.get_device_name(0) if cuda_available else None
    total_memory = (
        int(torch.cuda.get_device_properties(0).total_memory)
        if cuda_available
        else None
    )
    return {
        "requested_device": device,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": gpu,
        "gpu_total_memory": total_memory,
    }


def _stage_seconds(stage_metrics: dict[str, Any]) -> float:
    value = stage_metrics.get("seconds", stage_metrics.get("total_seconds", 0.0))
    return float(value) if isinstance(value, int | float) else 0.0


def _report_value(value: object) -> str:
    if value is None or value == "":
        return "NOT AVAILABLE"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    return str(value)


def _write_execution_report(
    path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Generate the required self-contained report on success or failure."""
    stage_metrics = cast(dict[str, dict[str, Any]], metrics.get("stages", {}))
    durations = {name: _stage_seconds(value) for name, value in stage_metrics.items()}
    measured = {name: value for name, value in durations.items() if value > 0.0}
    ranked_stages = sorted(measured, key=lambda name: measured[name], reverse=True)
    largest = ranked_stages[0] if ranked_stages else None
    second_largest = ranked_stages[1] if len(ranked_stages) > 1 else None
    total = float(metrics.get("T_total", 0.0) or 0.0)
    artifacts = cast(dict[str, dict[str, Any]], manifest.get("artifacts", {}))
    embedding = stage_metrics.get("embeddings", {})
    ingestion_metrics = stage_metrics.get("ingestion", {})
    chunking_metrics = stage_metrics.get("chunking", {})
    faiss_metrics = stage_metrics.get("faiss", {})
    retrieval_metrics = stage_metrics.get("retrieval", {})
    validation_metrics = stage_metrics.get("validation", {})
    runtime = cast(dict[str, Any], manifest.get("runtime_environment", {}))
    runtime_config = cast(dict[str, Any], manifest.get("runtime_config", {}))
    paths = cast(dict[str, Any], runtime_config.get("paths", {}))
    semantic = cast(dict[str, Any], manifest.get("semantic_config", {}))
    status = manifest.get("status")
    completed = status == "completed"
    failed = status == "failed"
    fresh = manifest.get("mode") == "fresh"
    resumed_fresh = manifest.get("mode") == "resume-fresh"
    verdict = "PASS" if completed else "FAIL" if failed else "INCOMPLETE"
    executed = cast(list[str], manifest.get("stages_executed", []))
    reused = cast(list[str], manifest.get("stages_reused", []))
    records = cast(dict[str, dict[str, Any]], manifest.get("stage_records", {}))
    last_successful = next(
        (
            name
            for name in reversed(STAGES)
            if records.get(name, {}).get("status") == "completed"
        ),
        None,
    )
    lines = [
        "NERV E2E EXECUTION REPORT",
        "=========================",
        "",
        "1. EXECUTIVE SUMMARY",
        (
            "The requested pipeline stages completed with identity validation."
            if completed
            else (
                "The run failed closed; completed-stage evidence is preserved below."
                if failed
                else "The run is in progress; incremental evidence is preserved below."
            )
        ),
        "",
        "2. FINAL VERDICT",
        verdict,
        "",
        "3. RUN ID",
        _report_value(manifest.get("run_id")),
        "",
        "4. GIT COMMIT",
        _report_value(manifest.get("git_commit")),
        "",
        "5. WORKTREE STATE",
        "DIRTY" if manifest.get("git_dirty") else "CLEAN",
        "",
        "6. HOST/PLATFORM",
        _report_value(manifest.get("platform")),
        "",
        "7. GPU",
        _report_value(runtime.get("gpu")),
        "",
        "8. TORCH/CUDA",
        (
            f"Torch={_report_value(runtime.get('torch_version'))}; "
            f"Torch CUDA={_report_value(runtime.get('torch_cuda'))}; "
            f"CUDA available={_report_value(runtime.get('cuda_available'))}"
        ),
        "",
        "9. SEMANTIC CONFIG",
        json.dumps(semantic, ensure_ascii=False, sort_keys=True),
        "",
        "10. STAGES REQUESTED",
        ", ".join(cast(list[str], manifest.get("stages_requested", []))) or "NONE",
        "",
        "11. STAGES REUSED",
        ", ".join(reused) or "NONE",
        "",
        "12. STAGES EXECUTED",
        ", ".join(executed) or "NONE",
        "",
        "13. DOCUMENT ARTIFACT",
        _report_value(
            artifacts.get("documents", {}).get("path", paths.get("documents"))
        ),
        "",
        "14. DOCUMENT COUNT/HASH",
        (
            f"count={_report_value(artifacts.get('documents', {}).get('count'))}; "
            f"sha256={_report_value(artifacts.get('documents', {}).get('sha256'))}"
        ),
        "",
        "15. CHUNK ARTIFACT",
        _report_value(artifacts.get("chunks", {}).get("path", paths.get("chunks"))),
        "",
        "16. CHUNK COUNT/HASH",
        (
            f"count={_report_value(artifacts.get('chunks', {}).get('count'))}; "
            f"sha256={_report_value(artifacts.get('chunks', {}).get('sha256'))}"
        ),
        "",
        "17. MAX STORED TOKENS",
        _report_value(stage_metrics.get("chunking", {}).get("max_stored_tokens")),
        "",
        "18. EMBEDDING MODEL",
        _report_value(semantic.get("encoder_model_name")),
        "",
        "19. EMBEDDING DEVICE",
        _report_value(embedding.get("device", manifest.get("device"))),
        "",
        "20. EMBEDDING BATCH",
        _report_value(
            embedding.get(
                "batch_size", runtime_config.get("embedding_batch_size")
            )
        ),
        "",
        "21. EMBEDDING ROWS",
        _report_value(embedding.get("rows")),
        "",
        "22. EMBEDDING DIMENSION",
        _report_value(embedding.get("dimension", semantic.get("embedding_dimension"))),
        "",
        "23. EMBEDDING RUNTIME",
        _report_value(embedding.get("seconds")),
        "",
        "24. EMBEDDING THROUGHPUT",
        _report_value(embedding.get("vectors_per_second")),
        "",
        "25. EMBEDDING ARTIFACT HASH",
        _report_value(artifacts.get("embeddings", {}).get("sha256")),
        "",
        "26. FAISS TYPE",
        _report_value(faiss_metrics.get("index_type")),
        "",
        "27. FAISS NTOTAL",
        _report_value(faiss_metrics.get("ntotal")),
        "",
        "28. FAISS BUILD RUNTIME",
        _report_value(faiss_metrics.get("seconds")),
        "",
        "29. QUERY SOURCE",
        _report_value(artifacts.get("queries", {}).get("path", paths.get("queries"))),
        "",
        "30. QUERY COUNT",
        _report_value(retrieval_metrics.get("query_count")),
        "",
        "31. RETRIEVAL RUNTIME",
        _report_value(retrieval_metrics.get("total_seconds")),
        "",
        "32. MEAN QUERY LATENCY",
        _report_value(retrieval_metrics.get("mean_ms_per_query")),
        "",
        "33. MEDIAN QUERY LATENCY",
        _report_value(retrieval_metrics.get("median_ms_per_query")),
        "",
        "34. P95 QUERY LATENCY",
        _report_value(retrieval_metrics.get("p95_ms_per_query")),
        "",
        "35. RESULT VALIDATION",
        _report_value(validation_metrics.get("valid")),
        "",
        "36. TOTAL MEASURED RUNTIME",
        f"{total:.6f} seconds",
        "",
        "37. STAGE RUNTIME TABLE",
    ]
    lines.extend(
        f"{name}: {durations.get(name, 0.0):.6f} seconds" for name in STAGES
    )
    lines.extend(["", "38. STAGE PERCENTAGE OF TOTAL"])
    lines.extend(
        f"{name}: {(durations.get(name, 0.0) / total * 100.0 if total else 0.0):.3f}%"
        for name in STAGES
    )
    lines.extend(
        [
            "",
            "39. LARGEST BOTTLENECK",
            _report_value(largest),
            "",
            "40. WARNINGS",
            "\n".join(cast(list[str], manifest.get("warnings", []))) or "NONE",
            "",
            "41. ERRORS",
            "\n".join(cast(list[str], manifest.get("errors", []))) or "NONE",
            "",
            "42. ARTIFACT PATHS",
            "\n".join(
                f"{name}: {value.get('path')}" for name, value in artifacts.items()
            )
            or "NONE",
            "",
            "43. RECOMMENDED NEXT OPTIMIZATION",
            (
                f"Fresh-run measured priority: {largest}."
                if fresh and completed and largest is not None
                else (
                    "Measure and review the production "
                    f"{largest} stage before optimizing."
                    if largest is not None and not fresh
                    else (
                        "No optimization recommendation without completed timing "
                        "evidence."
                    )
                )
            ),
            "",
            "FAILURE DETAILS",
            f"LAST SUCCESSFUL STAGE: {_report_value(last_successful)}",
            f"FAILED STAGE: {_report_value(manifest.get('failed_stage'))}",
            f"ERROR SUMMARY: {_report_value(manifest.get('error'))}",
            f"DETAILED LOG PATH: {_report_value(paths.get('pipeline_log'))}",
            "",
        ]
    )
    if fresh:
        fresh_evidence = cast(dict[str, Any], manifest.get("fresh_execution", {}))
        document_repro = cast(
            dict[str, Any], fresh_evidence.get("document_reproducibility", {})
        )
        chunk_repro = cast(
            dict[str, Any], fresh_evidence.get("chunk_reproducibility", {})
        )
        ocr_runtime = cast(dict[str, Any], fresh_evidence.get("ocr_runtime", {}))
        lines.extend(
            [
                "FRESH EXECUTION",
                (
                    "Raw corpus path: "
                    f"{_report_value(fresh_evidence.get('raw_corpus_path'))}"
                ),
                (
                    "Raw corpus file count: "
                    f"{_report_value(fresh_evidence.get('raw_corpus_file_count'))}"
                ),
                f"OCR runtime: {_report_value(ocr_runtime.get('status'))}",
                f"All stages executed: {', '.join(executed) or 'NONE'}",
                f"Any reused stages: {', '.join(reused) or 'NONE'}",
                (
                    "Document count: "
                    f"{_report_value(artifacts.get('documents', {}).get('count'))}"
                ),
                (
                    "Document SHA: "
                    f"{_report_value(artifacts.get('documents', {}).get('sha256'))}"
                ),
                (
                    "Historical document SHA: "
                    f"{_report_value(document_repro.get('historical_sha256'))}"
                ),
                (
                    "Document reproducibility classification: "
                    f"{_report_value(document_repro.get('classification'))}"
                ),
                (
                    "Document reproducibility explanation: "
                    f"{_report_value(document_repro.get('explanation'))}"
                ),
                (
                    "Chunk count: "
                    f"{_report_value(artifacts.get('chunks', {}).get('count'))}"
                ),
                (
                    "Chunk SHA: "
                    f"{_report_value(artifacts.get('chunks', {}).get('sha256'))}"
                ),
                (
                    "Historical chunk SHA: "
                    f"{_report_value(chunk_repro.get('historical_sha256'))}"
                ),
                (
                    "Chunk reproducibility classification: "
                    f"{_report_value(chunk_repro.get('classification'))}"
                ),
                (
                    "Chunk reproducibility explanation: "
                    f"{_report_value(chunk_repro.get('explanation'))}"
                ),
                f"Embedding rows: {_report_value(embedding.get('rows'))}",
                f"FAISS ntotal: {_report_value(faiss_metrics.get('ntotal'))}",
                "OFFICIAL QUERY RESULTS: NOT AVAILABLE",
                f"Largest fresh stage: {_report_value(largest if completed else None)}",
                (
                    "Second-largest fresh stage: "
                    f"{_report_value(second_largest if completed else None)}"
                ),
                (
                    "Optimization priority: "
                    f"{_report_value(largest if completed else None)}"
                ),
                "Fresh stage throughput:",
                (
                    "ingestion documents/second: "
                    f"{_report_value(ingestion_metrics.get('documents_per_second'))}"
                ),
                (
                    "chunking chunks/second: "
                    f"{_report_value(chunking_metrics.get('chunks_per_second'))}"
                ),
                (
                    "embeddings vectors/second: "
                    f"{_report_value(embedding.get('vectors_per_second'))}"
                ),
                (
                    "faiss vectors/second: "
                    f"{_report_value(faiss_metrics.get('vectors_indexed_per_second'))}"
                ),
                (
                    "retrieval queries/second: "
                    f"{_report_value(retrieval_metrics.get('queries_per_second'))}"
                ),
                (
                    "validation results/second: "
                    f"{_report_value(validation_metrics.get('results_per_second'))}"
                ),
                "",
            ]
        )
    if resumed_fresh:
        resume = cast(dict[str, Any], manifest.get("resume_fresh", {}))
        inherited = cast(
            dict[str, Any], resume.get("inherited_stage_seconds", {})
        )
        preflight = cast(dict[str, Any], resume.get("preflight", {}))
        inherited_stages = cast(list[str], manifest.get("stages_inherited", []))
        continuation = float(
            metrics.get("continuation_wall_clock_seconds", total) or 0.0
        )
        compute_equivalent = float(
            metrics.get("resumed_fresh_compute_equivalent_e2e_seconds", 0.0) or 0.0
        )
        lines.extend(
            [
                "RESUME-FRESH CONTINUATION",
                f"Parent run: {_report_value(resume.get('parent_run_id'))}",
                f"Parent preflight: {_report_value(preflight.get('status'))}",
                "Inherited stages: " + (", ".join(inherited_stages) or "NONE"),
                f"Executed in continuation: {', '.join(executed) or 'NONE'}",
                f"Reused production baseline stages: {', '.join(reused) or 'NONE'}",
                "ingestion: status=INHERITED_VALIDATED; "
                f"source_run={_report_value(resume.get('parent_run_id'))}; "
                f"measured_runtime={_report_value(inherited.get('ingestion'))}",
                "chunking: status=INHERITED_VALIDATED; "
                f"source_run={_report_value(resume.get('parent_run_id'))}; "
                f"measured_runtime={_report_value(inherited.get('chunking'))}",
                "embeddings: status="
                f"{_report_value(records.get('embeddings', {}).get('action'))}",
                "faiss: status="
                f"{_report_value(records.get('faiss', {}).get('action'))}",
                "retrieval: status="
                f"{_report_value(records.get('retrieval', {}).get('action'))}",
                "validation: status="
                f"{_report_value(records.get('validation', {}).get('action'))}",
                f"CONTINUATION WALL-CLOCK TIME: {continuation:.6f} seconds",
                "RESUMED FRESH COMPUTE-EQUIVALENT E2E TIME: "
                f"{compute_equivalent:.6f} seconds",
                "The compute-equivalent value combines inherited measured stages with "
                "this continuation; it is not one uninterrupted wall-clock execution.",
                f"Embedding rows: {_report_value(embedding.get('rows'))}",
                f"FAISS ntotal: {_report_value(faiss_metrics.get('ntotal'))}",
                "OFFICIAL QUERY RESULTS: NOT AVAILABLE",
                "",
            ]
        )
    _atomic_text(path, "\n".join(lines))


def _checkpoint(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Atomically persist incremental evidence and refresh the text report."""
    metrics["updated_at"] = _utc_now()
    manifest["updated_at"] = metrics["updated_at"]
    _atomic_json(args.metrics, metrics)
    _atomic_json(args.run_manifest, manifest)
    _write_execution_report(args.report, manifest, metrics)


def _stage_paths(args: argparse.Namespace, stage: str) -> tuple[list[str], list[str]]:
    mapping = {
        "ingestion": ([str(args.corpus)], [str(args.documents)]),
        "chunking": (
            [str(args.documents)],
            [str(args.chunks), str(args.chunk_config), str(args.chunk_metrics)],
        ),
        "embeddings": (
            [str(args.chunks)],
            [str(args.embeddings), str(args.embedding_manifest)],
        ),
        "faiss": (
            [str(args.embeddings), str(args.embedding_manifest)],
            [str(args.faiss_index), str(args.faiss_metadata)],
        ),
        "retrieval": (
            [str(args.faiss_index), str(args.queries)],
            [str(args.results)],
        ),
        "validation": ([str(args.results)], [str(args.report)]),
    }
    return mapping[stage]


def _begin_stage(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    stage: str,
) -> str:
    started_at = _utc_now()
    inputs, outputs = _stage_paths(args, stage)
    manifest["stage_records"][stage] = {
        "status": "running",
        "start_timestamp": started_at,
        "end_timestamp": None,
        "duration_seconds": None,
        "action": None,
        "input_artifacts": inputs,
        "output_artifacts": outputs,
        "counts": {},
        "hashes": {},
        "warnings": [],
        "errors": [],
    }
    _checkpoint(args, manifest, metrics)
    return started_at


def _complete_stage(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    stage: str,
    *,
    action: str,
    duration: float,
) -> None:
    record = manifest["stage_records"][stage]
    stage_data = cast(dict[str, Any], metrics["stages"].get(stage, {}))
    artifacts = cast(dict[str, dict[str, Any]], manifest.get("artifacts", {}))
    record.update(
        {
            "status": "completed",
            "end_timestamp": _utc_now(),
            "duration_seconds": duration,
            "action": (
                action.upper()
                if manifest.get("mode") in {"fresh", "resume-fresh"}
                else action
            ),
            "counts": {
                key: value
                for key, value in stage_data.items()
                if key
                in {
                    "documents_written",
                    "documents",
                    "chunks",
                    "rows",
                    "ntotal",
                    "query_count",
                    "error_count",
                    "warning_count",
                    "input_count",
                    "output_count",
                }
            },
            "hashes": {
                name: value["sha256"]
                for name, value in artifacts.items()
                if isinstance(value, dict) and "sha256" in value
            },
        }
    )
    _checkpoint(args, manifest, metrics)


def _atomic_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Atomically publish result JSONL without changing its record schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")
        _atomic_replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_ingestion_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Publish ingestion output through its existing serializer and atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        guardar_jsonl(records, temporary)
        _atomic_replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _git_state() -> tuple[str, bool]:
    """Return current commit and whether tracked/untracked state is dirty."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _artifact(path: Path, *, count: int | None = None) -> dict[str, object]:
    """Describe one file without embedding its contents."""
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if count is not None:
        value["count"] = count
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    """Return whether a resolved path is inside a resolved root."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _inspect_raw_corpus(path: Path) -> dict[str, object]:
    """Validate and count files accepted by the ingestion contract."""
    if not path.is_dir():
        raise FileNotFoundError(f"missing raw corpus directory: {path}")
    discovered = 0
    supported = 0
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        relative = item.relative_to(path)
        if any(part in DIRECTORIOS_IGNORADOS for part in relative.parts):
            continue
        if (
            item.name in BASURA_SILENCIOSA
            or item.suffix.lower() == ".tmp"
            or item.name.startswith(".~")
        ):
            continue
        discovered += 1
        if item.suffix.lower() in EXTENSIONES_SOPORTADAS:
            supported += 1
    if supported == 0:
        raise ValueError(
            "raw corpus contains no supported input files; supported suffixes: "
            f"{sorted(EXTENSIONES_SOPORTADAS)}"
        )
    return {
        "path": str(path.resolve()),
        "discovered_file_count": discovered,
        "supported_file_count": supported,
    }


def _validate_ocr_runtime() -> dict[str, object]:
    """Fail fast unless the canonical OCR executable and languages are usable."""
    if pytesseract is None:
        raise RuntimeError("fresh production OCR requires the pytesseract package.")
    try:
        available = set(pytesseract.get_languages(config=""))
    except Exception as error:
        raise RuntimeError(
            "fresh production OCR requires a working Tesseract executable."
        ) from error
    required = set(OCR_IDIOMAS.split("+"))
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(
            "fresh production OCR is missing required Tesseract languages: "
            f"{missing}."
        )
    return {
        "executable": str(pytesseract.pytesseract.tesseract_cmd),
        "required_languages": sorted(required),
        "available_languages": sorted(available),
    }


def _semantic_projection_sha256(path: Path, *, artifact_kind: str) -> str:
    """Hash ordered semantic content while excluding metadata-derived IDs."""
    if artifact_kind == "documents":
        excluded = {"doc_id"}
    elif artifact_kind == "chunks":
        excluded = {"doc_id", "chunk_id", "parent_chunk_id"}
    else:
        raise ValueError(f"unknown reproducibility artifact kind: {artifact_kind}")
    digest = hashlib.sha256()
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid {artifact_kind} JSON at line {line_number}."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{artifact_kind} line {line_number} must be an object."
                )
            projection = {
                key: value for key, value in record.items() if key not in excluded
            }
            digest.update(
                json.dumps(
                    projection,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
    return digest.hexdigest().upper()


def _classify_reproducibility(
    *,
    actual_path: Path,
    actual_count: int,
    actual_sha256: str,
    historical_path: Path | None,
    historical_count: int | None,
    historical_sha256: str | None,
    artifact_kind: str,
) -> dict[str, object]:
    """Classify a validated fresh artifact against accepted historical evidence."""
    expected_sha = historical_sha256.upper() if historical_sha256 else None
    actual_sha = actual_sha256.upper()
    if historical_count is not None and actual_count != historical_count:
        return {
            "classification": "DIFFERENT",
            "explanation": (
                f"count differs: fresh={actual_count}, historical={historical_count}."
            ),
            "historical_count": historical_count,
            "historical_sha256": expected_sha,
        }
    if expected_sha is not None and actual_sha == expected_sha:
        return {
            "classification": "BYTE_IDENTICAL",
            "explanation": "fresh SHA-256 matches the accepted historical SHA-256.",
            "historical_count": historical_count,
            "historical_sha256": expected_sha,
        }
    if historical_path is not None and historical_path.is_file():
        actual_semantic = _semantic_projection_sha256(
            actual_path, artifact_kind=artifact_kind
        )
        historical_semantic = _semantic_projection_sha256(
            historical_path, artifact_kind=artifact_kind
        )
        if actual_semantic == historical_semantic:
            return {
                "classification": "SEMANTICALLY_EQUIVALENT",
                "explanation": (
                    "byte identity differs, but ordered content matches after "
                    "excluding metadata-derived identity fields."
                ),
                "historical_count": historical_count,
                "historical_sha256": expected_sha,
                "fresh_semantic_sha256": actual_semantic,
                "historical_semantic_sha256": historical_semantic,
            }
    return {
        "classification": "STRUCTURALLY_EQUIVALENT",
        "explanation": (
            "validated count and schema match, but byte/semantic identity was not "
            "established against the accepted artifact."
        ),
        "historical_count": historical_count,
        "historical_sha256": expected_sha,
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return cast(dict[str, Any], value)


def _configuration_fingerprint(config: EncoderConfig) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def _validate_documents(path: Path) -> tuple[int, str]:
    """Validate ingestion schema, unique IDs, count, and exact file identity."""
    if not path.is_file():
        raise FileNotFoundError(f"missing documentos.jsonl: {path}")
    seen: set[str] = set()
    count = 0
    for line_number, document in read_documents(path):
        validate_document(document, line_number=line_number)
        doc_id = str(document["doc_id"])
        if doc_id in seen:
            raise ValueError(f"duplicate document doc_id: {doc_id!r}")
        seen.add(doc_id)
        count += 1
    if count == 0:
        raise ValueError("document artifact is empty.")
    return count, sha256_file(path)


def _iter_chunk_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid chunk JSON at line {line_number}."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"chunk line {line_number} must be an object.")
            yield cast(dict[str, Any], record)


def _validate_chunk_config(path: Path, semantic: EncoderConfig) -> dict[str, Any]:
    payload = _load_json_object(path, label="chunking config")
    expected: dict[str, object] = {
        "encoder_model_name": semantic["encoder_model_name"],
        "tokenizer_revision": semantic["tokenizer_revision"],
        "add_special_tokens": semantic["add_special_tokens"],
        "document_prefix": semantic["document_prefix"],
        "chunk_max_tokens": semantic["chunk_max_tokens"],
        "encoder_max_input_tokens": semantic["encoder_max_input_tokens"],
        "overlap_tokens": semantic["overlap_tokens"],
        "oversized_sentence_policy": semantic["oversized_sentence_policy"],
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"chunking config field {key!r} is incompatible.")
    overhead = payload.get("encoder_special_token_overhead")
    effective = payload.get("effective_content_max_tokens")
    if overhead != 2 or effective != 510:
        raise ValueError(
            "chunking config does not prove the accepted 512/2/510 contract."
        )
    return payload


def _validate_chunks(
    chunks_path: Path,
    config_path: Path,
    validation_path: Path | None,
) -> dict[str, object]:
    """Validate chunk identity, schema, bounds, config, and optional evidence."""
    if not chunks_path.is_file():
        raise FileNotFoundError(f"missing chunks.jsonl: {chunks_path}")
    semantic = load_encoder_config()
    chunk_config = _validate_chunk_config(config_path, semantic)
    rows, ordered_hash = ordered_chunk_identity(chunks_path)
    if rows == 0:
        raise ValueError("chunk artifact is empty.")
    maximum = 0
    hard_split_sources: set[str] = set()
    hard_split_generated = 0
    for record in _iter_chunk_records(chunks_path):
        count = record.get("num_tokens")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError("chunk num_tokens must be an integer.")
        if count > 510:
            raise ValueError("chunk exceeds effective stored capacity 510.")
        maximum = max(maximum, count)
        if record.get("hard_split") is True:
            hard_split_generated += 1
            parent = record.get("parent_chunk_id")
            if isinstance(parent, str):
                hard_split_sources.add(parent)
    evidence: dict[str, Any] | None = None
    if validation_path is not None:
        evidence = _load_json_object(validation_path, label="chunk validation evidence")
        if evidence.get("completion_status") != "completed":
            raise ValueError("chunk validation evidence is not completed.")
        if evidence.get("failure_count") != 0:
            raise ValueError("chunk validation evidence contains failures.")
        if evidence.get("valid_chunk_count") != rows:
            raise ValueError("chunk validation count does not match the artifact.")
        if evidence.get("maximum_num_tokens") != maximum:
            raise ValueError("chunk validation maximum does not match the artifact.")
        frozen = evidence.get("frozen_configuration")
        if frozen != semantic:
            raise ValueError("chunk validation semantic configuration is stale.")
        if evidence.get("configuration_fingerprint") != _configuration_fingerprint(
            semantic
        ):
            raise ValueError("chunk validation configuration fingerprint is stale.")
    return {
        "rows": rows,
        "sha256": sha256_file(chunks_path),
        "ordered_chunk_ids_sha256": ordered_hash,
        "max_stored_tokens": maximum,
        "hard_split_sources": len(hard_split_sources),
        "hard_split_generated": hard_split_generated,
        "chunking_config_sha256": sha256_file(config_path),
        "validation_evidence": str(validation_path.resolve())
        if validation_path is not None
        else None,
        "tokenizer_class": chunk_config.get("tokenizer_class"),
    }


def _apply_dependency_aware_chunk_policy(
    *,
    current_document_sha256: str,
    historical_document_sha256: str | None,
    current_chunk_count: int,
    current_chunk_sha256: str,
    historical_chunk_count: int | None,
    historical_chunk_sha256: str | None,
) -> dict[str, object]:
    """Enforce downstream historical identity only for identical upstream input."""
    historical_document_sha = (
        historical_document_sha256.upper() if historical_document_sha256 else None
    )
    upstream_identical = (
        historical_document_sha is not None
        and current_document_sha256.upper() == historical_document_sha
    )
    chunk_count_matches = (
        historical_chunk_count is None
        or current_chunk_count == historical_chunk_count
    )
    chunk_sha_matches = (
        historical_chunk_sha256 is None
        or current_chunk_sha256.upper() == historical_chunk_sha256.upper()
    )
    evidence: dict[str, object] = {
        "policy": (
            "STRICT_HISTORICAL_IDENTITY"
            if upstream_identical
            else "COMPARISON_ONLY_UPSTREAM_CHANGED"
        ),
        "upstream_document_identity_matches": upstream_identical,
        "historical_document_sha256": historical_document_sha,
        "historical_chunk_count": historical_chunk_count,
        "historical_chunk_sha256": (
            historical_chunk_sha256.upper() if historical_chunk_sha256 else None
        ),
        "chunk_count_matches_historical": chunk_count_matches,
        "chunk_sha256_matches_historical": chunk_sha_matches,
    }
    if upstream_identical and not chunk_count_matches:
        raise ValueError(
            "historical chunk count is strict because document SHA-256 is unchanged."
        )
    if upstream_identical and not chunk_sha_matches:
        raise ValueError(
            "historical chunk SHA-256 is strict because document SHA-256 is unchanged."
        )
    return evidence


def _require_approval_integer(
    value: object,
    *,
    label: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"resume approval field {label!r} must be a non-negative integer."
        )
    return value


def _validate_resume_parent(
    args: argparse.Namespace,
    semantic: EncoderConfig,
) -> dict[str, Any]:
    """Fail closed unless an approved FRESH parent exactly matches its evidence."""
    if not args.resume_run_id:
        raise ValueError("resume-fresh requires --resume-run-id.")
    if args.resume_run_directory is None or not args.resume_run_directory.is_dir():
        raise FileNotFoundError("resume-fresh parent run directory does not exist.")
    approval = _load_json_object(args.resume_approval, label="fresh resume approval")
    if approval.get("schema_version") != 1:
        raise ValueError("fresh resume approval schema is unsupported.")
    if approval.get("approval_status") != "REVIEWED_VALIDATED":
        raise ValueError("fresh resume approval is not reviewed and validated.")
    if approval.get("run_id") != args.resume_run_id:
        raise ValueError("fresh resume approval does not match --resume-run-id.")
    if approval.get("parent_mode") != "fresh":
        raise ValueError("fresh resume approval does not describe a FRESH parent.")

    parent_manifest_path = args.resume_run_directory / "run_manifest.json"
    parent = _load_json_object(parent_manifest_path, label="parent run manifest")
    if parent.get("run_id") != args.resume_run_id or parent.get("mode") != "fresh":
        raise ValueError("resume-fresh requires a matching FRESH parent run.")
    if parent.get("stages_executed") != ["ingestion", "chunking"]:
        raise ValueError(
            "resume-fresh parent must have executed ingestion then chunking only."
        )
    if parent.get("stages_reused") != []:
        raise ValueError("resume-fresh parent must contain zero reused stages.")
    records = cast(dict[str, dict[str, Any]], parent.get("stage_records", {}))
    if records.get("ingestion", {}).get("action") != "EXECUTED":
        raise ValueError("resume-fresh parent ingestion was not EXECUTED.")
    parent_error = str(parent.get("error") or "")
    if (
        records.get("chunking", {}).get("status") != "failed"
        or "chunk count does not match the explicitly expected identity"
        not in parent_error
    ):
        raise ValueError(
            "resume-fresh parent is not the reviewed post-chunk "
            "identity-policy failure."
        )

    expected_parent_artifacts = args.resume_run_directory / "artifacts"
    expected_paths = {
        "documents": expected_parent_artifacts / "documentos.jsonl",
        "chunks": expected_parent_artifacts / "chunks.jsonl",
        "chunk_config": expected_parent_artifacts / "chunking_config.json",
    }
    for name, expected_path in expected_paths.items():
        if getattr(args, name).resolve() != expected_path.resolve():
            raise ValueError(f"resume-fresh {name} path does not reference its parent.")

    documents_approval = cast(dict[str, Any], approval.get("documents", {}))
    chunks_approval = cast(dict[str, Any], approval.get("chunks", {}))
    config_approval = cast(dict[str, Any], approval.get("chunking_config", {}))
    validation = cast(
        dict[str, Any], approval.get("canonical_chunk_validation", {})
    )
    document_count, document_sha = _validate_documents(args.documents)
    chunk_info = _validate_chunks(args.chunks, args.chunk_config, None)
    approved_document_count = _require_approval_integer(
        documents_approval.get("count"), label="documents.count"
    )
    approved_chunk_count = _require_approval_integer(
        chunks_approval.get("count"), label="chunks.count"
    )
    approved_maximum = _require_approval_integer(
        chunks_approval.get("maximum_num_tokens"),
        label="chunks.maximum_num_tokens",
    )
    if document_count != approved_document_count:
        raise ValueError("resume-fresh document count differs from approval.")
    if document_sha.upper() != str(documents_approval.get("sha256", "")).upper():
        raise ValueError("resume-fresh document SHA-256 differs from approval.")
    if chunk_info["rows"] != approved_chunk_count:
        raise ValueError("resume-fresh chunk count differs from approval.")
    if str(chunk_info["sha256"]).upper() != str(
        chunks_approval.get("sha256", "")
    ).upper():
        raise ValueError("resume-fresh chunk SHA-256 differs from approval.")
    if chunk_info["max_stored_tokens"] != approved_maximum or approved_maximum > 510:
        raise ValueError("resume-fresh approved token maximum violates the contract.")
    if sha256_file(args.chunk_config).upper() != str(
        config_approval.get("sha256", "")
    ).upper():
        raise ValueError("resume-fresh chunking config SHA-256 differs from approval.")
    if config_approval.get("configuration_fingerprint") != _configuration_fingerprint(
        semantic
    ):
        raise ValueError("resume-fresh approval has a stale semantic fingerprint.")

    required_zero_fields = (
        "invalid_chunk_count",
        "failure_count",
        "token_count_mismatch_count",
        "encoder_hard_limit_exceeded_count",
        "hard_split_metadata_error_count",
        "duplicate_chunk_id_count",
        "position_error_count",
        "document_order_error_count",
        "unknown_document_count",
        "missing_output_document_count",
    )
    if validation.get("completion_status") != "completed":
        raise ValueError("trusted canonical chunk validation is not completed.")
    if any(validation.get(field) != 0 for field in required_zero_fields):
        raise ValueError("trusted canonical chunk validation contains failures.")
    if validation.get("valid_chunk_count") != approved_chunk_count:
        raise ValueError("trusted canonical chunk validation count is stale.")
    if validation.get("checked_document_count") != approved_document_count:
        raise ValueError("trusted canonical document coverage is stale.")
    if validation.get("maximum_num_tokens") != approved_maximum:
        raise ValueError("trusted canonical token maximum is stale.")
    evidence_report = Path(str(approval.get("evidence_report", "")))
    if not evidence_report.is_absolute():
        evidence_report = PROJECT_ROOT / evidence_report
    if not evidence_report.is_file():
        raise FileNotFoundError("trusted fresh chunk postmortem is missing.")

    parent_artifacts = cast(dict[str, dict[str, Any]], parent.get("artifacts", {}))
    for name, count, digest in (
        ("documents", document_count, document_sha),
        ("chunks", chunk_info["rows"], cast(str, chunk_info["sha256"])),
    ):
        recorded = parent_artifacts.get(name, {})
        recorded_sha = str(recorded.get("sha256", "")).upper()
        if recorded.get("count") != count or recorded_sha != digest.upper():
            raise ValueError(f"parent manifest/artifact identity mismatch for {name}.")

    inherited = cast(dict[str, Any], approval.get("inherited_stage_seconds", {}))
    ingestion_seconds = float(inherited.get("ingestion", -1.0))
    chunking_seconds = float(inherited.get("chunking", -1.0))
    if ingestion_seconds < 0.0 or chunking_seconds < 0.0:
        raise ValueError("resume-fresh inherited timing evidence is invalid.")
    return {
        "parent_run_id": args.resume_run_id,
        "parent_manifest": str(parent_manifest_path.resolve()),
        "approval": str(args.resume_approval.resolve()),
        "evidence_report": str(evidence_report.resolve()),
        "documents": {
            "path": str(args.documents.resolve()),
            "count": document_count,
            "sha256": document_sha,
        },
        "chunks": {
            "path": str(args.chunks.resolve()),
            "count": chunk_info["rows"],
            "sha256": chunk_info["sha256"],
            "max_stored_tokens": chunk_info["max_stored_tokens"],
        },
        "chunk_info": chunk_info,
        "inherited_stage_seconds": {
            "ingestion": ingestion_seconds,
            "chunking": chunking_seconds,
        },
    }


def _copy_metadata(chunks_path: Path, metadata_path: Path) -> None:
    """Atomically publish ordered chunk metadata as an exact byte copy."""
    if chunks_path.resolve() == metadata_path.resolve():
        return
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.", suffix=".tmp", dir=metadata_path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with chunks_path.open("rb") as source, temporary.open("wb") as destination:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(block)
        _atomic_replace_with_retry(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in _iter_chunk_records(path):
        required = {"doc_id", "chunk_id", "texto"}
        if not required.issubset(record):
            raise ValueError(f"metadata record is missing fields: {sorted(required)}")
        records.append(record)
    return records


def _truncate_respecting_sentences(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_boundary = max(
        truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! ")
    )
    if last_boundary == -1:
        return truncated
    return truncated[: last_boundary + 1].strip()


def _format_query_result(
    query_id: str,
    hits: list[tuple[int, float]],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format existing retrieval/ranking results in the competition schema."""
    from nerv.retrieval.ranking import (
        aggregate_scores_by_document,
        deduplicate_by_chunk,
        rank_results,
    )

    candidates = [
        {
            "doc_id": metadata[position]["doc_id"],
            "chunk_id": metadata[position]["chunk_id"],
            "text": metadata[position]["texto"],
            "score": score,
        }
        for position, score in hits
    ]
    candidates = deduplicate_by_chunk(candidates)
    fragments = rank_results(candidates, top_k=FRAGMENT_TOP_K)
    documents = rank_results(
        aggregate_scores_by_document(candidates), top_k=DOCUMENT_TOP_K
    )
    return {
        "query_id": query_id,
        "documents": [
            {"rank": rank, "doc_id": document["doc_id"]}
            for rank, document in enumerate(documents, start=1)
        ],
        "fragments": [
            {
                "rank": rank,
                "chunk_id": fragment["chunk_id"],
                "doc_id": fragment["doc_id"],
                "text": _truncate_respecting_sentences(
                    cast(str, fragment["text"]), MAX_WORDS_PER_FRAGMENT
                ),
            }
            for rank, fragment in enumerate(fragments, start=1)
        ],
    }


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing results artifact: {path}")
    return list(_iter_chunk_records(path))


def _stage_log(
    stage: str,
    event: str,
    *,
    detail: str | None = None,
) -> None:
    suffix = f" {detail}" if detail else ""
    LOGGER.info("[%s] %s%s", stage, event, suffix)
    print(f"[NERV] {stage}: {event.upper()}", flush=True)


def _stage_should_execute(
    stage: str,
    *,
    active: bool,
    reuse_existing: bool,
    forced: set[str],
    artifacts_exist: bool,
) -> bool:
    if not active:
        return False
    if stage in forced:
        return True
    if reuse_existing and artifacts_exist:
        return False
    return True


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in (
        "corpus",
        "documents",
        "chunks",
        "chunk_config",
        "chunk_validation",
        "chunk_metrics",
        "embeddings",
        "embedding_manifest",
        "faiss_index",
        "faiss_metadata",
        "queries",
        "results",
        "metrics",
        "run_manifest",
        "report",
        "pipeline_log",
        "historical_documents",
        "historical_chunks",
        "resume_run_directory",
        "resume_approval",
    ):
        value = getattr(args, name)
        if value is not None:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            setattr(args, name, path.resolve())
    if args.report is None:
        args.report = args.run_manifest.with_name("e2e_execution_report.txt")
    if args.pipeline_log is None:
        args.pipeline_log = args.run_manifest.with_name("pipeline.log")
    if args.mode == "fresh" and args.chunk_validation is None:
        args.chunk_validation = args.run_manifest.with_name(
            "canonical_chunk_validation.json"
        )


def _validate_cli(args: argparse.Namespace) -> tuple[int, int, set[str]]:
    if args.from_stage not in STAGES or args.to_stage not in STAGES:
        raise ValueError("unknown pipeline stage.")
    start = STAGES.index(args.from_stage)
    finish = STAGES.index(args.to_stage)
    if start > finish:
        raise ValueError("--from-stage must not come after --to-stage.")
    forced = set(args.force_stage)
    outside = [stage for stage in forced if not start <= STAGES.index(stage) <= finish]
    if outside:
        raise ValueError(f"forced stages are outside the requested range: {outside}")
    if args.embedding_batch_size <= 0:
        raise ValueError("--embedding-batch-size must be positive.")
    if args.device not in {"cpu", "cuda"}:
        raise ValueError("--device must be cpu or cuda.")
    if start > STAGES.index("retrieval"):
        raise ValueError(
            "validation-only E2E runs are prohibited because results lack a "
            "query/index identity sidecar; use validate_results for schema-only "
            "checks."
        )
    if args.mode in {"fresh", "resume-fresh"}:
        if args.device != "cuda" or args.embedding_batch_size != 16:
            raise ValueError(
                f"{args.mode} mode requires CUDA with embedding batch size 16."
            )
    if args.mode == "fresh":
        if args.from_stage != "ingestion" or args.to_stage != "validation":
            raise ValueError(
                "fresh mode requires the complete ingestion-to-validation range."
            )
        if args.reuse_existing:
            raise ValueError("fresh mode prohibits --reuse-existing.")
        if forced:
            raise ValueError("fresh mode does not accept --force-stage.")
        if args.fixture_validation:
            fixture_root = PROJECT_ROOT / "tests" / "fixtures"
            if not _is_relative_to(args.corpus, fixture_root):
                raise ValueError(
                    "--fixture-validation requires a corpus under tests/fixtures."
                )
        elif args.expected_document_count != EXPECTED_FULL_DOCUMENT_COUNT:
            raise ValueError(
                "production fresh mode requires the accepted document count."
            )
        if (
            (args.expected_document_sha256 or "").upper()
            != EXPECTED_FULL_DOCUMENT_SHA256
            or (args.expected_chunk_sha256 or "").upper()
            != EXPECTED_FULL_CHUNK_SHA256
        ):
            raise ValueError(
                "fresh mode requires the accepted historical SHA evidence."
            )
        if (
            args.historical_document_count != EXPECTED_FULL_DOCUMENT_COUNT
            or args.historical_chunk_count != EXPECTED_FULL_CHUNK_COUNT
        ):
            raise ValueError(
                "fresh mode requires the accepted historical count evidence."
            )
        run_root = args.run_manifest.parent.resolve()
        isolated_outputs = {
            "documents": args.documents,
            "chunks": args.chunks,
            "chunk_config": args.chunk_config,
            "chunk_validation": args.chunk_validation,
            "chunk_metrics": args.chunk_metrics,
            "embeddings": args.embeddings,
            "embedding_manifest": args.embedding_manifest,
            "faiss_index": args.faiss_index,
            "faiss_metadata": args.faiss_metadata,
            "results": args.results,
            "metrics": args.metrics,
            "run_manifest": args.run_manifest,
            "report": args.report,
            "pipeline_log": args.pipeline_log,
        }
        outside_run = [
            name
            for name, path in isolated_outputs.items()
            if not _is_relative_to(path, run_root)
        ]
        if outside_run:
            raise ValueError(
                "fresh output paths must be isolated under the run directory: "
                f"{outside_run}."
            )
    elif args.mode == "resume-fresh":
        if args.from_stage != "embeddings" or args.to_stage != "validation":
            raise ValueError(
                "resume-fresh requires the embeddings-to-validation range."
            )
        if args.reuse_existing:
            raise ValueError("resume-fresh prohibits --reuse-existing.")
        if forced:
            raise ValueError("resume-fresh does not accept --force-stage.")
        if args.fixture_validation:
            raise ValueError("resume-fresh does not accept --fixture-validation.")
        if args.resume_approval is None:
            raise ValueError(
                "resume-fresh requires an explicit --resume-approval path."
            )
        run_root = args.run_manifest.parent.resolve()
        isolated_outputs = {
            "embeddings": args.embeddings,
            "embedding_manifest": args.embedding_manifest,
            "faiss_index": args.faiss_index,
            "results": args.results,
            "metrics": args.metrics,
            "run_manifest": args.run_manifest,
            "report": args.report,
            "pipeline_log": args.pipeline_log,
        }
        outside_run = [
            name
            for name, path in isolated_outputs.items()
            if not _is_relative_to(path, run_root)
        ]
        if outside_run:
            raise ValueError(
                "resume-fresh continuation outputs must be isolated under the new "
                f"run directory: {outside_run}."
            )
        inherited_paths = {args.documents.resolve(), args.chunks.resolve()}
        continuation_paths = {path.resolve() for path in isolated_outputs.values()}
        if inherited_paths & continuation_paths:
            raise ValueError("resume-fresh outputs must not mutate parent artifacts.")
    output_paths = {
        "documents": args.documents,
        "chunks": args.chunks,
        "chunk_config": args.chunk_config,
        "chunk_metrics": args.chunk_metrics,
        "embeddings": args.embeddings,
        "embedding_manifest": args.embedding_manifest,
        "faiss_index": args.faiss_index,
        "faiss_metadata": args.faiss_metadata,
        "results": args.results,
        "metrics": args.metrics,
        "run_manifest": args.run_manifest,
        "report": args.report,
        "pipeline_log": args.pipeline_log,
    }
    if args.chunk_validation is not None:
        output_paths["chunk_validation"] = args.chunk_validation
    names_by_path: dict[Path, list[str]] = {}
    for name, path in output_paths.items():
        names_by_path.setdefault(path.resolve(), []).append(name)
    invalid_aliases = [
        names
        for names in names_by_path.values()
        if len(names) > 1 and set(names) != {"chunks", "faiss_metadata"}
    ]
    if invalid_aliases:
        raise ValueError(
            f"pipeline output paths must be distinct: {invalid_aliases}."
        )
    if "embeddings" in forced and (
        args.embeddings.exists() or args.embedding_manifest.exists()
    ):
        raise ValueError(
            "--force-stage embeddings never overwrites published artifacts; "
            "provide unused --embeddings and --embedding-manifest paths."
        )
    if args.fixture_validation and args.mode != "fresh":
        raise ValueError("--fixture-validation is available only in fresh mode.")
    return start, finish, forced


def _cuda_available() -> bool:
    """Return the actual torch CUDA capability without loading the encoder."""
    try:
        import torch
    except ImportError as error:
        raise ImportError("torch is required to validate CUDA availability.") from error
    return bool(torch.cuda.is_available())


def run(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the selected production stages and return metrics and manifest."""
    _resolve_paths(args)
    start, finish, forced = _validate_cli(args)
    _configure_file_logging(args.pipeline_log)
    commit, dirty = _git_state()
    run_started = time.perf_counter()
    run_id = args.run_id or str(uuid.uuid4())
    requested: list[str] = []
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "T_ingestion": 0.0,
        "T_chunking": 0.0,
        "T_embeddings": 0.0,
        "T_faiss_build": 0.0,
        "T_query_loading": 0.0,
        "T_query_encoding": 0.0,
        "T_retrieval": 0.0,
        "T_validation": 0.0,
        "T_total": 0.0,
        "stages": {},
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": _utc_now(),
        "finished_at": None,
        "status": "running",
        "failed_stage": None,
        "error": None,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "device": args.device,
        "mode": args.mode,
        "semantic_config_sha256": None,
        "semantic_config": {},
        "runtime_environment": _runtime_device_info(args.device),
        "runtime_config": {
            "embedding_batch_size": args.embedding_batch_size,
            "candidate_top_k": CANDIDATE_TOP_K,
            "fragment_top_k": FRAGMENT_TOP_K,
            "document_top_k": DOCUMENT_TOP_K,
            "local_files_only": args.local_files_only,
            "paths": {
                name: (
                    str(getattr(args, name))
                    if getattr(args, name) is not None
                    else None
                )
                for name in (
                    "corpus",
                    "documents",
                    "chunks",
                    "chunk_config",
                    "chunk_validation",
                    "chunk_metrics",
                    "embeddings",
                    "embedding_manifest",
                    "faiss_index",
                    "faiss_metadata",
                    "queries",
                    "results",
                    "metrics",
                    "run_manifest",
                    "report",
                    "pipeline_log",
                    "historical_documents",
                    "historical_chunks",
                    "resume_run_directory",
                    "resume_approval",
                )
            },
        },
        "stages_requested": requested,
        "stages_executed": [],
        "stages_reused": [],
        "stages_inherited": [],
        "artifacts": {},
        "stage_records": {},
        "fresh_execution": {
            "raw_corpus_path": str(args.corpus),
            "raw_corpus_file_count": None,
            "raw_corpus_discovered_file_count": None,
            "ocr_runtime": {},
            "document_reproducibility": {},
            "chunk_reproducibility": {},
        }
        if args.mode == "fresh"
        else {},
        "resume_fresh": {
            "parent_run_id": args.resume_run_id,
            "preflight": {},
            "inherited_artifacts": {},
            "inherited_stage_seconds": {},
        }
        if args.mode == "resume-fresh"
        else {},
        "warnings": [],
        "errors": [],
    }
    if "tests\\fixtures" in str(args.queries).lower():
        manifest["warnings"].append(
            "Official queries are unavailable; designated synthetic evaluation "
            "queries are being used."
        )
    _checkpoint(args, manifest, metrics)
    print(f"[NERV] Run ID: {run_id}", flush=True)
    runtime = cast(dict[str, Any], manifest["runtime_environment"])
    if runtime.get("gpu"):
        print(f"[NERV] CUDA: {runtime['gpu']}", flush=True)

    encoder: Any | None = None
    token_counter: TokenCounter | None = None
    resume_preflight: dict[str, Any] | None = None
    current_stage = "initialization"

    try:
        if args.device == "cuda" and not _cuda_available():
            raise RuntimeError("CUDA was explicitly requested but is not available.")
        if args.mode == "fresh":
            corpus_info = _inspect_raw_corpus(args.corpus)
            fresh_execution = cast(dict[str, Any], manifest["fresh_execution"])
            fresh_execution["raw_corpus_path"] = corpus_info["path"]
            fresh_execution["raw_corpus_file_count"] = corpus_info[
                "supported_file_count"
            ]
            fresh_execution["raw_corpus_discovered_file_count"] = corpus_info[
                "discovered_file_count"
            ]
            if args.fixture_validation:
                fresh_execution["ocr_runtime"] = {
                    "status": "NOT_REQUIRED_FOR_TEXT_FIXTURE"
                }
            else:
                fresh_execution["ocr_runtime"] = {
                    "status": "AVAILABLE",
                    **_validate_ocr_runtime(),
                }
        semantic = load_encoder_config()
        if args.mode == "resume-fresh":
            resume_preflight = _validate_resume_parent(args, semantic)
            manifest["resume_fresh"] = {
                "parent_run_id": resume_preflight["parent_run_id"],
                "preflight": {
                    "status": "PASSED",
                    "approval": resume_preflight["approval"],
                    "evidence_report": resume_preflight["evidence_report"],
                },
                "inherited_artifacts": {
                    "documents": resume_preflight["documents"],
                    "chunks": resume_preflight["chunks"],
                },
                "inherited_stage_seconds": resume_preflight[
                    "inherited_stage_seconds"
                ],
            }
        requested = list(STAGES[start : finish + 1])
        manifest["stages_requested"] = requested
        manifest["semantic_config_sha256"] = sha256_file(
            DEFAULT_ENCODER_CONFIG_PATH
        )
        manifest["semantic_config"] = dict(semantic)
        if finish >= STAGES.index("retrieval"):
            load_queries(
                args.queries,
                expected_ids=[f"q{number:03d}" for number in range(1, 51)],
            )
        _checkpoint(args, manifest, metrics)

        def get_encoder_and_counter() -> tuple[Any, TokenCounter]:
            nonlocal encoder, token_counter
            if encoder is None:
                encoder = load_encoder(
                    device=args.device,
                    local_files_only=args.local_files_only,
                )
            if token_counter is None:
                token_counter = TokenCounter(
                    semantic["encoder_model_name"],
                    revision=semantic["tokenizer_revision"],
                    add_special_tokens=semantic["add_special_tokens"],
                    document_prefix=semantic["document_prefix"],
                    local_files_only=args.local_files_only,
                    tokenizer=encoder.tokenizer,
                )
            return encoder, token_counter

        document_count = 0
        chunk_info: dict[str, object] = {}
        embedding_manifest: EmbeddingManifest | None = None
        index: Any | None = None
        query_records: list[QueryRecord] = []
        result_records: list[dict[str, Any]] = []

        current_stage = "ingestion"
        _begin_stage(args, manifest, metrics, current_stage)
        active = start <= 0 <= finish
        execute = _stage_should_execute(
            current_stage,
            active=active,
            reuse_existing=args.reuse_existing,
            forced=forced,
            artifacts_exist=args.documents.is_file(),
        )
        _stage_log(current_stage, "START", detail=f"output={args.documents}")
        stage_started = time.perf_counter()
        if execute:
            if not args.corpus.is_dir():
                raise FileNotFoundError(f"missing corpus directory: {args.corpus}")
            ingestion_cache = args.documents.with_suffix(
                ".ingestion_cache.jsonl"
            )
            documents, incidents, ingestion_stats = procesar_corpus(
                args.corpus,
                max_trabajadores=1,
                max_procesos_pesados=1,
                usar_ocr=False,
                ruta_cache=ingestion_cache,
                ignorar_cache_lectura=args.mode == "fresh",
            )
            pending_ocr = {
                str(incident["archivo"])
                for incident in incidents
                if incident.get("estado") == "pendiente_ocr"
            }
            if args.mode == "fresh" and pending_ocr:
                documents, incidents, ingestion_stats = resolver_pendientes_ocr(
                    args.corpus,
                    documents,
                    incidents,
                    pending_ocr,
                    ingestion_stats,
                    1,
                    1,
                    MODO_DOC_ID_RAPIDO,
                    ingestion_cache,
                )
            _atomic_ingestion_jsonl(args.documents, documents)
            errors_path = args.documents.with_name("errores.jsonl")
            _atomic_ingestion_jsonl(errors_path, incidents)
            document_count, _document_hash = _validate_documents(args.documents)
            elapsed = time.perf_counter() - stage_started
            metrics["T_ingestion"] = elapsed
            metrics["stages"][current_stage] = {
                "files_found": ingestion_stats.get("encontrados"),
                "input_count": ingestion_stats.get("encontrados"),
                "output_count": document_count,
                "documents_written": document_count,
                "incidents": len(incidents),
                "ocr_second_pass_files": len(pending_ocr),
                "pending_ocr_after": ingestion_stats.get("pendientes_ocr"),
                "documents_per_second": document_count / elapsed if elapsed else 0.0,
                "seconds": elapsed,
            }
            manifest["stages_executed"].append(current_stage)
            action = "executed"
            _stage_log(current_stage, "executed", detail=f"documents={document_count}")
        else:
            document_count, _document_hash = _validate_documents(args.documents)
            inherited_ingestion = (
                cast(dict[str, float], resume_preflight["inherited_stage_seconds"])[
                    "ingestion"
                ]
                if resume_preflight is not None
                else None
            )
            metrics["stages"][current_stage] = {
                "input_count": document_count,
                "output_count": document_count,
                "documents_written": document_count,
                "seconds": 0.0,
                "inherited_measured_seconds": inherited_ingestion,
            }
            if args.mode == "resume-fresh":
                manifest["stages_inherited"].append(current_stage)
                action = "inherited_validated"
                cast(dict[str, Any], manifest["stage_records"][current_stage]).update(
                    {
                        "source_run": args.resume_run_id,
                        "measured_runtime_seconds": inherited_ingestion,
                    }
                )
            else:
                manifest["stages_reused"].append(current_stage)
                action = "reused"
            _stage_log(current_stage, action, detail=f"documents={document_count}")
        document_hash = sha256_file(args.documents)
        document_reproducibility = _classify_reproducibility(
            actual_path=args.documents,
            actual_count=document_count,
            actual_sha256=document_hash,
            historical_path=args.historical_documents,
            historical_count=(
                args.historical_document_count
                if args.historical_document_count is not None
                else args.expected_document_count
            ),
            historical_sha256=args.expected_document_sha256,
            artifact_kind="documents",
        )
        if args.mode == "fresh":
            cast(dict[str, Any], manifest["fresh_execution"])[
                "document_reproducibility"
            ] = document_reproducibility
        manifest["artifacts"]["documents"] = _artifact(
            args.documents, count=document_count
        )
        errors_path = args.documents.with_name("errores.jsonl")
        if errors_path.is_file():
            manifest["artifacts"]["ingestion_errors"] = _artifact(errors_path)
        if (
            args.expected_document_count is not None
            and document_count != args.expected_document_count
        ):
            raise ValueError(
                "document count does not match the explicitly expected identity."
            )
        if (
            args.mode != "fresh"
            and
            args.expected_document_sha256 is not None
            and document_hash.upper() != args.expected_document_sha256.upper()
        ):
            raise ValueError(
                "document SHA-256 does not match the explicitly expected identity."
            )
        _stage_log(
            current_stage,
            "DONE",
            detail=f"elapsed={metrics['T_ingestion']:.3f}s status=PASS",
        )
        _complete_stage(
            args,
            manifest,
            metrics,
            current_stage,
            action=action,
            duration=float(metrics["T_ingestion"]),
        )

        if finish >= 1:
            current_stage = "chunking"
            _begin_stage(args, manifest, metrics, current_stage)
            active = start <= 1 <= finish
            execute = _stage_should_execute(
                current_stage,
                active=active,
                reuse_existing=args.reuse_existing,
                forced=forced,
                artifacts_exist=args.chunks.is_file() and args.chunk_config.is_file(),
            )
            _stage_log(current_stage, "START", detail=f"input={args.documents}")
            stage_started = time.perf_counter()
            if execute:
                _encoder, counter = get_encoder_and_counter()
                capacity = resolve_encoder_capacity(
                    config=semantic,
                    tokenizer_model_max_length=counter.reported_model_max_length,
                    encoder_special_token_overhead=(
                        counter.encoder_special_token_overhead
                    ),
                    local_files_only=args.local_files_only,
                )
                summary = run_chunking_pipeline(
                    args.documents,
                    args.chunks,
                    token_counter=counter,
                    max_tokens=semantic["chunk_max_tokens"],
                    overlap_tokens=semantic["overlap_tokens"],
                    encoder_max_input_tokens=capacity.effective_content_max_tokens,
                    config_path=args.chunk_config,
                    total_document_count=document_count,
                )
                elapsed = time.perf_counter() - stage_started
                metrics["T_chunking"] = elapsed
                if args.mode == "fresh":
                    validation_args = argparse.Namespace(
                        input=args.documents,
                        chunks=args.chunks,
                        metrics_output=args.chunk_validation,
                        local_files_only=args.local_files_only,
                        validation_batch_size=256,
                        failure_example_limit=100,
                    )
                    validation_exit = validate_chunk_artifact(
                        validation_args,
                        token_counter=counter,
                    )
                    if validation_exit != 0:
                        raise ValueError(
                            "canonical chunk validation failed with exit code "
                            f"{validation_exit}."
                        )
                chunk_info = _validate_chunks(
                    args.chunks,
                    args.chunk_config,
                    args.chunk_validation if args.mode == "fresh" else None,
                )
                metrics["stages"][current_stage] = _chunk_metrics(summary, elapsed)
                manifest["stages_executed"].append(current_stage)
                action = "executed"
                _stage_log(
                    current_stage,
                    "executed",
                    detail=f"chunks={summary.chunk_count}",
                )
            else:
                chunk_info = _validate_chunks(
                    args.chunks, args.chunk_config, args.chunk_validation
                )
                inherited_chunking = (
                    cast(
                        dict[str, float],
                        resume_preflight["inherited_stage_seconds"],
                    )["chunking"]
                    if resume_preflight is not None
                    else None
                )
                metrics["stages"][current_stage] = {
                    "input_count": document_count,
                    "output_count": chunk_info["rows"],
                    "documents": document_count,
                    "chunks": chunk_info["rows"],
                    "hard_split_sources": chunk_info["hard_split_sources"],
                    "hard_split_generated": chunk_info["hard_split_generated"],
                    "max_stored_tokens": chunk_info["max_stored_tokens"],
                    "seconds": 0.0,
                    "inherited_measured_seconds": inherited_chunking,
                    "chunks_per_second": None,
                }
                if args.mode == "resume-fresh":
                    manifest["stages_inherited"].append(current_stage)
                    action = "inherited_validated"
                    cast(
                        dict[str, Any], manifest["stage_records"][current_stage]
                    ).update(
                        {
                            "source_run": args.resume_run_id,
                            "measured_runtime_seconds": inherited_chunking,
                        }
                    )
                else:
                    manifest["stages_reused"].append(current_stage)
                    action = "reused"
                _stage_log(
                    current_stage,
                    action,
                    detail=f"chunks={chunk_info['rows']}",
                )
            chunk_stage_metrics = cast(
                dict[str, Any], metrics["stages"][current_stage]
            )
            chunk_stage_metrics.setdefault("input_count", document_count)
            chunk_stage_metrics.setdefault("output_count", chunk_info["rows"])
            chunk_reproducibility = _classify_reproducibility(
                actual_path=args.chunks,
                actual_count=cast(int, chunk_info["rows"]),
                actual_sha256=cast(str, chunk_info["sha256"]),
                historical_path=args.historical_chunks,
                historical_count=(
                    args.historical_chunk_count
                    if args.historical_chunk_count is not None
                    else args.expected_chunk_count
                ),
                historical_sha256=args.expected_chunk_sha256,
                artifact_kind="chunks",
            )
            if args.mode == "fresh":
                fresh_execution = cast(dict[str, Any], manifest["fresh_execution"])
                fresh_execution["chunk_reproducibility"] = chunk_reproducibility
                chunk_policy = _apply_dependency_aware_chunk_policy(
                    current_document_sha256=document_hash,
                    historical_document_sha256=args.expected_document_sha256,
                    current_chunk_count=cast(int, chunk_info["rows"]),
                    current_chunk_sha256=cast(str, chunk_info["sha256"]),
                    historical_chunk_count=args.historical_chunk_count,
                    historical_chunk_sha256=args.expected_chunk_sha256,
                )
                fresh_execution["chunk_identity_policy"] = chunk_policy
                if chunk_policy["policy"] == "COMPARISON_ONLY_UPSTREAM_CHANGED":
                    manifest["warnings"].append(
                        "Historical chunk identity is comparison-only because the "
                        "validated document SHA-256 changed."
                    )
                _atomic_json(args.chunk_metrics, chunk_stage_metrics)
            manifest["artifacts"]["chunks"] = _artifact(
                args.chunks, count=cast(int, chunk_info["rows"])
            ) | {
                "ordered_chunk_ids_sha256": chunk_info[
                    "ordered_chunk_ids_sha256"
                ]
            }
            manifest["artifacts"]["chunking_config"] = _artifact(args.chunk_config)
            if args.mode == "fresh":
                manifest["artifacts"]["chunking_metrics"] = _artifact(
                    args.chunk_metrics
                )
                manifest["artifacts"]["canonical_chunk_validation"] = _artifact(
                    args.chunk_validation
                )
            if (
                args.mode != "fresh"
                and args.mode != "resume-fresh"
                and
                args.expected_chunk_count is not None
                and chunk_info["rows"] != args.expected_chunk_count
            ):
                raise ValueError(
                    "chunk count does not match the explicitly expected identity."
                )
            if (
                args.mode != "fresh"
                and
                args.expected_chunk_sha256 is not None
                and str(chunk_info["sha256"]).upper()
                != args.expected_chunk_sha256.upper()
            ):
                raise ValueError(
                    "chunk SHA-256 does not match the explicitly expected identity."
                )
            _stage_log(
                current_stage,
                "DONE",
                detail=f"elapsed={metrics['T_chunking']:.3f}s status=PASS",
            )
            _complete_stage(
                args,
                manifest,
                metrics,
                current_stage,
                action=action,
                duration=float(metrics["T_chunking"]),
            )

        if finish >= 2:
            current_stage = "embeddings"
            _begin_stage(args, manifest, metrics, current_stage)
            active = start <= 2 <= finish
            execute = _stage_should_execute(
                current_stage,
                active=active,
                reuse_existing=args.reuse_existing,
                forced=forced,
                artifacts_exist=(
                    args.embeddings.is_file() and args.embedding_manifest.is_file()
                ),
            )
            _stage_log(
                current_stage,
                "START",
                detail=f"device={args.device} batch_size={args.embedding_batch_size}",
            )
            stage_started = time.perf_counter()
            if execute:
                active_encoder, counter = get_encoder_and_counter()
                embedding_manifest = generate_embedding_artifact(
                    args.chunks,
                    args.embeddings,
                    args.embedding_manifest,
                    encoder=active_encoder,
                    token_counter=counter,
                    batch_size=args.embedding_batch_size,
                    device=args.device,
                )
                elapsed = time.perf_counter() - stage_started
                metrics["T_embeddings"] = elapsed
                manifest["stages_executed"].append(current_stage)
                action = "executed"
            else:
                _matrix, embedding_manifest = validate_embedding_artifact(
                    args.embeddings,
                    args.embedding_manifest,
                    args.chunks,
                )
                if isinstance(_matrix, np.memmap):
                    _release_memmap(_matrix, flush=False)
                manifest["stages_reused"].append(current_stage)
                elapsed = 0.0
                action = "reused"
            rows = embedding_manifest["row_count"]
            metrics["stages"][current_stage] = {
                "input_count": rows,
                "output_count": rows,
                "rows": rows,
                "dimension": embedding_manifest["embedding_dimension"],
                "batch_size": args.embedding_batch_size,
                "device": embedding_manifest["device"],
                "seconds": elapsed,
                "vectors_per_second": rows / elapsed if elapsed else None,
            }
            manifest["artifacts"]["embeddings"] = _artifact(
                args.embeddings, count=rows
            )
            manifest["artifacts"]["embedding_manifest"] = _artifact(
                args.embedding_manifest
            )
            _stage_log(current_stage, action, detail=f"rows={rows}")
            _stage_log(
                current_stage,
                "DONE",
                detail=f"elapsed={metrics['T_embeddings']:.3f}s status=PASS",
            )
            _complete_stage(
                args,
                manifest,
                metrics,
                current_stage,
                action=action,
                duration=float(metrics["T_embeddings"]),
            )

        if finish >= 3:
            if embedding_manifest is None:
                embedding_manifest = load_embedding_manifest(args.embedding_manifest)
            current_stage = "faiss"
            _begin_stage(args, manifest, metrics, current_stage)
            active = start <= 3 <= finish
            sidecar = index_manifest_path(args.faiss_index)
            execute = _stage_should_execute(
                current_stage,
                active=active,
                reuse_existing=args.reuse_existing,
                forced=forced,
                artifacts_exist=(
                    args.faiss_index.is_file()
                    and sidecar.is_file()
                    and args.faiss_metadata.is_file()
                ),
            )
            _stage_log(current_stage, "START", detail=f"output={args.faiss_index}")
            stage_started = time.perf_counter()
            if execute:
                _copy_metadata(args.chunks, args.faiss_metadata)
                index, validated_manifest = build_index_from_artifact(
                    args.embeddings,
                    args.embedding_manifest,
                    args.chunks,
                    args.faiss_metadata,
                )
                sidecar = save_index(
                    index,
                    args.faiss_index,
                    embedding_manifest=validated_manifest,
                    metadata_path=args.faiss_metadata,
                )
                elapsed = time.perf_counter() - stage_started
                metrics["T_faiss_build"] = elapsed
                manifest["stages_executed"].append(current_stage)
                action = "executed"
            else:
                index = load_index(
                    args.faiss_index, metadata_path=args.faiss_metadata
                )
                sidecar_payload = _load_json_object(
                    sidecar, label="FAISS identity manifest"
                )
                if (
                    sidecar_payload.get("embeddings_sha256")
                    != embedding_manifest["embeddings_sha256"]
                ):
                    raise ValueError(
                        "FAISS embedding identity does not match manifest."
                    )
                manifest["stages_reused"].append(current_stage)
                elapsed = 0.0
                action = "reused"
            if type(index).__name__ != "IndexFlatIP":
                raise ValueError("FAISS index type must be IndexFlatIP.")
            if index.d != 384 or index.ntotal != embedding_manifest["row_count"]:
                raise ValueError("FAISS dimension or row count violates the contract.")
            metrics["stages"][current_stage] = {
                "input_count": embedding_manifest["row_count"],
                "output_count": index.ntotal,
                "index_type": type(index).__name__,
                "dimension": index.d,
                "ntotal": index.ntotal,
                "seconds": elapsed,
                "vectors_indexed_per_second": index.ntotal / elapsed
                if elapsed
                else None,
            }
            manifest["artifacts"]["faiss_index"] = _artifact(
                args.faiss_index, count=index.ntotal
            )
            manifest["artifacts"]["faiss_manifest"] = _artifact(sidecar)
            manifest["artifacts"]["faiss_metadata"] = _artifact(
                args.faiss_metadata, count=index.ntotal
            ) if args.faiss_metadata.resolve() != args.chunks.resolve() else dict(
                cast(dict[str, Any], manifest["artifacts"]["chunks"])
            )
            _stage_log(current_stage, action, detail=f"ntotal={index.ntotal}")
            _stage_log(
                current_stage,
                "DONE",
                detail=f"elapsed={metrics['T_faiss_build']:.3f}s status=PASS",
            )
            _complete_stage(
                args,
                manifest,
                metrics,
                current_stage,
                action=action,
                duration=float(metrics["T_faiss_build"]),
            )

        if finish >= 4:
            current_stage = "retrieval"
            _begin_stage(args, manifest, metrics, current_stage)
            active = start <= 4 <= finish
            # Results have no identity sidecar binding them to query/index inputs.
            # Retrieval is cheap (50 queries), so an active stage always executes.
            execute = active
            _stage_log(current_stage, "START", detail=f"queries={args.queries}")
            if execute:
                if index is None:
                    index = load_index(
                        args.faiss_index, metadata_path=args.faiss_metadata
                    )
                metadata = _load_metadata(args.faiss_metadata)
                if len(metadata) != index.ntotal:
                    raise ValueError("retrieval metadata count does not match FAISS.")
                query_started = time.perf_counter()
                query_records = load_queries(
                    args.queries,
                    expected_ids=[f"q{number:03d}" for number in range(1, 51)],
                )
                metrics["T_query_loading"] = time.perf_counter() - query_started
                active_encoder, _counter = get_encoder_and_counter()
                result_records = []
                latencies: list[float] = []
                encoding_total = 0.0
                search_total = 0.0
                for query in query_records:
                    query_total_started = time.perf_counter()
                    encoding_started = time.perf_counter()
                    vector = encode_query(query["text"], active_encoder)
                    encoding_total += time.perf_counter() - encoding_started
                    search_started = time.perf_counter()
                    hits = retrieve(vector, index, top_k=CANDIDATE_TOP_K)
                    result_records.append(
                        _format_query_result(query["query_id"], hits, metadata)
                    )
                    search_total += time.perf_counter() - search_started
                    latencies.append(time.perf_counter() - query_total_started)
                metrics["T_query_encoding"] = encoding_total
                metrics["T_retrieval"] = search_total
                _atomic_jsonl(args.results, result_records)
                manifest["stages_executed"].append(current_stage)
                action = "executed"
                metrics["stages"][current_stage] = _retrieval_metrics(
                    query_records, latencies, search_total
                )
            else:
                query_records = load_queries(
                    args.queries,
                    expected_ids=[f"q{number:03d}" for number in range(1, 51)],
                )
                result_records = _read_results(args.results)
                manifest["stages_reused"].append(current_stage)
                action = "reused"
                metrics["stages"][current_stage] = {
                    "query_count": len(query_records),
                    "candidate_top_k": CANDIDATE_TOP_K,
                    "fragment_top_k": FRAGMENT_TOP_K,
                    "document_top_k": DOCUMENT_TOP_K,
                    "total_seconds": 0.0,
                    "mean_ms_per_query": None,
                    "median_ms_per_query": None,
                    "p95_ms_per_query": None,
                }
            manifest["artifacts"]["queries"] = _artifact(
                args.queries, count=len(query_records)
            )
            manifest["artifacts"]["results"] = _artifact(
                args.results, count=len(result_records)
            )
            retrieval_stage_metrics = cast(
                dict[str, Any], metrics["stages"][current_stage]
            )
            retrieval_stage_metrics["input_count"] = len(query_records)
            retrieval_stage_metrics["output_count"] = len(result_records)
            _stage_log(
                current_stage, action, detail=f"queries={len(query_records)}"
            )
            _stage_log(
                current_stage,
                "DONE",
                detail=(
                    "elapsed="
                    f"{metrics['T_query_encoding'] + metrics['T_retrieval']:.3f}s "
                    "status=PASS"
                ),
            )
            retrieval_duration = float(
                metrics["T_query_loading"]
                + metrics["T_query_encoding"]
                + metrics["T_retrieval"]
            )
            metrics["stages"][current_stage]["seconds"] = retrieval_duration
            metrics["stages"][current_stage]["queries_per_second"] = (
                len(query_records) / retrieval_duration if retrieval_duration else 0.0
            )
            _complete_stage(
                args,
                manifest,
                metrics,
                current_stage,
                action=action,
                duration=retrieval_duration,
            )

        if finish >= 5:
            current_stage = "validation"
            _begin_stage(args, manifest, metrics, current_stage)
            _stage_log(current_stage, "START", detail=f"input={args.results}")
            stage_started = time.perf_counter()
            if not result_records:
                result_records = _read_results(args.results)
            validate_results(result_records)
            elapsed = time.perf_counter() - stage_started
            metrics["T_validation"] = elapsed
            metrics["stages"][current_stage] = {
                "input_count": len(result_records),
                "output_count": len(result_records),
                "valid": True,
                "error_count": 0,
                "warning_count": 0,
                "seconds": elapsed,
                "results_per_second": len(result_records) / elapsed if elapsed else 0.0,
            }
            if start <= 5 <= finish:
                manifest["stages_executed"].append(current_stage)
                action = "executed"
            else:
                manifest["stages_reused"].append(current_stage)
                action = "reused"
            _stage_log(
                current_stage,
                "DONE",
                detail=f"elapsed={elapsed:.3f}s status=PASS",
            )
            _complete_stage(
                args,
                manifest,
                metrics,
                current_stage,
                action=action,
                duration=elapsed,
            )

        if args.mode == "fresh":
            if manifest["stages_reused"]:
                raise RuntimeError(
                    "fresh mode detected reused stages and failed closed."
                )
            if manifest["stages_executed"] != list(STAGES):
                raise RuntimeError(
                    "fresh mode did not execute every stage in canonical order."
                )
        if args.mode == "resume-fresh":
            if manifest["stages_reused"]:
                raise RuntimeError(
                    "resume-fresh detected reused production artifacts and "
                    "failed closed."
                )
            if manifest["stages_inherited"] != ["ingestion", "chunking"]:
                raise RuntimeError(
                    "resume-fresh did not inherit exactly ingestion and chunking."
                )
            if manifest["stages_executed"] != list(STAGES[2:]):
                raise RuntimeError(
                    "resume-fresh did not execute embeddings through validation "
                    "in order."
                )
        metrics["T_total"] = time.perf_counter() - run_started
        if args.mode == "resume-fresh":
            if resume_preflight is None:
                raise RuntimeError(
                    "resume-fresh completed without validated preflight."
                )
            inherited_seconds = cast(
                dict[str, float], resume_preflight["inherited_stage_seconds"]
            )
            metrics["continuation_wall_clock_seconds"] = metrics["T_total"]
            metrics["resumed_fresh_compute_equivalent_e2e_seconds"] = (
                inherited_seconds["ingestion"]
                + inherited_seconds["chunking"]
                + _stage_seconds(cast(dict[str, Any], metrics["stages"]["embeddings"]))
                + _stage_seconds(cast(dict[str, Any], metrics["stages"]["faiss"]))
                + _stage_seconds(cast(dict[str, Any], metrics["stages"]["retrieval"]))
                + _stage_seconds(cast(dict[str, Any], metrics["stages"]["validation"]))
            )
        timed = {
            key: value
            for key, value in metrics.items()
            if key.startswith("T_") and key != "T_total" and isinstance(value, float)
        }
        metrics["stage_percentages"] = {
            key: (value / metrics["T_total"] * 100.0 if metrics["T_total"] else 0.0)
            for key, value in timed.items()
        }
        manifest["status"] = "completed"
        manifest["finished_at"] = _utc_now()
        _checkpoint(args, manifest, metrics)
        print(f"[NERV] report: {args.report}", flush=True)
        return metrics, manifest
    except Exception as error:
        metrics["T_total"] = time.perf_counter() - run_started
        manifest["status"] = "failed"
        manifest["failed_stage"] = current_stage
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["errors"].append(cast(str, manifest["error"]))
        manifest["finished_at"] = _utc_now()
        stage_record = manifest["stage_records"].get(current_stage)
        if isinstance(stage_record, dict):
            stage_record["status"] = "failed"
            stage_record["end_timestamp"] = manifest["finished_at"]
            stage_record["errors"] = [manifest["error"]]
        LOGGER.exception("Pipeline failed during %s", current_stage)
        _checkpoint(args, manifest, metrics)
        _stage_log(current_stage, "FAILED", detail=manifest["error"])
        print(f"[NERV] report: {args.report}", flush=True)
        raise


def _chunk_metrics(summary: PipelineSummary, elapsed: float) -> dict[str, object]:
    return {
        "input_count": summary.document_count,
        "output_count": summary.chunk_count,
        "documents": summary.document_count,
        "chunks": summary.chunk_count,
        "hard_split_sources": summary.hard_split_source_unit_count,
        "hard_split_generated": summary.hard_split_generated_chunk_count,
        "max_stored_tokens": summary.maximum_tokens_per_chunk,
        "seconds": elapsed,
        "chunks_per_second": summary.chunk_count / elapsed if elapsed else 0.0,
    }


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return ordered[index]


def _retrieval_metrics(
    queries: Sequence[QueryRecord],
    latencies: Sequence[float],
    retrieval_seconds: float,
) -> dict[str, object]:
    return {
        "query_count": len(queries),
        "candidate_top_k": CANDIDATE_TOP_K,
        "fragment_top_k": FRAGMENT_TOP_K,
        "document_top_k": DOCUMENT_TOP_K,
        "total_seconds": retrieval_seconds,
        "mean_ms_per_query": statistics.fmean(latencies) * 1000.0
        if latencies
        else 0.0,
        "median_ms_per_query": statistics.median(latencies) * 1000.0
        if latencies
        else 0.0,
        "p95_ms_per_query": _percentile_95(latencies) * 1000.0,
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the stable minimal Phase-2 CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run the identity-safe NERV end-to-end pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full", "fresh", "resume-fresh", "custom"),
        default="custom",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--resume-run-id")
    parser.add_argument("--resume-run-directory", type=Path)
    parser.add_argument(
        "--resume-approval",
        type=Path,
    )
    parser.add_argument("--from-stage", choices=STAGES, default="ingestion")
    parser.add_argument("--to-stage", choices=STAGES, default="validation")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--force-stage", choices=STAGES, action="append", default=[]
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--corpus", type=Path, default=Path("corpus/raw"))
    parser.add_argument(
        "--documents", type=Path, default=Path("corpus/processed/documentos.jsonl")
    )
    parser.add_argument("--chunks", type=Path, default=Path("outputs/chunks.jsonl"))
    parser.add_argument(
        "--chunk-config",
        type=Path,
        default=Path("outputs/chunking_config.json"),
    )
    parser.add_argument("--chunk-validation", type=Path)
    parser.add_argument(
        "--chunk-metrics", type=Path, default=Path("outputs/chunking_metrics.json")
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("outputs/embeddings.npy")
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=Path("outputs/embeddings.manifest.json"),
    )
    parser.add_argument(
        "--faiss-index", type=Path, default=Path("outputs/index.faiss")
    )
    parser.add_argument(
        "--faiss-metadata", type=Path, default=Path("outputs/metadata.jsonl")
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("corpus/queries/queries.jsonl")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("outputs/resultados.jsonl")
    )
    parser.add_argument(
        "--metrics", type=Path, default=Path("outputs/run_metrics.json")
    )
    parser.add_argument(
        "--run-manifest", type=Path, default=Path("outputs/run_manifest.json")
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--pipeline-log", type=Path)
    parser.add_argument("--expected-document-count", type=int)
    parser.add_argument("--expected-document-sha256")
    parser.add_argument("--expected-chunk-count", type=int)
    parser.add_argument("--expected-chunk-sha256")
    parser.add_argument("--historical-documents", type=Path)
    parser.add_argument("--historical-chunks", type=Path)
    parser.add_argument("--historical-document-count", type=int)
    parser.add_argument("--historical-chunk-count", type=int)
    parser.add_argument("--fixture-validation", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI and return a non-zero process status on fail-closed errors."""
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as error:
        print(f"[pipeline] FAILED {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
