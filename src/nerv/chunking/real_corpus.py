"""Independent production, validation, and benchmark commands for chunking."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .atomic_io import _atomic_replace_with_retry
from .configuration import (
    EncoderCapacity,
    EncoderConfig,
    load_encoder_config,
    resolve_encoder_capacity,
)
from .pipeline import (
    PipelineDocumentProgress,
    PipelineSummary,
    configure_logging,
    read_documents,
    run_pipeline,
    validate_document,
)
from .token_counter import TokenCounter

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "outputs/resultados/documentos.jsonl"
DEFAULT_CHUNKS = ROOT / "outputs/resultados/chunks.jsonl"
DEFAULT_PRODUCTION_METRICS = (
    ROOT / "outputs/resultados/chunking_production_metrics.json"
)
DEFAULT_VALIDATION_METRICS = (
    ROOT / "outputs/resultados/chunking_validation_metrics.json"
)
DEFAULT_BENCHMARK_METRICS = (
    ROOT / "outputs/resultados/chunking_benchmark_metrics.json"
)
DEFAULT_PROGRESS = ROOT / "outputs/resultados/chunking_progress.json"
DEFAULT_CONFIG_OUTPUT = ROOT / "outputs/resultados/chunking_config.json"
DEFAULT_MANIFEST = (
    ROOT / "tests/chunking/fixtures/real_corpus_benchmark_manifest.json"
)
DEFAULT_LOG_PATH = ROOT / "outputs/logs/chunking_real_corpus.log"
REQUIRED_CHUNK_FIELDS = frozenset(
    {
        "doc_id",
        "chunk_id",
        "fuente",
        "formato",
        "fenomeno",
        "posicion",
        "num_tokens",
        "texto",
    }
)
HARD_SPLIT_TRACE_FIELDS = frozenset(
    {
        "hard_split",
        "parent_chunk_id",
        "hard_split_strategy",
        "hard_split_part_index",
        "hard_split_part_count",
        "hard_split_overlap_tokens",
        "hard_split_parent_num_tokens",
        "hard_split_source_start",
        "hard_split_source_end",
        "hard_split_emitted_start",
    }
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, value: object) -> None:
    """Write one JSON document without exposing a partial destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    ready_for_publication = False
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
            json.dump(value, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        ready_for_publication = True
        _atomic_replace_with_retry(temporary_path, path)
    except BaseException:
        if temporary_path is not None and not ready_for_publication:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "Could not remove partial JSON temporary %s while preserving "
                    "the active exception.",
                    temporary_path,
                    exc_info=True,
                )
        elif temporary_path is not None:
            LOGGER.error(
                "Atomic JSON publication failed; the complete temporary was "
                "preserved at %s.",
                temporary_path,
            )
        raise


def _attempt_secondary_persistence(
    action: Callable[[], None],
    *,
    description: str,
) -> None:
    """Log secondary persistence failure without masking the active exception."""
    try:
        action()
    except Exception:
        LOGGER.exception(
            "Could not persist %s; the primary failure remains authoritative.",
            description,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _configuration_fingerprint(config: EncoderConfig) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    item = path.stat()
    return {
        "path": str(path),
        "size_bytes": item.st_size,
        "modified_time_ns": item.st_mtime_ns,
        "device": item.st_dev,
        "inode": item.st_ino,
    }


def _optional_file_identity(path: Path) -> dict[str, object] | None:
    """Return file identity when present without treating absence as failure."""
    return _file_identity(path) if path.is_file() else None


def _environment_summary() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_processor_count": os.cpu_count(),
    }


def _dependency_versions() -> dict[str, str | None]:
    """Return benchmark-relevant package versions without importing packages."""
    versions: dict[str, str | None] = {}
    for package in ("nerv", "transformers", "tokenizers", "pysbd"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _workflow_code_fingerprint() -> str:
    """Fingerprint workflow and pipeline source used by resumable benchmarks."""
    digest = hashlib.sha256()
    module_directory = Path(__file__).parent
    paths = (
        Path(__file__),
        module_directory / "pipeline.py",
        module_directory / "hard_limit.py",
        module_directory / "chunker.py",
        module_directory / "token_counter.py",
        module_directory / "sentence_splitter.py",
        module_directory / "language_detector.py",
        module_directory / "configuration.py",
        module_directory / "resources/language_markers.json",
    )
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _count_documents(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def _require_distinct_paths(**paths: Path) -> None:
    """Reject aliases between read-only inputs and writable artifacts."""
    resolved_paths: list[tuple[str, Path]] = []
    for name, path in paths.items():
        resolved = path.resolve()
        for previous_name, previous_path in resolved_paths:
            same_existing_file = False
            if resolved.exists() and previous_path.exists():
                try:
                    same_existing_file = os.path.samefile(
                        resolved,
                        previous_path,
                    )
                except OSError:
                    same_existing_file = False
            if resolved == previous_path or same_existing_file:
                raise ValueError(
                    f"{name} must differ from {previous_name}: {path}."
                )
        resolved_paths.append((name, resolved))


def _load_counter(args: argparse.Namespace, config: EncoderConfig) -> TokenCounter:
    counter = TokenCounter(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
        local_files_only=args.local_files_only,
    )
    _resolve_capacity(counter, config, local_files_only=args.local_files_only)
    return counter


def _resolve_capacity(
    counter: TokenCounter,
    config: EncoderConfig,
    *,
    local_files_only: bool,
) -> EncoderCapacity:
    """Resolve the total and effective capacity for production or validation."""
    return resolve_encoder_capacity(
        config=config,
        tokenizer_model_max_length=counter.reported_model_max_length,
        encoder_special_token_overhead=counter.encoder_special_token_overhead,
        local_files_only=local_files_only,
    )


def _base_run_metrics(
    *,
    mode: str,
    source_path: Path,
    config: EncoderConfig,
) -> dict[str, Any]:
    source_identity = _optional_file_identity(source_path)
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "mode": mode,
        "source_path": str(source_path),
        "frozen_configuration": config,
        "configuration_fingerprint": _configuration_fingerprint(config),
        "input_size_bytes": (
            source_identity["size_bytes"] if source_identity else None
        ),
        "input_modified_time_ns": (
            source_identity["modified_time_ns"] if source_identity else None
        ),
        "environment": _environment_summary(),
        "completion_status": "not_started",
        "warnings": [],
        "failures": [],
    }


def _save_production_state(
    *,
    progress_path: Path,
    metrics_path: Path,
    state: dict[str, Any],
) -> None:
    state["generated_at"] = _utc_now()
    _write_json_atomic(progress_path, state)
    _write_json_atomic(metrics_path, state)


def produce_command(
    args: argparse.Namespace,
    *,
    token_counter: TokenCounter | None = None,
    pipeline_runner: Callable[..., PipelineSummary] = run_pipeline,
) -> int:
    """Run exactly one production pass and persist progress independently."""
    config = load_encoder_config()
    source_path: Path = args.input
    output_path: Path = args.output
    metrics_path: Path = args.metrics_output
    progress_path: Path = args.progress_output
    config_path: Path = args.config_output
    _require_distinct_paths(
        input_path=source_path,
        output_path=output_path,
        metrics_output=metrics_path,
        progress_output=progress_path,
        config_output=config_path,
    )
    output_identity_before = _optional_file_identity(output_path)
    total_documents: int | None = None
    state = _base_run_metrics(
        mode="produce",
        source_path=source_path,
        config=config,
    )
    state.update(
        {
            "output_path": str(output_path),
            "work_directory": str(args.work_dir) if args.work_dir else None,
            "document_count": None,
            "processed_document_count": 0,
            "failed_document_count": 0,
            "chunk_count": 0,
            "generated_chunk_count": 0,
            "oversized_chunk_count": 0,
            "hard_split_source_unit_count": 0,
            "hard_split_generated_chunk_count": 0,
            "hard_split_by_format": {},
            "hard_split_by_strategy": {},
            "maximum_tokens_per_chunk": None,
            "detected_language_counts": {},
            "accumulated_stage_durations": {
                "language_detection_seconds": 0.0,
                "splitting_seconds": 0.0,
                "chunking_seconds": 0.0,
                "record_construction_seconds": 0.0,
                "writing_seconds": 0.0,
            },
            "processing_duration_seconds": 0.0,
            "tokenizer_load_duration_seconds": 0.0,
            "output_size_bytes": None,
            "output_identity_before": output_identity_before,
            "output_identity_after": output_identity_before,
            "output_publication_status": "not_started",
            "last_completed_document_index": 0,
            "last_completed_doc_id": None,
            "failure_count": 0,
            "temporary_file_path": None,
            "completion_status": "running",
            "multiprocessing_enabled": args.workers == 1,
            "worker_count": args.workers,
            "start_method": "spawn" if args.workers == 1 else None,
            "workers_initialized": 0,
            "worker_initialization_seconds": 0.0,
            "worker_task_count": 0,
            "parallel_processing_wall_seconds": 0.0,
            "worker_service_seconds_sum": 0.0,
            "worker_temp_bytes_total": 0,
            "worker_temp_write_seconds": 0.0,
            "ordered_merge_seconds": 0.0,
            "merged_document_count": 0,
            "last_merged_document_index": 0,
            "peak_pending_documents": 0,
            "peak_pending_bytes": 0,
            "output_sha256": None,
        }
    )
    _save_production_state(
        progress_path=progress_path,
        metrics_path=metrics_path,
        state=state,
    )
    started = time.perf_counter()
    language_counts: Counter[str] = Counter()
    hard_split_format_counts: Counter[str] = Counter()
    hard_split_strategy_counts: Counter[str] = Counter()

    def record_output_state_after_failure() -> None:
        output_identity_after = _optional_file_identity(output_path)
        output_changed = output_identity_after != output_identity_before
        temporary_file_path = state["temporary_file_path"]
        temporary_file_exists = bool(
            temporary_file_path
            and Path(str(temporary_file_path)).is_file()
        )
        state["output_identity_after"] = output_identity_after
        state["temporary_file_exists"] = temporary_file_exists
        state["output_publication_status"] = (
            "published_before_failure" if output_changed else "previous_preserved"
        )
        if output_changed:
            state["warnings"].append(
                "A completed chunks output was published before a later "
                "artifact step failed. Inspect output_identity_after."
            )

    def record_temporary_path(path: Path) -> None:
        state["temporary_file_path"] = str(path)
        _save_production_state(
            progress_path=progress_path,
            metrics_path=metrics_path,
            state=state,
        )

    def record_progress(progress: PipelineDocumentProgress) -> None:
        language_counts[progress.detected_language or "not_detected"] += 1
        hard_split_format_counts.update(progress.hard_split_by_format)
        hard_split_strategy_counts.update(progress.hard_split_by_strategy)
        stage_durations = state["accumulated_stage_durations"]
        assert isinstance(stage_durations, dict)
        stage_durations["language_detection_seconds"] += (
            progress.language_detection_duration_seconds
        )
        stage_durations["splitting_seconds"] += progress.splitting_duration_seconds
        stage_durations["chunking_seconds"] += progress.chunking_duration_seconds
        stage_durations["record_construction_seconds"] += (
            progress.record_construction_duration_seconds
        )
        stage_durations["writing_seconds"] += progress.writing_duration_seconds
        state.update(
            {
                "processed_document_count": progress.processed_document_count,
                "last_completed_document_index": progress.document_index,
                "last_completed_doc_id": progress.doc_id,
                "generated_chunk_count": progress.total_generated_chunk_count,
                "chunk_count": progress.total_generated_chunk_count,
                "oversized_chunk_count": (
                    int(state["oversized_chunk_count"])
                    + progress.oversized_chunk_count
                ),
                "hard_split_source_unit_count": (
                    int(state["hard_split_source_unit_count"])
                    + progress.hard_split_source_unit_count
                ),
                "hard_split_generated_chunk_count": (
                    int(state["hard_split_generated_chunk_count"])
                    + progress.hard_split_generated_chunk_count
                ),
                "hard_split_by_format": dict(
                    sorted(hard_split_format_counts.items())
                ),
                "hard_split_by_strategy": dict(
                    sorted(hard_split_strategy_counts.items())
                ),
                "detected_language_counts": dict(sorted(language_counts.items())),
                "processing_duration_seconds": progress.elapsed_time_seconds,
                "estimated_remaining_seconds": (
                    progress.estimated_remaining_seconds
                ),
            }
        )
        _save_production_state(
            progress_path=progress_path,
            metrics_path=metrics_path,
            state=state,
        )

    def record_architecture_metrics(metrics: dict[str, object]) -> None:
        state.update(metrics)

    try:
        total_documents = _count_documents(source_path)
        state["document_count"] = total_documents
        _save_production_state(
            progress_path=progress_path,
            metrics_path=metrics_path,
            state=state,
        )
        if args.workers == 1:
            if token_counter is not None or pipeline_runner is not run_pipeline:
                raise ValueError(
                    "the Phase-1 workers path owns worker-local tokenizer "
                    "initialization and does not accept injected pipeline objects."
                )
            if args.work_dir is None:
                raise ValueError("--workers 1 requires --work-dir.")
            from .parallel_pipeline import run_parallel_pipeline_phase_1

            parallel_run = run_parallel_pipeline_phase_1(
                source_path,
                output_path,
                config=config,
                local_files_only=args.local_files_only,
                config_path=config_path,
                total_document_count=total_documents,
                progress_callback=record_progress,
                global_progress_interval=args.global_progress_interval,
                work_dir=args.work_dir,
                preserve_incomplete=True,
                temporary_path_callback=record_temporary_path,
                architecture_metrics_callback=record_architecture_metrics,
            )
            summary = parallel_run.summary
            parallel_metrics = asdict(parallel_run.metrics)
            state.update(parallel_metrics)
            state["tokenizer_load_duration_seconds"] = (
                parallel_run.metrics.worker_initialization_seconds
            )
        else:
            if token_counter is None:
                tokenizer_started = time.perf_counter()
                token_counter = _load_counter(args, config)
                _ = token_counter.tokenizer_class
                state["tokenizer_load_duration_seconds"] = (
                    time.perf_counter() - tokenizer_started
                )
            capacity = _resolve_capacity(
                token_counter,
                config,
                local_files_only=args.local_files_only,
            )
            state.update(asdict(capacity))
            summary = pipeline_runner(
                source_path,
                output_path,
                token_counter=token_counter,
                max_tokens=config["chunk_max_tokens"],
                overlap_tokens=config["overlap_tokens"],
                encoder_max_input_tokens=(
                    capacity.effective_content_max_tokens
                ),
                config_path=config_path,
                total_document_count=total_documents,
                progress_callback=record_progress,
                global_progress_interval=args.global_progress_interval,
                work_dir=args.work_dir,
                preserve_incomplete=True,
                temporary_path_callback=record_temporary_path,
            )
    except KeyboardInterrupt:
        record_output_state_after_failure()
        state["processing_duration_seconds"] = time.perf_counter() - started
        state["completion_status"] = "interrupted"
        state["failure_count"] = int(state["failure_count"]) + 1
        state["failures"].append(
            {
                "category": "keyboard_interrupt",
                "message": "Production was interrupted by the operator.",
            }
        )
        _attempt_secondary_persistence(
            lambda: _save_production_state(
                progress_path=progress_path,
                metrics_path=metrics_path,
                state=state,
            ),
            description="interrupted production state",
        )
        if state["output_publication_status"] == "published_before_failure":
            LOGGER.error(
                "Production was interrupted after a completed chunks output "
                "was published. Inspect output_identity_after."
            )
        elif state["temporary_file_exists"]:
            LOGGER.error(
                "Production interrupted after document %s. Incomplete "
                "temporary file: %s",
                state["last_completed_document_index"],
                state["temporary_file_path"],
            )
        else:
            LOGGER.error(
                "Production interrupted before publication; the previous "
                "output was preserved."
            )
        return 130
    except Exception as error:
        record_output_state_after_failure()
        state["processing_duration_seconds"] = time.perf_counter() - started
        state["completion_status"] = "failed"
        state["failed_document_count"] = 1
        state["failure_count"] = int(state["failure_count"]) + 1
        state["failures"].append(
            {
                "category": type(error).__name__,
                "message": str(error),
            }
        )
        _attempt_secondary_persistence(
            lambda: _save_production_state(
                progress_path=progress_path,
                metrics_path=metrics_path,
                state=state,
            ),
            description="failed production state",
        )
        if state["output_publication_status"] == "published_before_failure":
            LOGGER.exception(
                "Production failed after a completed chunks output was "
                "published. Inspect output_identity_after."
            )
        elif state["temporary_file_exists"]:
            LOGGER.exception(
                "Production failed; incomplete temporary output was "
                "preserved at %s.",
                state["temporary_file_path"],
            )
        else:
            LOGGER.exception(
                "Production failed before publication; the previous output "
                "was preserved."
            )
        return 1

    state.update(
        {
            "processed_document_count": summary.document_count,
            "chunk_count": summary.chunk_count,
            "generated_chunk_count": summary.chunk_count,
            "oversized_chunk_count": summary.oversized_chunk_count,
            "hard_split_source_unit_count": (
                summary.hard_split_source_unit_count
            ),
            "hard_split_generated_chunk_count": (
                summary.hard_split_generated_chunk_count
            ),
            "hard_split_by_format": dict(
                sorted(summary.hard_split_by_format.items())
            ),
            "hard_split_by_strategy": dict(
                sorted(summary.hard_split_by_strategy.items())
            ),
            "maximum_tokens_per_chunk": summary.maximum_tokens_per_chunk,
            "processing_duration_seconds": summary.processing_duration_seconds,
            "output_size_bytes": output_path.stat().st_size,
            "output_identity_after": _file_identity(output_path),
            "output_publication_status": "published",
            "completion_status": "completed",
            "estimated_remaining_seconds": 0.0,
            "temporary_file_exists": bool(
                state["temporary_file_path"]
                and Path(str(state["temporary_file_path"])).exists()
            ),
        }
    )
    _save_production_state(
        progress_path=progress_path,
        metrics_path=metrics_path,
        state=state,
    )
    LOGGER.info(
        "Production completed: documents=%d chunks=%d output=%s.",
        summary.document_count,
        summary.chunk_count,
        output_path,
    )
    return 0


class _ValidationFailures:
    """Keep complete counters while storing only bounded examples."""

    def __init__(self, example_limit: int) -> None:
        self.example_limit = example_limit
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, object]] = []

    def add(self, category: str, **details: object) -> None:
        self.counts[category] += 1
        if len(self.examples) < self.example_limit:
            self.examples.append({"category": category, **details})

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _load_source_contract(
    source_path: Path,
) -> tuple[dict[str, tuple[int, object, object, object]], list[str]]:
    contract: dict[str, tuple[int, object, object, object]] = {}
    order: list[str] = []
    for line_number, document in read_documents(source_path):
        validate_document(document, line_number=line_number)
        doc_id = str(document["doc_id"])
        if doc_id in contract:
            raise ValueError(
                f"duplicate source doc_id {doc_id!r} on line {line_number}."
            )
        contract[doc_id] = (
            len(order),
            document.get("fuente"),
            document.get("formato"),
            document.get("fenomeno"),
        )
        order.append(doc_id)
    return contract, order


def validate_command(
    args: argparse.Namespace,
    *,
    token_counter: TokenCounter | None = None,
) -> int:
    """Validate an existing chunks file without invoking production."""
    config = load_encoder_config()
    source_path: Path = args.input
    chunks_path: Path = args.chunks
    metrics_path: Path = args.metrics_output
    _require_distinct_paths(
        input_path=source_path,
        chunks_path=chunks_path,
        metrics_output=metrics_path,
    )
    started = time.perf_counter()
    source_identity_before = _optional_file_identity(source_path)
    chunks_identity_before = _optional_file_identity(chunks_path)
    failures = _ValidationFailures(args.failure_example_limit)

    metrics: dict[str, Any] = {
        **_base_run_metrics(
            mode="validate",
            source_path=source_path,
            config=config,
        ),
        "chunks_path": str(chunks_path),
        "source_identity_before": source_identity_before,
        "chunks_identity_before": chunks_identity_before,
        "valid_chunk_count": 0,
        "invalid_chunk_count": 0,
        "duplicate_chunk_id_count": 0,
        "position_error_count": 0,
        "document_order_error_count": 0,
        "metadata_error_count": 0,
        "token_count_mismatch_count": 0,
        "encoder_hard_limit_exceeded_count": 0,
        "hard_split_metadata_error_count": 0,
        "chunks_le_soft_limit": 0,
        "chunks_gt_soft_limit_but_le_hard_limit": 0,
        "chunks_gt_hard_limit": 0,
        "maximum_num_tokens": None,
        "empty_chunk_count": 0,
        "stored_prefix_error_count": 0,
        "unknown_document_count": 0,
        "missing_output_document_count": 0,
        "checked_document_count": 0,
        "validation_duration_seconds": 0.0,
        "failure_count": 0,
        "failure_counts": {},
        "failure_examples": [],
        "failure_example_limit": args.failure_example_limit,
        "completion_status": "running",
        "limitations": [
            "Sentence-integrity validation is partial in this optimization stage.",
            "The validator does not perform substring-based sentence reconstruction.",
            "Overlap semantics are not re-evaluated in this mode.",
        ],
    }
    _write_json_atomic(metrics_path, metrics)
    try:
        source_contract, _ = _load_source_contract(source_path)
        if token_counter is None:
            token_counter = _load_counter(args, config)
    except KeyboardInterrupt:
        metrics["completion_status"] = "interrupted"
        metrics["failure_count"] = 1
        metrics["failure_counts"] = {"keyboard_interrupt": 1}
        metrics["failure_examples"] = [
            {"category": "keyboard_interrupt"}
        ][: args.failure_example_limit]
        metrics["validation_duration_seconds"] = time.perf_counter() - started
        metrics["generated_at"] = _utc_now()
        _attempt_secondary_persistence(
            lambda: _write_json_atomic(metrics_path, metrics),
            description="interrupted validation setup metrics",
        )
        LOGGER.error("Validation setup was interrupted; metrics were preserved.")
        return 130
    except Exception as error:
        category = type(error).__name__
        metrics["completion_status"] = "failed"
        metrics["failure_count"] = 1
        metrics["failure_counts"] = {category: 1}
        metrics["failure_examples"] = [
            {"category": category, "message": str(error)}
        ][: args.failure_example_limit]
        metrics["validation_duration_seconds"] = time.perf_counter() - started
        metrics["generated_at"] = _utc_now()
        _attempt_secondary_persistence(
            lambda: _write_json_atomic(metrics_path, metrics),
            description="failed validation setup metrics",
        )
        LOGGER.exception("Validation setup failed; metrics were preserved.")
        return 1
    capacity = _resolve_capacity(
        token_counter,
        config,
        local_files_only=args.local_files_only,
    )
    metrics.update(asdict(capacity))
    unique_chunk_ids: set[str] = set()
    checked_documents: set[str] = set()
    expected_positions: Counter[str] = Counter()
    last_source_index = -1
    batch: list[dict[str, Any]] = []
    hard_split_groups: dict[str, list[dict[str, Any]]] = {}

    def update_failure_metrics() -> None:
        metrics["duplicate_chunk_id_count"] = failures.counts[
            "duplicate_chunk_id"
        ]
        metrics["position_error_count"] = failures.counts["position_error"]
        metrics["document_order_error_count"] = failures.counts[
            "document_order_error"
        ]
        metrics["metadata_error_count"] = failures.counts["metadata_error"]
        metrics["token_count_mismatch_count"] = failures.counts[
            "token_count_mismatch"
        ]
        metrics["encoder_hard_limit_exceeded_count"] = failures.counts[
            "encoder_hard_limit_exceeded"
        ]
        metrics["hard_split_metadata_error_count"] = failures.counts[
            "hard_split_metadata_error"
        ]
        metrics["empty_chunk_count"] = failures.counts["empty_chunk"]
        metrics["stored_prefix_error_count"] = failures.counts[
            "stored_document_prefix"
        ]
        metrics["unknown_document_count"] = failures.counts["unknown_document"]
        metrics["missing_output_document_count"] = failures.counts[
            "missing_output_document"
        ]
        metrics["failure_count"] = failures.total
        metrics["failure_counts"] = dict(sorted(failures.counts.items()))
        metrics["failure_examples"] = failures.examples
        metrics["checked_document_count"] = len(checked_documents)
        metrics["validation_duration_seconds"] = time.perf_counter() - started
        metrics["generated_at"] = _utc_now()

    def flush_batch() -> None:
        if not batch:
            return
        texts = [str(item["record"]["texto"]) for item in batch]
        counts = token_counter.count_many(
            texts,
            include_document_prefix=True,
            batch_size=args.validation_batch_size,
        )
        for item, actual_count in zip(batch, counts, strict=True):
            record = item["record"]
            maximum_num_tokens = metrics["maximum_num_tokens"]
            metrics["maximum_num_tokens"] = (
                actual_count
                if maximum_num_tokens is None
                else max(int(maximum_num_tokens), actual_count)
            )
            if actual_count <= config["chunk_max_tokens"]:
                metrics["chunks_le_soft_limit"] = (
                    int(metrics["chunks_le_soft_limit"]) + 1
                )
            elif actual_count <= capacity.effective_content_max_tokens:
                metrics["chunks_gt_soft_limit_but_le_hard_limit"] = (
                    int(metrics["chunks_gt_soft_limit_but_le_hard_limit"]) + 1
                )
            else:
                metrics["chunks_gt_hard_limit"] = (
                    int(metrics["chunks_gt_hard_limit"]) + 1
                )
                item["valid"] = False
                failures.add(
                    "encoder_hard_limit_exceeded",
                    line=item["line"],
                    chunk_id=record.get("chunk_id"),
                    actual=actual_count,
                    hard_limit=capacity.effective_content_max_tokens,
                )
            if record.get("num_tokens") != actual_count:
                item["valid"] = False
                failures.add(
                    "token_count_mismatch",
                    line=item["line"],
                    chunk_id=record.get("chunk_id"),
                    stored=record.get("num_tokens"),
                    actual=actual_count,
                )
            key = "valid_chunk_count" if item["valid"] else "invalid_chunk_count"
            metrics[key] = int(metrics[key]) + 1
        batch.clear()
        update_failure_metrics()
        _write_json_atomic(metrics_path, metrics)

    def validate_hard_split_groups() -> None:
        """Validate complete parent traces after every part has been recounted."""
        for parent_chunk_id, items in hard_split_groups.items():
            ordered = sorted(
                items,
                key=lambda item: int(
                    item["record"]["hard_split_part_index"]
                ),
            )
            records = [item["record"] for item in ordered]
            expected_part_count = records[0]["hard_split_part_count"]
            group_valid = (
                len(records) == expected_part_count
                and [record["hard_split_part_index"] for record in records]
                == list(range(expected_part_count))
                and len({record["doc_id"] for record in records}) == 1
                and len(
                    {record["hard_split_strategy"] for record in records}
                )
                == 1
                and len(
                    {
                        record["hard_split_parent_num_tokens"]
                        for record in records
                    }
                )
                == 1
            )
            reconstructed_parts: list[str] = []
            previous_end = 0
            for record in records:
                source_start = record["hard_split_source_start"]
                source_end = record["hard_split_source_end"]
                emitted_start = record["hard_split_emitted_start"]
                text = record["texto"]
                local_start = source_start - emitted_start
                local_end = source_end - emitted_start
                if (
                    source_start != previous_end
                    or not isinstance(text, str)
                    or local_start < 0
                    or local_end <= local_start
                    or local_end > len(text)
                ):
                    group_valid = False
                    break
                reconstructed_parts.append(text[local_start:local_end])
                previous_end = source_end
            if group_valid:
                reconstructed = "".join(reconstructed_parts)
                group_valid = (
                    token_counter.count(
                        reconstructed,
                        include_document_prefix=True,
                    )
                    == records[0]["hard_split_parent_num_tokens"]
                )
            if group_valid:
                continue
            failures.add(
                "hard_split_metadata_error",
                parent_chunk_id=parent_chunk_id,
                message="hard-split parts do not form one complete parent trace.",
            )
            for item in items:
                if item["valid"]:
                    item["valid"] = False
                    metrics["valid_chunk_count"] = (
                        int(metrics["valid_chunk_count"]) - 1
                    )
                    metrics["invalid_chunk_count"] = (
                        int(metrics["invalid_chunk_count"]) + 1
                    )

    try:
        with chunks_path.open("r", encoding="utf-8", errors="strict") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    metrics["invalid_chunk_count"] = (
                        int(metrics["invalid_chunk_count"]) + 1
                    )
                    failures.add("empty_jsonl_line", line=line_number)
                    continue
                try:
                    value: object = json.loads(line)
                except json.JSONDecodeError as error:
                    metrics["invalid_chunk_count"] = (
                        int(metrics["invalid_chunk_count"]) + 1
                    )
                    failures.add(
                        "malformed_json",
                        line=line_number,
                        message=error.msg,
                    )
                    continue
                if not isinstance(value, dict):
                    metrics["invalid_chunk_count"] = (
                        int(metrics["invalid_chunk_count"]) + 1
                    )
                    failures.add("non_object_chunk", line=line_number)
                    continue
                record: dict[str, Any] = value
                valid = True
                trace_parent_id: str | None = None
                missing = sorted(REQUIRED_CHUNK_FIELDS - record.keys())
                if missing:
                    valid = False
                    failures.add(
                        "missing_chunk_fields",
                        line=line_number,
                        fields=missing,
                    )
                present_trace_fields = HARD_SPLIT_TRACE_FIELDS & record.keys()
                if present_trace_fields:
                    trace_valid = record.get("hard_split") is True
                    trace_valid = trace_valid and not (
                        HARD_SPLIT_TRACE_FIELDS - record.keys()
                    )
                    integer_trace_fields = (
                        "hard_split_part_index",
                        "hard_split_part_count",
                        "hard_split_overlap_tokens",
                        "hard_split_parent_num_tokens",
                        "hard_split_source_start",
                        "hard_split_source_end",
                        "hard_split_emitted_start",
                    )
                    trace_valid = trace_valid and all(
                        not isinstance(record.get(field), bool)
                        and isinstance(record.get(field), int)
                        for field in integer_trace_fields
                    )
                    part_index = record.get("hard_split_part_index")
                    part_count = record.get("hard_split_part_count")
                    overlap_count = record.get("hard_split_overlap_tokens")
                    parent_count = record.get("hard_split_parent_num_tokens")
                    source_start = record.get("hard_split_source_start")
                    source_end = record.get("hard_split_source_end")
                    emitted_start = record.get("hard_split_emitted_start")
                    trace_valid = trace_valid and (
                        isinstance(record.get("parent_chunk_id"), str)
                        and bool(record.get("parent_chunk_id"))
                        and isinstance(record.get("hard_split_strategy"), str)
                        and bool(record.get("hard_split_strategy"))
                        and isinstance(part_index, int)
                        and isinstance(part_count, int)
                        and 0 <= part_index < part_count
                        and part_count >= 2
                        and isinstance(overlap_count, int)
                        and 0 <= overlap_count <= config["overlap_tokens"]
                        and isinstance(parent_count, int)
                        and parent_count > capacity.effective_content_max_tokens
                        and isinstance(source_start, int)
                        and isinstance(source_end, int)
                        and isinstance(emitted_start, int)
                        and 0 <= emitted_start <= source_start < source_end
                    )
                    if not trace_valid:
                        valid = False
                        failures.add(
                            "hard_split_metadata_error",
                            line=line_number,
                            chunk_id=record.get("chunk_id"),
                        )
                    else:
                        trace_parent_id = str(record["parent_chunk_id"])
                doc_id = record.get("doc_id")
                chunk_id = record.get("chunk_id")
                if isinstance(chunk_id, str):
                    if chunk_id in unique_chunk_ids:
                        valid = False
                        failures.add(
                            "duplicate_chunk_id",
                            line=line_number,
                            chunk_id=chunk_id,
                        )
                    unique_chunk_ids.add(chunk_id)
                else:
                    valid = False
                    failures.add(
                        "invalid_chunk_id",
                        line=line_number,
                        chunk_id=chunk_id,
                    )
                for field in ("fuente", "formato"):
                    field_value = record.get(field)
                    if not isinstance(field_value, str) or not field_value.strip():
                        valid = False
                        failures.add(
                            "invalid_field_type",
                            line=line_number,
                            field=field,
                        )
                for field in ("fenomeno", "posicion", "num_tokens"):
                    field_value = record.get(field)
                    if (
                        isinstance(field_value, bool)
                        or not isinstance(field_value, int)
                        or field_value < 0
                    ):
                        valid = False
                        failures.add(
                            "invalid_field_type",
                            line=line_number,
                            field=field,
                        )
                if isinstance(doc_id, str) and doc_id in source_contract:
                    checked_documents.add(doc_id)
                    source_index, fuente, formato, fenomeno = source_contract[doc_id]
                    if source_index < last_source_index:
                        valid = False
                        failures.add(
                            "document_order_error",
                            line=line_number,
                            doc_id=doc_id,
                        )
                    last_source_index = max(last_source_index, source_index)
                    expected_position = expected_positions[doc_id]
                    if record.get("posicion") != expected_position:
                        valid = False
                        failures.add(
                            "position_error",
                            line=line_number,
                            doc_id=doc_id,
                            expected=expected_position,
                            actual=record.get("posicion"),
                        )
                    expected_chunk_id = (
                        f"{doc_id}-chunk-{expected_position:04d}"
                    )
                    if chunk_id != expected_chunk_id:
                        valid = False
                        failures.add(
                            "chunk_id_format_error",
                            line=line_number,
                            doc_id=doc_id,
                            expected=expected_chunk_id,
                            actual=chunk_id,
                        )
                    expected_positions[doc_id] += 1
                    expected_metadata = {
                        "fuente": fuente,
                        "formato": formato,
                        "fenomeno": fenomeno,
                    }
                    for field, expected in expected_metadata.items():
                        if record.get(field) != expected:
                            valid = False
                            failures.add(
                                "metadata_error",
                                line=line_number,
                                doc_id=doc_id,
                                field=field,
                                expected=expected,
                                actual=record.get(field),
                            )
                else:
                    valid = False
                    failures.add(
                        "unknown_document",
                        line=line_number,
                        doc_id=doc_id,
                    )
                text = record.get("texto")
                if not isinstance(text, str) or not text.strip():
                    valid = False
                    failures.add(
                        "empty_chunk",
                        line=line_number,
                        chunk_id=chunk_id,
                    )
                elif text.startswith(config["document_prefix"]):
                    valid = False
                    failures.add(
                        "stored_document_prefix",
                        line=line_number,
                        chunk_id=chunk_id,
                    )
                if isinstance(text, str) and text.strip():
                    batch_item = {
                        "line": line_number,
                        "record": record,
                        "valid": valid,
                    }
                    batch.append(batch_item)
                    if trace_parent_id is not None:
                        hard_split_groups.setdefault(trace_parent_id, []).append(
                            batch_item
                        )
                    if len(batch) >= args.validation_batch_size:
                        flush_batch()
                else:
                    key = "valid_chunk_count" if valid else "invalid_chunk_count"
                    metrics[key] = int(metrics[key]) + 1
        flush_batch()
        validate_hard_split_groups()
        missing_document_ids = sorted(
            set(source_contract) - checked_documents,
            key=lambda doc_id: source_contract[doc_id][0],
        )
        for missing_doc_id in missing_document_ids:
            failures.add(
                "missing_output_document",
                doc_id=missing_doc_id,
            )
        source_identity_after = _file_identity(source_path)
        chunks_identity_after = _file_identity(chunks_path)
        metrics["source_identity_after"] = source_identity_after
        metrics["chunks_identity_after"] = chunks_identity_after
        if source_identity_after != source_identity_before:
            failures.add("source_changed_during_validation")
        if chunks_identity_after != chunks_identity_before:
            failures.add("chunks_changed_during_validation")
        update_failure_metrics()
        metrics["completion_status"] = (
            "completed_with_failures" if failures.total else "completed"
        )
    except KeyboardInterrupt:
        update_failure_metrics()
        metrics["completion_status"] = "interrupted"
        failures.add("keyboard_interrupt")
        update_failure_metrics()
        _attempt_secondary_persistence(
            lambda: _write_json_atomic(metrics_path, metrics),
            description="interrupted validation metrics",
        )
        LOGGER.error("Validation interrupted; partial metrics were preserved.")
        return 130
    except Exception as error:
        failures.add(type(error).__name__, message=str(error))
        update_failure_metrics()
        metrics["completion_status"] = "failed"
        _attempt_secondary_persistence(
            lambda: _write_json_atomic(metrics_path, metrics),
            description="failed validation metrics",
        )
        LOGGER.exception("Validation failed; partial metrics were preserved.")
        return 1
    _write_json_atomic(metrics_path, metrics)
    LOGGER.info(
        "Validation completed: valid=%d invalid=%d failures=%d.",
        metrics["valid_chunk_count"],
        metrics["invalid_chunk_count"],
        metrics["failure_count"],
    )
    return 0 if not failures.total else 2


def _load_manifest(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("benchmark manifest must be a JSON object.")
    doc_ids = value.get("doc_ids")
    if not isinstance(doc_ids, list) or not doc_ids or not all(
        isinstance(doc_id, str) and doc_id for doc_id in doc_ids
    ):
        raise ValueError("benchmark manifest doc_ids must be a non-empty list.")
    return value


def _prepare_benchmark_sample(
    source_path: Path,
    manifest: dict[str, Any],
    destination_path: Path,
) -> int:
    requested = set(manifest["doc_ids"])
    selected: list[tuple[int, dict[str, object]]] = []
    for line_number, document in read_documents(source_path):
        if document.get("doc_id") in requested:
            selected.append((line_number, document))
    found = {str(document["doc_id"]) for _, document in selected}
    missing = sorted(requested - found)
    if missing:
        raise ValueError(f"benchmark manifest references missing doc_ids: {missing}.")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    ready_for_publication = False
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for _, document in selected:
                temporary.write(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        ready_for_publication = True
        _atomic_replace_with_retry(temporary_path, destination_path)
    except BaseException:
        if temporary_path is not None and not ready_for_publication:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "Could not remove partial benchmark temporary %s while "
                    "preserving the active exception.",
                    temporary_path,
                    exc_info=True,
                )
        elif temporary_path is not None:
            LOGGER.error(
                "Benchmark sample publication failed; the complete temporary "
                "was preserved at %s.",
                temporary_path,
            )
        raise
    return len(selected)


def _refresh_benchmark_statistics(metrics: dict[str, Any]) -> None:
    durations = [
        float(repetition["duration_seconds"])
        for repetition in metrics["repetitions"]
    ]
    metrics["completed_repetitions"] = len(durations)
    metrics["duration_per_repetition"] = durations
    metrics["minimum_duration"] = min(durations) if durations else None
    metrics["maximum_duration"] = max(durations) if durations else None
    metrics["average_duration"] = statistics.fmean(durations) if durations else None
    metrics["median_duration"] = statistics.median(durations) if durations else None
    metrics["standard_deviation"] = (
        statistics.pstdev(durations) if durations else None
    )
    hashes = [
        str(repetition["output_sha256"])
        for repetition in metrics["repetitions"]
    ]
    metrics["output_hash_per_repetition"] = hashes
    metrics["deterministic_output"] = (
        len(set(hashes)) == 1 if len(hashes) >= 2 else None
    )
    metrics["generated_at"] = _utc_now()


def _benchmark_work_directory(args: argparse.Namespace) -> Path:
    """Return a stable isolated default directory for one metrics artifact."""
    if args.work_dir is not None:
        configured_work_dir: Path = args.work_dir
        return configured_work_dir
    key = hashlib.sha256(
        str(args.metrics_output.resolve()).encode("utf-8")
    ).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "nerv_chunking_benchmark" / key


def _benchmark_command_unlocked(
    args: argparse.Namespace,
    *,
    token_counter: TokenCounter | None = None,
    pipeline_runner: Callable[..., PipelineSummary] = run_pipeline,
) -> int:
    """Benchmark a manifest-selected sample and checkpoint each repetition."""
    config = load_encoder_config()
    work_dir = _benchmark_work_directory(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    benchmark_output = work_dir / "benchmark_chunks.jsonl"
    benchmark_config = work_dir / "benchmark_chunking_config.json"
    metrics_path: Path = args.metrics_output
    manifest = (
        {
            "schema_version": 1,
            "description": "Explicit full-corpus benchmark.",
            "doc_ids": [],
        }
        if args.full_corpus
        else _load_manifest(args.manifest)
    )
    if args.full_corpus:
        LOGGER.warning(
            "A full-corpus benchmark was explicitly requested. This is expensive."
        )
        benchmark_input = args.input
        document_count = _count_documents(benchmark_input)
    else:
        benchmark_input = work_dir / "benchmark_documents.jsonl"
        _require_distinct_paths(
            input_path=args.input,
            manifest_path=args.manifest,
            metrics_output=metrics_path,
            benchmark_input=benchmark_input,
            benchmark_output=benchmark_output,
            benchmark_config=benchmark_config,
        )
        document_count = _prepare_benchmark_sample(
            args.input,
            manifest,
            benchmark_input,
        )
    if args.full_corpus:
        _require_distinct_paths(
            input_path=args.input,
            metrics_output=metrics_path,
            benchmark_output=benchmark_output,
            benchmark_config=benchmark_config,
        )
    if token_counter is None:
        token_counter = _load_counter(args, config)
    capacity = _resolve_capacity(
        token_counter,
        config,
        local_files_only=args.local_files_only,
    )
    compatibility = {
        "source_identity": _file_identity(args.input),
        "benchmark_input_sha256": _sha256(benchmark_input),
        "configuration_fingerprint": _configuration_fingerprint(config),
        "workflow_code_fingerprint": _workflow_code_fingerprint(),
        "dependency_versions": _dependency_versions(),
        "environment": _environment_summary(),
        "work_directory": str(work_dir.resolve()),
        "tokenizer": {
            "model_name": token_counter.model_name,
            "revision": token_counter.revision,
            "class": token_counter.tokenizer_class,
            "add_special_tokens": token_counter.add_special_tokens,
        },
        "manifest": str(args.manifest) if not args.full_corpus else None,
        "manifest_sha256": (
            _sha256(args.manifest) if not args.full_corpus else None
        ),
        "full_corpus": args.full_corpus,
        "requested_repetitions": args.repetitions,
    }
    metrics: dict[str, Any]
    if metrics_path.exists():
        previous: object = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            isinstance(previous, dict)
            and previous.get("compatibility") == compatibility
        ):
            metrics = previous
            LOGGER.info(
                "Resuming benchmark after %d measured repetitions.",
                len(metrics.get("repetitions", [])),
            )
            metrics["completion_status"] = "running"
            metrics["resumed_at"] = _utc_now()
            metrics["resume_from_repetition"] = len(
                metrics.get("repetitions", [])
            )
            _write_json_atomic(metrics_path, metrics)
        else:
            metrics = {}
    else:
        metrics = {}
    if not metrics:
        metrics = {
            **_base_run_metrics(
                mode="benchmark",
                source_path=args.input,
                config=config,
            ),
            "compatibility": compatibility,
            "manifest": manifest,
            "benchmark_input_path": str(benchmark_input),
            "benchmark_output_path": str(benchmark_output),
            "work_directory": str(work_dir),
            "warmup_completed": False,
            "warmup_duration_seconds": None,
            "completed_repetitions": 0,
            "requested_repetitions": args.repetitions,
            "repetitions": [],
            "duration_per_repetition": [],
            "minimum_duration": None,
            "maximum_duration": None,
            "average_duration": None,
            "median_duration": None,
            "standard_deviation": None,
            "documents_per_second": [],
            "chunks_per_second": [],
            "bytes_per_second": [],
            "output_hash_per_repetition": [],
            "deterministic_output": None,
            "completion_status": "running",
            "resumed_at": None,
            "resume_from_repetition": 0,
        }
        _write_json_atomic(metrics_path, metrics)

    def execute_once() -> tuple[PipelineSummary, float, str, int]:
        started = time.perf_counter()
        summary = pipeline_runner(
            benchmark_input,
            benchmark_output,
            token_counter=token_counter,
            max_tokens=config["chunk_max_tokens"],
            overlap_tokens=config["overlap_tokens"],
            encoder_max_input_tokens=capacity.effective_content_max_tokens,
            config_path=benchmark_config,
            total_document_count=document_count,
            global_progress_interval=args.global_progress_interval,
            work_dir=work_dir,
        )
        duration = time.perf_counter() - started
        return (
            summary,
            duration,
            _sha256(benchmark_output),
            benchmark_output.stat().st_size,
        )

    try:
        if not metrics["warmup_completed"]:
            _, warmup_duration, _, _ = execute_once()
            metrics["warmup_completed"] = True
            metrics["warmup_duration_seconds"] = warmup_duration
            metrics["generated_at"] = _utc_now()
            _write_json_atomic(metrics_path, metrics)
        while len(metrics["repetitions"]) < args.repetitions:
            repetition_number = len(metrics["repetitions"]) + 1
            summary, duration, output_hash, output_size = execute_once()
            metrics["repetitions"].append(
                {
                    "repetition": repetition_number,
                    "duration_seconds": duration,
                    "document_count": summary.document_count,
                    "chunk_count": summary.chunk_count,
                    "documents_per_second": summary.documents_per_second,
                    "chunks_per_second": summary.chunks_per_second,
                    "bytes_per_second": output_size / duration if duration else 0.0,
                    "output_size_bytes": output_size,
                    "output_sha256": output_hash,
                    "maximum_tokens_per_chunk": (
                        summary.maximum_tokens_per_chunk
                    ),
                    "hard_split_source_unit_count": (
                        summary.hard_split_source_unit_count
                    ),
                    "hard_split_generated_chunk_count": (
                        summary.hard_split_generated_chunk_count
                    ),
                    "hard_split_by_format": summary.hard_split_by_format,
                    "hard_split_by_strategy": summary.hard_split_by_strategy,
                    "completed_at": _utc_now(),
                }
            )
            metrics["documents_per_second"].append(summary.documents_per_second)
            metrics["chunks_per_second"].append(summary.chunks_per_second)
            metrics["bytes_per_second"].append(
                output_size / duration if duration else 0.0
            )
            _refresh_benchmark_statistics(metrics)
            _write_json_atomic(metrics_path, metrics)
    except KeyboardInterrupt:
        _refresh_benchmark_statistics(metrics)
        metrics["completion_status"] = "interrupted"
        _attempt_secondary_persistence(
            lambda: _write_json_atomic(metrics_path, metrics),
            description="interrupted benchmark metrics",
        )
        LOGGER.error("Benchmark interrupted; completed repetitions were preserved.")
        return 130
    except Exception as error:
        _refresh_benchmark_statistics(metrics)
        metrics["completion_status"] = "failed"
        metrics["failures"].append(
            {"category": type(error).__name__, "message": str(error)}
        )
        _attempt_secondary_persistence(
            lambda: _write_json_atomic(metrics_path, metrics),
            description="failed benchmark metrics",
        )
        LOGGER.exception("Benchmark failed; completed repetitions were preserved.")
        return 1
    _refresh_benchmark_statistics(metrics)
    metrics["completion_status"] = "completed"
    _write_json_atomic(metrics_path, metrics)
    LOGGER.info(
        "Benchmark completed: repetitions=%d deterministic=%s.",
        metrics["completed_repetitions"],
        metrics["deterministic_output"],
    )
    return 0


def _save_benchmark_preparation_failure(
    args: argparse.Namespace,
    *,
    status: str,
    error: BaseException,
) -> None:
    """Replace absent, stale, or corrupt setup state with a failed checkpoint."""
    metrics_path: Path = args.metrics_output
    previous: dict[str, Any] = {}
    if metrics_path.is_file():
        try:
            decoded: object = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(decoded, dict):
                previous = decoded
        except (OSError, UnicodeError, json.JSONDecodeError):
            previous = {}
    failures = previous.get("failures")
    if not isinstance(failures, list):
        failures = []
    failures.append(
        {
            "category": type(error).__name__,
            "message": str(error),
            "stage": "preparation",
        }
    )
    source_identity = _optional_file_identity(args.input)
    previous.update(
        {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "mode": "benchmark",
            "source_path": str(args.input),
            "source_identity": source_identity,
            "manifest_path": None if args.full_corpus else str(args.manifest),
            "work_directory": str(_benchmark_work_directory(args)),
            "requested_repetitions": args.repetitions,
            "completion_status": status,
            "failure_stage": "preparation",
            "failures": failures,
        }
    )
    _write_json_atomic(metrics_path, previous)


def benchmark_command(
    args: argparse.Namespace,
    *,
    token_counter: TokenCounter | None = None,
    pipeline_runner: Callable[..., PipelineSummary] = run_pipeline,
) -> int:
    """Run one benchmark owner at a time for the selected work directory."""
    work_dir = _benchmark_work_directory(args)
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / ".benchmark.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError:
        LOGGER.error(
            "Benchmark work directory is already locked: %s. Verify that no "
            "benchmark is active before removing a stale lock.",
            lock_path,
        )
        return 1
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as lock_file:
            lock_file.write(f"pid={os.getpid()} started_at={_utc_now()}\n")
            lock_file.flush()
            os.fsync(lock_file.fileno())
        try:
            return _benchmark_command_unlocked(
                args,
                token_counter=token_counter,
                pipeline_runner=pipeline_runner,
            )
        except KeyboardInterrupt as error:
            interrupt_error = error
            _attempt_secondary_persistence(
                lambda: _save_benchmark_preparation_failure(
                    args,
                    status="interrupted",
                    error=interrupt_error,
                ),
                description="interrupted benchmark preparation metrics",
            )
            LOGGER.error(
                "Benchmark preparation was interrupted; metrics were preserved."
            )
            return 130
        except Exception as error:
            preparation_error = error
            _attempt_secondary_persistence(
                lambda: _save_benchmark_preparation_failure(
                    args,
                    status="failed",
                    error=preparation_error,
                ),
                description="failed benchmark preparation metrics",
            )
            LOGGER.exception(
                "Benchmark preparation failed; metrics were preserved."
            )
            return 1
    finally:
        lock_path.unlink(missing_ok=True)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Input documentos.jsonl path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the frozen tokenizer to be available in the local cache.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Shared UTF-8 log path for all chunking components.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level for console and file handlers.",
    )


def _add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Add options used only by pipeline-executing commands."""
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Local directory for large incomplete temporary files.",
    )
    parser.add_argument(
        "--global-progress-interval",
        type=int,
        default=25,
        help="Emit global progress after this many documents.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the independent real-corpus command structure."""
    parser = argparse.ArgumentParser(
        description="Produce, validate, or benchmark NERV chunking independently."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    produce_parser = subparsers.add_parser(
        "produce",
        help="Run exactly one production pass and persist progress.",
    )
    _add_common_arguments(produce_parser)
    _add_execution_arguments(produce_parser)
    produce_parser.add_argument("--output", type=Path, default=DEFAULT_CHUNKS)
    produce_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_PRODUCTION_METRICS,
    )
    produce_parser.add_argument(
        "--progress-output",
        type=Path,
        default=DEFAULT_PROGRESS,
    )
    produce_parser.add_argument(
        "--config-output",
        type=Path,
        default=DEFAULT_CONFIG_OUTPUT,
    )
    produce_parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Use the Phase-1 spawn architecture. Only --workers 1 is enabled; "
            "omit this option for the unchanged sequential path."
        ),
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate existing chunks without running production.",
    )
    _add_common_arguments(validate_parser)
    validate_parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    validate_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_VALIDATION_METRICS,
    )
    validate_parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=256,
        help="Maximum number of chunk texts tokenized in one validation batch.",
    )
    validate_parser.add_argument(
        "--failure-example-limit",
        type=int,
        default=100,
        help="Maximum stored examples; complete failure counters are retained.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark a representative manifest-selected sample.",
    )
    _add_common_arguments(benchmark_parser)
    _add_execution_arguments(benchmark_parser)
    benchmark_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    benchmark_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_BENCHMARK_METRICS,
    )
    benchmark_parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Measured repetitions after one warm-up.",
    )
    benchmark_parser.add_argument(
        "--full-corpus",
        action="store_true",
        help="Benchmark the full corpus. WARNING: this is extremely expensive.",
    )
    args = parser.parse_args(argv)
    if args.command != "validate" and args.global_progress_interval <= 0:
        parser.error("--global-progress-interval must be greater than zero.")
    if args.command == "produce" and args.workers is not None:
        if args.workers != 1:
            parser.error(
                "Phase 1 enables only --workers 1; concurrent workers are not "
                "enabled until a later phase."
            )
        if args.work_dir is None:
            parser.error("Phase 1 requires --work-dir with --workers 1.")
    if args.command == "validate":
        if args.validation_batch_size <= 0:
            parser.error("--validation-batch-size must be greater than zero.")
        if args.failure_example_limit < 0:
            parser.error("--failure-example-limit cannot be negative.")
    if args.command == "benchmark" and args.repetitions <= 0:
        parser.error("--repetitions must be greater than zero.")
    return args


def _validate_cli_artifact_paths(args: argparse.Namespace) -> None:
    """Protect command inputs and outputs before logging opens its file."""
    paths = {
        "input_path": args.input,
        "log_path": args.log_path,
        "metrics_output": args.metrics_output,
    }
    if args.command == "produce":
        paths.update(
            {
                "output_path": args.output,
                "progress_output": args.progress_output,
                "config_output": args.config_output,
            }
        )
    elif args.command == "validate":
        paths["chunks_path"] = args.chunks
    else:
        work_dir = _benchmark_work_directory(args)
        paths.update(
            {
                "benchmark_output": work_dir / "benchmark_chunks.jsonl",
                "benchmark_config": work_dir / "benchmark_chunking_config.json",
                "benchmark_lock": work_dir / ".benchmark.lock",
            }
        )
        if not args.full_corpus:
            paths["manifest_path"] = args.manifest
            paths["benchmark_input"] = work_dir / "benchmark_documents.jsonl"
    _require_distinct_paths(**paths)


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Configure shared logging and dispatch exactly one command."""
    args = parse_args(argv)
    try:
        _validate_cli_artifact_paths(args)
    except ValueError as error:
        print(f"Real-corpus command rejected: {error}", file=sys.stderr)
        return 2
    configure_logging(args.log_path, level=getattr(logging, args.log_level))
    try:
        if args.command == "produce":
            return produce_command(args)
        if args.command == "validate":
            return validate_command(args)
        return benchmark_command(args)
    except KeyboardInterrupt:
        LOGGER.error("Command interrupted before mode-specific state was available.")
        return 130
    except Exception:
        LOGGER.exception("Command failed before mode-specific recovery completed.")
        return 1


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the selected command and return a meaningful process status."""
    raise SystemExit(run_cli(argv))


if __name__ == "__main__":
    main()
