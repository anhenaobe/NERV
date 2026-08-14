"""Spawn-safe Phase-1 document processing with one persistent worker."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, cast

from .configuration import EncoderCapacity, EncoderConfig, resolve_encoder_capacity
from .language_detector import _load_language_markers
from .pipeline import (
    Document,
    PipelineDocumentProgress,
    PipelineSummary,
    ProgressCallback,
    TemporaryPathCallback,
    _atomic_text_writer,
    _json_safe_value,
    _package_version,
    _process_document,
    _publish_completed_file,
    _validate_pipeline_configuration,
    read_documents,
    validate_document,
)
from .sentence_splitter import (
    _PYSBD_LANGUAGE_CODES,
    _get_pysbd_segmenter,
    split_sentences,
)
from .token_counter import TokenCounter

LOGGER = logging.getLogger(__name__)
PHASE_1_WORKER_COUNT = 1
MANIFEST_SCHEMA_VERSION = 1

TokenCounterFactory = Callable[[EncoderConfig, bool], TokenCounter]
ArchitectureMetricsCallback = Callable[[dict[str, object]], None]


@dataclass(frozen=True)
class WorkerMetadata:
    """Compact metadata for one initialized persistent worker."""

    pid: int
    initialization_seconds: float
    encoder_model_name: str
    tokenizer_class: str
    tokenizer_revision: str | None
    use_fast: bool
    trust_remote_code: bool
    add_special_tokens: bool
    document_prefix: str
    local_files_only: bool
    reported_model_max_length: object
    encoder_special_token_overhead: int
    effective_content_max_tokens: int


@dataclass(frozen=True)
class WorkerDocumentResult:
    """Bounded manifest for one closed per-document JSONL temporary."""

    schema_version: int
    document_index: int
    doc_id: str
    document_format: str
    temporary_path: str
    output_byte_count: int
    output_sha256: str
    record_count: int
    sentence_count: int
    detected_language: str | None
    oversized_chunk_count: int
    token_total: int
    minimum_tokens_per_chunk: int | None
    maximum_tokens_per_chunk: int | None
    hard_split_source_unit_count: int
    hard_split_generated_chunk_count: int
    hard_split_by_format: dict[str, int]
    hard_split_by_strategy: dict[str, int]
    language_detection_seconds: float
    splitting_seconds: float
    chunking_seconds: float
    record_construction_seconds: float
    worker_temp_write_seconds: float
    worker_service_seconds: float
    worker_pid: int
    character_count: int
    line_count: int
    bounded_block_count: int | None = None
    bounded_largest_block_characters: int | None = None
    bounded_maximum_carry_characters: int | None = None
    bounded_carry_warning_count: int | None = None


@dataclass(frozen=True)
class ParallelPipelineMetrics:
    """Small aggregate measurements for the Phase-1 orchestration."""

    multiprocessing_enabled: bool
    worker_count: int
    start_method: str
    workers_initialized: int
    worker_initialization_seconds: float
    worker_task_count: int
    parallel_processing_wall_seconds: float
    worker_service_seconds_sum: float
    worker_temp_bytes_total: int
    worker_temp_write_seconds: float
    ordered_merge_seconds: float
    merged_document_count: int
    last_merged_document_index: int
    peak_pending_documents: int
    peak_pending_bytes: int
    output_sha256: str


@dataclass(frozen=True)
class ParallelPipelineRun:
    """Successful Phase-1 summary plus architecture-specific metrics."""

    summary: PipelineSummary
    metrics: ParallelPipelineMetrics


@dataclass
class _WorkerState:
    config: EncoderConfig
    config_fingerprint: str
    token_counter: TokenCounter
    metadata: WorkerMetadata
    capacity: EncoderCapacity


_WORKER_STATE: _WorkerState | None = None
_WORKER_INITIALIZATION_ERROR: tuple[str, str] | None = None


def configuration_fingerprint(config: EncoderConfig) -> str:
    """Return a deterministic fingerprint for immutable worker configuration."""
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _create_production_counter(
    config: EncoderConfig,
    local_files_only: bool,
) -> TokenCounter:
    """Load and capacity-check the canonical tokenizer inside a worker."""
    counter = TokenCounter(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
        local_files_only=local_files_only,
    )
    resolve_encoder_capacity(
        config=config,
        tokenizer_model_max_length=counter.reported_model_max_length,
        encoder_special_token_overhead=counter.encoder_special_token_overhead,
        local_files_only=local_files_only,
    )
    return counter


def initialize_worker(
    config: EncoderConfig,
    expected_config_fingerprint: str,
    local_files_only: bool,
    counter_factory: TokenCounterFactory = _create_production_counter,
) -> None:
    """Initialize one process-local tokenizer and language resources exactly once."""
    global _WORKER_INITIALIZATION_ERROR, _WORKER_STATE
    started = time.perf_counter()
    _WORKER_STATE = None
    _WORKER_INITIALIZATION_ERROR = None
    try:
        frozen_config = cast(EncoderConfig, dict(config))
        actual_fingerprint = configuration_fingerprint(frozen_config)
        if actual_fingerprint != expected_config_fingerprint:
            raise ValueError("worker configuration fingerprint does not match.")
        counter = counter_factory(frozen_config, local_files_only)
        capacity = resolve_encoder_capacity(
            config=frozen_config,
            tokenizer_model_max_length=counter.reported_model_max_length,
            encoder_special_token_overhead=(
                counter.encoder_special_token_overhead
            ),
            local_files_only=local_files_only,
        )
        _load_language_markers()
        _get_pysbd_segmenter("es")
        _get_pysbd_segmenter("en")
        metadata = WorkerMetadata(
            pid=os.getpid(),
            initialization_seconds=time.perf_counter() - started,
            encoder_model_name=counter.model_name,
            tokenizer_class=counter.tokenizer_class,
            tokenizer_revision=counter.revision,
            use_fast=counter.use_fast,
            trust_remote_code=counter.trust_remote_code,
            add_special_tokens=counter.add_special_tokens,
            document_prefix=counter.document_prefix,
            local_files_only=counter.local_files_only,
            reported_model_max_length=_json_safe_value(
                counter.reported_model_max_length
            ),
            encoder_special_token_overhead=(
                capacity.encoder_special_token_overhead
            ),
            effective_content_max_tokens=(
                capacity.effective_content_max_tokens
            ),
        )
        _WORKER_STATE = _WorkerState(
            config=frozen_config,
            config_fingerprint=actual_fingerprint,
            token_counter=counter,
            metadata=metadata,
            capacity=capacity,
        )
    except Exception as error:
        _WORKER_INITIALIZATION_ERROR = (
            type(error).__name__,
            str(error)[:1000],
        )


def worker_metadata_task() -> WorkerMetadata:
    """Return initialized worker metadata without exposing worker resources."""
    if _WORKER_INITIALIZATION_ERROR is not None:
        error_type, message = _WORKER_INITIALIZATION_ERROR
        raise RuntimeError(
            f"worker initialization failed: {error_type}: {message}"
        )
    if _WORKER_STATE is None:
        raise RuntimeError("worker state was not initialized.")
    return _WORKER_STATE.metadata


def _confined_document_path(
    run_documents_dir: Path,
    *,
    document_index: int,
    doc_id: str,
) -> Path:
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:12]
    name = f"{document_index:08d}-{digest}-{uuid.uuid4().hex}.jsonl.tmp"
    resolved_directory = run_documents_dir.resolve()
    path = (resolved_directory / name).resolve()
    if path.parent != resolved_directory:
        raise ValueError("document temporary path escaped the run directory.")
    return path


def process_document_task(
    document_index: int,
    document: Document,
    run_documents_dir: str,
    expected_config_fingerprint: str,
) -> WorkerDocumentResult:
    """Process one document and return only its compact closed-file manifest."""
    if _WORKER_INITIALIZATION_ERROR is not None:
        error_type, message = _WORKER_INITIALIZATION_ERROR
        raise RuntimeError(
            f"worker initialization failed: {error_type}: {message}"
        )
    state = _WORKER_STATE
    if state is None:
        raise RuntimeError("worker state was not initialized.")
    if state.config_fingerprint != expected_config_fingerprint:
        raise ValueError("task configuration fingerprint does not match worker state.")
    doc_id = str(document.get("doc_id", "unknown"))
    document_format = str(document.get("formato", "unknown"))
    text = document.get("texto")
    if not isinstance(text, str):
        raise TypeError(
            f"document_index={document_index} doc_id={doc_id}: texto must be a string."
        )
    service_started = time.perf_counter()
    temporary_path = _confined_document_path(
        Path(run_documents_dir),
        document_index=document_index,
        doc_id=doc_id,
    )
    try:
        processed = _process_document(
            document,
            sentence_splitter=split_sentences,
            token_counter=state.token_counter,
            max_tokens=state.config["chunk_max_tokens"],
            overlap_tokens=state.config["overlap_tokens"],
            encoder_max_input_tokens=(
                state.capacity.effective_content_max_tokens
            ),
        )
    except Exception as error:
        raise RuntimeError(
            f"document_index={document_index} doc_id={doc_id} failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    output_hash = hashlib.sha256()
    output_bytes = 0
    token_counts: list[int] = []
    oversized_count = 0
    write_started = time.perf_counter()
    try:
        with temporary_path.open("xb") as destination:
            for record in processed.records:
                token_count = record.get("num_tokens")
                if not isinstance(token_count, int):
                    raise TypeError("chunk field 'num_tokens' must be an integer.")
                if token_count > state.capacity.effective_content_max_tokens:
                    raise RuntimeError(
                        "refusing to write a chunk above the encoder hard limit."
                    )
                encoded = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                destination.write(encoded)
                output_hash.update(encoded)
                output_bytes += len(encoded)
                token_counts.append(token_count)
                oversized_count += int(
                    token_count > state.config["chunk_max_tokens"]
                )
            destination.flush()
            os.fsync(destination.fileno())
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"document_index={document_index} doc_id={doc_id} failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    write_seconds = time.perf_counter() - write_started
    bounded = processed.bounded_segmentation_stats
    return WorkerDocumentResult(
        schema_version=MANIFEST_SCHEMA_VERSION,
        document_index=document_index,
        doc_id=doc_id,
        document_format=document_format,
        temporary_path=str(temporary_path),
        output_byte_count=output_bytes,
        output_sha256=output_hash.hexdigest(),
        record_count=len(processed.records),
        sentence_count=len(processed.sentences),
        detected_language=processed.detected_language,
        oversized_chunk_count=oversized_count,
        token_total=sum(token_counts),
        minimum_tokens_per_chunk=min(token_counts) if token_counts else None,
        maximum_tokens_per_chunk=max(token_counts) if token_counts else None,
        hard_split_source_unit_count=processed.hard_split_source_unit_count,
        hard_split_generated_chunk_count=processed.hard_split_generated_chunk_count,
        hard_split_by_format=dict(processed.hard_split_by_format),
        hard_split_by_strategy=dict(processed.hard_split_by_strategy),
        language_detection_seconds=processed.language_detection_duration_seconds,
        splitting_seconds=processed.splitting_duration_seconds,
        chunking_seconds=processed.chunking_duration_seconds,
        record_construction_seconds=(
            processed.record_construction_duration_seconds
        ),
        worker_temp_write_seconds=write_seconds,
        worker_service_seconds=time.perf_counter() - service_started,
        worker_pid=os.getpid(),
        character_count=len(text),
        line_count=text.count("\n") + 1,
        bounded_block_count=bounded.block_count if bounded else None,
        bounded_largest_block_characters=(
            bounded.largest_block_characters if bounded else None
        ),
        bounded_maximum_carry_characters=(
            bounded.maximum_carry_characters if bounded else None
        ),
        bounded_carry_warning_count=(
            bounded.oversized_carry_warning_count if bounded else None
        ),
    )


def verify_worker_result(
    result: WorkerDocumentResult,
    *,
    run_documents_dir: Path,
) -> Path:
    """Verify schema, confinement, size, and digest before ordered merge."""
    if result.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported worker manifest schema version.")
    integer_counters = {
        "document_index": result.document_index,
        "output_byte_count": result.output_byte_count,
        "record_count": result.record_count,
        "sentence_count": result.sentence_count,
        "oversized_chunk_count": result.oversized_chunk_count,
        "token_total": result.token_total,
        "hard_split_source_unit_count": result.hard_split_source_unit_count,
        "hard_split_generated_chunk_count": result.hard_split_generated_chunk_count,
    }
    if any(value < 0 for value in integer_counters.values()):
        raise ValueError("worker manifest contains a negative counter.")
    documents_dir = run_documents_dir.resolve()
    temporary_path = Path(result.temporary_path).resolve()
    if temporary_path.parent != documents_dir:
        raise ValueError("worker temporary path is outside the run directory.")
    if not temporary_path.is_file():
        raise FileNotFoundError(
            f"worker temporary is missing for document_index={result.document_index}."
        )
    if temporary_path.stat().st_size != result.output_byte_count:
        message = (
            "worker temporary size mismatch for "
            f"document_index={result.document_index}."
        )
        raise ValueError(message)
    digest = hashlib.sha256()
    newline_count = 0
    with temporary_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
            newline_count += block.count(b"\n")
    if digest.hexdigest() != result.output_sha256:
        message = (
            "worker temporary hash mismatch for "
            f"document_index={result.document_index}."
        )
        raise ValueError(message)
    if newline_count != result.record_count:
        raise ValueError(
            "worker temporary record count does not match its manifest."
        )
    return temporary_path


def verify_result_identity(
    result: WorkerDocumentResult,
    *,
    expected_document_index: int,
    expected_doc_id: str,
) -> None:
    """Bind a returned manifest to the task that produced it."""
    if result.document_index != expected_document_index:
        raise ValueError("worker manifest returned an unexpected document index.")
    if result.doc_id != expected_doc_id:
        raise ValueError("worker manifest returned an unexpected doc_id.")


def merge_pending_results(
    pending: dict[int, WorkerDocumentResult],
    *,
    next_merge_index: int,
    destination: BinaryIO,
    run_documents_dir: Path,
    on_merged: Callable[[WorkerDocumentResult, float], None] | None = None,
) -> int:
    """Merge every available canonical result and return the next index."""
    while next_merge_index in pending:
        result = pending.pop(next_merge_index)
        started = time.perf_counter()
        temporary_path = verify_worker_result(
            result,
            run_documents_dir=run_documents_dir,
        )
        with temporary_path.open("rb") as source:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        merge_seconds = time.perf_counter() - started
        temporary_path.unlink()
        if on_merged is not None:
            on_merged(result, merge_seconds)
        next_merge_index += 1
    return next_merge_index


def _write_parallel_chunking_config(
    path: Path,
    *,
    metadata: WorkerMetadata,
    config: EncoderConfig,
    generated_at: str,
) -> None:
    configuration: dict[str, object] = {
        "encoder_model_name": metadata.encoder_model_name,
        "tokenizer_revision": metadata.tokenizer_revision,
        "tokenizer_class": metadata.tokenizer_class,
        "use_fast": metadata.use_fast,
        "trust_remote_code": metadata.trust_remote_code,
        "add_special_tokens": metadata.add_special_tokens,
        "document_prefix": metadata.document_prefix,
        "local_files_only": metadata.local_files_only,
        "reported_model_max_length": metadata.reported_model_max_length,
        "chunk_max_tokens": config["chunk_max_tokens"],
        "encoder_max_input_tokens": config["encoder_max_input_tokens"],
        "encoder_special_token_overhead": (
            metadata.encoder_special_token_overhead
        ),
        "effective_content_max_tokens": metadata.effective_content_max_tokens,
        "overlap_tokens": config["overlap_tokens"],
        "oversized_sentence_policy": (
            "preserve_through_encoder_limit_then_hierarchical_subdivide"
        ),
        "language_detector": "nerv.chunking.language_detector.detect_language",
        "splitter_backend_by_language": dict(_PYSBD_LANGUAGE_CODES),
        "generated_at": generated_at,
        "code_version": _package_version(),
    }
    with _atomic_text_writer(path) as destination:
        json.dump(configuration, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def run_parallel_pipeline_phase_1(
    input_path: Path,
    output_path: Path,
    *,
    config: EncoderConfig,
    local_files_only: bool,
    config_path: Path | None = None,
    total_document_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
    global_progress_interval: int = 25,
    work_dir: Path,
    preserve_incomplete: bool = False,
    temporary_path_callback: TemporaryPathCallback | None = None,
    counter_factory: TokenCounterFactory = _create_production_counter,
    architecture_metrics_callback: ArchitectureMetricsCallback | None = None,
) -> ParallelPipelineRun:
    """Run the parallel-ready architecture with exactly one spawn worker."""
    if global_progress_interval <= 0:
        raise ValueError("global_progress_interval must be greater than zero.")
    _validate_pipeline_configuration(
        input_path,
        output_path,
        config["chunk_max_tokens"],
        config["overlap_tokens"],
        config["encoder_max_input_tokens"],
    )
    if config_path is not None and config_path.resolve() in {
        input_path.resolve(),
        output_path.resolve(),
    }:
        raise ValueError("config_path must differ from input_path and output_path.")
    if total_document_count is not None and total_document_count < 0:
        raise ValueError("total_document_count cannot be negative.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    run_directory = work_dir / f"produce-{uuid.uuid4().hex}"
    documents_directory = run_directory / "documents"
    run_directory.mkdir(parents=True, exist_ok=False)
    documents_directory.mkdir()
    global_temporary = run_directory / "chunks.global.tmp"
    fingerprint = configuration_fingerprint(config)
    pending: dict[int, WorkerDocumentResult] = {}
    next_merge_index = 1
    document_count = 0
    chunk_count = 0
    sentence_count = 0
    oversized_count = 0
    token_total = 0
    minimum_tokens: int | None = None
    maximum_tokens: int | None = None
    hard_source_count = 0
    hard_generated_count = 0
    hard_by_format: dict[str, int] = {}
    hard_by_strategy: dict[str, int] = {}
    worker_service_sum = 0.0
    worker_temp_bytes = 0
    worker_temp_write_sum = 0.0
    merge_seconds_total = 0.0
    peak_pending_documents = 0
    peak_pending_bytes = 0
    output_hash = hashlib.sha256()
    worker_metadata: WorkerMetadata | None = None
    published = False

    def publish_architecture_metrics() -> None:
        if architecture_metrics_callback is None:
            return
        architecture_metrics_callback(
            {
                "multiprocessing_enabled": True,
                "worker_count": PHASE_1_WORKER_COUNT,
                "start_method": "spawn",
                "workers_initialized": int(worker_metadata is not None),
                "worker_initialization_seconds": (
                    worker_metadata.initialization_seconds
                    if worker_metadata is not None
                    else 0.0
                ),
                "worker_task_count": document_count,
                "parallel_processing_wall_seconds": time.perf_counter() - started,
                "worker_service_seconds_sum": worker_service_sum,
                "worker_temp_bytes_total": worker_temp_bytes,
                "worker_temp_write_seconds": worker_temp_write_sum,
                "ordered_merge_seconds": merge_seconds_total,
                "merged_document_count": document_count,
                "last_merged_document_index": document_count,
                "peak_pending_documents": peak_pending_documents,
                "peak_pending_bytes": peak_pending_bytes,
            }
        )

    def on_merged(result: WorkerDocumentResult, merge_seconds: float) -> None:
        nonlocal document_count, chunk_count, sentence_count, oversized_count
        nonlocal token_total, minimum_tokens, maximum_tokens
        nonlocal hard_source_count, hard_generated_count, worker_service_sum
        nonlocal worker_temp_bytes, worker_temp_write_sum, merge_seconds_total
        document_count += 1
        chunk_count += result.record_count
        sentence_count += result.sentence_count
        oversized_count += result.oversized_chunk_count
        token_total += result.token_total
        minimum_tokens = (
            result.minimum_tokens_per_chunk
            if minimum_tokens is None
            else min(
                minimum_tokens,
                result.minimum_tokens_per_chunk
                if result.minimum_tokens_per_chunk is not None
                else minimum_tokens,
            )
        )
        maximum_tokens = (
            result.maximum_tokens_per_chunk
            if maximum_tokens is None
            else max(
                maximum_tokens,
                result.maximum_tokens_per_chunk
                if result.maximum_tokens_per_chunk is not None
                else maximum_tokens,
            )
        )
        hard_source_count += result.hard_split_source_unit_count
        hard_generated_count += result.hard_split_generated_chunk_count
        for key, value in result.hard_split_by_format.items():
            hard_by_format[key] = hard_by_format.get(key, 0) + value
        for key, value in result.hard_split_by_strategy.items():
            hard_by_strategy[key] = hard_by_strategy.get(key, 0) + value
        worker_service_sum += result.worker_service_seconds
        worker_temp_bytes += result.output_byte_count
        worker_temp_write_sum += result.worker_temp_write_seconds
        merge_seconds_total += merge_seconds
        publish_architecture_metrics()
        elapsed = time.perf_counter() - started
        remaining = None
        if total_document_count is not None and document_count:
            remaining = elapsed / document_count * max(
                total_document_count - document_count,
                0,
            )
        if progress_callback is not None:
            progress_callback(
                PipelineDocumentProgress(
                    document_index=result.document_index,
                    total_document_count=total_document_count,
                    doc_id=result.doc_id,
                    document_format=result.document_format,
                    character_count=result.character_count,
                    line_count=result.line_count,
                    detected_language=result.detected_language,
                    language_detection_duration_seconds=(
                        result.language_detection_seconds
                    ),
                    splitting_duration_seconds=result.splitting_seconds,
                    chunking_duration_seconds=result.chunking_seconds,
                    record_construction_duration_seconds=(
                        result.record_construction_seconds
                    ),
                    writing_duration_seconds=(
                        result.worker_temp_write_seconds + merge_seconds
                    ),
                    document_duration_seconds=result.worker_service_seconds,
                    generated_chunk_count=result.record_count,
                    oversized_chunk_count=result.oversized_chunk_count,
                    processed_document_count=document_count,
                    total_generated_chunk_count=chunk_count,
                    elapsed_time_seconds=elapsed,
                    estimated_remaining_seconds=remaining,
                    hard_split_source_unit_count=(
                        result.hard_split_source_unit_count
                    ),
                    hard_split_generated_chunk_count=(
                        result.hard_split_generated_chunk_count
                    ),
                    hard_split_by_format=dict(result.hard_split_by_format),
                    hard_split_by_strategy=dict(result.hard_split_by_strategy),
                )
            )

    try:
        if temporary_path_callback is not None:
            temporary_path_callback(global_temporary)
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=PHASE_1_WORKER_COUNT,
            initializer=initialize_worker,
            initargs=(config, fingerprint, local_files_only, counter_factory),
        ) as pool:
            worker_metadata = pool.apply(worker_metadata_task)
            publish_architecture_metrics()
            with global_temporary.open("xb") as destination:
                for document_index, (line_number, document) in enumerate(
                    read_documents(input_path),
                    start=1,
                ):
                    validate_document(document, line_number=line_number)
                    result = pool.apply(
                        process_document_task,
                        (
                            document_index,
                            document,
                            str(documents_directory),
                            fingerprint,
                        ),
                    )
                    expected_doc_id = str(document["doc_id"])
                    verify_result_identity(
                        result,
                        expected_document_index=document_index,
                        expected_doc_id=expected_doc_id,
                    )
                    if result.document_index in pending:
                        raise ValueError(
                            "worker manifest returned a duplicate document index."
                        )
                    pending[result.document_index] = result
                    peak_pending_documents = max(
                        peak_pending_documents,
                        len(pending),
                    )
                    peak_pending_bytes = max(
                        peak_pending_bytes,
                        sum(item.output_byte_count for item in pending.values()),
                    )
                    next_merge_index = merge_pending_results(
                        pending,
                        next_merge_index=next_merge_index,
                        destination=destination,
                        run_documents_dir=documents_directory,
                        on_merged=on_merged,
                    )
                    if (
                        document_count % global_progress_interval == 0
                        or (
                            total_document_count is not None
                            and document_count == total_document_count
                        )
                    ):
                        LOGGER.info(
                            "Phase-1 progress: documents=%d chunks=%d elapsed=%.1fs.",
                            document_count,
                            chunk_count,
                            time.perf_counter() - started,
                        )
                if pending:
                    raise RuntimeError("ordered merge ended with pending documents.")
                destination.flush()
                os.fsync(destination.fileno())
            with global_temporary.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output_hash.update(block)
            _publish_completed_file(global_temporary, output_path)
            published = True
    except BaseException:
        if not preserve_incomplete:
            global_temporary.unlink(missing_ok=True)
        for path in documents_directory.glob("*.jsonl.tmp"):
            path.unlink(missing_ok=True)
        raise
    finally:
        if documents_directory.exists() and not any(documents_directory.iterdir()):
            documents_directory.rmdir()
        if run_directory.exists() and not any(run_directory.iterdir()):
            run_directory.rmdir()

    if worker_metadata is None or not published:
        raise RuntimeError("parallel pipeline did not initialize or publish.")
    if config_path is not None:
        _write_parallel_chunking_config(
            config_path,
            metadata=worker_metadata,
            config=config,
            generated_at=generated_at,
        )
    duration = time.perf_counter() - started
    summary = PipelineSummary(
        document_count=document_count,
        chunk_count=chunk_count,
        sentence_count=sentence_count,
        oversized_chunk_count=oversized_count,
        minimum_tokens_per_chunk=minimum_tokens,
        maximum_tokens_per_chunk=maximum_tokens,
        average_tokens_per_chunk=token_total / chunk_count if chunk_count else 0.0,
        processing_duration_seconds=duration,
        documents_per_second=document_count / duration if duration else 0.0,
        chunks_per_second=chunk_count / duration if duration else 0.0,
        input_path=input_path,
        output_path=output_path,
        encoder_model_name=worker_metadata.encoder_model_name,
        tokenizer_class=worker_metadata.tokenizer_class,
        tokenizer_revision=worker_metadata.tokenizer_revision,
        max_tokens=config["chunk_max_tokens"],
        overlap_tokens=config["overlap_tokens"],
        encoder_max_input_tokens=config["encoder_max_input_tokens"],
        encoder_special_token_overhead=(
            worker_metadata.encoder_special_token_overhead
        ),
        effective_content_max_tokens=(
            worker_metadata.effective_content_max_tokens
        ),
        hard_split_source_unit_count=hard_source_count,
        hard_split_generated_chunk_count=hard_generated_count,
        hard_split_by_format=hard_by_format,
        hard_split_by_strategy=hard_by_strategy,
    )
    return ParallelPipelineRun(
        summary=summary,
        metrics=ParallelPipelineMetrics(
            multiprocessing_enabled=True,
            worker_count=PHASE_1_WORKER_COUNT,
            start_method=context.get_start_method(),
            workers_initialized=1,
            worker_initialization_seconds=(
                worker_metadata.initialization_seconds
            ),
            worker_task_count=document_count,
            parallel_processing_wall_seconds=duration,
            worker_service_seconds_sum=worker_service_sum,
            worker_temp_bytes_total=worker_temp_bytes,
            worker_temp_write_seconds=worker_temp_write_sum,
            ordered_merge_seconds=merge_seconds_total,
            merged_document_count=document_count,
            last_merged_document_index=document_count,
            peak_pending_documents=peak_pending_documents,
            peak_pending_bytes=peak_pending_bytes,
            output_sha256=output_hash.hexdigest(),
        ),
    )
