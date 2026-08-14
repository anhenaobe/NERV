"""Streaming JSONL orchestration for tokenizer-based document chunking."""

import argparse
import json
import logging
import os
import shutil
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, cast

from .atomic_io import _atomic_replace_with_retry
from .chunker import create_chunks
from .configuration import (
    derive_effective_content_max_tokens,
    load_encoder_config,
    resolve_encoder_capacity,
)
from .hard_limit import split_hard_limited_text
from .language_detector import detect_language
from .sentence_splitter import (
    _PYSBD_LANGUAGE_CODES,
    _BoundedSegmentationStats,
    _split_sentences_bounded_with_stats,
    split_sentences,
)
from .token_counter import TokenCounter

LOGGER = logging.getLogger(__name__)
DEFAULT_LOG_PATH = Path("outputs/logs/chunking_pipeline.log")
_HANDLER_PREFIX = "nerv.chunking.pipeline."
_REQUIRED_FIELDS = frozenset({"doc_id", "fuente", "formato", "fenomeno", "texto"})
_TABULAR_FORMATS = frozenset({"csv", "tsv", "xls", "xlsx"})


def _safe_log_text(value: str) -> str:
    """Escape control characters without changing the stored contract value."""
    return "".join(
        character
        if character >= " " and character != "\x7f"
        else f"\\x{ord(character):02x}"
        for character in value
    )


Document = dict[str, object]
ChunkRecord = dict[str, object]
SentenceSplitter = Callable[[str], list[str]]
TemporaryPathCallback = Callable[[Path], None]


@dataclass(frozen=True)
class PipelineDocumentProgress:
    """Observed timings and counters for one completed input document."""

    document_index: int
    total_document_count: int | None
    doc_id: str
    document_format: str
    character_count: int
    line_count: int
    detected_language: str | None
    language_detection_duration_seconds: float
    splitting_duration_seconds: float
    chunking_duration_seconds: float
    record_construction_duration_seconds: float
    writing_duration_seconds: float
    document_duration_seconds: float
    generated_chunk_count: int
    oversized_chunk_count: int
    processed_document_count: int
    total_generated_chunk_count: int
    elapsed_time_seconds: float
    estimated_remaining_seconds: float | None
    hard_split_source_unit_count: int = 0
    hard_split_generated_chunk_count: int = 0
    hard_split_by_format: dict[str, int] = dataclass_field(default_factory=dict)
    hard_split_by_strategy: dict[str, int] = dataclass_field(default_factory=dict)


ProgressCallback = Callable[[PipelineDocumentProgress], None]


@dataclass(frozen=True)
class PipelineSummary:
    """Immutable summary of a successful pipeline execution."""

    document_count: int
    chunk_count: int
    sentence_count: int
    oversized_chunk_count: int
    minimum_tokens_per_chunk: int | None
    maximum_tokens_per_chunk: int | None
    average_tokens_per_chunk: float
    processing_duration_seconds: float
    documents_per_second: float
    chunks_per_second: float
    input_path: Path
    output_path: Path
    encoder_model_name: str
    tokenizer_class: str
    tokenizer_revision: str | None
    max_tokens: int
    overlap_tokens: int
    encoder_max_input_tokens: int = 512
    encoder_special_token_overhead: int = 2
    effective_content_max_tokens: int = 510
    hard_split_source_unit_count: int = 0
    hard_split_generated_chunk_count: int = 0
    hard_split_by_format: dict[str, int] = dataclass_field(default_factory=dict)
    hard_split_by_strategy: dict[str, int] = dataclass_field(default_factory=dict)

    @property
    def oversized_sentence_chunk_count(self) -> int:
        """Return the explicit complete-sentence oversized chunk count."""
        return self.oversized_chunk_count

    @property
    def configured_overlap_tokens(self) -> int:
        """Return the configured complete-sentence overlap target."""
        return self.overlap_tokens


@dataclass(frozen=True)
class _ProcessedDocument:
    sentences: list[str]
    records: list[ChunkRecord]
    detected_language: str | None
    bounded_segmentation_stats: _BoundedSegmentationStats | None
    language_detection_duration_seconds: float
    splitting_duration_seconds: float
    chunking_duration_seconds: float
    record_construction_duration_seconds: float
    hard_split_source_unit_count: int
    hard_split_generated_chunk_count: int
    hard_split_by_format: dict[str, int]
    hard_split_by_strategy: dict[str, int]


def configure_logging(
    log_path: Path = DEFAULT_LOG_PATH,
    *,
    level: int = logging.INFO,
) -> None:
    """Configure console and UTF-8 file logging for CLI execution."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    chunking_logger = logging.getLogger("nerv.chunking")
    for handler in tuple(chunking_logger.handlers):
        if handler.get_name().startswith(_HANDLER_PREFIX):
            chunking_logger.removeHandler(handler)
            handler.close()

    console_handler = logging.StreamHandler()
    console_handler.set_name(f"{_HANDLER_PREFIX}console")
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.set_name(f"{_HANDLER_PREFIX}file")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    chunking_logger.setLevel(level)
    chunking_logger.propagate = False
    chunking_logger.addHandler(console_handler)
    chunking_logger.addHandler(file_handler)


def _publish_completed_file(temporary_path: Path, destination_path: Path) -> None:
    """Publish one completed file atomically at its destination."""
    if temporary_path.parent.resolve() == destination_path.parent.resolve():
        _atomic_replace_with_retry(temporary_path, destination_path)
        return

    adjacent_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination_path.name}.",
            suffix=".publish.tmp",
            dir=destination_path.parent,
            delete=False,
        ) as adjacent:
            adjacent_path = Path(adjacent.name)
            with temporary_path.open("rb") as source:
                shutil.copyfileobj(source, adjacent, length=1024 * 1024)
            adjacent.flush()
            os.fsync(adjacent.fileno())
        _atomic_replace_with_retry(adjacent_path, destination_path)
        temporary_path.unlink(missing_ok=True)
    except BaseException:
        if adjacent_path is not None:
            try:
                adjacent_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "Could not remove adjacent publication temporary %s while "
                    "preserving the active exception.",
                    adjacent_path,
                    exc_info=True,
                )
        raise


@contextmanager
def _atomic_text_writer(
    path: Path,
    *,
    work_dir: Path | None = None,
    preserve_incomplete: bool = False,
    temporary_path_callback: TemporaryPathCallback | None = None,
) -> Iterator[IO[str]]:
    """Yield a UTF-8 temporary file and publish it only after completion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = work_dir or path.parent
    temporary_directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=temporary_directory,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            if temporary_path_callback is not None:
                temporary_path_callback(temporary_path)
            yield temporary
            temporary.flush()
            os.fsync(temporary.fileno())
        _publish_completed_file(temporary_path, path)
    except BaseException:
        if temporary_path is not None and not preserve_incomplete:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning(
                    "Could not remove incomplete temporary %s while preserving "
                    "the active exception.",
                    temporary_path,
                    exc_info=True,
                )
        raise


def read_documents(path: Path) -> Iterator[tuple[int, Document]]:
    """Yield each non-empty JSON object with its one-based source line."""
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    LOGGER.warning(
                        "Empty line %d in %s was skipped.",
                        line_number,
                        path,
                    )
                    continue

                try:
                    decoded: object = json.loads(line)
                except json.JSONDecodeError as error:
                    LOGGER.exception(
                        "Invalid JSON on line %d of %s.",
                        line_number,
                        path,
                    )
                    raise ValueError(
                        f"invalid JSON on line {line_number}: {error.msg}"
                    ) from error

                if not isinstance(decoded, dict) or not all(
                    isinstance(key, str) for key in decoded
                ):
                    raise ValueError(
                        f"line {line_number} must contain one JSON object."
                    )
                yield line_number, cast(Document, decoded)
    except (OSError, UnicodeError):
        LOGGER.exception("Could not read %s.", path)
        raise


def validate_document(document: Document, *, line_number: int) -> None:
    """Validate the established input contract for one document."""
    missing_fields = sorted(_REQUIRED_FIELDS - document.keys())
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(f"line {line_number} is missing required fields: {missing}.")

    for field in ("doc_id", "fuente", "formato", "texto"):
        value = document[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"line {line_number} field {field!r} must be a non-empty string."
            )

    fenomeno = document["fenomeno"]
    if isinstance(fenomeno, bool) or not isinstance(fenomeno, int):
        raise ValueError(f"line {line_number} field 'fenomeno' must be an integer.")


def build_chunk_record(
    document: Document,
    *,
    chunk_text: str,
    position: int,
    token_count: int,
) -> ChunkRecord:
    """Build one output record while preserving source metadata."""
    doc_id = document["doc_id"]
    if not isinstance(doc_id, str):
        raise TypeError("validated document field 'doc_id' must be a string.")
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}-chunk-{position:04d}",
        "fuente": document["fuente"],
        "formato": document["formato"],
        "fenomeno": document["fenomeno"],
        "posicion": position,
        "num_tokens": token_count,
        "texto": chunk_text,
    }


def process_document(
    document: Document,
    *,
    sentence_splitter: SentenceSplitter = split_sentences,
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    encoder_max_input_tokens: int | None = None,
) -> list[ChunkRecord]:
    """Split, chunk, and convert one validated document into output records."""
    return _process_document(
        document,
        sentence_splitter=sentence_splitter,
        token_counter=token_counter,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        encoder_max_input_tokens=(
            encoder_max_input_tokens
            if encoder_max_input_tokens is not None
            else load_encoder_config()["encoder_max_input_tokens"]
        ),
    ).records


def _split_document_sentences(
    document: Document,
    *,
    sentence_splitter: SentenceSplitter,
) -> list[str]:
    """Preserve tabular rows or use linguistic sentence segmentation."""
    text = document["texto"]
    document_format = document["formato"]
    if not isinstance(text, str):
        raise TypeError("validated document field 'texto' must be a string.")
    if not isinstance(document_format, str):
        raise TypeError("validated document field 'formato' must be a string.")
    if document_format.casefold() in _TABULAR_FORMATS:
        return [line.strip() for line in text.splitlines() if line.strip()]
    return sentence_splitter(text)


def _create_additive_chunks(
    units: list[str],
    *,
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[str], list[int]] | None:
    """Pack complete units when sampled whitespace-join counts are additive."""
    if not units:
        return [], []

    unit_counts = token_counter.count_many(
        units,
        include_document_prefix=True,
    )
    prefix_count = token_counter.count("", include_document_prefix=True)
    contributions = [count - prefix_count for count in unit_counts]
    if any(contribution < 0 for contribution in contributions):
        return None

    pair_indexes = list(range(min(8, len(units) - 1)))
    pair_indexes.extend(range(max(0, len(units) - 9), max(0, len(units) - 1)))
    pair_indexes = sorted(set(pair_indexes))
    pair_texts = [f"{units[index]} {units[index + 1]}" for index in pair_indexes]
    pair_counts = token_counter.count_many(
        pair_texts,
        include_document_prefix=True,
    )
    for index, pair_count in zip(pair_indexes, pair_counts, strict=True):
        expected = prefix_count + contributions[index] + contributions[index + 1]
        if pair_count != expected:
            return None

    groups: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_tokens = prefix_count
    for unit, unit_count, contribution in zip(
        units,
        unit_counts,
        contributions,
        strict=True,
    ):
        if unit_count > max_tokens:
            if current:
                groups.append(current)
                current = []
                current_tokens = prefix_count
            groups.append([(unit, contribution)])
            continue
        if current and current_tokens + contribution > max_tokens:
            groups.append(current)
            current = []
            current_tokens = prefix_count
        current.append((unit, contribution))
        current_tokens += contribution
    if current:
        groups.append(current)

    chunks: list[str] = []
    chunk_counts: list[int] = []
    for index, group in enumerate(groups):
        overlap: list[tuple[str, int]] = []
        if index > 0 and overlap_tokens:
            for previous_unit in reversed(groups[index - 1]):
                candidate = [previous_unit, *overlap]
                candidate_tokens = prefix_count + sum(
                    contribution for _, contribution in candidate
                )
                if candidate_tokens > overlap_tokens:
                    break
                overlap = candidate

        group_tokens = prefix_count + sum(contribution for _, contribution in group)
        chunk_tokens = group_tokens + sum(contribution for _, contribution in overlap)
        while overlap and chunk_tokens > max_tokens:
            overlap.pop(0)
            chunk_tokens = group_tokens + sum(
                contribution for _, contribution in overlap
            )
        chunk_units = [*overlap, *group]
        chunk_text = " ".join(unit for unit, _ in chunk_units)
        chunks.append(chunk_text)
        chunk_counts.append(chunk_tokens)
    return chunks, chunk_counts


def _process_document(
    document: Document,
    *,
    sentence_splitter: SentenceSplitter,
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    encoder_max_input_tokens: int,
) -> _ProcessedDocument:
    """Return records and already-computed sentences for pipeline accounting."""
    text = document["texto"]
    document_format = document["formato"]
    if not isinstance(text, str):
        raise TypeError("validated document field 'texto' must be a string.")
    if not isinstance(document_format, str):
        raise TypeError("validated document field 'formato' must be a string.")

    detected_language: str | None = None
    bounded_segmentation_stats: _BoundedSegmentationStats | None = None
    language_detection_duration = 0.0
    split_started = time.perf_counter()
    if document_format.casefold() in _TABULAR_FORMATS:
        sentences = _split_document_sentences(
            document,
            sentence_splitter=sentence_splitter,
        )
    elif sentence_splitter is split_sentences:
        detection_started = time.perf_counter()
        detection = detect_language(text)
        language_detection_duration = time.perf_counter() - detection_started
        detected_language = detection.language
        segmentation = _split_sentences_bounded_with_stats(
            text,
            language=detected_language,
        )
        sentences = segmentation.sentences
        bounded_segmentation_stats = segmentation.stats
    else:
        sentences = sentence_splitter(text)
    splitting_duration = time.perf_counter() - split_started
    splitting_duration = max(
        splitting_duration - language_detection_duration,
        0.0,
    )

    def count_encoder_input(chunk_text: str) -> int:
        return token_counter.count(
            chunk_text,
            include_document_prefix=True,
        )

    chunking_started = time.perf_counter()
    batched = _create_additive_chunks(
        sentences,
        token_counter=token_counter,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    if batched is None:
        LOGGER.warning(
            "Tokenizer counts are not additive for document %s; "
            "using exact scalar grouping.",
            _safe_log_text(str(document["doc_id"])),
        )
        chunks = create_chunks(
            sentences,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            count_tokens=count_encoder_input,
        )
        chunk_token_counts = token_counter.count_many(
            chunks,
            include_document_prefix=True,
        )
    else:
        chunks, estimated_token_counts = batched
        chunk_token_counts = token_counter.count_many(
            chunks,
            include_document_prefix=True,
        )
        if chunk_token_counts != estimated_token_counts:
            LOGGER.warning(
                "A non-additive final count was detected for document %s; "
                "using exact scalar grouping.",
                _safe_log_text(str(document["doc_id"])),
            )
            chunks = create_chunks(
                sentences,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                count_tokens=count_encoder_input,
            )
            chunk_token_counts = token_counter.count_many(
                chunks,
                include_document_prefix=True,
            )
    expanded_chunks: list[str] = []
    expanded_token_counts: list[int] = []
    split_metadata: list[dict[str, object] | None] = []
    hard_split_source_unit_count = 0
    hard_split_generated_chunk_count = 0
    hard_split_by_format: dict[str, int] = {}
    hard_split_by_strategy: dict[str, int] = {}
    for logical_position, (chunk, token_count) in enumerate(
        zip(chunks, chunk_token_counts, strict=True)
    ):
        if token_count <= encoder_max_input_tokens:
            expanded_chunks.append(chunk)
            expanded_token_counts.append(token_count)
            split_metadata.append(None)
            continue
        split_result = split_hard_limited_text(
            chunk,
            document_format=document_format,
            token_counter=token_counter,
            hard_limit=encoder_max_input_tokens,
            overlap_tokens=overlap_tokens,
        )
        hard_split_source_unit_count += 1
        hard_split_generated_chunk_count += len(split_result.parts)
        normalized_format = document_format.casefold()
        hard_split_by_format[normalized_format] = (
            hard_split_by_format.get(normalized_format, 0) + 1
        )
        hard_split_by_strategy[split_result.strategy] = (
            hard_split_by_strategy.get(split_result.strategy, 0) + 1
        )
        parent_chunk_id = f"{document['doc_id']}-logical-chunk-{logical_position:04d}"
        for part_index, part in enumerate(split_result.parts):
            expanded_chunks.append(part.text)
            expanded_token_counts.append(part.token_count)
            split_metadata.append(
                {
                    "hard_split": True,
                    "parent_chunk_id": parent_chunk_id,
                    "hard_split_strategy": split_result.strategy,
                    "hard_split_part_index": part_index,
                    "hard_split_part_count": len(split_result.parts),
                    "hard_split_overlap_tokens": part.overlap_tokens,
                    "hard_split_parent_num_tokens": split_result.source_token_count,
                    "hard_split_source_start": part.coverage_start,
                    "hard_split_source_end": part.coverage_end,
                    "hard_split_emitted_start": part.emitted_start,
                }
            )
    chunking_duration = time.perf_counter() - chunking_started

    record_construction_started = time.perf_counter()
    records: list[ChunkRecord] = []
    for position, (chunk, token_count, metadata) in enumerate(
        zip(
            expanded_chunks,
            expanded_token_counts,
            split_metadata,
            strict=True,
        )
    ):
        if token_count > encoder_max_input_tokens:
            raise AssertionError("an emitted chunk exceeds the encoder hard limit.")
        record = build_chunk_record(
            document,
            chunk_text=chunk,
            position=position,
            token_count=token_count,
        )
        if metadata is not None:
            record.update(metadata)
        records.append(record)
    record_construction_duration = time.perf_counter() - record_construction_started
    return _ProcessedDocument(
        sentences=sentences,
        records=records,
        detected_language=detected_language,
        bounded_segmentation_stats=bounded_segmentation_stats,
        language_detection_duration_seconds=language_detection_duration,
        splitting_duration_seconds=splitting_duration,
        chunking_duration_seconds=chunking_duration,
        record_construction_duration_seconds=record_construction_duration,
        hard_split_source_unit_count=hard_split_source_unit_count,
        hard_split_generated_chunk_count=hard_split_generated_chunk_count,
        hard_split_by_format=hard_split_by_format,
        hard_split_by_strategy=hard_split_by_strategy,
    )


def _validate_pipeline_configuration(
    input_path: Path,
    output_path: Path,
    max_tokens: int,
    overlap_tokens: int,
    encoder_max_input_tokens: int,
) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an integer.")
    if isinstance(overlap_tokens, bool) or not isinstance(overlap_tokens, int):
        raise TypeError("overlap_tokens must be an integer.")
    if isinstance(encoder_max_input_tokens, bool) or not isinstance(
        encoder_max_input_tokens, int
    ):
        raise TypeError("encoder_max_input_tokens must be an integer.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError(
            "overlap_tokens must be non-negative and smaller than max_tokens."
        )
    if encoder_max_input_tokens < max_tokens:
        raise ValueError(
            "encoder_max_input_tokens must be greater than or equal to max_tokens."
        )
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input_path and output_path must be different.")


def _package_version() -> str:
    try:
        return version("nerv")
    except PackageNotFoundError:
        return "uninstalled"


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_chunking_config(
    path: Path,
    *,
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
    encoder_max_input_tokens: int,
    generated_at: str,
) -> None:
    """Write the tokenizer and chunking configuration atomically."""
    frozen_config = load_encoder_config()
    configuration: dict[str, object] = {
        "encoder_model_name": token_counter.model_name,
        "tokenizer_revision": token_counter.revision,
        "tokenizer_class": token_counter.tokenizer_class,
        "use_fast": token_counter.use_fast,
        "trust_remote_code": token_counter.trust_remote_code,
        "add_special_tokens": token_counter.add_special_tokens,
        "document_prefix": token_counter.document_prefix,
        "local_files_only": token_counter.local_files_only,
        "reported_model_max_length": _json_safe_value(
            token_counter.reported_model_max_length
        ),
        "chunk_max_tokens": max_tokens,
        "encoder_max_input_tokens": frozen_config["encoder_max_input_tokens"],
        "encoder_special_token_overhead": (
            token_counter.encoder_special_token_overhead
        ),
        "effective_content_max_tokens": encoder_max_input_tokens,
        "overlap_tokens": overlap_tokens,
        "oversized_sentence_policy": (
            "preserve_through_encoder_limit_then_hierarchical_subdivide"
        ),
        "language_detector": "nerv.chunking.language_detector.detect_language",
        "splitter_backend_by_language": dict(_PYSBD_LANGUAGE_CODES),
        "generated_at": generated_at,
        "code_version": _package_version(),
    }
    with _atomic_text_writer(path) as destination:
        json.dump(
            configuration,
            destination,
            ensure_ascii=False,
            indent=2,
        )
        destination.write("\n")


def run_pipeline(
    input_path: Path,
    output_path: Path,
    *,
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int | None = None,
    encoder_max_input_tokens: int | None = None,
    config_path: Path | None = None,
    total_document_count: int | None = None,
    progress_callback: ProgressCallback | None = None,
    global_progress_interval: int = 25,
    work_dir: Path | None = None,
    preserve_incomplete: bool = False,
    temporary_path_callback: TemporaryPathCallback | None = None,
) -> PipelineSummary:
    """Stream documents into atomically published tokenizer-based chunks."""
    frozen_config = load_encoder_config()
    if overlap_tokens is None:
        overlap_tokens = frozen_config["overlap_tokens"]
    if encoder_max_input_tokens is None:
        capacity = derive_effective_content_max_tokens(
            encoder_max_input_tokens=frozen_config["encoder_max_input_tokens"],
            encoder_special_token_overhead=(
                token_counter.encoder_special_token_overhead
            ),
            tokenizer_model_max_length=token_counter.reported_model_max_length,
            model_max_position_embeddings=frozen_config[
                "encoder_max_input_tokens"
            ],
        )
        encoder_max_input_tokens = capacity.effective_content_max_tokens
    reported_limit = token_counter.reported_model_max_length
    if isinstance(reported_limit, int) and reported_limit < encoder_max_input_tokens:
        raise ValueError("encoder_max_input_tokens exceeds tokenizer.model_max_length.")
    _validate_pipeline_configuration(
        input_path,
        output_path,
        max_tokens,
        overlap_tokens,
        encoder_max_input_tokens,
    )
    if config_path is not None and config_path.resolve() in {
        input_path.resolve(),
        output_path.resolve(),
    }:
        raise ValueError("config_path must differ from input_path and output_path.")
    if total_document_count is not None and total_document_count < 0:
        raise ValueError("total_document_count cannot be negative.")
    if global_progress_interval <= 0:
        raise ValueError("global_progress_interval must be greater than zero.")

    document_count = 0
    chunk_count = 0
    sentence_count = 0
    oversized_chunk_count = 0
    token_total = 0
    minimum_tokens: int | None = None
    maximum_tokens: int | None = None
    hard_split_source_unit_count = 0
    hard_split_generated_chunk_count = 0
    hard_split_by_format: dict[str, int] = {}
    hard_split_by_strategy: dict[str, int] = {}
    generated_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()

    with _atomic_text_writer(
        output_path,
        work_dir=work_dir,
        preserve_incomplete=preserve_incomplete,
        temporary_path_callback=temporary_path_callback,
    ) as destination:
        for line_number, document in read_documents(input_path):
            document_started = time.perf_counter()
            validate_document(document, line_number=line_number)
            document_index = document_count + 1
            doc_id = str(document["doc_id"])
            document_format = str(document["formato"])
            log_doc_id = _safe_log_text(doc_id)
            log_document_format = _safe_log_text(document_format)
            document_text = str(document["texto"])
            position_label = (
                f"{document_index}/{total_document_count}"
                if total_document_count is not None
                else str(document_index)
            )
            LOGGER.info(
                "[%s] doc_id=%s format=%s chars=%d lines=%d stage=start.",
                position_label,
                log_doc_id,
                log_document_format,
                len(document_text),
                document_text.count("\n") + 1,
            )
            processed = _process_document(
                document,
                sentence_splitter=split_sentences,
                token_counter=token_counter,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                encoder_max_input_tokens=encoder_max_input_tokens,
            )
            document_count += 1
            sentence_count += len(processed.sentences)
            hard_split_source_unit_count += processed.hard_split_source_unit_count
            hard_split_generated_chunk_count += (
                processed.hard_split_generated_chunk_count
            )
            for key, value in processed.hard_split_by_format.items():
                hard_split_by_format[key] = hard_split_by_format.get(key, 0) + value
            for key, value in processed.hard_split_by_strategy.items():
                hard_split_by_strategy[key] = hard_split_by_strategy.get(key, 0) + value

            writing_started = time.perf_counter()
            document_oversized_count = 0
            for record in processed.records:
                token_count = record["num_tokens"]
                if not isinstance(token_count, int):
                    raise TypeError("chunk field 'num_tokens' must be an integer.")
                if token_count > encoder_max_input_tokens:
                    raise RuntimeError(
                        "refusing to write a chunk above the encoder hard limit."
                    )
                chunk_count += 1
                is_oversized = int(token_count > max_tokens)
                oversized_chunk_count += is_oversized
                document_oversized_count += is_oversized
                token_total += token_count
                minimum_tokens = (
                    token_count
                    if minimum_tokens is None
                    else min(minimum_tokens, token_count)
                )
                maximum_tokens = (
                    token_count
                    if maximum_tokens is None
                    else max(maximum_tokens, token_count)
                )
                destination.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                destination.write("\n")
            writing_duration = time.perf_counter() - writing_started
            document_duration = time.perf_counter() - document_started
            elapsed = time.perf_counter() - started
            estimated_remaining: float | None = None
            if total_document_count is not None and document_count:
                remaining_documents = max(
                    total_document_count - document_count,
                    0,
                )
                estimated_remaining = elapsed / document_count * remaining_documents

            progress = PipelineDocumentProgress(
                document_index=document_index,
                total_document_count=total_document_count,
                doc_id=doc_id,
                document_format=document_format,
                character_count=len(document_text),
                line_count=document_text.count("\n") + 1,
                detected_language=processed.detected_language,
                language_detection_duration_seconds=(
                    processed.language_detection_duration_seconds
                ),
                splitting_duration_seconds=processed.splitting_duration_seconds,
                chunking_duration_seconds=processed.chunking_duration_seconds,
                record_construction_duration_seconds=(
                    processed.record_construction_duration_seconds
                ),
                writing_duration_seconds=writing_duration,
                document_duration_seconds=document_duration,
                generated_chunk_count=len(processed.records),
                oversized_chunk_count=document_oversized_count,
                processed_document_count=document_count,
                total_generated_chunk_count=chunk_count,
                elapsed_time_seconds=elapsed,
                estimated_remaining_seconds=estimated_remaining,
                hard_split_source_unit_count=(processed.hard_split_source_unit_count),
                hard_split_generated_chunk_count=(
                    processed.hard_split_generated_chunk_count
                ),
                hard_split_by_format=dict(processed.hard_split_by_format),
                hard_split_by_strategy=dict(processed.hard_split_by_strategy),
            )
            LOGGER.info(
                "[%s] doc_id=%s stage=detect duration=%.3fs language=%s.",
                position_label,
                log_doc_id,
                progress.language_detection_duration_seconds,
                progress.detected_language or "not_detected",
            )
            LOGGER.info(
                "[%s] doc_id=%s stage=split duration=%.3fs units=%d.",
                position_label,
                log_doc_id,
                progress.splitting_duration_seconds,
                len(processed.sentences),
            )
            if processed.bounded_segmentation_stats is not None:
                segmentation_stats = processed.bounded_segmentation_stats
                LOGGER.info(
                    "[%s] doc_id=%s stage=split_blocks blocks=%d "
                    "largest_block_chars=%d maximum_carry_chars=%d "
                    "carry_warnings=%d.",
                    position_label,
                    log_doc_id,
                    segmentation_stats.block_count,
                    segmentation_stats.largest_block_characters,
                    segmentation_stats.maximum_carry_characters,
                    segmentation_stats.oversized_carry_warning_count,
                )
            LOGGER.info(
                "[%s] doc_id=%s stage=chunk duration=%.3fs chunks=%d oversized=%d.",
                position_label,
                log_doc_id,
                progress.chunking_duration_seconds,
                progress.generated_chunk_count,
                progress.oversized_chunk_count,
            )
            LOGGER.info(
                "[%s] doc_id=%s stage=records duration=%.3fs.",
                position_label,
                log_doc_id,
                progress.record_construction_duration_seconds,
            )
            LOGGER.info(
                "[%s] doc_id=%s stage=write duration=%.3fs total=%.3fs.",
                position_label,
                log_doc_id,
                progress.writing_duration_seconds,
                progress.document_duration_seconds,
            )
            if progress_callback is not None:
                progress_callback(progress)
            if document_count % global_progress_interval == 0 or (
                total_document_count is not None
                and document_count == total_document_count
            ):
                LOGGER.info(
                    "Global progress: documents=%d chunks=%d elapsed=%.1fs "
                    "documents_per_second=%.4f estimated_remaining=%.1fs "
                    "(estimate).",
                    document_count,
                    chunk_count,
                    elapsed,
                    document_count / elapsed if elapsed else 0.0,
                    estimated_remaining or 0.0,
                )

    if config_path is not None:
        write_chunking_config(
            config_path,
            token_counter=token_counter,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            encoder_max_input_tokens=encoder_max_input_tokens,
            generated_at=generated_at,
        )

    duration = time.perf_counter() - started
    return PipelineSummary(
        document_count=document_count,
        chunk_count=chunk_count,
        sentence_count=sentence_count,
        oversized_chunk_count=oversized_chunk_count,
        minimum_tokens_per_chunk=minimum_tokens,
        maximum_tokens_per_chunk=maximum_tokens,
        average_tokens_per_chunk=token_total / chunk_count if chunk_count else 0.0,
        processing_duration_seconds=duration,
        documents_per_second=document_count / duration if duration else 0.0,
        chunks_per_second=chunk_count / duration if duration else 0.0,
        input_path=input_path,
        output_path=output_path,
        encoder_model_name=token_counter.model_name,
        tokenizer_class=token_counter.tokenizer_class,
        tokenizer_revision=token_counter.revision,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        encoder_max_input_tokens=frozen_config["encoder_max_input_tokens"],
        encoder_special_token_overhead=(
            token_counter.encoder_special_token_overhead
        ),
        effective_content_max_tokens=encoder_max_input_tokens,
        hard_split_source_unit_count=hard_split_source_unit_count,
        hard_split_generated_chunk_count=hard_split_generated_chunk_count,
        hard_split_by_format=hard_split_by_format,
        hard_split_by_strategy=hard_split_by_strategy,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments without executing the pipeline."""
    frozen = load_encoder_config()
    parser = argparse.ArgumentParser(
        description="Create tokenizer-limited chunks from UTF-8 JSONL documents."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input documentos.jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output chunks.jsonl file.",
    )
    parser.add_argument(
        "--encoder-model",
        default=frozen["encoder_model_name"],
        help="Exact Hugging Face model identifier shared with embeddings.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=frozen["chunk_max_tokens"],
        help="Maximum content-token count per chunk.",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=frozen["overlap_tokens"],
        help="Whole-sentence overlap token budget.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        help="Optional pinned Hugging Face tokenizer revision.",
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="Request the Python tokenizer implementation instead of a fast one.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote tokenizer code. Disabled by default.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the tokenizer to be available in the local cache.",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        help="Optional path for chunking_config.json.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional path for the pipeline summary JSON.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="CLI logging level.",
    )
    return parser.parse_args(argv)


def _validate_cli_artifact_paths(
    args: argparse.Namespace,
    *,
    config_path: Path,
) -> None:
    """Reject destructive aliases before logging or pipeline writes begin."""
    paths: dict[str, Path] = {
        "input_path": args.input,
        "output_path": args.output,
        "config_output": config_path,
        "log_path": DEFAULT_LOG_PATH,
    }
    if args.metrics_output is not None:
        paths["metrics_output"] = args.metrics_output
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
                raise ValueError(f"{name} must differ from {previous_name}: {path}.")
        resolved_paths.append((name, resolved))


def main(argv: Sequence[str] | None = None) -> None:
    """Load one tokenizer and execute the CLI pipeline."""
    args = parse_args(argv)
    config_path = args.config_output or args.output.with_name("chunking_config.json")
    _validate_cli_artifact_paths(args, config_path=config_path)
    configure_logging(level=getattr(logging, args.log_level))
    token_counter = TokenCounter(
        args.encoder_model,
        revision=args.tokenizer_revision,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
        add_special_tokens=False,
        document_prefix=load_encoder_config()["document_prefix"],
        local_files_only=args.local_files_only,
    )
    frozen_config = load_encoder_config()
    capacity = resolve_encoder_capacity(
        config=frozen_config,
        tokenizer_model_max_length=token_counter.reported_model_max_length,
        encoder_special_token_overhead=(
            token_counter.encoder_special_token_overhead
        ),
        local_files_only=args.local_files_only,
    )

    LOGGER.info(
        "Starting chunking: input=%s output=%s encoder=%s max_tokens=%d "
        "overlap_tokens=%d.",
        args.input,
        args.output,
        args.encoder_model,
        args.max_tokens,
        args.overlap_tokens,
    )
    try:
        summary = run_pipeline(
            args.input,
            args.output,
            token_counter=token_counter,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            encoder_max_input_tokens=capacity.effective_content_max_tokens,
            config_path=config_path,
        )
    except Exception:
        LOGGER.exception("The chunking pipeline failed.")
        raise

    if args.metrics_output is not None:
        summary_data = asdict(summary)
        summary_data["input_path"] = str(summary.input_path)
        summary_data["output_path"] = str(summary.output_path)
        with _atomic_text_writer(args.metrics_output) as destination:
            json.dump(summary_data, destination, ensure_ascii=False, indent=2)
            destination.write("\n")

    LOGGER.info(
        "Chunking completed: documents=%d chunks=%d oversized=%d.",
        summary.document_count,
        summary.chunk_count,
        summary.oversized_chunk_count,
    )


if __name__ == "__main__":
    main()
