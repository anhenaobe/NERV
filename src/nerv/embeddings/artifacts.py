"""Identity-safe chunk-to-embedding artifact production and validation."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np

from nerv.chunking.configuration import (
    DEFAULT_ENCODER_CONFIG_PATH,
    EncoderConfig,
    load_encoder_config,
)
from nerv.chunking.token_counter import TokenCounter
from nerv.embeddings.embedding_generator import generate_embeddings
from nerv.embeddings.encoder import encoder_content_capacity, validate_encoder_inputs

LOGGER = logging.getLogger(__name__)


class EmbeddingManifest(TypedDict):
    """Versioned identity and semantic contract for one embedding matrix."""

    schema_version: int
    source_chunks_path: str
    source_chunks_sha256: str
    ordered_chunk_ids_sha256: str
    row_count: int
    embedding_dimension: int
    dtype: str
    encoder_model_name: str
    tokenizer_revision: str | None
    tokenizer_class: str
    document_prefix: str
    add_special_tokens: bool
    sentence_transformer_special_tokens: int
    normalize_embeddings: bool
    normalization_method: str
    encoder_max_input_tokens: int
    semantic_config_sha256: str
    embeddings_sha256: str
    device: str
    generated_at: str


_REQUIRED_CHUNK_FIELDS = frozenset({"chunk_id", "num_tokens", "texto"})


def sha256_file(path: Path) -> str:
    """Hash a file exactly as stored."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _release_memmap(matrix: np.memmap, *, flush: bool) -> None:
    """Explicitly release a NumPy mapping before Windows file publication."""
    if flush:
        matrix.flush()
    mapping = getattr(matrix, "_mmap", None)
    if mapping is None:
        raise RuntimeError("NumPy memmap does not expose its backing mmap handle.")
    mapping.close()


def _fsync_file(path: Path) -> None:
    """Synchronize one completed file after its memory mapping is closed."""
    with Path(path).open("rb+") as artifact:
        artifact.flush()
        os.fsync(artifact.fileno())


def _best_effort_unlink(path: Path) -> None:
    """Clean an incomplete temporary without masking the primary failure."""
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        LOGGER.warning("Could not remove incomplete temporary %s: %s", path, error)


def _iter_chunks(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid chunk JSON at line {line_number}."
                ) from error
            if not isinstance(item, dict):
                raise ValueError(f"chunk line {line_number} must be an object.")
            missing = _REQUIRED_CHUNK_FIELDS - item.keys()
            if missing:
                raise ValueError(
                    f"chunk line {line_number} is missing: {sorted(missing)}."
                )
            chunk_id = item["chunk_id"]
            text = item["texto"]
            token_count = item["num_tokens"]
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(f"chunk line {line_number} has an invalid chunk_id.")
            if not isinstance(text, str) or not text:
                raise ValueError(f"chunk line {line_number} has invalid texto.")
            if (
                isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count < 0
            ):
                raise ValueError(f"chunk line {line_number} has invalid num_tokens.")
            yield item


def ordered_chunk_identity(path: Path) -> tuple[int, str]:
    """Return row count and deterministic ordered chunk-ID hash."""
    digest = hashlib.sha256()
    seen: set[str] = set()
    row_count = 0
    for item in _iter_chunks(path):
        chunk_id = cast(str, item["chunk_id"])
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {chunk_id!r}.")
        seen.add(chunk_id)
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\n")
        row_count += 1
    return row_count, digest.hexdigest().upper()


def _validate_source_counts(
    chunks_path: Path,
    counter: TokenCounter,
    config: EncoderConfig,
    effective_content_max_tokens: int,
) -> None:
    for item in _iter_chunks(chunks_path):
        count = cast(int, item["num_tokens"])
        if count > effective_content_max_tokens:
            raise ValueError(
                f"chunk {item['chunk_id']!r} exceeds "
                "effective_content_max_tokens."
            )
        actual = counter.count(
            cast(str, item["texto"]),
            include_document_prefix=True,
        )
        if actual != count:
            raise ValueError(
                f"chunk {item['chunk_id']!r} num_tokens={count} but recount={actual}."
            )


def _batches(path: Path, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in _iter_chunks(path):
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _special_token_count(encoder: Any) -> int:
    tokenizer = getattr(encoder, "tokenizer", None)
    function = getattr(tokenizer, "num_special_tokens_to_add", None)
    if not callable(function):
        raise ValueError("encoder tokenizer cannot report special-token overhead.")
    value = function(pair=False)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("encoder tokenizer reported invalid special-token overhead.")
    return value


def generate_embedding_artifact(
    chunks_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
    *,
    encoder: Any,
    token_counter: TokenCounter,
    batch_size: int,
    device: str,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> EmbeddingManifest:
    """Stream chunks twice and atomically publish a bound matrix/manifest pair.

    The manifest is the commit marker.  If publication is interrupted between
    the matrix replacement and manifest replacement, the stored embedding hash
    makes the inconsistent pair unusable rather than silently reusable.
    """
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    chunks_path = Path(chunks_path)
    embeddings_path = Path(embeddings_path)
    manifest_path = Path(manifest_path)
    if not chunks_path.is_file():
        raise FileNotFoundError(chunks_path)
    if embeddings_path.resolve() == manifest_path.resolve():
        raise ValueError("embeddings_path and manifest_path must differ.")
    existing_outputs = [
        path for path in (embeddings_path, manifest_path) if path.exists()
    ]
    if existing_outputs:
        formatted = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Embedding generation requires unused output paths; "
            f"refusing to overwrite: {formatted}"
        )

    config = load_encoder_config(config_path)
    effective_content_max_tokens = encoder_content_capacity(encoder, config)
    row_count, ordered_ids_hash = ordered_chunk_identity(chunks_path)
    source_hash = sha256_file(chunks_path)
    _validate_source_counts(
        chunks_path,
        token_counter,
        config,
        effective_content_max_tokens,
    )

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{embeddings_path.name}.",
        suffix=".tmp",
        dir=embeddings_path.parent,
    )
    os.close(descriptor)
    temporary_embeddings = Path(temporary_name)
    temporary_manifest: Path | None = None
    matrix: np.memmap | None = None
    preserve_completed_temporaries = False
    try:
        matrix = np.lib.format.open_memmap(
            temporary_embeddings,
            mode="w+",
            dtype=np.float32,
            shape=(row_count, config["embedding_dimension"]),
        )
        offset = 0
        for batch in _batches(chunks_path, batch_size):
            raw_texts = [cast(str, item["texto"]) for item in batch]
            counts = [cast(int, item["num_tokens"]) for item in batch]
            prefixed = [f"{config['document_prefix']}{text}" for text in raw_texts]
            validate_encoder_inputs(prefixed, counts, encoder, config)
            vectors = generate_embeddings(
                raw_texts,
                encoder,
                batch_size,
                kind="document",
                config=config,
            )
            matrix[offset : offset + len(batch)] = vectors
            offset += len(batch)
        _release_memmap(matrix, flush=True)
        matrix = None
        _fsync_file(temporary_embeddings)
        if offset != row_count:
            raise ValueError("chunk source changed while embeddings were generated.")
        current_rows, current_ids_hash = ordered_chunk_identity(chunks_path)
        if (
            current_rows != row_count
            or current_ids_hash != ordered_ids_hash
            or sha256_file(chunks_path) != source_hash
        ):
            raise ValueError("chunk source changed while embeddings were generated.")

        tokenizer = getattr(encoder, "tokenizer", None)
        manifest: EmbeddingManifest = {
            "schema_version": 1,
            "source_chunks_path": str(chunks_path.resolve()),
            "source_chunks_sha256": source_hash,
            "ordered_chunk_ids_sha256": ordered_ids_hash,
            "row_count": row_count,
            "embedding_dimension": config["embedding_dimension"],
            "dtype": "float32",
            "encoder_model_name": config["encoder_model_name"],
            "tokenizer_revision": config["tokenizer_revision"],
            "tokenizer_class": type(tokenizer).__name__,
            "document_prefix": config["document_prefix"],
            "add_special_tokens": config["add_special_tokens"],
            "sentence_transformer_special_tokens": _special_token_count(encoder),
            "normalize_embeddings": config["normalize_embeddings"],
            "normalization_method": config["normalization_method"],
            "encoder_max_input_tokens": config["encoder_max_input_tokens"],
            "semantic_config_sha256": sha256_file(config_path),
            "embeddings_sha256": sha256_file(temporary_embeddings),
            "device": device,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        descriptor, temporary_manifest_name = tempfile.mkstemp(
            prefix=f".{manifest_path.name}.",
            suffix=".tmp",
            dir=manifest_path.parent,
        )
        os.close(descriptor)
        temporary_manifest = Path(temporary_manifest_name)
        temporary_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _fsync_file(temporary_manifest)
        preserve_completed_temporaries = True
        os.link(temporary_embeddings, embeddings_path)
        temporary_embeddings.unlink()
        os.link(temporary_manifest, manifest_path)
        temporary_manifest.unlink()
        return manifest
    finally:
        if matrix is not None:
            _release_memmap(matrix, flush=False)
        if not preserve_completed_temporaries:
            _best_effort_unlink(temporary_embeddings)
            if temporary_manifest is not None:
                _best_effort_unlink(temporary_manifest)


def validate_recoverable_embedding_artifact(
    temporary_embeddings: Path,
    temporary_manifest: Path,
    chunks_path: Path,
    *,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> EmbeddingManifest:
    """Validate a completed failed-run pair and explicitly release its mapping."""
    temporary_embeddings = Path(temporary_embeddings)
    temporary_manifest = Path(temporary_manifest)
    chunks_path = Path(chunks_path)
    if not temporary_embeddings.is_file() or not temporary_manifest.is_file():
        raise FileNotFoundError("completed embedding recovery pair is missing.")
    matrix: np.ndarray | None = None
    try:
        matrix, manifest = validate_embedding_artifact(
            temporary_embeddings,
            temporary_manifest,
            chunks_path,
            config_path=config_path,
        )
    finally:
        if isinstance(matrix, np.memmap):
            _release_memmap(matrix, flush=False)
    return manifest


def recover_embedding_artifact(
    temporary_embeddings: Path,
    temporary_manifest: Path,
    embeddings_path: Path,
    manifest_path: Path,
    chunks_path: Path,
    *,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> EmbeddingManifest:
    """Validate and publish a completed failed-run state without recomputation.

    Recovery accepts either a complete temporary pair or the exact half-commit
    left when the matrix was published but manifest publication failed. Every
    path must be distinct after resolution. Final creation uses hard links so
    an independently created destination is never overwritten; the manifest
    remains the final commit marker.
    """
    temporary_embeddings = Path(temporary_embeddings)
    temporary_manifest = Path(temporary_manifest)
    embeddings_path = Path(embeddings_path)
    manifest_path = Path(manifest_path)
    chunks_path = Path(chunks_path)
    paths = {
        "temporary_embeddings": temporary_embeddings.resolve(strict=False),
        "temporary_manifest": temporary_manifest.resolve(strict=False),
        "embeddings": embeddings_path.resolve(strict=False),
        "manifest": manifest_path.resolve(strict=False),
        "chunks": chunks_path.resolve(strict=False),
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError("embedding recovery paths must be pairwise distinct.")
    if manifest_path.exists():
        raise FileExistsError(
            "recovery refuses to overwrite a published embedding manifest."
        )

    # Generation publishes the matrix before the manifest. If only that first
    # move completed, validate the final matrix against the preserved manifest.
    if embeddings_path.exists():
        if temporary_embeddings.exists():
            raise FileExistsError(
                "recovery found both final and temporary embedding matrices."
            )
        manifest = validate_recoverable_embedding_artifact(
            embeddings_path,
            temporary_manifest,
            chunks_path,
            config_path=config_path,
        )
        os.link(temporary_manifest, manifest_path)
        temporary_manifest.unlink()
        return manifest

    manifest = validate_recoverable_embedding_artifact(
        temporary_embeddings,
        temporary_manifest,
        chunks_path,
        config_path=config_path,
    )
    os.link(temporary_embeddings, embeddings_path)
    temporary_embeddings.unlink()
    os.link(temporary_manifest, manifest_path)
    temporary_manifest.unlink()
    return manifest


def load_embedding_manifest(path: Path) -> EmbeddingManifest:
    """Load a manifest and reject missing, unknown, or malformed fields."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("embedding manifest must be an object.")
    required = set(EmbeddingManifest.__required_keys__)
    if set(data) != required:
        raise ValueError("embedding manifest fields do not match schema 1.")
    manifest = cast(EmbeddingManifest, data)
    if manifest["schema_version"] != 1:
        raise ValueError("embedding manifest schema_version must be 1.")
    return manifest


def validate_embedding_artifact(
    embeddings_path: Path,
    manifest_path: Path,
    chunks_path: Path,
    *,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> tuple[np.ndarray, EmbeddingManifest]:
    """Validate semantic identity, chunk order, and every vector property."""
    manifest = load_embedding_manifest(manifest_path)
    config = load_encoder_config(config_path)
    rows, ordered_hash = ordered_chunk_identity(chunks_path)
    if sha256_file(chunks_path) != manifest["source_chunks_sha256"]:
        raise ValueError("chunk source SHA-256 does not match embedding manifest.")
    if ordered_hash != manifest["ordered_chunk_ids_sha256"]:
        raise ValueError("ordered chunk IDs do not match embedding manifest.")
    if rows != manifest["row_count"]:
        raise ValueError("chunk row count does not match embedding manifest.")
    if sha256_file(config_path) != manifest["semantic_config_sha256"]:
        raise ValueError("semantic config hash does not match embedding manifest.")
    semantic_pairs = {
        "embedding_dimension": config["embedding_dimension"],
        "encoder_model_name": config["encoder_model_name"],
        "tokenizer_revision": config["tokenizer_revision"],
        "document_prefix": config["document_prefix"],
        "add_special_tokens": config["add_special_tokens"],
        "normalize_embeddings": config["normalize_embeddings"],
        "normalization_method": config["normalization_method"],
        "encoder_max_input_tokens": config["encoder_max_input_tokens"],
    }
    for key, expected in semantic_pairs.items():
        if manifest[key] != expected:  # type: ignore[literal-required]
            raise ValueError(f"manifest field {key!r} conflicts with semantic config.")
    if sha256_file(embeddings_path) != manifest["embeddings_sha256"]:
        raise ValueError("embedding artifact SHA-256 does not match manifest.")
    matrix = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    try:
        if matrix.dtype != np.float32:
            raise ValueError("embedding dtype must be float32.")
        expected_shape = (manifest["row_count"], manifest["embedding_dimension"])
        if matrix.shape != expected_shape:
            raise ValueError(
                f"embedding shape={matrix.shape}; expected {expected_shape}."
            )
        if not np.isfinite(matrix).all():
            raise ValueError("embedding artifact contains NaN or infinite values.")
        if manifest["normalize_embeddings"] and len(matrix):
            norms = np.linalg.norm(matrix, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError("embedding artifact is not L2-normalized.")
    except Exception:
        if isinstance(matrix, np.memmap):
            _release_memmap(matrix, flush=False)
        raise
    return matrix, manifest
