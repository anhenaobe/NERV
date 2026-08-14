"""Construct FAISS indexes only from validated, identity-bound vectors."""

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from nerv.chunking.configuration import (
    DEFAULT_ENCODER_CONFIG_PATH,
    load_encoder_config,
)
from nerv.embeddings.artifacts import (
    EmbeddingManifest,
    ordered_chunk_identity,
    validate_embedding_artifact,
)


def build_index(
    embeddings: np.ndarray,
    *,
    expected_row_count: int | None = None,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> Any:
    """Build the frozen IndexFlatIP after strict vector validation."""
    config = load_encoder_config(config_path)
    if config["future_faiss_index"] != "IndexFlatIP":
        raise ValueError("semantic config does not authorize IndexFlatIP.")
    if config["future_similarity_metric"] != "inner_product":
        raise ValueError("semantic config does not authorize inner product.")
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray with dtype float32.")
    if embeddings.dtype != np.float32:
        raise ValueError("embedding dtype must be float32.")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D; received shape={embeddings.shape}.")
    if embeddings.shape[1] != config["embedding_dimension"]:
        raise ValueError(
            f"embedding dimension must be {config['embedding_dimension']}; "
            f"received {embeddings.shape[1]}."
        )
    if expected_row_count is not None and embeddings.shape[0] != expected_row_count:
        raise ValueError("embedding row count does not match the manifest.")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or infinite values.")
    if config["normalize_embeddings"] and len(embeddings):
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise ValueError("embeddings must be L2-normalized for IndexFlatIP cosine.")

    index = faiss.IndexFlatIP(config["embedding_dimension"])
    index.add(np.ascontiguousarray(embeddings))
    return index


def build_index_from_artifact(
    embeddings_path: Path,
    manifest_path: Path,
    chunks_path: Path,
    metadata_path: Path,
    *,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> tuple[Any, EmbeddingManifest]:
    """Validate chunk and metadata identities before constructing FAISS."""
    embeddings, manifest = validate_embedding_artifact(
        embeddings_path,
        manifest_path,
        chunks_path,
        config_path=config_path,
    )
    metadata_rows, metadata_ids_hash = ordered_chunk_identity(metadata_path)
    if metadata_rows != manifest["row_count"]:
        raise ValueError("metadata row count does not match embedding manifest.")
    if metadata_ids_hash != manifest["ordered_chunk_ids_sha256"]:
        raise ValueError("metadata ordered chunk IDs do not match embedding manifest.")
    index = build_index(
        np.asarray(embeddings),
        expected_row_count=manifest["row_count"],
        config_path=config_path,
    )
    return index, manifest
