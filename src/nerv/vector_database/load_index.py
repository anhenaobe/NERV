"""Load a persisted FAISS index only after sidecar identity validation."""

import json
from pathlib import Path
from typing import Any

import faiss

from nerv.chunking.configuration import DEFAULT_ENCODER_CONFIG_PATH
from nerv.embeddings.artifacts import ordered_chunk_identity, sha256_file
from nerv.vector_database.save_index import index_manifest_path


def load_index(
    path: Path,
    *,
    metadata_path: Path,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> Any:
    """Validate index bytes, semantic config, and metadata order before use."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el índice FAISS en: {path}")
    sidecar = index_manifest_path(path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"Falta el manifiesto obligatorio: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "index_type",
        "dimension",
        "row_count",
        "ordered_chunk_ids_sha256",
        "embeddings_sha256",
        "semantic_config_sha256",
        "index_sha256",
        "generated_at",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("FAISS manifest fields do not match schema 1.")
    if payload["schema_version"] != 1 or payload["index_type"] != "IndexFlatIP":
        raise ValueError("FAISS manifest schema or index type is unsupported.")
    if sha256_file(path) != payload["index_sha256"]:
        raise ValueError("FAISS index hash does not match its manifest.")
    if sha256_file(config_path) != payload["semantic_config_sha256"]:
        raise ValueError("FAISS semantic config identity is stale.")
    rows, ordered_hash = ordered_chunk_identity(metadata_path)
    if (
        rows != payload["row_count"]
        or ordered_hash != payload["ordered_chunk_ids_sha256"]
    ):
        raise ValueError("FAISS metadata identity does not match its manifest.")
    index = faiss.read_index(str(path))
    if type(index).__name__ != "IndexFlatIP":
        raise ValueError("persisted FAISS index is not IndexFlatIP.")
    if index.d != payload["dimension"] or index.ntotal != payload["row_count"]:
        raise ValueError("persisted FAISS shape does not match its manifest.")
    return index
