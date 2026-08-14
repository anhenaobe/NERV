"""Atomically persist FAISS with its embedding/metadata identity sidecar."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss

from nerv.embeddings.artifacts import (
    EmbeddingManifest,
    ordered_chunk_identity,
    sha256_file,
)


def index_manifest_path(index_path: Path) -> Path:
    """Return the required sidecar location for a persisted index."""
    return Path(f"{index_path}.manifest.json")


def save_index(
    index: Any,
    path: Path,
    *,
    embedding_manifest: EmbeddingManifest,
    metadata_path: Path,
) -> Path:
    """Persist an IndexFlatIP only when metadata identity is aligned."""
    path = Path(path)
    rows, ordered_hash = ordered_chunk_identity(metadata_path)
    if rows != embedding_manifest["row_count"]:
        raise ValueError("metadata row count does not match embedding manifest.")
    if ordered_hash != embedding_manifest["ordered_chunk_ids_sha256"]:
        raise ValueError("metadata order does not match embedding manifest.")
    if type(index).__name__ != "IndexFlatIP":
        raise ValueError("only IndexFlatIP can be persisted by this contract.")
    if index.d != embedding_manifest["embedding_dimension"]:
        raise ValueError("FAISS dimension does not match embedding manifest.")
    if index.ntotal != embedding_manifest["row_count"]:
        raise ValueError("FAISS row count does not match embedding manifest.")

    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = index_manifest_path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_index = Path(temporary_name)
    descriptor, temporary_manifest_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_manifest = Path(temporary_manifest_name)
    try:
        faiss.write_index(index, str(temporary_index))
        payload = {
            "schema_version": 1,
            "index_type": "IndexFlatIP",
            "dimension": index.d,
            "row_count": index.ntotal,
            "ordered_chunk_ids_sha256": ordered_hash,
            "embeddings_sha256": embedding_manifest["embeddings_sha256"],
            "semantic_config_sha256": embedding_manifest["semantic_config_sha256"],
            "index_sha256": sha256_file(temporary_index),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        temporary_manifest.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_index, path)
        os.replace(temporary_manifest, sidecar)
        return sidecar
    finally:
        temporary_index.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
