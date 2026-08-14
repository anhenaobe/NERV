"""Strict FAISS vector and identity contract tests."""

import json
from pathlib import Path

import numpy as np
import pytest

from nerv.chunking.configuration import load_encoder_config
from nerv.chunking.pipeline import process_document, validate_document
from nerv.chunking.token_counter import TokenCounter
from nerv.embeddings.artifacts import generate_embedding_artifact
from nerv.retrieval.query_encoder import encode_query
from nerv.retrieval.retriever import retrieve
from nerv.vector_database.build_index import build_index, build_index_from_artifact
from nerv.vector_database.load_index import load_index
from nerv.vector_database.save_index import save_index

from .test_embeddings import FakeEncoder, WordTokenizer


def _vectors(rows: int = 2) -> np.ndarray:
    vectors = np.zeros((rows, 384), dtype=np.float32)
    vectors[:, 0] = 1.0
    return vectors


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_faiss_rejects_non_finite(value: float) -> None:
    vectors = _vectors()
    vectors[0, 0] = value
    with pytest.raises(ValueError, match="NaN or infinite"):
        build_index(vectors)


def test_faiss_rejects_wrong_dimension_dtype_norm_and_row_count() -> None:
    with pytest.raises(ValueError, match="dimension"):
        build_index(np.zeros((2, 383), dtype=np.float32))
    with pytest.raises(ValueError, match="dtype"):
        build_index(_vectors().astype(np.float64))
    with pytest.raises(ValueError, match="L2-normalized"):
        build_index(np.ones((2, 384), dtype=np.float32))
    with pytest.raises(ValueError, match="row count"):
        build_index(_vectors(), expected_row_count=3)


def test_faiss_builds_indexflatip() -> None:
    index = build_index(_vectors())
    assert type(index).__name__ == "IndexFlatIP"
    assert index.ntotal == 2
    assert index.d == 384


def test_metadata_order_must_match_embedding_manifest(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    records = [
        {
            "chunk_id": f"D-chunk-{index:04d}",
            "num_tokens": 2,
            "texto": f"word{index}",
        }
        for index in range(2)
    ]
    chunks.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    encoder = FakeEncoder()
    config = load_encoder_config()
    counter = TokenCounter(
        "fake",
        tokenizer=WordTokenizer(),
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
    )
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "manifest.json"
    generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=counter,
        batch_size=2,
        device="cpu",
    )
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        "".join(json.dumps(record) + "\n" for record in reversed(records)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordered chunk IDs"):
        build_index_from_artifact(
            embeddings,
            manifest,
            chunks,
            metadata,
        )


def test_small_contract_smoke_from_document_to_retrieval(tmp_path: Path) -> None:
    """Cross every Phase-1 boundary without a production corpus run."""
    config = load_encoder_config()
    tokenizer = WordTokenizer()
    counter = TokenCounter(
        "fake",
        tokenizer=tokenizer,
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
    )
    document = {
        "doc_id": "SYN-001",
        "fuente": "synthetic.txt",
        "formato": "txt",
        "fenomeno": 1,
        "texto": "orbital evidence is stable",
    }
    validate_document(document, line_number=1)  # type: ignore[arg-type]
    chunks = process_document(  # type: ignore[arg-type]
        document,
        sentence_splitter=lambda text: [text],
        token_counter=counter,
        max_tokens=config["chunk_max_tokens"],
        overlap_tokens=config["overlap_tokens"],
    )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"
    encoder = FakeEncoder()
    generate_embedding_artifact(
        chunks_path,
        embeddings_path,
        manifest_path,
        encoder=encoder,
        token_counter=counter,
        batch_size=1,
        device="cpu",
    )
    index, _manifest = build_index_from_artifact(
        embeddings_path,
        manifest_path,
        chunks_path,
        chunks_path,
    )
    index_path = tmp_path / "index.faiss"
    save_index(
        index,
        index_path,
        embedding_manifest=_manifest,
        metadata_path=chunks_path,
    )
    reloaded = load_index(index_path, metadata_path=chunks_path)
    query = encode_query("orbital evidence", encoder)
    assert retrieve(query, reloaded, top_k=1) == [(0, pytest.approx(1.0))]
