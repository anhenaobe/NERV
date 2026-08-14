"""Focused semantic and embedding artifact contract tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import nerv
from nerv.chunking.atomic_io import _atomic_replace_with_retry
from nerv.chunking.configuration import load_encoder_config
from nerv.chunking.pipeline import validate_document
from nerv.chunking.token_counter import TokenCounter
from nerv.embeddings import artifacts as embedding_artifacts
from nerv.embeddings.artifacts import (
    generate_embedding_artifact,
    recover_embedding_artifact,
    validate_embedding_artifact,
)
from nerv.embeddings.embedding_generator import generate_embeddings
from nerv.embeddings.encoder import load_encoder, validate_encoder_inputs
from nerv.retrieval.queries import load_queries
from nerv.retrieval.query_encoder import encode_query


class WordTokenizer:
    """Deterministic tokenizer exposing the relevant HF surface."""

    def __init__(self, model_max_length: int = 514) -> None:
        self.model_max_length = model_max_length

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        **options: Any,
    ) -> dict[str, Any]:
        values = [text] if isinstance(text, str) else text
        encoded: list[list[int]] = []
        for value in values:
            count = len(value.split()) + (2 if add_special_tokens else 0)
            encoded.append(list(range(count)))
        result: dict[str, Any] = {
            "input_ids": encoded[0] if isinstance(text, str) else encoded
        }
        if options.get("return_length"):
            result["length"] = [len(ids) for ids in encoded]
        return result


class FakeEncoder:
    """Encoder that records prefixed text and returns normalized float32 rows."""

    def __init__(self, model_max_length: int = 514) -> None:
        self.tokenizer = WordTokenizer(model_max_length)
        self.max_seq_length = model_max_length
        self.encoded_texts: list[str] = []

    def encode(self, texts: list[str], **options: Any) -> np.ndarray:
        self.encoded_texts.extend(texts)
        matrix = np.zeros((len(texts), 384), dtype=np.float32)
        matrix[:, 0] = 1.0
        return matrix


def _counter(tokenizer: WordTokenizer) -> TokenCounter:
    config = load_encoder_config()
    return TokenCounter(
        "fake",
        tokenizer=tokenizer,
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
    )


def _write_chunks(path: Path, counts: list[int]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for index, count in enumerate(counts):
            raw_words = max(0, count - 1)  # passage: contributes one token
            record = {
                "doc_id": f"D{index}",
                "chunk_id": f"D{index}-chunk-0000",
                "fuente": "synthetic.txt",
                "formato": "txt",
                "fenomeno": 1,
                "posicion": 0,
                "num_tokens": count,
                "texto": " ".join(f"w{number}" for number in range(raw_words)),
            }
            destination.write(json.dumps(record) + "\n")


def _real_exact_raw_text(tokenizer: Any, target: int) -> str:
    config = load_encoder_config()
    prefix_count = len(
        tokenizer(
            config["document_prefix"],
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    raw = " ".join("x" for _ in range(target - prefix_count))
    actual = len(
        tokenizer(
            f"{config['document_prefix']}{raw}",
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
    )
    if actual != target:
        raise AssertionError(f"could not construct {target} real tokens: {actual}")
    return raw


def test_authoritative_import_and_config_source() -> None:
    package_path = Path(nerv.__file__).resolve()
    assert package_path == Path(__file__).parents[1] / "src" / "nerv" / "__init__.py"
    config = load_encoder_config()
    assert config["schema_version"] == 2
    assert config["chunk_max_tokens"] == 256
    assert config["encoder_max_input_tokens"] == 512


@pytest.mark.parametrize("count", [255, 256, 257, 509, 510])
def test_validated_boundary_is_encoded_without_truncation(count: int) -> None:
    config = load_encoder_config()
    encoder = FakeEncoder(model_max_length=514)
    text = f"{config['document_prefix']}" + " ".join("w" for _ in range(count - 1))
    validate_encoder_inputs([text], [count], encoder, config)


@pytest.mark.parametrize("count", [511, 512, 513])
def test_sentence_transformer_special_token_capacity_is_not_falsified(
    count: int,
) -> None:
    config = load_encoder_config()
    encoder = FakeEncoder(model_max_length=512)
    text = f"{config['document_prefix']}" + " ".join(
        "w" for _ in range(count - 1)
    )
    with pytest.raises(ValueError, match="safe_capacity=510"):
        validate_encoder_inputs([text], [count], encoder, config)


def test_document_and_query_prefix_are_each_applied_once() -> None:
    encoder = FakeEncoder()
    generate_embeddings(["raw document"], encoder, 1)
    encode_query("raw query", encoder)
    assert encoder.encoded_texts == ["passage: raw document", "query: raw query"]


def test_embedding_generator_rejects_shape_and_non_finite_values() -> None:
    class BadEncoder(FakeEncoder):
        def encode(self, texts: list[str], **options: Any) -> np.ndarray:
            return np.full((len(texts), 384), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="NaN"):
        generate_embeddings(["text"], BadEncoder(), 1)


def test_artifact_boundaries_hashes_and_reuse(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    _write_chunks(chunks, [255, 256, 257, 509, 510])
    encoder = FakeEncoder(model_max_length=514)
    generated = generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=_counter(encoder.tokenizer),
        batch_size=2,
        device="cpu",
    )
    matrix, validated = validate_embedding_artifact(embeddings, manifest, chunks)
    assert matrix.shape == (5, 384)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert validated == generated
    assert isinstance(matrix, np.memmap)
    embedding_artifacts._release_memmap(matrix, flush=False)

    lines = chunks.read_text(encoding="utf-8").splitlines()
    chunks.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256|ordered chunk"):
        validate_embedding_artifact(embeddings, manifest, chunks)


def test_memmap_is_released_before_matrix_then_manifest_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    _write_chunks(chunks, [8, 9])
    encoder = FakeEncoder()
    opened: list[np.memmap] = []
    actual_open_memmap = np.lib.format.open_memmap
    actual_link = os.link
    publication_order: list[str] = []

    def tracking_open_memmap(*args: Any, **kwargs: Any) -> np.memmap:
        matrix = actual_open_memmap(*args, **kwargs)
        opened.append(matrix)
        return matrix

    def observing_link(source: Path, destination: Path) -> None:
        assert opened and opened[0]._mmap.closed
        if destination == embeddings:
            assert not manifest.exists()
            publication_order.append("embeddings")
        elif destination == manifest:
            assert embeddings.is_file()
            publication_order.append("manifest")
        actual_link(source, destination)

    monkeypatch.setattr(np.lib.format, "open_memmap", tracking_open_memmap)
    monkeypatch.setattr(os, "link", observing_link)
    generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=_counter(encoder.tokenizer),
        batch_size=1,
        device="cpu",
    )

    assert publication_order == ["embeddings", "manifest"]
    matrix, _validated = validate_embedding_artifact(embeddings, manifest, chunks)
    assert matrix.shape == (2, 384)
    assert isinstance(matrix, np.memmap)
    embedding_artifacts._release_memmap(matrix, flush=False)


@pytest.mark.skipif(os.name != "nt", reason="requires Windows file locking")
def test_windows_replace_fails_while_mapped_then_passes_after_release(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / ".embeddings.npy.test.tmp"
    destination = tmp_path / "embeddings.npy"
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(2, 384),
    )
    matrix[:] = 0.0
    matrix[:, 0] = 1.0
    matrix.flush()

    with pytest.raises(PermissionError) as captured:
        os.replace(temporary, destination)
    assert captured.value.winerror == 32

    embedding_artifacts._release_memmap(matrix, flush=False)
    _atomic_replace_with_retry(temporary, destination)
    loaded = np.load(destination, allow_pickle=False)
    assert loaded.shape == (2, 384)


def test_completed_temporary_pair_can_be_recovered_without_reencoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    temporary_embeddings = tmp_path / ".embeddings.npy.failed.tmp"
    temporary_manifest = tmp_path / ".embeddings.manifest.json.failed.tmp"
    _write_chunks(chunks, [8, 9])
    encoder = FakeEncoder()
    generated = generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=_counter(encoder.tokenizer),
        batch_size=1,
        device="cpu",
    )
    os.replace(embeddings, temporary_embeddings)
    os.replace(manifest, temporary_manifest)
    actual_link = os.link
    publication_order: list[str] = []

    def observing_link(source: Path, destination: Path) -> None:
        if destination == embeddings:
            assert not manifest.exists()
            publication_order.append("embeddings")
        elif destination == manifest:
            assert embeddings.is_file()
            publication_order.append("manifest")
        actual_link(source, destination)

    monkeypatch.setattr(os, "link", observing_link)
    recovered = recover_embedding_artifact(
        temporary_embeddings,
        temporary_manifest,
        embeddings,
        manifest,
        chunks,
    )

    assert recovered == generated
    assert publication_order == ["embeddings", "manifest"]
    assert not temporary_embeddings.exists()
    assert not temporary_manifest.exists()
    matrix, _validated = validate_embedding_artifact(embeddings, manifest, chunks)
    assert isinstance(matrix, np.memmap)
    embedding_artifacts._release_memmap(matrix, flush=False)


def test_half_committed_matrix_can_publish_manifest_without_reencoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    _write_chunks(chunks, [8, 9])
    encoder = FakeEncoder()
    actual_link = os.link
    temporary_sources: list[Path] = []

    def fail_manifest_publication(source: Path, destination: Path) -> None:
        temporary_sources.append(source)
        if destination == manifest:
            raise PermissionError("controlled manifest publication failure")
        actual_link(source, destination)

    monkeypatch.setattr(
        os,
        "link",
        fail_manifest_publication,
    )
    with pytest.raises(PermissionError, match="controlled"):
        generate_embedding_artifact(
            chunks,
            embeddings,
            manifest,
            encoder=encoder,
            token_counter=_counter(encoder.tokenizer),
            batch_size=1,
            device="cpu",
        )

    temporary_embeddings, temporary_manifest = temporary_sources
    assert embeddings.is_file()
    assert not temporary_embeddings.exists()
    assert temporary_manifest.is_file()
    assert not manifest.exists()

    monkeypatch.setattr(os, "link", actual_link)
    recovered = recover_embedding_artifact(
        temporary_embeddings,
        temporary_manifest,
        embeddings,
        manifest,
        chunks,
    )
    assert recovered["row_count"] == 2
    assert manifest.is_file()
    assert not temporary_manifest.exists()


def test_generation_refuses_to_replace_published_pair(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    _write_chunks(chunks, [8])
    embeddings.write_bytes(b"existing matrix")
    manifest.write_bytes(b"existing manifest")
    encoder = FakeEncoder()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate_embedding_artifact(
            chunks,
            embeddings,
            manifest,
            encoder=encoder,
            token_counter=_counter(encoder.tokenizer),
            batch_size=1,
            device="cpu",
        )

    assert embeddings.read_bytes() == b"existing matrix"
    assert manifest.read_bytes() == b"existing manifest"


def test_generation_never_clobbers_destination_created_during_encoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    _write_chunks(chunks, [8])
    encoder = FakeEncoder()
    actual_link = os.link

    def racing_link(source: Path, destination: Path) -> None:
        if destination == embeddings:
            destination.write_bytes(b"independent publisher")
        actual_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        generate_embedding_artifact(
            chunks,
            embeddings,
            manifest,
            encoder=encoder,
            token_counter=_counter(encoder.tokenizer),
            batch_size=1,
            device="cpu",
        )

    assert embeddings.read_bytes() == b"independent publisher"
    assert not manifest.exists()


def test_recovery_manifest_failure_retains_retryable_half_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    temporary_embeddings = tmp_path / ".embeddings.npy.failed.tmp"
    temporary_manifest = tmp_path / ".embeddings.manifest.json.failed.tmp"
    _write_chunks(chunks, [8])
    encoder = FakeEncoder()
    generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=_counter(encoder.tokenizer),
        batch_size=1,
        device="cpu",
    )
    os.replace(embeddings, temporary_embeddings)
    os.replace(manifest, temporary_manifest)
    actual_link = os.link

    def fail_manifest_link(source: Path, destination: Path) -> None:
        if destination == manifest:
            raise PermissionError("controlled manifest link failure")
        actual_link(source, destination)

    monkeypatch.setattr(os, "link", fail_manifest_link)
    with pytest.raises(PermissionError, match="controlled"):
        recover_embedding_artifact(
            temporary_embeddings,
            temporary_manifest,
            embeddings,
            manifest,
            chunks,
        )
    assert embeddings.is_file()
    assert not temporary_embeddings.exists()
    assert temporary_manifest.is_file()
    assert not manifest.exists()

    monkeypatch.setattr(os, "link", actual_link)
    recovered = recover_embedding_artifact(
        temporary_embeddings,
        temporary_manifest,
        embeddings,
        manifest,
        chunks,
    )
    assert recovered["row_count"] == 1
    assert manifest.is_file()
    assert not temporary_manifest.exists()


def test_recovery_never_deletes_matrix_replaced_during_manifest_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    temporary_embeddings = tmp_path / ".embeddings.npy.failed.tmp"
    temporary_manifest = tmp_path / ".embeddings.manifest.json.failed.tmp"
    independent = tmp_path / "independent.npy"
    _write_chunks(chunks, [8])
    encoder = FakeEncoder()
    generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=_counter(encoder.tokenizer),
        batch_size=1,
        device="cpu",
    )
    os.replace(embeddings, temporary_embeddings)
    os.replace(manifest, temporary_manifest)
    independent.write_bytes(b"independently published matrix")
    actual_link = os.link

    def replace_then_fail(source: Path, destination: Path) -> None:
        if destination == manifest:
            os.replace(independent, embeddings)
            raise PermissionError("controlled concurrent replacement")
        actual_link(source, destination)

    monkeypatch.setattr(os, "link", replace_then_fail)
    with pytest.raises(PermissionError, match="controlled"):
        recover_embedding_artifact(
            temporary_embeddings,
            temporary_manifest,
            embeddings,
            manifest,
            chunks,
        )

    assert embeddings.read_bytes() == b"independently published matrix"
    assert temporary_manifest.is_file()
    assert not manifest.exists()


def test_recovery_rejects_aliases_and_never_clobbers_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    temporary_embeddings = tmp_path / ".embeddings.npy.failed.tmp"
    temporary_manifest = tmp_path / ".embeddings.manifest.json.failed.tmp"
    _write_chunks(chunks, [8])
    encoder = FakeEncoder()
    generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=_counter(encoder.tokenizer),
        batch_size=1,
        device="cpu",
    )
    os.replace(embeddings, temporary_embeddings)
    os.replace(manifest, temporary_manifest)

    with pytest.raises(ValueError, match="pairwise distinct"):
        recover_embedding_artifact(
            temporary_embeddings,
            temporary_manifest,
            embeddings,
            embeddings,
            chunks,
        )

    actual_link = os.link

    def racing_link(source: Path, destination: Path) -> None:
        if destination == embeddings:
            destination.write_bytes(b"independent publisher")
        actual_link(source, destination)

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(FileExistsError):
        recover_embedding_artifact(
            temporary_embeddings,
            temporary_manifest,
            embeddings,
            manifest,
            chunks,
        )
    assert embeddings.read_bytes() == b"independent publisher"
    assert temporary_embeddings.is_file()
    assert temporary_manifest.is_file()


def test_changed_chunks_and_config_hash_reject_reuse(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    _write_chunks(chunks, [256])
    encoder = FakeEncoder()
    generate_embedding_artifact(
        chunks,
        embeddings,
        manifest,
        encoder=encoder,
        token_counter=_counter(encoder.tokenizer),
        batch_size=1,
        device="cpu",
    )
    chunks.write_text(chunks.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_embedding_artifact(embeddings, manifest, chunks)

    _write_chunks(chunks, [256])
    alternate_config = tmp_path / "encoder_config.json"
    canonical = Path(__file__).parents[1] / "config" / "encoder_config.json"
    alternate_config.write_text(
        canonical.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="config hash"):
        validate_embedding_artifact(
            embeddings,
            manifest,
            chunks,
            config_path=alternate_config,
        )


@pytest.mark.parametrize("count", [511, 512, 513])
def test_artifact_rejects_count_above_hard_limit(
    tmp_path: Path,
    count: int,
) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(chunks, [count])
    encoder = FakeEncoder(model_max_length=515)
    with pytest.raises(ValueError, match="exceeds effective_content_max_tokens"):
        generate_embedding_artifact(
            chunks,
            tmp_path / "embeddings.npy",
            tmp_path / "manifest.json",
            encoder=encoder,
            token_counter=_counter(encoder.tokenizer),
            batch_size=1,
            device="cpu",
        )


def test_ingestion_schema_is_directly_accepted_by_chunking() -> None:
    document = {
        "doc_id": "DOC-001",
        "fuente": "corpus/sample.txt",
        "formato": "txt",
        "fenomeno": 1,
        "texto": "Texto de prueba.",
        "idioma": "es",
        "metadata": {"origin": "outer ingestion"},
    }
    validate_document(document, line_number=1)  # type: ignore[arg-type]


def test_strict_query_loader_rejects_missing_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query_id":"q001","text":"one"}\n{"query_id":"q001","text":"two"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_queries(path)
    path.write_text('{"text":"missing id"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="query_id and text"):
        load_queries(path)


@pytest.mark.integration
def test_real_sentence_transformer_safe_max_and_preencode_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = os.environ.get("NERV_TEST_ENCODER_MODEL")
    if model_name != "intfloat/multilingual-e5-small":
        pytest.skip("NERV_TEST_ENCODER_MODEL is not configured for the frozen model.")
    config = load_encoder_config()
    encoder = load_encoder(device="cpu", local_files_only=True)
    assert type(encoder.tokenizer).__name__ == "XLMRobertaTokenizer"
    assert encoder.max_seq_length == 512
    assert encoder.tokenizer.num_special_tokens_to_add(pair=False) == 2

    safe_raw = _real_exact_raw_text(encoder.tokenizer, 510)
    safe_prefixed = f"{config['document_prefix']}{safe_raw}"
    validate_encoder_inputs([safe_prefixed], [510], encoder, config)
    matrix = generate_embeddings([safe_raw], encoder, 1, config=config)
    assert matrix.shape == (1, 384)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)

    unsafe_raw = _real_exact_raw_text(encoder.tokenizer, 511)
    unsafe_prefixed = f"{config['document_prefix']}{unsafe_raw}"
    with pytest.raises(ValueError, match="safe_capacity=510"):
        validate_encoder_inputs([unsafe_prefixed], [511], encoder, config)

    chunks = tmp_path / "unsafe.jsonl"
    chunks.write_text(
        json.dumps(
            {
                "chunk_id": "unsafe-chunk-0000",
                "num_tokens": 511,
                "texto": unsafe_raw,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    encode_called = False

    def fail_if_encoded(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal encode_called
        encode_called = True
        raise AssertionError("unsafe text reached SentenceTransformer.encode")

    monkeypatch.setattr(encoder, "encode", fail_if_encoded)
    counter = TokenCounter(
        config["encoder_model_name"],
        tokenizer=encoder.tokenizer,
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
    )
    with pytest.raises(ValueError, match="effective_content_max_tokens"):
        generate_embedding_artifact(
            chunks,
            tmp_path / "unsafe.npy",
            tmp_path / "unsafe.manifest.json",
            encoder=encoder,
            token_counter=counter,
            batch_size=1,
            device="cpu",
        )
    assert not encode_called
