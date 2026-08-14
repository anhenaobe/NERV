"""Local-cache integration test for the frozen E5 chunking pipeline."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from nerv.chunking import TokenCounter, split_sentences
from nerv.chunking.pipeline import run_pipeline

MODEL_NAME = "intfloat/multilingual-e5-small"
FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "codefest_fictional_documents.jsonl"
)


def _real_tokenizer() -> object:
    try:
        transformers = pytest.importorskip("transformers")
        return transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
    except (OSError, ValueError) as error:
        pytest.skip(f"Frozen tokenizer is not locally available: {error}")


def _real_counter() -> TokenCounter:
    return TokenCounter(MODEL_NAME, tokenizer=_real_tokenizer())


class _InputIdsOnlyTokenizer:
    """Expose the pre-optimization call surface around a real tokenizer."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer
        self.model_max_length = getattr(tokenizer, "model_max_length", None)

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> Any:
        return self._tokenizer(  # type: ignore[operator]
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return self._tokenizer.num_special_tokens_to_add(  # type: ignore[union-attr]
            pair=pair
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.integration
def test_frozen_real_tokenizer_pipeline(tmp_path: Path) -> None:
    """Validate the full synthetic flow with the selected real tokenizer."""
    counter = _real_counter()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    summary = run_pipeline(
        FIXTURE,
        first,
        token_counter=counter,
        max_tokens=256,
        overlap_tokens=32,
    )
    run_pipeline(
        FIXTURE,
        second,
        token_counter=counter,
        max_tokens=256,
        overlap_tokens=32,
    )
    records = _read_jsonl(first)
    documents = {record["doc_id"]: record for record in _read_jsonl(FIXTURE)}

    assert summary.document_count == 30
    assert first.read_bytes() == second.read_bytes()
    assert {record["doc_id"] for record in records} == set(documents)
    assert all(
        record["texto"]
        and not str(record["texto"]).startswith("passage: ")
        for record in records
    )
    assert all(
        record["num_tokens"]
        == counter.count(str(record["texto"]), include_document_prefix=True)
        for record in records
    )
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["doc_id"]), []).append(record)
    for doc_id, chunks in grouped.items():
        assert [chunk["posicion"] for chunk in chunks] == list(range(len(chunks)))
        assert [chunk["chunk_id"] for chunk in chunks] == [
            f"{doc_id}-chunk-{position:04d}" for position in range(len(chunks))
        ]
        sentences = split_sentences(str(documents[doc_id]["texto"]))
        for chunk in chunks:
            if int(chunk["num_tokens"]) > 256:
                assert str(chunk["texto"]) in sentences
        assert all(
            any(sentence in str(chunk["texto"]) for chunk in chunks)
            for sentence in sentences
        )


@pytest.mark.integration
def test_real_tokenizer_is_multilingual_and_deterministic() -> None:
    """Encode Spanish, English, and Portuguese without truncation."""
    counter = _real_counter()
    for text in (
        "La órbita permanece estable.",
        "The orbit remains stable.",
        "A órbita permanece estável.",
    ):
        assert counter.encode(text)
        assert counter.encode(text) == counter.encode(text)
        assert counter.count(text, include_document_prefix=True) >= counter.count(text)


@pytest.mark.integration
@pytest.mark.parametrize("include_document_prefix", [False, True])
def test_real_return_length_matches_reference_matrix(
    include_document_prefix: bool,
) -> None:
    """The frozen tokenizer returns exact lengths for controlled semantics."""
    counter = _real_counter()
    texts = [
        "",
        "prueba",
        "La órbita permanece estable.",
        "The orbit remains stable.",
        "A órbita permanece estável.",
        "acentos: áéíóú ñ ç ã",
        "¿Puntuación?! -- yes...",
        "1.234,56 y 3.14",
        "palabra_muy_larga_" * 500,
        "espacios     repetidos",
        "primera línea\nsegunda línea",
        "texto sintético " * 2_000,
    ]
    expected = [
        len(
            counter.encode(
                text,
                include_document_prefix=include_document_prefix,
            )
        )
        for text in texts
    ]

    assert counter.count_many(
        texts,
        include_document_prefix=include_document_prefix,
        batch_size=5,
    ) == expected
    assert [
        counter.count(
            text,
            include_document_prefix=include_document_prefix,
        )
        for text in texts
    ] == expected


@pytest.mark.integration
def test_real_optimized_pipeline_matches_input_id_fallback(
    tmp_path: Path,
) -> None:
    """Optimization preserves byte-identical synthetic pipeline output."""
    tokenizer = _real_tokenizer()
    optimized_counter = TokenCounter(MODEL_NAME, tokenizer=tokenizer)
    reference_counter = TokenCounter(
        MODEL_NAME,
        tokenizer=_InputIdsOnlyTokenizer(tokenizer),
    )
    reference_output = tmp_path / "reference.jsonl"
    optimized_output = tmp_path / "optimized.jsonl"

    reference_summary = run_pipeline(
        FIXTURE,
        reference_output,
        token_counter=reference_counter,
        max_tokens=256,
        overlap_tokens=32,
    )
    optimized_summary = run_pipeline(
        FIXTURE,
        optimized_output,
        token_counter=optimized_counter,
        max_tokens=256,
        overlap_tokens=32,
    )

    reference_bytes = reference_output.read_bytes()
    optimized_bytes = optimized_output.read_bytes()
    assert optimized_summary.document_count == reference_summary.document_count
    assert optimized_summary.chunk_count == reference_summary.chunk_count
    assert optimized_bytes == reference_bytes
    assert hashlib.sha256(optimized_bytes).hexdigest() == hashlib.sha256(
        reference_bytes
    ).hexdigest()
