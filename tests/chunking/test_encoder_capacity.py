"""Capacity derivation and Phase-1B hard-split boundary contracts."""

from collections.abc import Sequence

import pytest

from nerv.chunking.configuration import derive_effective_content_max_tokens
from nerv.chunking.hard_limit import split_hard_limited_text
from nerv.chunking.pipeline import process_document


class _CharacterCounter:
    """Count code points plus the frozen two-token document prefix."""

    reported_model_max_length = 512

    def count(self, text: str, *, include_document_prefix: bool = False) -> int:
        return len(text) + (2 if include_document_prefix else 0)

    def count_many(
        self,
        texts: Sequence[str],
        *,
        include_document_prefix: bool = False,
        batch_size: int = 256,
    ) -> list[int]:
        return [
            self.count(text, include_document_prefix=include_document_prefix)
            for text in texts
        ]

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(text))]


def _capacity() -> int:
    return derive_effective_content_max_tokens(
        encoder_max_input_tokens=512,
        encoder_special_token_overhead=2,
        tokenizer_model_max_length=512,
        model_max_position_embeddings=512,
    ).effective_content_max_tokens


def test_effective_capacity_is_derived_not_configured() -> None:
    capacity = derive_effective_content_max_tokens(
        encoder_max_input_tokens=512,
        encoder_special_token_overhead=2,
        tokenizer_model_max_length=512,
        model_max_position_embeddings=512,
    )
    assert capacity.encoder_max_input_tokens == 512
    assert capacity.encoder_special_token_overhead == 2
    assert capacity.effective_content_max_tokens == 510


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("encoder_special_token_overhead", None, "positive integer"),
        ("tokenizer_model_max_length", 511, "tokenizer.model_max_length"),
        ("model_max_position_embeddings", 511, "model.max_position_embeddings"),
    ],
)
def test_capacity_derivation_fails_closed(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "encoder_max_input_tokens": 512,
        "encoder_special_token_overhead": 2,
        "tokenizer_model_max_length": 512,
        "model_max_position_embeddings": 512,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        derive_effective_content_max_tokens(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("stored_count", [255, 256, 257, 509, 510, 511, 512, 513])
def test_hard_split_uses_derived_effective_boundary(stored_count: int) -> None:
    counter = _CharacterCounter()
    text = "x" * (stored_count - 2)
    result = split_hard_limited_text(
        text,
        document_format="txt",
        token_counter=counter,  # type: ignore[arg-type]
        hard_limit=_capacity(),
        overlap_tokens=32,
    )
    if stored_count <= 510:
        assert result.strategy == "unchanged"
        assert [part.text for part in result.parts] == [text]
    else:
        assert result.strategy == "token_offsets"
        assert len(result.parts) >= 2
        reconstructed = "".join(
            text[part.coverage_start : part.coverage_end] for part in result.parts
        )
        assert reconstructed == text
        assert all(part.token_count <= 510 for part in result.parts)


def test_pipeline_preserves_metadata_when_511_token_parent_is_split() -> None:
    counter = _CharacterCounter()
    document = {
        "doc_id": "BOUNDARY-511",
        "fuente": "synthetic/boundary.txt",
        "formato": "txt",
        "fenomeno": 7,
        "texto": "x" * 509,
    }
    records = process_document(
        document,  # type: ignore[arg-type]
        sentence_splitter=lambda text: [text],
        token_counter=counter,  # type: ignore[arg-type]
        max_tokens=256,
        overlap_tokens=32,
        encoder_max_input_tokens=_capacity(),
    )
    assert len(records) >= 2
    assert all(record["doc_id"] == document["doc_id"] for record in records)
    assert all(record["fuente"] == document["fuente"] for record in records)
    assert all(record["formato"] == document["formato"] for record in records)
    assert all(record["fenomeno"] == document["fenomeno"] for record in records)
    assert all(record["num_tokens"] <= 510 for record in records)
    assert all(record["hard_split_parent_num_tokens"] == 511 for record in records)
    assert [record["posicion"] for record in records] == list(range(len(records)))
