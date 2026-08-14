"""Tests for tokenizer-injected candidate counting in the chunker."""

import pytest

from nerv.chunking import TokenCounter, create_chunks

from .fakes import DeterministicFakeTokenizer


def _counter() -> TokenCounter:
    return TokenCounter(
        "fake/offline-tokenizer",
        tokenizer=DeterministicFakeTokenizer(),
    )


def test_joined_candidate_text_is_counted() -> None:
    """Candidate limits use the joined text instead of summed sentence counts."""
    calls: list[str] = []

    def boundary_sensitive_count(text: str) -> int:
        calls.append(text)
        return len(text.replace(" ", "")) + text.count(". A")

    chunks = create_chunks(
        ["AA.", "AB."],
        max_tokens=6,
        count_tokens=boundary_sensitive_count,
    )

    assert chunks == ["AA.", "AB."]
    assert "AA. AB." in calls


def test_real_counter_dependency_controls_chunk_limits() -> None:
    """The injected tokenizer counter determines deterministic boundaries."""
    counter = _counter()
    sentences = ["Alpha beta.", "Gamma.", "Delta epsilon."]

    chunks = create_chunks(
        sentences,
        max_tokens=16,
        count_tokens=counter.count,
    )

    assert chunks == ["Alpha beta. Gamma.", "Delta epsilon."]
    assert all(counter.count(chunk) <= 16 for chunk in chunks)


def test_oversized_sentence_remains_complete() -> None:
    """An oversized tokenizer result is preserved as one complete chunk."""
    counter = _counter()
    oversized = "This complete sentence is deliberately oversized."

    chunks = create_chunks(
        ["Short.", oversized, "Final."],
        max_tokens=20,
        count_tokens=counter.count,
    )

    assert chunks == ["Short.", oversized, "Final."]
    assert counter.count(oversized) > 20


def test_tokenizer_based_overlap_is_bounded() -> None:
    """Overlap copies complete suffix sentences without preventing progress."""
    counter = _counter()
    sentences = ["One.", "Two.", "Three.", "Four."]

    chunks = create_chunks(
        sentences,
        max_tokens=11,
        overlap_tokens=6,
        count_tokens=counter.count,
    )

    assert chunks == ["One. Two.", "Two. Three.", "Three. Four."]
    assert len(chunks) == 3


@pytest.mark.parametrize("invalid_count", [-1, True, 1.5])
def test_invalid_counter_results_are_rejected(invalid_count: object) -> None:
    """The chunker rejects counters that violate the count contract."""

    def invalid_counter(text: str) -> int:
        return invalid_count  # type: ignore[return-value]

    with pytest.raises(ValueError, match="non-negative integer"):
        create_chunks(
            ["Sentence."],
            max_tokens=10,
            count_tokens=invalid_counter,
        )
