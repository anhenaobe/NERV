"""Focused tests for bounded, carry-based narrative segmentation."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from nerv.chunking.language_detector import detect_language
from nerv.chunking.sentence_splitter import (
    _iter_text_blocks,
    _normalize_for_integrity,
    _split_sentences_bounded_with_stats,
    split_sentences,
)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Short text.",
        "First paragraph.\n\nSecond paragraph.\nThird line.",
        "First line.\nSecond line.\nThird line.",
        "alpha beta gamma delta epsilon",
        "x" * 97,
        "First line.\r\nSecond line.\r\n\r\nThird line.",
        "El análisis técnico continúa sin pérdida de información.",
        "A revisão técnica mantém informação íntegra e ordenada.",
        "Dr. Smith measured 3.14 units on 2026-08-05.",
        "A" * 80 + " " + "B" * 80 + ".",
    ],
)
def test_block_generation_is_bounded_lossless_and_deterministic(text: str) -> None:
    """Every block sequence is stable, contiguous, non-empty, and bounded."""
    first = list(
        _iter_text_blocks(
            text,
            target_block_characters=12,
            maximum_block_characters=20,
        )
    )
    second = list(
        _iter_text_blocks(
            text,
            target_block_characters=12,
            maximum_block_characters=20,
        )
    )

    assert first == second
    assert "".join(first) == text
    assert all(first)
    assert all(len(block) <= 20 for block in first)


def test_block_boundary_preference_is_paragraph_then_newline_then_space() -> None:
    """The first usable boundary type wins even when a weaker one is nearer."""
    paragraph_text = "aaaa bbbb\ncccc\n\ndddd eeee"
    newline_text = "aaaa bbbb\ncccc dddd eeee"
    whitespace_text = "aaaa bbbb cccc dddd eeee"

    paragraph_block = next(
        _iter_text_blocks(
            paragraph_text,
            target_block_characters=5,
            maximum_block_characters=20,
        )
    )
    newline_block = next(
        _iter_text_blocks(
            newline_text,
            target_block_characters=5,
            maximum_block_characters=15,
        )
    )
    whitespace_block = next(
        _iter_text_blocks(
            whitespace_text,
            target_block_characters=5,
            maximum_block_characters=10,
        )
    )

    assert paragraph_block.endswith("\n\n")
    assert newline_block.endswith("\n")
    assert whitespace_block.endswith(" ")


@pytest.mark.parametrize(
    ("language", "text", "target", "maximum"),
    [
        ("es", "La prueba termina aquí. Otra empieza ahora.", 21, 27),
        ("es", "La prueba termina aquí. Otra empieza ahora.", 22, 28),
        ("es", "El Dr. Pérez revisó el valor. Todo quedó estable.", 5, 11),
        ("en", "The value was 3.14 units. The reading was stable.", 17, 23),
        ("en", 'She said, "keep this sentence intact." Then she left.', 12, 19),
        ("en", "One clause continues because another clause follows.", 15, 21),
        (
            "es",
            "Esta oración no tiene puntuación final y sigue siendo una unidad",
            9,
            14,
        ),
        ("es", "Primer párrafo completo.\n\nSegundo párrafo completo.", 20, 27),
        ("pt", "O Dr. Almeida mediu 3.14 pontos. A revisão continua.", 8, 15),
        ("en", "Dr. Smith arrived. The test passed on Aug. 5, 2026.", 7, 13),
    ],
)
def test_adversarial_boundaries_match_whole_document_segmentation(
    language: str,
    text: str,
    target: int,
    maximum: int,
) -> None:
    """Tiny execution blocks preserve the controlled PySBD sentence list."""
    whole = split_sentences(text, language=language, bounded=False)
    bounded = split_sentences(
        text,
        language=language,
        target_block_characters=target,
        maximum_block_characters=maximum,
    )

    assert bounded == whole
    assert _normalize_for_integrity(" ".join(bounded)) == (
        _normalize_for_integrity(text)
    )


def test_one_oversized_sentence_is_never_cut_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An ever-growing carry may exceed the block maximum without data loss."""
    text = " ".join(["continuous"] * 40)

    with caplog.at_level(logging.WARNING):
        result = _split_sentences_bounded_with_stats(
            text,
            language="en",
            target_block_characters=12,
            maximum_block_characters=18,
        )

    assert result.sentences == [text]
    assert result.stats.maximum_carry_characters > 18
    assert result.stats.oversized_carry_warning_count == 1
    assert (
        sum("Sentence carry exceeded" in record.message for record in caplog.records)
        == 1
    )


def test_language_and_backend_are_resolved_once_for_many_blocks() -> None:
    """Bounded execution does not repeat document-level language work."""
    text = "The report is complete. " * 30

    with (
        patch(
            "nerv.chunking.sentence_splitter.detect_language",
            side_effect=detect_language,
        ) as detector,
        patch(
            "nerv.chunking.sentence_splitter._get_segmenter",
            wraps=__import__(
                "nerv.chunking.sentence_splitter",
                fromlist=["_get_segmenter"],
            )._get_segmenter,
        ) as backend,
    ):
        result = split_sentences(
            text,
            target_block_characters=20,
            maximum_block_characters=30,
        )

    assert result
    detector.assert_called_once()
    backend.assert_called_once_with("en")


def test_portuguese_fallback_warns_once_per_document(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Backend fallback is resolved once, rather than once per block."""
    text = "O relatório está completo. A revisão continua. " * 20

    with caplog.at_level(logging.WARNING):
        result = split_sentences(
            text,
            language="pt",
            target_block_characters=20,
            maximum_block_characters=30,
        )

    assert result
    assert (
        sum(
            "does not support Portuguese" in record.message
            for record in caplog.records
        )
        == 1
    )


@pytest.mark.parametrize(
    ("target", "maximum", "exception"),
    [
        (0, 10, ValueError),
        (11, 10, ValueError),
        (True, 10, TypeError),
        (5, 1.5, TypeError),
    ],
)
def test_block_size_controls_are_validated(
    target: object,
    maximum: object,
    exception: type[Exception],
) -> None:
    """Invalid execution controls fail before segmentation begins."""
    with pytest.raises(exception):
        split_sentences(
            "A complete sentence.",
            language="en",
            target_block_characters=target,  # type: ignore[arg-type]
            maximum_block_characters=maximum,  # type: ignore[arg-type]
        )


def test_large_generated_fixture_preserves_content_and_reports_bounds() -> None:
    """A generated 100 KB fixture exercises scale without a binary asset."""
    paragraph = (
        "Heading\nDr. Smith measured 3.14 units. "
        "The bounded splitter preserves this paragraph.\n\n"
    )
    text = (paragraph * ((100 * 1024 // len(paragraph)) + 1))[: 100 * 1024]

    whole = split_sentences(text, language="en", bounded=False)
    bounded = _split_sentences_bounded_with_stats(text, language="en")

    assert bounded.sentences == whole
    assert _normalize_for_integrity(" ".join(bounded.sentences)) == (
        _normalize_for_integrity(text)
    )
    assert bounded.stats.block_count > 1
    assert bounded.stats.largest_block_characters <= 65_536
