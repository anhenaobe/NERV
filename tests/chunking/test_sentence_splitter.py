"""Tests for multilingual sentence segmentation."""

from unittest.mock import patch

import pytest

from nerv.chunking.language_detector import detect_language
from nerv.chunking.sentence_splitter import (
    _PYSBD_LANGUAGE_CODES,
    _get_pysbd_segmenter,
    _split_sentences_bounded_with_stats,
    split_sentences,
)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for content-preservation comparisons."""
    return " ".join(text.split())


@pytest.mark.parametrize(
    ("language", "text", "expected"),
    [
        (
            "es",
            "El informe está completo. La revisión continúa.",
            ["El informe está completo.", "La revisión continúa."],
        ),
        (
            "es",
            "El Dr. Pérez midió 3.14 puntos. ¿Terminó? ¡Sí, terminó!",
            ["El Dr. Pérez midió 3.14 puntos.", "¿Terminó?", "¡Sí, terminó!"],
        ),
        (
            "es",
            "Primer párrafo completo.\n\nSegundo párrafo activo. Tercera etapa",
            [
                "Primer párrafo completo.",
                "Segundo párrafo activo.",
                "Tercera etapa",
            ],
        ),
        (
            "en",
            "The report is complete. The review continues.",
            ["The report is complete.", "The review continues."],
        ),
        (
            "en",
            "Dr. Smith measured 3.14 points. Don't stop now. Is it stable?",
            [
                "Dr. Smith measured 3.14 points.",
                "Don't stop now.",
                "Is it stable?",
            ],
        ),
        (
            "en",
            "The first stage ended. The second stage continues",
            ["The first stage ended.", "The second stage continues"],
        ),
        (
            "pt",
            "O relatório está completo. A revisão continua.",
            ["O relatório está completo.", "A revisão continua."],
        ),
        (
            "pt",
            "O Dr. Almeida falou com a Sra. Costa. A taxa foi 3.14.",
            ["O Dr. Almeida falou com a Sra. Costa.", "A taxa foi 3.14."],
        ),
        (
            "pt",
            "O sistema processa dados, logs, etc. A equipe confirmou o teste.",
            [
                "O sistema processa dados, logs, etc.",
                "A equipe confirmou o teste.",
            ],
        ),
        (
            "pt",
            "O sensor enviou dados ao servidor. A conexão usa TLS. Está ativo?",
            [
                "O sensor enviou dados ao servidor.",
                "A conexão usa TLS.",
                "Está ativo?",
            ],
        ),
        (
            "pt",
            "A primeira etapa terminou. A segunda continua",
            ["A primeira etapa terminou.", "A segunda continua"],
        ),
    ],
)
def test_representative_segmentation(
    language: str,
    text: str,
    expected: list[str],
) -> None:
    """Preserve boundaries, punctuation, content, and order."""
    result = split_sentences(text, language=language)
    assert result == expected
    assert normalize_whitespace(" ".join(result)) == normalize_whitespace(text)
    assert all(result)


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        (
            "El informe contiene evidencia clara. Además, la revisión continúa.",
            "es",
        ),
        (
            "The report contains clear evidence. However, the review continues.",
            "en",
        ),
        (
            (
                "A análise dos documentos permite compreender as condições. "
                "Embora existam diferenças, também há elementos comparáveis. "
                "Além disso, as pessoas devem rever os resultados."
            ),
            "pt",
        ),
    ],
)
def test_automatic_language_detection(
    text: str,
    expected_language: str,
) -> None:
    """Select a backend from exactly one detector call."""
    with patch(
        "nerv.chunking.sentence_splitter.detect_language",
        side_effect=detect_language,
    ) as detector:
        result = split_sentences(text)

    assert len(result) >= 2
    assert detector.call_count == 1
    assert detect_language(text).language == expected_language


def test_explicit_language_skips_detection() -> None:
    """Do not execute detection when the caller supplies a language."""
    with patch("nerv.chunking.sentence_splitter.detect_language") as detector:
        result = split_sentences(
            "Primera oración. Segunda oración.",
            language="es",
        )
    detector.assert_not_called()
    assert result == ["Primera oración.", "Segunda oración."]


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_text(text: str) -> None:
    """Return no sentences when the input has no content."""
    assert split_sentences(text) == []


def test_invalid_text_and_language() -> None:
    """Validate the text value and supported language code."""
    with pytest.raises(TypeError, match="text must be a string"):
        split_sentences(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Language not supported"):
        split_sentences("Bonjour.", language="fr")


def test_portuguese_shares_the_cached_spanish_backend() -> None:
    """Use only two real PySBD configurations for three logical languages."""
    _get_pysbd_segmenter.cache_clear()
    spanish = _get_pysbd_segmenter(_PYSBD_LANGUAGE_CODES["es"])
    portuguese = _get_pysbd_segmenter(_PYSBD_LANGUAGE_CODES["pt"])
    english = _get_pysbd_segmenter(_PYSBD_LANGUAGE_CODES["en"])
    assert portuguese is spanish
    assert english is not spanish
    assert _get_pysbd_segmenter.cache_info().maxsize == 2
    assert _get_pysbd_segmenter.cache_info().currsize == 2


def test_pdf_page_and_line_wraps_do_not_cut_continuing_prose() -> None:
    """Reconcile synthetic page labels and extraction line wraps in place."""
    text = (
        "[Pagina 1]\nThe extracted sentence reaches the end of this page and\n\n"
        "[Pagina 2]\ncontinues on the next page without invented punctuation. "
        "A genuine sentence follows."
    )

    result = _split_sentences_bounded_with_stats(
        text,
        language="en",
        document_format="pdf",
    ).sentences

    assert result == [
        (
            "[Pagina 1] The extracted sentence reaches the end of this page and "
            "[Pagina 2] continues on the next page without invented punctuation."
        ),
        "A genuine sentence follows.",
    ]


def test_pdf_short_table_cells_do_not_chain_into_one_prose_unit() -> None:
    """Keep dense newline-delimited table cells independently packable."""
    text = (
        "AI Skill Cluster\n"
        "Skill\n"
        "Artificial Intelligence\n"
        "Expert System\n"
        "IBM Watson\n"
        "Skill Cluster\n"
        "AI\n"
        "AI\n"
        "In addition, the taxonomy assigns skills to clusters."
    )

    result = _split_sentences_bounded_with_stats(
        text,
        language="en",
        document_format="pdf",
    ).sentences

    assert result[:8] == [
        "AI Skill Cluster",
        "Skill",
        "Artificial Intelligence",
        "Expert System",
        "IBM Watson",
        "Skill Cluster",
        "AI",
        "AI In addition, the taxonomy assigns skills to clusters.",
    ]


def test_continuous_whitespace_pysbd_enumeration_is_reconciled() -> None:
    """Merge the confirmed lowercase enumeration pattern emitted by PySBD."""
    text = (
        "The framework has three parts: i) detection, ii) review, and "
        "iii) remediation. The next sentence remains independent."
    )

    result = _split_sentences_bounded_with_stats(
        text,
        language="en",
        document_format="json",
    ).sentences

    assert result == [
        (
            "The framework has three parts: i) detection, ii) review, and "
            "iii) remediation."
        ),
        "The next sentence remains independent.",
    ]


def test_structural_json_field_boundary_is_not_reconciled() -> None:
    """Keep newline-delimited JSON field records independent without punctuation."""
    text = "title: Structural heading\nbody_paragraphs[0]: prose starts here"

    result = _split_sentences_bounded_with_stats(
        text,
        language="en",
        document_format="json",
    ).sentences

    assert result == [
        "title: Structural heading",
        "body_paragraphs[0]: prose starts here",
    ]
