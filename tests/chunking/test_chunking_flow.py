"""Pruebas del flujo detector, splitter y chunker."""

import pytest

from nerv.chunking import create_chunks, detect_language, split_sentences

FLOW_CASES = [
    (
        "es",
        (
            "El informe presenta evidencia clara. Además, el equipo revisa los "
            "datos. La metodología conserva cada registro. Finalmente, los "
            "resultados quedan documentados."
        ),
        [
            "El informe presenta evidencia clara.",
            "Además, el equipo revisa los datos.",
            "La metodología conserva cada registro.",
            "Finalmente, los resultados quedan documentados.",
        ],
    ),
    (
        "en",
        (
            "The report presents clear evidence. However, the team reviews the "
            "data. The method preserves every record. Finally, the results are "
            "documented."
        ),
        [
            "The report presents clear evidence.",
            "However, the team reviews the data.",
            "The method preserves every record.",
            "Finally, the results are documented.",
        ],
    ),
    (
        "pt",
        (
            "A análise dos documentos permite compreender as condições. "
            "Embora existam diferenças, também há elementos comparáveis. "
            "Além disso, as pessoas devem rever os resultados. Finalmente, "
            "a informação fica documentada."
        ),
        [
            "A análise dos documentos permite compreender as condições.",
            "Embora existam diferenças, também há elementos comparáveis.",
            "Além disso, as pessoas devem rever os resultados.",
            "Finalmente, a informação fica documentada.",
        ],
    ),
]


@pytest.mark.parametrize(("expected_language", "text", "expected"), FLOW_CASES)
def test_flujo_integrado_multilingue(
    expected_language: str,
    text: str,
    expected: list[str],
) -> None:
    """El flujo completo conserva oraciones y genera varios chunks."""
    detection = detect_language(text)
    sentences = split_sentences(text)
    chunks = create_chunks(sentences, max_tokens=8)

    assert detection.language == expected_language
    assert sentences == expected
    assert 2 <= len(chunks) <= len(sentences)
    assert " ".join(chunks) == " ".join(expected)
    assert all(any(sentence in chunk for chunk in chunks) for sentence in expected)

    repeated = create_chunks(split_sentences(text), max_tokens=8)
    assert repeated == chunks
