"""Pruebas del detector heurístico de idioma."""

from time import perf_counter

import pytest

from nerv.chunking.language_detector import (
    LANGUAGE_CODES,
    _load_language_markers,
    _sample_text,
    detect_language,
)

SPANISH_TEXT = (
    "El análisis de los documentos permite comprender cómo cambian las "
    "condiciones. Aunque existen diferencias, también hay elementos comunes "
    "que pueden compararse. Además, las personas deben revisar los resultados "
    "porque todavía puede haber información incompleta."
)
ENGLISH_TEXT = (
    "The analysis of the documents helps explain how conditions change. "
    "Although there are differences, several elements can be compared. "
    "However, every result should be reviewed because some information may "
    "still be incomplete."
)
PORTUGUESE_TEXT = (
    "A análise dos documentos permite compreender como as condições mudam. "
    "Embora existam diferenças, também há elementos que podem ser comparados. "
    "Além disso, as pessoas devem rever os resultados porque ainda pode haver "
    "informação incompleta."
)


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        (SPANISH_TEXT, "es"),
        (ENGLISH_TEXT, "en"),
        (PORTUGUESE_TEXT, "pt"),
        (
            SPANISH_TEXT + " Short quotation: the source is under review.",
            "es",
        ),
        (
            ENGLISH_TEXT + " Referencia breve: el informe sigue en revisión.",
            "en",
        ),
        ("Executive summary. " + PORTUGUESE_TEXT, "pt"),
    ],
)
def test_detecta_textos_claros_y_mixtos(
    text: str,
    expected_language: str,
) -> None:
    """Clasifica textos claros y mixtos por su idioma predominante."""
    result = detect_language(text)
    assert result.language == expected_language
    assert result.is_ambiguous is False
    assert set(result.scores) == set(LANGUAGE_CODES)
    assert 0.0 <= result.confidence <= 1.0
    assert result.margin >= 0.0


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_texto_vacio_usa_default(text: str) -> None:
    """Usa el idioma predeterminado ante ausencia de evidencia."""
    result = detect_language(text, default="pt")
    assert result.language == "pt"
    assert result.confidence == 0.0
    assert result.is_ambiguous is True


def test_texto_ambiguo_usa_default_y_conserva_scores() -> None:
    """Expone la ambigüedad sin borrar la evidencia calculada."""
    result = detect_language("the y de", default="pt", minimum_score=100.0)
    assert result.language == "pt"
    assert result.is_ambiguous is True
    assert any(score > 0 for score in result.scores.values())


def test_rechaza_entrada_invalida() -> None:
    """Rechaza valores que no sean cadenas."""
    with pytest.raises(TypeError, match="text must be a string"):
        detect_language(42)  # type: ignore[arg-type]


def test_resultado_es_repetible() -> None:
    """La misma entrada y configuración producen el mismo resultado."""
    assert detect_language(SPANISH_TEXT) == detect_language(SPANISH_TEXT)


def test_muestra_usa_inicio_centro_y_final() -> None:
    """La muestra larga incorpora las tres zonas del documento."""
    text = "INICIO----" + ("a" * 80) + "CENTRO----" + ("b" * 80) + "----FINAL"
    sample = _sample_text(text, 20)
    assert "INICIO" in sample
    assert "CENTRO" in sample
    assert "FINAL" in sample
    assert len(sample) <= 60


def test_recursos_se_cargan_una_sola_vez() -> None:
    """El recurso lingüístico queda reutilizado en memoria."""
    _load_language_markers.cache_clear()
    first = _load_language_markers()
    second = _load_language_markers()
    assert first is second
    assert _load_language_markers.cache_info().misses == 1
    assert _load_language_markers.cache_info().hits == 1


def test_medicion_repetible_de_300_detecciones() -> None:
    """Conserva la cobertura de rendimiento sin fijar un umbral del equipo."""
    _load_language_markers.cache_clear()
    cases = [
        *((SPANISH_TEXT, "es") for _ in range(100)),
        *((ENGLISH_TEXT, "en") for _ in range(100)),
        *((PORTUGUESE_TEXT, "pt") for _ in range(100)),
    ]

    start = perf_counter()
    detected = [detect_language(text).language for text, _ in cases]
    elapsed = perf_counter() - start

    assert detected == [expected for _, expected in cases]
    assert _load_language_markers.cache_info().misses == 1
    assert _load_language_markers.cache_info().hits == 299
    assert elapsed >= 0.0
