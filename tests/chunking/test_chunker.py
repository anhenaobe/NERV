"""Pruebas del agrupador de oraciones completas."""

import pytest

from nerv.chunking import count_words_as_tokens, create_chunks


def test_lista_vacia_y_entradas_vacias() -> None:
    """No genera chunks vacíos."""
    assert create_chunks([], max_tokens=5) == []
    assert create_chunks(["", "  ", "\n"], max_tokens=5) == []


def test_una_oracion_y_varias_en_un_chunk() -> None:
    """Agrupa entradas pequeñas sin alterar su contenido."""
    assert create_chunks(["Una oración breve."], max_tokens=5) == ["Una oración breve."]
    assert create_chunks(
        ["Uno dos.", "Tres cuatro.", "Cinco."],
        max_tokens=5,
    ) == ["Uno dos. Tres cuatro. Cinco."]


def test_genera_varios_chunks_y_preserva_orden() -> None:
    """Respeta el límite aproximado y el orden lógico."""
    sentences = ["Uno dos tres.", "Cuatro cinco.", "Seis siete.", "Ocho."]
    assert create_chunks(sentences, max_tokens=5) == [
        "Uno dos tres. Cuatro cinco.",
        "Seis siete. Ocho.",
    ]


def test_omite_oraciones_vacias_intermedias() -> None:
    """Las entradas vacías no alteran los límites."""
    assert create_chunks(
        ["Uno dos.", "", "  ", "Tres cuatro."],
        max_tokens=4,
    ) == ["Uno dos. Tres cuatro."]


def test_oracion_oversized_se_conserva_completa() -> None:
    """Una oración sobredimensionada se emite sola y sin cortes."""
    oversized = "uno dos tres cuatro cinco seis"
    chunks = create_chunks(
        ["Inicio breve.", oversized, "Final breve."],
        max_tokens=4,
    )
    assert chunks == ["Inicio breve.", oversized, "Final breve."]
    assert count_words_as_tokens(chunks[1]) > 4


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_rechaza_max_tokens_invalido(max_tokens: int) -> None:
    """El límite debe ser positivo."""
    with pytest.raises(ValueError, match="max_tokens"):
        create_chunks(["Texto."], max_tokens=max_tokens)


@pytest.mark.parametrize("overlap_tokens", [-1, 5, 6])
def test_rechaza_overlap_invalido(overlap_tokens: int) -> None:
    """El solapamiento debe ser no negativo y menor que el límite."""
    with pytest.raises(ValueError, match="overlap_tokens"):
        create_chunks(
            ["Texto."],
            max_tokens=5,
            overlap_tokens=overlap_tokens,
        )


def test_rechaza_tipos_invalidos() -> None:
    """Valida el contenedor y cada oración."""
    with pytest.raises(TypeError, match="sequence"):
        create_chunks("No es una secuencia válida.", max_tokens=5)
    with pytest.raises(TypeError, match="every sentence"):
        create_chunks(["Válida.", 3], max_tokens=5)  # type: ignore[list-item]


def test_overlap_duplica_solo_oraciones_completas_y_termina() -> None:
    """El overlap usa un sufijo acotado sin ciclos ni fragmentos parciales."""
    sentences = [
        "Uno dos tres.",
        "Cuatro cinco.",
        "Seis siete.",
        "Ocho nueve.",
    ]
    chunks = create_chunks(sentences, max_tokens=5, overlap_tokens=3)
    assert chunks == [
        "Uno dos tres. Cuatro cinco.",
        "Cuatro cinco. Seis siete.",
        "Seis siete. Ocho nueve.",
    ]
    assert len(chunks) == 3


def test_sin_overlap_conserva_todo_el_contenido() -> None:
    """Cada oración aparece exactamente una vez cuando overlap es cero."""
    sentences = ["Uno dos tres.", "Cuatro cinco.", "Seis siete.", "Ocho."]
    chunks = create_chunks(sentences, max_tokens=5)
    assert " ".join(chunks) == " ".join(sentences)


def test_resultado_es_repetible() -> None:
    """Misma entrada y configuración producen chunks idénticos."""
    sentences = ["Uno dos tres.", "Cuatro cinco.", "Seis siete.", "Ocho."]
    first = create_chunks(sentences, max_tokens=5, overlap_tokens=2)
    second = create_chunks(sentences, max_tokens=5, overlap_tokens=2)
    assert first == second
