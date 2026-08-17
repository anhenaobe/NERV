"""Focused contracts for the deterministic Stage-1 delivery formatter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_generator_module() -> object:
    path = Path(__file__).resolve().parents[1] / "entrega" / "generador.py"
    spec = importlib.util.spec_from_file_location("stage1_delivery_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(text: str, formato: str = "pdf") -> dict[str, object]:
    return {
        "doc_id": "DOC-001",
        "chunk_id": "DOC-001-chunk-0000",
        "fuente": "source.pdf",
        "formato": formato,
        "fenomeno": 1,
        "posicion": 0,
        "num_tokens": 10,
        "texto": text,
    }


def test_bounded_fragment_text_preserves_corrected_text_within_limit() -> None:
    generator = _load_generator_module()
    text = "Primera oración completa. Segunda unidad estructural"

    rendered = generator._bounded_fragment_text(  # type: ignore[attr-defined]
        _record(text), 250
    )

    assert rendered == text


def test_bounded_fragment_text_retains_structural_json_metadata() -> None:
    generator = _load_generator_module()
    text = "Resumen completo. keywords[0]: inteligencia artificial issue: 12"

    rendered = generator._bounded_fragment_text(  # type: ignore[attr-defined]
        _record(text, formato="json"), 250
    )

    assert rendered == text


def test_bounded_fragment_text_uses_last_safe_boundary_above_limit() -> None:
    generator = _load_generator_module()
    text = " ".join(["oración"] * 200 + ["completa."] + ["resto"] * 100)

    rendered = generator._bounded_fragment_text(  # type: ignore[attr-defined]
        _record(text), 250
    )

    assert rendered.endswith("completa.")
    assert len(rendered.split()) == 201


def test_bounded_fragment_text_fails_closed_above_limit_without_boundary() -> None:
    generator = _load_generator_module()
    text = " ".join(["prosa"] * 251)

    with pytest.raises(ValueError, match="no complete source sentence"):
        generator._bounded_fragment_text(  # type: ignore[attr-defined]
            _record(text), 250
        )
