"""Evalúa de forma reproducible el flujo de chunking con fixtures sintéticos."""

import json
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any

from nerv.chunking import (
    count_words_as_tokens,
    create_chunks,
    detect_language,
    split_sentences,
)
from nerv.chunking.language_detector import LANGUAGE_CODES

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"
RESULT_PATH = HERE / "results" / "chunking_metrics.json"
MAX_WORD_UNITS = 12
OVERLAP_WORD_UNITS = 3


def normalize_whitespace(text: str) -> str:
    """Normaliza únicamente los espacios para comparar contenido."""
    return " ".join(text.split())


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _load_language_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with (FIXTURES_DIR / "language_samples.jsonl").open(
        encoding="utf-8"
    ) as fixture:
        for line in fixture:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def _load_sentence_samples() -> list[dict[str, Any]]:
    with (FIXTURES_DIR / "sentence_samples.json").open(
        encoding="utf-8"
    ) as fixture:
        data: list[dict[str, Any]] = json.load(fixture)
    return data


def _boundary_offsets(sentences: list[str]) -> set[int]:
    """Representa cada límite por su posición en el texto normalizado unido."""
    normalized = [normalize_whitespace(sentence) for sentence in sentences]
    boundaries: set[int] = set()
    offset = 0
    for sentence in normalized[:-1]:
        offset += len(sentence)
        boundaries.add(offset)
        offset += 1
    return boundaries


def _lost_character_count(original: str, reconstructed: str) -> int:
    """Cuenta caracteres originales sin correspondencia tras normalizar espacios."""
    source = normalize_whitespace(original)
    candidate = normalize_whitespace(reconstructed)
    matched = sum(
        block.size
        for block in SequenceMatcher(None, source, candidate).get_matching_blocks()
    )
    return max(len(source) - matched, 0)


def evaluate_detector(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Calcula exactitud, confusión y señales heurísticas del detector."""
    confusion = {
        expected: {predicted: 0 for predicted in LANGUAGE_CODES}
        for expected in LANGUAGE_CODES
    }
    by_language_raw = {
        language: {
            "total": 0,
            "correct": 0,
            "confidences": [],
            "margins": [],
            "ambiguous": 0,
        }
        for language in LANGUAGE_CODES
    }
    confidences: list[float] = []
    margins: list[float] = []
    failures: list[dict[str, Any]] = []
    correct = 0
    ambiguous = 0

    for sample in samples:
        expected = str(sample["expected_language"])
        result = detect_language(str(sample["text"]))
        predicted = result.language
        confusion[expected][predicted] += 1
        language_data = by_language_raw[expected]
        language_data["total"] += 1
        language_data["confidences"].append(result.confidence)
        language_data["margins"].append(result.margin)
        confidences.append(result.confidence)
        margins.append(result.margin)

        if result.is_ambiguous:
            ambiguous += 1
            language_data["ambiguous"] += 1
        if predicted == expected:
            correct += 1
            language_data["correct"] += 1
        else:
            failures.append(
                {
                    "component": "detector",
                    "sample_id": sample["sample_id"],
                    "expected": expected,
                    "actual": predicted,
                    "is_ambiguous": result.is_ambiguous,
                    "confidence": result.confidence,
                    "margin": result.margin,
                }
            )

    by_language = {}
    for language, raw in by_language_raw.items():
        total = int(raw["total"])
        by_language[language] = {
            "total": total,
            "accuracy": _safe_ratio(int(raw["correct"]), total),
            "ambiguous_rate": _safe_ratio(int(raw["ambiguous"]), total),
            "average_confidence": _mean(raw["confidences"]),
            "average_margin": _mean(raw["margins"]),
        }

    metrics = {
        "sample_count": len(samples),
        "accuracy": _safe_ratio(correct, len(samples)),
        "confusion_matrix": confusion,
        "ambiguous_rate": _safe_ratio(ambiguous, len(samples)),
        "average_confidence": _mean(confidences),
        "average_margin": _mean(margins),
        "by_language": by_language,
        "confidence_note": (
            "La confianza es una proporción heurística interna, no una "
            "probabilidad calibrada."
        ),
    }
    return metrics, failures


def _empty_splitter_accumulator() -> dict[str, Any]:
    return {
        "cases": 0,
        "exact": 0,
        "true_boundaries": 0,
        "predicted_boundaries": 0,
        "expected_boundaries": 0,
        "original_characters": 0,
        "lost_characters": 0,
    }


def _finalize_splitter_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    precision = _safe_ratio(
        raw["true_boundaries"],
        raw["predicted_boundaries"],
    )
    recall = _safe_ratio(
        raw["true_boundaries"],
        raw["expected_boundaries"],
    )
    return {
        "case_count": raw["cases"],
        "exact_match_rate": _safe_ratio(raw["exact"], raw["cases"]),
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": _safe_ratio(2 * precision * recall, precision + recall),
        "content_loss_rate": _safe_ratio(
            raw["lost_characters"],
            raw["original_characters"],
        ),
        "lost_character_count": raw["lost_characters"],
    }


def evaluate_splitter(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compara oraciones y límites esperados por idioma."""
    overall = _empty_splitter_accumulator()
    by_language_raw = {
        language: _empty_splitter_accumulator()
        for language in LANGUAGE_CODES
    }
    failures: list[dict[str, Any]] = []

    for sample in samples:
        language = str(sample["language"])
        text = str(sample["text"])
        expected = [str(value) for value in sample["expected_sentences"]]
        predicted = split_sentences(text, language=language)
        expected_boundaries = _boundary_offsets(expected)
        predicted_boundaries = _boundary_offsets(predicted)
        true_boundaries = len(expected_boundaries & predicted_boundaries)
        reconstructed = " ".join(predicted)
        lost_characters = _lost_character_count(text, reconstructed)
        original_characters = len(normalize_whitespace(text))

        for accumulator in (overall, by_language_raw[language]):
            accumulator["cases"] += 1
            accumulator["exact"] += int(predicted == expected)
            accumulator["true_boundaries"] += true_boundaries
            accumulator["predicted_boundaries"] += len(predicted_boundaries)
            accumulator["expected_boundaries"] += len(expected_boundaries)
            accumulator["original_characters"] += original_characters
            accumulator["lost_characters"] += lost_characters

        if predicted != expected or lost_characters:
            failures.append(
                {
                    "component": "splitter",
                    "sample_id": sample["sample_id"],
                    "language": language,
                    "expected": expected,
                    "actual": predicted,
                    "lost_characters": lost_characters,
                }
            )

    metrics = _finalize_splitter_metrics(overall)
    metrics["by_language"] = {
        language: _finalize_splitter_metrics(raw)
        for language, raw in by_language_raw.items()
    }
    return metrics, failures


def _decompose_chunk(
    chunk: str,
    sentences: list[str],
) -> tuple[list[str], bool]:
    """Comprueba si un chunk se compone solo de oraciones completas conocidas."""
    remaining = chunk
    matched: list[str] = []
    while remaining:
        sentence = next(
            (
                candidate
                for candidate in sentences
                if remaining == candidate or remaining.startswith(candidate + " ")
            ),
            None,
        )
        if sentence is None:
            return matched, False
        matched.append(sentence)
        remaining = remaining[len(sentence) :].lstrip()
    return matched, True


def evaluate_chunker(
    samples: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Mide límites, conservación, overlap y repetibilidad del chunker."""
    all_chunks: list[str] = []
    lost_sentence_count = 0
    cut_sentence_count = 0
    reproducible = True
    failures: list[dict[str, Any]] = []
    per_language_counts = {
        language: {"documents": 0, "chunks": 0}
        for language in LANGUAGE_CODES
    }

    for sample in samples:
        language = str(sample["language"])
        sentences = [str(value) for value in sample["expected_sentences"]]
        chunks = create_chunks(sentences, max_tokens=MAX_WORD_UNITS)
        repeated = create_chunks(sentences, max_tokens=MAX_WORD_UNITS)
        reproducible = reproducible and chunks == repeated
        observed_sentences: list[str] = []

        for chunk in chunks:
            matched, complete = _decompose_chunk(chunk, sentences)
            observed_sentences.extend(matched)
            if not complete:
                cut_sentence_count += 1

        expected_counts = Counter(sentences)
        observed_counts = Counter(observed_sentences)
        lost = sum(
            max(expected_counts[sentence] - observed_counts[sentence], 0)
            for sentence in expected_counts
        )
        lost_sentence_count += lost
        all_chunks.extend(chunks)
        per_language_counts[language]["documents"] += 1
        per_language_counts[language]["chunks"] += len(chunks)

        if lost or chunks != repeated:
            failures.append(
                {
                    "component": "chunker",
                    "sample_id": sample["sample_id"],
                    "lost_sentences": lost,
                    "reproducible": chunks == repeated,
                }
            )

    oversized_sentence = " ".join(f"unidad{i}" for i in range(15))
    oversized_chunks = create_chunks(
        [oversized_sentence],
        max_tokens=MAX_WORD_UNITS,
    )
    all_chunks.extend(oversized_chunks)
    sizes = [count_words_as_tokens(chunk) for chunk in all_chunks]

    overlap_sentences = [
        "Uno dos tres.",
        "Cuatro cinco.",
        "Seis siete.",
        "Ocho nueve.",
    ]
    overlap_chunks = create_chunks(
        overlap_sentences,
        max_tokens=5,
        overlap_tokens=OVERLAP_WORD_UNITS,
    )
    baseline_units = count_words_as_tokens(" ".join(overlap_sentences))
    overlap_total_units = sum(count_words_as_tokens(chunk) for chunk in overlap_chunks)
    duplicated_units = max(overlap_total_units - baseline_units, 0)

    exceeding = sum(size > MAX_WORD_UNITS for size in sizes)
    metrics = {
        "document_count": len(samples),
        "total_chunks": len(all_chunks),
        "average_counted_units_per_chunk": _mean(
            [float(size) for size in sizes]
        ),
        "minimum_counted_units": min(sizes, default=0),
        "maximum_counted_units": max(sizes, default=0),
        "exceeding_limit_rate": _safe_ratio(exceeding, len(all_chunks)),
        "exceeding_limit_count": exceeding,
        "cut_sentence_count": cut_sentence_count,
        "lost_sentence_count": lost_sentence_count,
        "empty_chunk_count": sum(not chunk.strip() for chunk in all_chunks),
        "oversized_sentence_chunk_count": len(oversized_chunks),
        "overlap_evaluation": {
            "max_counted_units": 5,
            "overlap_counted_units": OVERLAP_WORD_UNITS,
            "chunks": overlap_chunks,
            "duplicated_counted_units": duplicated_units,
            "duplicated_content_rate": _safe_ratio(
                duplicated_units,
                overlap_total_units,
            ),
        },
        "reproducible": reproducible,
        "by_language": per_language_counts,
        "counting_unit": "palabras separadas por espacios (aproximación)",
    }
    return metrics, failures


def build_results() -> dict[str, Any]:
    """Ejecuta todas las evaluaciones y compone el artefacto final."""
    language_samples = _load_language_samples()
    sentence_samples = _load_sentence_samples()
    detector, detector_failures = evaluate_detector(language_samples)
    splitter, splitter_failures = evaluate_splitter(sentence_samples)
    chunker, chunker_failures = evaluate_chunker(sentence_samples)
    failures = [
        *detector_failures,
        *splitter_failures,
        *chunker_failures,
    ]

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": {
            "languages": list(LANGUAGE_CODES),
            "splitter_backends": {"es": "es", "en": "en", "pt": "es"},
            "chunker_max_counted_units": MAX_WORD_UNITS,
            "overlap_evaluation_counted_units": OVERLAP_WORD_UNITS,
            "counting_unit": "palabras separadas por espacios (aproximación)",
        },
        "detector": detector,
        "splitter": splitter,
        "chunker": chunker,
        "failed_cases": failures,
        "by_language": {
            language: {
                "detector": detector["by_language"][language],
                "splitter": splitter["by_language"][language],
                "chunker": chunker["by_language"][language],
            }
            for language in LANGUAGE_CODES
        },
        "limitations": [
            "Los fixtures son sintéticos y no representan el corpus CODEFEST.",
            "La confianza del detector no es una probabilidad calibrada.",
            "Portugués usa provisionalmente las reglas españolas de PySBD.",
            "El límite del chunker usa palabras aproximadas, no tokens del encoder.",
            "El pipeline JSONL permanece fuera del flujo evaluado.",
        ],
    }


def main() -> None:
    """Guarda las métricas y muestra un resumen legible."""
    results = build_results()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    detector = results["detector"]
    splitter = results["splitter"]
    chunker = results["chunker"]
    print("Evaluación de chunking completada")
    print(f"- Detector: accuracy={detector['accuracy']:.3f}")
    print(
        "- Splitter: "
        f"exact_match={splitter['exact_match_rate']:.3f}, "
        f"boundary_f1={splitter['boundary_f1']:.3f}"
    )
    print(
        "- Chunker: "
        f"chunks={chunker['total_chunks']}, "
        f"cut_sentences={chunker['cut_sentence_count']}, "
        f"lost_sentences={chunker['lost_sentence_count']}"
    )
    print(f"- Casos fallidos: {len(results['failed_cases'])}")
    print(f"- Resultado: {RESULT_PATH.relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
