"""Heuristic language detection for Spanish, English and Portuguese."""

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

logger = logging.getLogger(__name__)

LANGUAGE_CODES = ("es", "en", "pt")
COMMON_WORD_WEIGHT = 0.5
DISTINCTIVE_WORD_WEIGHT = 2.0
PATTERN_WEIGHT = 1.5
MAX_PATTERN_OCCURRENCES = 3

_WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?", re.UNICODE)


@dataclass(frozen=True)
class LanguageDetectionResult:
    """Resultado observable con scores y métricas de confianza."""

    language: str
    confidence: float
    margin: float
    scores: dict[str, float]
    sample_size: int
    is_ambiguous: bool


@dataclass(frozen=True)
class _LanguageMarkers:
    common_words: frozenset[str]
    distinctive_words: frozenset[str]
    patterns: tuple[str, ...]


def _read_string_collection(
    value: object,
    *,
    language: str,
    category: str,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(
            f"Invalid markers for {language}.{category}: "
            "expected a list of non-empty strings."
        )
    return tuple(value)


@lru_cache(maxsize=1)
def _load_language_markers() -> dict[str, _LanguageMarkers]:
    """Load and prepare the linguistic markers once."""
    resource = files("nerv.chunking").joinpath(
        "resources",
        "language_markers.json",
    )
    raw_data: object = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict) or set(raw_data) != set(LANGUAGE_CODES):
        raise ValueError("language_markers.json must contain exactly es, en and pt.")

    markers: dict[str, _LanguageMarkers] = {}
    for language in LANGUAGE_CODES:
        language_data = raw_data[language]
        if not isinstance(language_data, dict):
            raise ValueError(f"Invalid markers for {language}: expected an object.")

        expected_categories = {
            "common_words",
            "distinctive_words",
            "patterns",
        }
        if set(language_data) != expected_categories:
            raise ValueError(
                f"Invalid markers for {language}: expected "
                "common_words, distinctive_words and patterns."
            )

        common_words = _read_string_collection(
            language_data["common_words"],
            language=language,
            category="common_words",
        )
        distinctive_words = _read_string_collection(
            language_data["distinctive_words"],
            language=language,
            category="distinctive_words",
        )
        patterns = _read_string_collection(
            language_data["patterns"],
            language=language,
            category="patterns",
        )
        markers[language] = _LanguageMarkers(
            common_words=frozenset(common_words),
            distinctive_words=frozenset(distinctive_words),
            patterns=patterns,
        )

    return markers


def _sample_text(text: str, sample_characters: int) -> str:
    """Takes a stable sample from the beginning, center and end of the text."""
    if len(text) <= 3 * sample_characters:
        return text

    center_start = (len(text) - sample_characters) // 2
    end_start = len(text) - sample_characters
    sections = (
        text[:sample_characters],
        text[center_start : center_start + sample_characters],
        text[end_start:],
    )
    if sample_characters < 3:
        return "".join(sections)

    start, center, end = sections
    return "\n".join((start, center, end[2:]))


def _extract_words(text: str) -> list[str]:
    """Extracts Unicode words without removing diacritical marks."""
    return _WORD_PATTERN.findall(text.casefold())


def _score_language(
    normalized_sample: str,
    word_frequencies: Counter[str],
    markers: _LanguageMarkers,
) -> float:
    common_score = sum(
        frequency * COMMON_WORD_WEIGHT
        for word, frequency in word_frequencies.items()
        if word in markers.common_words
    )
    distinctive_score = sum(
        frequency * DISTINCTIVE_WORD_WEIGHT
        for word, frequency in word_frequencies.items()
        if word in markers.distinctive_words
    )
    pattern_score = sum(
        min(normalized_sample.count(pattern), MAX_PATTERN_OCCURRENCES) * PATTERN_WEIGHT
        for pattern in markers.patterns
    )
    return common_score + distinctive_score + pattern_score


def _validate_configuration(
    default: str,
    sample_characters: int,
    minimum_score: float,
    minimum_confidence: float,
    minimum_margin: float,
) -> None:
    if default not in LANGUAGE_CODES:
        raise ValueError("default must be 'es', 'en' or 'pt'.")
    if sample_characters <= 0:
        raise ValueError("sample_characters must be greater than zero.")
    if minimum_score < 0:
        raise ValueError("minimum_score cannot be negative.")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be between 0.0 and 1.0.")
    if minimum_margin < 0:
        raise ValueError("minimum_margin cannot be negative.")


def detect_language(
    text: str,
    default: str = "es",
    sample_characters: int = 2000,
    minimum_score: float = 3.0,
    minimum_confidence: float = 0.50,
    minimum_margin: float = 1.5,
) -> LanguageDetectionResult:
    """Detect the predominant language among Spanish, English and Portuguese."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    _validate_configuration(
        default,
        sample_characters,
        minimum_score,
        minimum_confidence,
        minimum_margin,
    )

    sample = _sample_text(text, sample_characters)
    normalized_sample = " ".join(sample.casefold().split())
    scores = {language: 0.0 for language in LANGUAGE_CODES}

    if normalized_sample:
        word_frequencies = Counter(_extract_words(normalized_sample))
        markers = _load_language_markers()
        scores = {
            language: _score_language(
                normalized_sample,
                word_frequencies,
                markers[language],
            )
            for language in LANGUAGE_CODES
        }

    ranked_languages = sorted(
        LANGUAGE_CODES,
        key=lambda language: (-scores[language], LANGUAGE_CODES.index(language)),
    )
    best_language, second_language = ranked_languages[:2]
    best_score = scores[best_language]
    second_score = scores[second_language]
    total_score = sum(score for score in scores.values() if score > 0)
    confidence = best_score / total_score if total_score else 0.0
    margin = best_score - second_score
    is_ambiguous = (
        best_score < minimum_score
        or confidence < minimum_confidence
        or margin < minimum_margin
    )
    selected_language = default if is_ambiguous else best_language

    if is_ambiguous:
        logger.warning(
            "Ambiguous detection; %s will be used. "
            "confidence=%.3f margin=%.3f sample=%d scores=%s",
            selected_language,
            confidence,
            margin,
            len(sample),
            scores,
        )

    return LanguageDetectionResult(
        language=selected_language,
        confidence=confidence,
        margin=margin,
        scores=scores,
        sample_size=len(sample),
        is_ambiguous=is_ambiguous,
    )
