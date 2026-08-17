"""Sentence segmentation using cached PySBD backends and bounded text blocks."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

import pysbd  # type: ignore[import-untyped]

from .language_detector import detect_language

logger = logging.getLogger(__name__)

DEFAULT_TARGET_BLOCK_CHARACTERS = 32_768
DEFAULT_MAXIMUM_BLOCK_CHARACTERS = 65_536

_SUPPORTED_LANGUAGE_CODES = frozenset({"es", "en", "pt"})
_PARAGRAPH_BOUNDARY = re.compile(r"(?:\r\n|\r|\n)[^\S\r\n]*(?:\r\n|\r|\n)+")
_NEWLINE_BOUNDARY = re.compile(r"\r\n|\r|\n")
_WHITESPACE_BOUNDARY = re.compile(r"\s+")
_BOUNDARY_PATTERNS = (
    _PARAGRAPH_BOUNDARY,
    _NEWLINE_BOUNDARY,
    _WHITESPACE_BOUNDARY,
)

# PySBD has no Portuguese rules, so Portuguese uses Spanish rules as a fallback.
_PYSBD_LANGUAGE_CODES = {
    "es": "es",
    "en": "en",
    "pt": "es",
}

_TERMINAL_SENTENCE_END = re.compile(r"[.!?…][\"'”’»\)\]\}]*$")
_PDF_PAGE_MARKER = re.compile(r"\[Pagina\s+\d+\]")
_JSON_FIELD_START = re.compile(
    r"^(?:title|date|authors?\[\d+\]|abstract|keywords?\[\d+\]|doi|issue|"
    r"body_paragraphs\[\d+\]|lists\[\d+\]|alerta_meta\.[^: ]+):",
    re.IGNORECASE,
)
_NUMBERED_CONTINUATION = re.compile(
    r"^[\"'“‘«]*(?:\(?[a-zivx]+\)|\(\d+\))\s*",
    re.IGNORECASE,
)
_NUMBERED_HEADING_TAIL = re.compile(
    r"(?:^|[.!?…]\s+)\d+(?:\.\d+)*\.\s+[^.!?…]{1,100}$"
)
_FOOTNOTE_START = re.compile(r"^\d{1,3}\s+\S")
_PDF_SOURCE_LINE = re.compile(r"^(?:Fuente|Source):", re.IGNORECASE)
_PDF_TABLE_CODE = re.compile(r"\([A-Z]{2,4}\)")
_PDF_NUMERIC_CELL = re.compile(r"^[-+]?\d[\d.,%]*$")
_LEGAL_RECITAL_START = re.compile(
    r"^(?:Tomando\s+nota|Recordando|Reconociendo|Considerando|Rappelant|"
    r"Considérant|Reconnaissant|Recalling|Recognizing|Considering)\b",
    re.IGNORECASE,
)
_CONTINUATION_CONNECTORS = frozenset(
    {
        "a",
        "al",
        "and",
        "as",
        "con",
        "da",
        "das",
        "de",
        "del",
        "do",
        "dos",
        "e",
        "el",
        "en",
        "for",
        "in",
        "la",
        "las",
        "los",
        "o",
        "of",
        "on",
        "or",
        "para",
        "por",
        "que",
        "the",
        "to",
        "um",
        "uma",
        "y",
    }
)


@dataclass(frozen=True)
class _BoundedSegmentationStats:
    """Operational measurements for one bounded segmentation call."""

    block_count: int
    largest_block_characters: int
    maximum_carry_characters: int
    oversized_carry_warning_count: int
    duration_seconds: float


@dataclass(frozen=True)
class _BoundedSegmentationResult:
    """Sentences and measurements produced by bounded segmentation."""

    sentences: list[str]
    stats: _BoundedSegmentationStats
    traces: tuple[_BoundedSegmentationTrace, ...] = ()


@dataclass(frozen=True)
class _BoundedSegmentationTrace:
    """Content-safe measurements for one diagnostic execution block."""

    block_number: int
    source_start_offset: int
    source_end_offset: int
    raw_block_characters: int
    combined_carry_block_characters: int
    segmented_item_count: int
    emitted_item_count: int
    carry_before_characters: int
    carry_after_characters: int
    oversized_carry_warning: bool
    carry_before_normalized_digest: str
    carry_after_normalized_digest: str
    emitted_content_normalized_digest: str


@lru_cache(maxsize=2)
def _get_pysbd_segmenter(pysbd_language_code: str) -> pysbd.Segmenter:
    """Obtain a cached PySBD segmenter for the given language code."""
    return pysbd.Segmenter(language=pysbd_language_code, clean=False)


def _get_segmenter(language_code: str) -> pysbd.Segmenter:
    """Resolve a supported language to one reusable backend segmenter."""
    if language_code not in _SUPPORTED_LANGUAGE_CODES:
        raise ValueError(
            f"Language code not supported: {language_code!r}. Use 'es', 'en' or 'pt'."
        )

    pysbd_language_code = _PYSBD_LANGUAGE_CODES[language_code]
    if language_code == "pt":
        logger.warning("PySBD does not support Portuguese; Spanish rules will be used.")

    return _get_pysbd_segmenter(pysbd_language_code)


def _validate_block_sizes(
    target_block_characters: int,
    maximum_block_characters: int,
) -> None:
    """Validate bounded splitter execution parameters."""
    for name, value in (
        ("target_block_characters", target_block_characters),
        ("maximum_block_characters", maximum_block_characters),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
    if target_block_characters > maximum_block_characters:
        raise ValueError(
            "target_block_characters cannot exceed maximum_block_characters."
        )


def _first_boundary_at_or_after(
    text: str,
    *,
    target: int,
    limit: int,
) -> int | None:
    """Return the first preferred boundary between target and limit."""
    for pattern in _BOUNDARY_PATTERNS:
        match = pattern.search(text, target, limit)
        if match is not None:
            return match.end()
    return None


def _last_boundary_before(
    text: str,
    *,
    start: int,
    target: int,
) -> int | None:
    """Return the last preferred boundary before the target position."""
    for pattern in _BOUNDARY_PATTERNS:
        last_end: int | None = None
        for match in pattern.finditer(text, start, target):
            last_end = match.end()
        if last_end is not None and last_end > start:
            return last_end
    return None


def _iter_text_blocks(
    text: str,
    *,
    target_block_characters: int = DEFAULT_TARGET_BLOCK_CHARACTERS,
    maximum_block_characters: int = DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
) -> Iterator[str]:
    """Yield deterministic contiguous blocks without discarding characters.

    Boundaries are preferred after blank-line paragraphs, then newlines, then
    other whitespace. A hard maximum-character boundary is used only when no
    preferred boundary is available.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    _validate_block_sizes(
        target_block_characters,
        maximum_block_characters,
    )
    if not text:
        return

    start = 0
    text_length = len(text)
    while start < text_length:
        limit = min(start + maximum_block_characters, text_length)
        if limit == text_length:
            yield text[start:]
            return

        target = min(start + target_block_characters, limit)
        boundary = _first_boundary_at_or_after(
            text,
            target=target,
            limit=limit,
        )
        if boundary is None:
            boundary = _last_boundary_before(
                text,
                start=start,
                target=target,
            )
        end = boundary if boundary is not None else limit
        if end <= start:
            end = limit
        block = text[start:end]
        if block:
            yield block
        start = end


def _clean_segments(segments: Sequence[str]) -> list[str]:
    """Strip PySBD boundary whitespace and discard empty items."""
    return [segment.strip() for segment in segments if segment.strip()]


def _prepare_document_text(text: str, *, document_format: str) -> str:
    """Normalize PDF newlines without discarding extraction provenance.

    The accepted ingestion artifact retains line breaks and synthetic page
    labels. They remain available to the deterministic reconciliation pass so
    it can distinguish line wraps, paragraphs, and page labels.
    """
    if not isinstance(document_format, str):
        raise TypeError("document_format must be a string.")
    if document_format.casefold() != "pdf":
        return text

    return text.replace("\r\n", "\n").replace("\r", "\n")


def _first_alpha(text: str) -> str:
    """Return the first alphabetic character, or an empty string."""
    return next((character for character in text if character.isalpha()), "")


def _last_word(text: str) -> str:
    """Return a normalized final word for continuation checks."""
    words = text.split()
    if not words:
        return ""
    return words[-1].casefold().strip(".,;:!?…()[]{}\"'”’»-")


def _looks_like_prose_line(text: str) -> bool:
    """Reject short or all-caps structural lines from PDF wrap merging."""
    words = text.split()
    if len(words) < 5:
        return False
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    if len(alpha_words) / len(words) < 0.60:
        return False
    upper_words = sum(word.strip(".,:;()[]").isupper() for word in alpha_words)
    return upper_words / len(alpha_words) < 0.70


def _looks_like_spaced_pdf_heading(text: str) -> bool:
    """Recognize extraction-split headings rendered as spaced capitals."""
    words = text.split()[:16]
    return sum(len(word) == 1 and word.isupper() for word in words) >= 4


def _looks_like_pdf_table_continuation(left: str, right: str) -> bool:
    """Recognize a country-code table followed by its numeric value cells."""
    if len(_PDF_TABLE_CODE.findall(left)) < 4:
        return False
    return sum(
        _PDF_NUMERIC_CELL.match(word) is not None for word in right.split()
    ) >= 4


def _should_reconcile_boundary(
    left: str,
    right: str,
    *,
    separator: str,
    document_format: str,
) -> bool:
    """Recognize only deterministic impossible-continuation boundaries."""
    if not left or not right or _TERMINAL_SENTENCE_END.search(left.rstrip()):
        return False
    if _JSON_FIELD_START.match(right.lstrip()):
        return False

    first_alpha = _first_alpha(right)
    continuous_whitespace = "\n" not in separator and "\r" not in separator
    lexical_continuation = (
        first_alpha.islower()
        or _last_word(left) in _CONTINUATION_CONNECTORS
        or left.rstrip().endswith((",", ";", ":", "-", "–"))
        or _NUMBERED_CONTINUATION.match(right.lstrip()) is not None
    )
    if continuous_whitespace:
        return lexical_continuation

    if document_format.casefold() != "pdf":
        return False
    if _PDF_PAGE_MARKER.match(right.lstrip()):
        return True
    line_break_count = separator.count("\n") + separator.count("\r")
    if line_break_count == 1 and lexical_continuation:
        return True
    if line_break_count >= 2 and _LEGAL_RECITAL_START.match(right.lstrip()):
        return True
    if line_break_count == 1 and (
        _PDF_SOURCE_LINE.match(right.lstrip())
        or _looks_like_spaced_pdf_heading(right)
        or _looks_like_pdf_table_continuation(left, right)
    ):
        return True
    if (
        line_break_count == 1
        and _looks_like_prose_line(left)
        and _looks_like_prose_line(right)
    ):
        return True
    if (
        line_break_count >= 1
        and len(left.split()) <= 12
        and _looks_like_prose_line(right)
    ):
        # A short non-terminal PDF heading/label is a structural prefix for
        # following prose. Two adjacent short structural cells are not a
        # heading/body pair and must remain independently packable.
        return True
    if _NUMBERED_HEADING_TAIL.search(left.rstrip()):
        return True
    return _FOOTNOTE_START.match(right.lstrip()) is not None and len(left.split()) >= 8


def _segment_offsets(text: str, sentences: Sequence[str]) -> list[tuple[int, int]]:
    """Locate clean PySBD items in their unchanged source order."""
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start < 0:
            raise ValueError(
                "sentence reconciliation could not preserve source ordering; "
                "the PySBD output was not found in the prepared document text."
            )
        end = start + len(sentence)
        offsets.append((start, end))
        cursor = end
    return offsets


def _reconcile_invalid_boundaries(
    text: str,
    sentences: Sequence[str],
    *,
    document_format: str,
) -> list[str]:
    """Merge proven invalid PySBD boundaries without inventing content."""
    if len(sentences) < 2:
        return list(sentences)

    offsets = _segment_offsets(text, sentences)
    reconciled = [sentences[0]]
    previous_end = offsets[0][1]
    for index, sentence in enumerate(sentences[1:], start=1):
        start, end = offsets[index]
        separator = text[previous_end:start]
        if _should_reconcile_boundary(
            reconciled[-1],
            sentence,
            separator=separator,
            document_format=document_format,
        ):
            reconciled[-1] = " ".join((reconciled[-1].rstrip(), sentence.lstrip()))
        else:
            reconciled.append(sentence)
        previous_end = end
    return reconciled


def _normalized_digest(text: str) -> str:
    """Hash normalized text for content-safe diagnostic correlation."""
    normalized = " ".join(text.split())
    return sha256(normalized.encode("utf-8")).hexdigest()


def _split_sentences_whole_document(
    text: str,
    *,
    language: str,
) -> list[str]:
    """Preserve the former whole-document PySBD behavior for comparisons."""
    normalized_text = text.strip()
    if not normalized_text:
        return []
    segmenter = _get_segmenter(language)
    return _clean_segments(segmenter.segment(normalized_text))


def _split_sentences_bounded_with_stats(
    text: str,
    *,
    language: str,
    document_format: str | None = None,
    target_block_characters: int = DEFAULT_TARGET_BLOCK_CHARACTERS,
    maximum_block_characters: int = DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
    trace: bool = False,
) -> _BoundedSegmentationResult:
    """Segment bounded blocks while retaining each uncertain final item.

    The final PySBD item from every non-final block remains as carry. It is
    prepended to the next contiguous block and resegmented with the additional
    context. A carry may exceed the normal maximum rather than cutting a
    complete sentence.
    """
    _validate_block_sizes(
        target_block_characters,
        maximum_block_characters,
    )
    if not isinstance(trace, bool):
        raise TypeError("trace must be a boolean.")
    prepared_text = (
        _prepare_document_text(text, document_format=document_format)
        if document_format is not None
        else text
    )
    normalized_text = prepared_text.strip()
    started = time.perf_counter()
    if not normalized_text:
        return _BoundedSegmentationResult(
            sentences=[],
            stats=_BoundedSegmentationStats(
                block_count=0,
                largest_block_characters=0,
                maximum_carry_characters=0,
                oversized_carry_warning_count=0,
                duration_seconds=time.perf_counter() - started,
            ),
        )

    segmenter = _get_segmenter(language)
    block_iterator = iter(
        _iter_text_blocks(
            normalized_text,
            target_block_characters=target_block_characters,
            maximum_block_characters=maximum_block_characters,
        )
    )
    current_block = next(block_iterator)
    carry = ""
    sentences: list[str] = []
    block_count = 0
    largest_block_characters = 0
    maximum_carry_characters = 0
    oversized_carry_warning_count = 0
    oversized_carry_active = False
    source_start_offset = 0
    traces: list[_BoundedSegmentationTrace] = []

    while True:
        try:
            next_block = next(block_iterator)
            is_final_block = False
        except StopIteration:
            next_block = None
            is_final_block = True

        block_count += 1
        source_end_offset = source_start_offset + len(current_block)
        largest_block_characters = max(
            largest_block_characters,
            len(current_block),
        )
        carry_before = carry
        combined_text = f"{carry_before}{current_block}"
        raw_segments = [
            segment for segment in segmenter.segment(combined_text) if segment.strip()
        ]

        emitted_segments: list[str]
        if is_final_block:
            if raw_segments:
                emitted_segments = _clean_segments(raw_segments)
            elif combined_text.strip():
                emitted_segments = [combined_text.strip()]
            else:
                emitted_segments = []
            sentences.extend(emitted_segments)
            carry = ""
        else:
            if raw_segments:
                emitted_segments = _clean_segments(raw_segments[:-1])
                carry = raw_segments[-1]
            else:
                emitted_segments = []
                carry = combined_text
            sentences.extend(emitted_segments)
        maximum_carry_characters = max(
            maximum_carry_characters,
            len(carry),
        )
        oversized_carry_warning = len(carry) > maximum_block_characters
        if oversized_carry_warning:
            if not oversized_carry_active:
                oversized_carry_warning_count += 1
                logger.warning(
                    "Sentence carry exceeded the bounded segmentation maximum; "
                    "the complete sentence will remain intact. carry_chars=%d "
                    "maximum_block_chars=%d.",
                    len(carry),
                    maximum_block_characters,
                )
            oversized_carry_active = True
        else:
            oversized_carry_active = False

        if trace:
            traces.append(
                _BoundedSegmentationTrace(
                    block_number=block_count,
                    source_start_offset=source_start_offset,
                    source_end_offset=source_end_offset,
                    raw_block_characters=len(current_block),
                    combined_carry_block_characters=len(combined_text),
                    segmented_item_count=len(raw_segments),
                    emitted_item_count=len(emitted_segments),
                    carry_before_characters=len(carry_before),
                    carry_after_characters=len(carry),
                    oversized_carry_warning=oversized_carry_warning,
                    carry_before_normalized_digest=_normalized_digest(carry_before),
                    carry_after_normalized_digest=_normalized_digest(carry),
                    emitted_content_normalized_digest=_normalized_digest(
                        " ".join(emitted_segments)
                    ),
                )
            )

        if is_final_block:
            break

        if next_block is None:
            break
        current_block = next_block
        source_start_offset = source_end_offset

    duration = time.perf_counter() - started
    stats = _BoundedSegmentationStats(
        block_count=block_count,
        largest_block_characters=largest_block_characters,
        maximum_carry_characters=maximum_carry_characters,
        oversized_carry_warning_count=oversized_carry_warning_count,
        duration_seconds=duration,
    )
    logger.debug(
        "Bounded segmentation completed: characters=%d blocks=%d "
        "largest_block=%d maximum_carry=%d carry_warnings=%d duration=%.3fs.",
        len(normalized_text),
        stats.block_count,
        stats.largest_block_characters,
        stats.maximum_carry_characters,
        stats.oversized_carry_warning_count,
        stats.duration_seconds,
    )
    reconciled_sentences = (
        _reconcile_invalid_boundaries(
            normalized_text,
            sentences,
            document_format=document_format,
        )
        if document_format is not None
        else sentences
    )
    return _BoundedSegmentationResult(
        sentences=reconciled_sentences,
        stats=stats,
        traces=tuple(traces),
    )


def _split_sentences_bounded(
    text: str,
    *,
    language: str,
    target_block_characters: int = DEFAULT_TARGET_BLOCK_CHARACTERS,
    maximum_block_characters: int = DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
) -> list[str]:
    """Return sentences from bounded carry-based segmentation."""
    return _split_sentences_bounded_with_stats(
        text,
        language=language,
        target_block_characters=target_block_characters,
        maximum_block_characters=maximum_block_characters,
    ).sentences


def _normalize_for_integrity(text: str) -> str:
    """Normalize whitespace linearly for controlled integrity comparisons."""
    return " ".join(text.split())


def split_sentences(
    text: str,
    language: str | None = None,
    *,
    bounded: bool = True,
    target_block_characters: int = DEFAULT_TARGET_BLOCK_CHARACTERS,
    maximum_block_characters: int = DEFAULT_MAXIMUM_BLOCK_CHARACTERS,
) -> list[str]:
    """Divide text into complete sentences with one language resolution."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(bounded, bool):
        raise TypeError("bounded must be a boolean.")

    normalized_text = text.strip()
    if not normalized_text:
        return []

    if language is None:
        detection = detect_language(normalized_text)
        language_code = detection.language
        logger.debug(
            "Detected language: language=%s confidence=%.3f margin=%.3f ambiguous=%s.",
            detection.language,
            detection.confidence,
            detection.margin,
            detection.is_ambiguous,
        )
    elif language in _SUPPORTED_LANGUAGE_CODES:
        language_code = language
    else:
        raise ValueError(
            f"Language not supported: {language!r}. Use 'es', 'en' or 'pt'."
        )

    if bounded:
        return _split_sentences_bounded(
            normalized_text,
            language=language_code,
            target_block_characters=target_block_characters,
            maximum_block_characters=maximum_block_characters,
        )
    return _split_sentences_whole_document(
        normalized_text,
        language=language_code,
    )
