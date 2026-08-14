"""Deterministic subdivision of logical chunks above the encoder hard limit."""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol


class _HardLimitCounter(Protocol):
    """Tokenizer operations required by hard-limit subdivision."""

    def count(self, text: str, *, include_document_prefix: bool = False) -> int: ...

    def token_offsets(self, text: str) -> list[tuple[int, int]]: ...


@dataclass(frozen=True)
class HardSplitPart:
    """One emitted part plus its exact, non-overlapping source coverage."""

    text: str
    token_count: int
    coverage_start: int
    coverage_end: int
    emitted_start: int
    overlap_tokens: int


@dataclass(frozen=True)
class HardSplitResult:
    """A complete ordered subdivision of one logical chunk."""

    parts: tuple[HardSplitPart, ...]
    strategy: str
    source_token_count: int


_TABULAR_FORMATS = frozenset({"csv", "tsv", "xls", "xlsx"})
_BOUNDARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("newline", re.compile(r"\r\n|[\r\n]")),
    ("semicolon", re.compile(r";")),
    ("colon", re.compile(r":")),
    ("comma", re.compile(r",")),
    ("punctuation", re.compile(r"[.!?。！？]")),
    ("whitespace", re.compile(r"\s+")),
)


def _count_encoder_input(counter: _HardLimitCounter, text: str) -> int:
    return counter.count(text, include_document_prefix=True)


def _boundary_ends(pattern: re.Pattern[str], text: str) -> set[int]:
    return {match.end() for match in pattern.finditer(text) if match.end() < len(text)}


def _tabular_boundary_ends(text: str) -> set[int]:
    """Return common cell delimiters outside RFC-style double-quoted cells."""
    boundaries: set[int] = set()
    in_quotes = False
    index = 0
    while index < len(text):
        character = text[index]
        if character == '"':
            if in_quotes and index + 1 < len(text) and text[index + 1] == '"':
                index += 2
                continue
            in_quotes = not in_quotes
        elif not in_quotes and character in {",", "\t", "|"}:
            boundaries.add(index + 1)
        index += 1
    return {boundary for boundary in boundaries if boundary < len(text)}


def _json_boundary_ends(text: str) -> set[int]:
    """Return structural separators outside JSON string literals."""
    boundaries: set[int] = set()
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in {",", "}", "]"} and index + 1 < len(text):
            boundaries.add(index + 1)
    return boundaries


def _partition_at_boundaries(
    text: str,
    *,
    candidate_ends: set[int],
    counter: _HardLimitCounter,
    hard_limit: int,
) -> list[tuple[int, int]] | None:
    """Greedily partition at the farthest available safe boundary."""
    ordered = sorted(candidate_ends | {len(text)})
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        if _count_encoder_input(counter, text[start:]) <= hard_limit:
            ranges.append((start, len(text)))
            break
        first_candidate = bisect_right(ordered, start)
        low = first_candidate
        high = len(ordered) - 1
        safe_end: int | None = None
        while low <= high:
            middle = (low + high) // 2
            end = ordered[middle]
            if _count_encoder_input(counter, text[start:end]) <= hard_limit:
                safe_end = end
                low = middle + 1
            else:
                high = middle - 1
        if safe_end is None or safe_end == len(text):
            return None
        ranges.append((start, safe_end))
        start = safe_end
    return ranges


def _largest_safe_character_end(
    text: str,
    *,
    start: int,
    counter: _HardLimitCounter,
    hard_limit: int,
) -> int:
    """Find a safe code-point boundary when offsets are unavailable or sparse."""
    low = start + 1
    high = len(text)
    best = start
    while low <= high:
        middle = (low + high) // 2
        if _count_encoder_input(counter, text[start:middle]) <= hard_limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best == start:
        raise ValueError(
            "a single source code point cannot fit within the encoder hard limit."
        )
    return best


def _partition_by_token_offsets(
    text: str,
    *,
    counter: _HardLimitCounter,
    hard_limit: int,
) -> list[tuple[int, int]]:
    """Partition using tokenizer offsets, with exact-count character fallback."""
    try:
        offsets = counter.token_offsets(text)
    except (AttributeError, ValueError):
        offsets = []
    token_ends = sorted({end for _, end in offsets})
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        if _count_encoder_input(counter, text[start:]) <= hard_limit:
            ranges.append((start, len(text)))
            break
        first_token = bisect_right(token_ends, start)
        candidate_index = min(
            first_token + hard_limit - 1,
            len(token_ends) - 1,
        )
        safe_end = start
        while candidate_index >= first_token:
            candidate_end = token_ends[candidate_index]
            if candidate_end < len(text) and (
                _count_encoder_input(counter, text[start:candidate_end])
                <= hard_limit
            ):
                safe_end = candidate_end
                break
            candidate_index -= 1
        while candidate_index + 1 < len(token_ends):
            candidate_end = token_ends[candidate_index + 1]
            if candidate_end >= len(text) or (
                _count_encoder_input(counter, text[start:candidate_end])
                > hard_limit
            ):
                break
            safe_end = candidate_end
            candidate_index += 1
        if safe_end == start:
            safe_end = _largest_safe_character_end(
                text,
                start=start,
                counter=counter,
                hard_limit=hard_limit,
            )
        ranges.append((start, safe_end))
        start = safe_end
    return ranges


def _overlap_start(
    text: str,
    *,
    coverage_start: int,
    coverage_end: int,
    counter: _HardLimitCounter,
    hard_limit: int,
    overlap_tokens: int,
) -> tuple[int, int]:
    if coverage_start == 0 or overlap_tokens == 0:
        return coverage_start, 0
    try:
        offsets = counter.token_offsets(text[:coverage_start])
        starts = [start for start, _ in offsets]
        emitted_start = (
            starts[max(0, len(starts) - overlap_tokens)]
            if starts and overlap_tokens
            else coverage_start
        )
    except (AttributeError, ValueError):
        emitted_start = max(0, coverage_start - overlap_tokens)
    while emitted_start < coverage_start:
        overlap_count = counter.count(text[emitted_start:coverage_start])
        if overlap_count <= overlap_tokens and (
            _count_encoder_input(counter, text[emitted_start:coverage_end])
            <= hard_limit
        ):
            return emitted_start, overlap_count
        emitted_start += 1
    return coverage_start, 0


def _materialize_parts(
    text: str,
    *,
    ranges: list[tuple[int, int]],
    counter: _HardLimitCounter,
    hard_limit: int,
    overlap_tokens: int,
) -> tuple[HardSplitPart, ...]:
    parts: list[HardSplitPart] = []
    for coverage_start, coverage_end in ranges:
        emitted_start, actual_overlap = _overlap_start(
            text,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            counter=counter,
            hard_limit=hard_limit,
            overlap_tokens=overlap_tokens,
        )
        emitted_text = text[emitted_start:coverage_end]
        token_count = _count_encoder_input(counter, emitted_text)
        if token_count > hard_limit:
            raise AssertionError("hard-limit subdivision emitted an oversized part.")
        parts.append(
            HardSplitPart(
                text=emitted_text,
                token_count=token_count,
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                emitted_start=emitted_start,
                overlap_tokens=actual_overlap,
            )
        )
    if "".join(text[part.coverage_start : part.coverage_end] for part in parts) != text:
        raise AssertionError("hard-limit subdivision did not preserve full coverage.")
    return tuple(parts)


def split_hard_limited_text(
    text: str,
    *,
    document_format: str,
    token_counter: _HardLimitCounter,
    hard_limit: int,
    overlap_tokens: int,
) -> HardSplitResult:
    """Keep safe text intact or subdivide it with hierarchical boundaries."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text:
        raise ValueError("text must be non-empty.")
    if hard_limit <= 0:
        raise ValueError("hard_limit must be greater than zero.")
    if overlap_tokens < 0 or overlap_tokens >= hard_limit:
        raise ValueError(
            "overlap_tokens must be non-negative and smaller than hard_limit."
        )
    source_token_count = _count_encoder_input(token_counter, text)
    if source_token_count <= hard_limit:
        return HardSplitResult(
            parts=(
                HardSplitPart(
                    text=text,
                    token_count=source_token_count,
                    coverage_start=0,
                    coverage_end=len(text),
                    emitted_start=0,
                    overlap_tokens=0,
                ),
            ),
            strategy="unchanged",
            source_token_count=source_token_count,
        )

    candidates: set[int] = set()
    normalized_format = document_format.casefold()
    if normalized_format in _TABULAR_FORMATS:
        candidates.update(_tabular_boundary_ends(text))
        ranges = _partition_at_boundaries(
            text,
            candidate_ends=candidates,
            counter=token_counter,
            hard_limit=hard_limit,
        )
        if ranges is not None:
            strategy = "tabular_cell"
        else:
            strategy = ""
    elif normalized_format == "json":
        candidates.update(_json_boundary_ends(text))
        ranges = _partition_at_boundaries(
            text,
            candidate_ends=candidates,
            counter=token_counter,
            hard_limit=hard_limit,
        )
        if ranges is not None:
            strategy = "json_structure"
        else:
            strategy = ""
    else:
        ranges = None
        strategy = ""

    if ranges is None:
        for boundary_name, pattern in _BOUNDARY_PATTERNS:
            candidates.update(_boundary_ends(pattern, text))
            ranges = _partition_at_boundaries(
                text,
                candidate_ends=candidates,
                counter=token_counter,
                hard_limit=hard_limit,
            )
            if ranges is not None:
                strategy = boundary_name
                break
    if ranges is None:
        ranges = _partition_by_token_offsets(
            text,
            counter=token_counter,
            hard_limit=hard_limit,
        )
        strategy = "token_offsets"

    parts = _materialize_parts(
        text,
        ranges=ranges,
        counter=token_counter,
        hard_limit=hard_limit,
        overlap_tokens=overlap_tokens,
    )
    if len(parts) < 2:
        raise AssertionError("an oversized logical chunk was not subdivided.")
    return HardSplitResult(
        parts=parts,
        strategy=strategy,
        source_token_count=source_token_count,
    )
