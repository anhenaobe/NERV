"""Focused contracts for deterministic encoder hard-limit subdivision."""

import pytest

from nerv.chunking.hard_limit import split_hard_limited_text


class _CharacterCounter:
    """Count each code point and expose exact code-point offsets."""

    def __init__(self, prefix_tokens: int = 2) -> None:
        self.prefix_tokens = prefix_tokens

    def count(self, text: str, *, include_document_prefix: bool = False) -> int:
        return len(text) + (self.prefix_tokens if include_document_prefix else 0)

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(text))]


def _reconstruct(result: object, text: str) -> str:
    parts = result.parts  # type: ignore[attr-defined]
    return "".join(text[part.coverage_start : part.coverage_end] for part in parts)


@pytest.mark.parametrize("content_length", [254, 255, 509, 510])
def test_safe_ranges_remain_byte_for_byte_intact(content_length: int) -> None:
    text = "á" * content_length
    result = split_hard_limited_text(
        text,
        document_format="txt",
        token_counter=_CharacterCounter(),
        hard_limit=512,
        overlap_tokens=32,
    )
    assert result.strategy == "unchanged"
    assert len(result.parts) == 1
    assert result.parts[0].text == text


@pytest.mark.parametrize("content_length", [511, 575, 998, 1598, 4198, 10237])
def test_token_offset_fallback_covers_extreme_inputs(content_length: int) -> None:
    text = "x" * content_length
    counter = _CharacterCounter()
    first = split_hard_limited_text(
        text,
        document_format="txt",
        token_counter=counter,
        hard_limit=512,
        overlap_tokens=32,
    )
    second = split_hard_limited_text(
        text,
        document_format="txt",
        token_counter=counter,
        hard_limit=512,
        overlap_tokens=32,
    )
    assert first == second
    assert first.source_token_count == content_length + 2
    assert first.strategy == "token_offsets"
    assert _reconstruct(first, text) == text
    assert all(part.token_count <= 512 for part in first.parts)
    assert all(part.overlap_tokens <= 32 for part in first.parts)
    assert all(
        part.text == text[part.emitted_start : part.coverage_end]
        for part in first.parts
    )
    assert all(
        counter.count(text[part.emitted_start : part.coverage_start])
        == part.overlap_tokens
        for part in first.parts
    )


@pytest.mark.parametrize(
    ("document_format", "text", "expected_strategy"),
    [
        ("csv", "left|middle|right|tail", "tabular_cell"),
        ("json", '{"a":"111","b":"222","c":"333"}', "json_structure"),
        ("txt", "First sentence. Second sentence. Third sentence.", "punctuation"),
        ("txt", "alpha beta gamma delta epsilon", "whitespace"),
    ],
)
def test_hierarchy_prefers_semantic_boundaries(
    document_format: str,
    text: str,
    expected_strategy: str,
) -> None:
    result = split_hard_limited_text(
        text,
        document_format=document_format,
        token_counter=_CharacterCounter(prefix_tokens=1),
        hard_limit=18,
        overlap_tokens=3,
    )
    assert result.strategy == expected_strategy
    assert _reconstruct(result, text) == text
    assert all(part.token_count <= 18 for part in result.parts)


def test_invalid_hard_limit_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="smaller than hard_limit"):
        split_hard_limited_text(
            "oversized",
            document_format="txt",
            token_counter=_CharacterCounter(),
            hard_limit=4,
            overlap_tokens=4,
        )


@pytest.mark.parametrize(
    ("document_format", "text", "strategy"),
    [
        ("csv", ",".join("c" * 180 for _ in range(8)), "tabular_cell"),
        (
            "json",
            "{" + ",".join(f'\"k{index}\":\"{"v" * 180}\"' for index in range(8)) + "}",
            "json_structure",
        ),
    ],
)
def test_very_large_structured_units_use_format_boundaries(
    document_format: str,
    text: str,
    strategy: str,
) -> None:
    result = split_hard_limited_text(
        text,
        document_format=document_format,
        token_counter=_CharacterCounter(),
        hard_limit=512,
        overlap_tokens=32,
    )
    assert result.strategy == strategy
    assert _reconstruct(result, text) == text
    assert all(part.token_count <= 512 for part in result.parts)
