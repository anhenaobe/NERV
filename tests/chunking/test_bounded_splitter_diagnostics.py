"""Regression tests for compact bounded-splitter correctness diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from nerv.chunking.sentence_splitter import (
    _normalize_for_integrity,
    _split_sentences_bounded_with_stats,
    _split_sentences_whole_document,
)
from tests.chunking import diagnose_bounded_splitter as diagnostic

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "bounded_splitter_diagnostic_cases.json"
)
RESULT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "bounded_splitter_diagnostics_v1.json"
)


def _cases() -> dict[str, dict[str, Any]]:
    payload = cast(
        dict[str, Any], json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    )
    return {cast(str, item["case_id"]): item for item in payload["cases"]}


def test_trace_is_disabled_by_default_and_content_safe_when_enabled() -> None:
    """Normal calls retain no trace; diagnostic traces contain no source text."""
    text = "First complete sentence. Second complete sentence. Third one."

    normal = _split_sentences_bounded_with_stats(
        text,
        language="en",
        target_block_characters=15,
        maximum_block_characters=24,
    )
    traced = _split_sentences_bounded_with_stats(
        text,
        language="en",
        target_block_characters=15,
        maximum_block_characters=24,
        trace=True,
    )

    assert normal.traces == ()
    assert traced.sentences == normal.sentences
    assert traced.traces
    assert traced.traces[0].source_start_offset == 0
    assert traced.traces[-1].source_end_offset == len(text)
    assert all(
        left.source_end_offset == right.source_start_offset
        for left, right in zip(traced.traces, traced.traces[1:], strict=False)
    )
    trace_keys = asdict(traced.traces[0]).keys()
    assert not {"text", "block_text", "carry", "segments"} & trace_keys


def test_first_divergence_report_is_sequential_and_excerpt_bounded() -> None:
    """The utility reports the first mismatch and at most five items per side."""
    source = "One. Two. Three changed. Four. Five. Six. Seven. Eight."
    baseline = ["One.", "Two.", "Three.", "Four.", "Five.", "Six.", "Seven."]
    bounded = [
        "One.",
        "Two.",
        "Three changed.",
        "Four.",
        "Five.",
        "Six.",
        "Seven.",
    ]

    result = diagnostic.compare_sentence_sequences(
        baseline,
        bounded,
        source=source,
        excerpt_characters=12,
    )

    assert result["first_differing_sentence_index"] == 2
    assert len(result["matching_before"]) == 2
    assert len(result["baseline_after"]) == 5
    assert len(result["bounded_after"]) == 5
    assert result["baseline_after"][0]["length"] == len("Three.")


def test_large_probe_rules_out_a_local_carry_or_boundary_error() -> None:
    """The sanitized first-divergence region agrees with deliberately tiny blocks."""
    case = _cases()["large_remote_context_probe"]
    text = cast(str, case["text"])
    whole = _split_sentences_whole_document(text, language="en")
    bounded = _split_sentences_bounded_with_stats(
        text,
        language="en",
        target_block_characters=64,
        maximum_block_characters=96,
        trace=True,
    )

    assert bounded.sentences == whole
    assert bounded.stats.block_count > 1
    assert diagnostic._classify_sentence_difference(
        {"exact_match": False},
        [{"exact_sentence_list_match": True}],
    ) == cast(str, case["classification"])


def test_extreme_probe_is_an_inherited_pysbd_transformation() -> None:
    """Both routes insert the same space at the sanitized OCR separator."""
    case = _cases()["extreme_ocr_separator_probe"]
    text = cast(str, case["text"])
    whole = _split_sentences_whole_document(text, language="en")
    bounded = _split_sentences_bounded_with_stats(
        text,
        language="en",
        target_block_characters=24,
        maximum_block_characters=40,
        trace=True,
    )
    normalized_source = _normalize_for_integrity(text)

    assert bounded.sentences == whole
    assert _normalize_for_integrity(" ".join(whole)) != normalized_source
    reconstruction = diagnostic.diagnose_reconstruction(
        text,
        bounded.sentences,
        traces=bounded.traces,
        excerpt_characters=120,
    )
    assert reconstruction["matches"] is False
    assert reconstruction["change_kind"] == "added"
    assert reconstruction["nearest_block"] is not None
    assert diagnostic._classify_reconstruction(
        [
            {
                "exact_sentence_list_match": True,
                "whole_reconstruction_match": False,
                "bounded_reconstruction_match": False,
                "whole_first_reconstruction_difference": 77,
                "bounded_first_reconstruction_difference": 77,
                "whole_reconstruction_difference_digest": "same",
                "bounded_reconstruction_difference_digest": "same",
            }
        ]
    ) == cast(str, case["classification"])


def test_affected_block_capture_limits_pysbd_excerpts() -> None:
    """Detailed replay exposes only nearby, individually bounded excerpts."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    traced = _split_sentences_bounded_with_stats(
        text,
        language="en",
        target_block_characters=18,
        maximum_block_characters=25,
        trace=True,
    )
    target = traced.traces[1]
    result = diagnostic._capture_affected_block(
        text,
        language="en",
        block_number=target.block_number,
        source_offset=target.source_start_offset,
        excerpt_characters=10,
        target_block_characters=18,
        maximum_block_characters=25,
    )

    assert result["block_number"] == target.block_number
    assert len(result["pysbd_output_near_affected_item"]) <= 11
    assert all(
        len(item["sanitized_excerpt_shape"]) <= 10 + len("...[truncated]...")
        for item in result["pysbd_output_near_affected_item"]
    )


def test_diagnostic_evidence_does_not_publish_source_words() -> None:
    """Human-readable evidence preserves shape, not confidential source text."""
    secret = "ConfidentialProjectZephyr 2042. AnotherSecretToken."
    evidence = diagnostic._sentence_evidence(secret, secret, 200)
    reconstruction = diagnostic.diagnose_reconstruction(
        "Alpha.SecretOCR",
        ["Alpha.", "SecretOCR"],
        traces=(),
        excerpt_characters=200,
    )
    serialized = json.dumps(
        {"evidence": evidence, "reconstruction": reconstruction}
    )

    assert "ConfidentialProjectZephyr" not in serialized
    assert "AnotherSecretToken" not in serialized
    assert "SecretOCR" not in serialized
    assert "sanitized_text_shape" in evidence


def test_classifiers_require_positive_matching_evidence() -> None:
    """Empty windows and same-offset/different-context cases remain unresolved."""
    assert diagnostic._classify_sentence_difference(
        {"exact_match": False}, []
    ) == "unresolved"
    assert diagnostic._classify_reconstruction(
        [
            {
                "whole_reconstruction_match": False,
                "bounded_reconstruction_match": False,
                "whole_first_reconstruction_difference": 10,
                "bounded_first_reconstruction_difference": 10,
                "whole_reconstruction_difference_digest": "whole",
                "bounded_reconstruction_difference_digest": "bounded",
            }
        ]
    ) == "unresolved"


def test_existing_diagnostic_checkpoint_is_never_replaced(tmp_path: Path) -> None:
    """Normal execution refuses terminal, error, and interrupted artifacts."""
    output = tmp_path / "diagnostics.json"
    output.write_text('{"status":"error"}\n', encoding="utf-8")
    args = argparse.Namespace(
        input=tmp_path / "documents.jsonl",
        output=output,
        whole_timeout_seconds=900.0,
        excerpt_characters=320,
    )

    try:
        diagnostic._new_payload(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing error checkpoint was not protected.")

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "error"


def test_published_diagnostic_artifact_contains_no_raw_excerpts() -> None:
    """The complete checked-in result retains shapes/digests, not corpus text."""
    payload = cast(
        dict[str, Any], json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    )
    forbidden_keys = {
        "text",
        "normalized",
        "original_context",
        "reconstructed_context",
        "excerpt",
        "sentences",
        "block_and_carry_context",
        "affected_block_pysbd",
        "controlled_local_windows",
    }

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            assert not (forbidden_keys & value.keys())
            for key, child in value.items():
                if key.endswith("_shape") and isinstance(child, str):
                    structural = child.replace("...[truncated]...", "")
                    assert all(
                        not character.isalpha() or character in {"L", "D"}
                        for character in structural
                    )
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(payload)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "Space Superiority" not in serialized
    assert "This Act is organized" not in serialized
