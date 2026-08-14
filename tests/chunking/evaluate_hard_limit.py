"""Run controlled exact-size hard-limit checks with the frozen tokenizer."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from transformers import AutoConfig

from nerv.chunking.configuration import load_encoder_config, resolve_encoder_capacity
from nerv.chunking.hard_limit import split_hard_limited_text
from nerv.chunking.token_counter import TokenCounter

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "tests/chunking/results/hard_limit_synthetic_metrics.json"
TARGET_TOTAL_IDS = (255, 256, 257, 509, 510, 511, 512, 513, 1000, 10239)


def _text_with_exact_total_ids(counter: TokenCounter, target: int) -> str:
    """Build deterministic whitespace-delimited text with an exact total count."""
    low = 1
    high = target * 2
    while low <= high:
        middle = (low + high) // 2
        text = " x" * middle
        count = counter.count(text, include_document_prefix=True)
        if count == target:
            return text
        if count < target:
            low = middle + 1
        else:
            high = middle - 1
    raise RuntimeError(f"could not construct an exact {target}-ID input.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_encoder_config()
    counter = TokenCounter(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
        local_files_only=True,
    )
    model_config = AutoConfig.from_pretrained(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        local_files_only=True,
    )
    model_max_position_embeddings = getattr(
        model_config,
        "max_position_embeddings",
        None,
    )
    if (
        counter.reported_model_max_length != config["encoder_max_input_tokens"]
        or model_max_position_embeddings != config["encoder_max_input_tokens"]
    ):
        raise AssertionError("the frozen tokenizer/model hard limit is not 512.")
    capacity = resolve_encoder_capacity(
        config=config,
        tokenizer_model_max_length=counter.reported_model_max_length,
        encoder_special_token_overhead=counter.encoder_special_token_overhead,
        local_files_only=True,
    )
    cases: list[dict[str, object]] = []
    for target in TARGET_TOTAL_IDS:
        text = _text_with_exact_total_ids(counter, target)
        first = split_hard_limited_text(
            text,
            document_format="txt",
            token_counter=counter,
            hard_limit=capacity.effective_content_max_tokens,
            overlap_tokens=config["overlap_tokens"],
        )
        second = split_hard_limited_text(
            text,
            document_format="txt",
            token_counter=counter,
            hard_limit=capacity.effective_content_max_tokens,
            overlap_tokens=config["overlap_tokens"],
        )
        reconstructed = "".join(
            text[part.coverage_start : part.coverage_end]
            for part in first.parts
        )
        exact_recounts = [
            counter.count(part.text, include_document_prefix=True)
            for part in first.parts
        ]
        case = {
            "requested_total_ids": target,
            "actual_source_total_ids": first.source_token_count,
            "strategy": first.strategy,
            "subchunk_count": len(first.parts),
            "minimum_subchunk_total_ids": min(exact_recounts),
            "maximum_subchunk_total_ids": max(exact_recounts),
            "maximum_internal_overlap_tokens": max(
                part.overlap_tokens for part in first.parts
            ),
            "coverage_preserved": reconstructed == text,
            "deterministic": first == second,
            "all_subchunks_within_hard_limit": all(
                count <= capacity.effective_content_max_tokens
                for count in exact_recounts
            ),
            "stored_counts_match_recount": exact_recounts
            == [part.token_count for part in first.parts],
        }
        if not all(
            (
                case["actual_source_total_ids"] == target,
                case["coverage_preserved"],
                case["deterministic"],
                case["all_subchunks_within_hard_limit"],
                case["stored_counts_match_recount"],
            )
        ):
            raise AssertionError(f"hard-limit synthetic case failed: {case}")
        cases.append(case)

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "encoder_model_name": config["encoder_model_name"],
        "tokenizer_class": counter.tokenizer_class,
        "tokenizer_model_max_length": counter.reported_model_max_length,
        "model_max_position_embeddings": model_max_position_embeddings,
        "soft_limit": config["chunk_max_tokens"],
        "encoder_max_input_tokens": capacity.encoder_max_input_tokens,
        "encoder_special_token_overhead": (
            capacity.encoder_special_token_overhead
        ),
        "effective_content_max_tokens": capacity.effective_content_max_tokens,
        "overlap_target": config["overlap_tokens"],
        "add_special_tokens": config["add_special_tokens"],
        "document_prefix": config["document_prefix"],
        "cases": cases,
        "all_cases_passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
