"""Focused real-tokenizer benchmark for TokenCounter length output."""

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

from transformers import AutoTokenizer

from nerv.chunking.configuration import load_encoder_config
from nerv.chunking.token_counter import TokenCounter

DEFAULT_OUTPUT = (
    Path(__file__).parent
    / "results"
    / "token_counter_optimization_metrics_v1.json"
)
DEFAULT_BATCH_SIZE = 256
DEFAULT_REPETITIONS = 5


class _ReferenceTokenCounter:
    """Freeze the previous list-copy and validated-input-IDs implementation."""

    def __init__(
        self,
        tokenizer: object,
        *,
        add_special_tokens: bool,
        document_prefix: str,
    ) -> None:
        self._tokenizer = tokenizer
        self._add_special_tokens = add_special_tokens
        self._document_prefix = document_prefix

    def count_many(
        self,
        texts: Sequence[str],
        *,
        include_document_prefix: bool = False,
        batch_size: int = 256,
    ) -> list[int]:
        normalized = list(texts)
        if any(not isinstance(text, str) for text in normalized):
            raise TypeError("every reference text must be a string")
        counts: list[int] = []
        tokenizer_call = cast(Callable[..., object], self._tokenizer)
        for start in range(0, len(normalized), batch_size):
            batch = normalized[start : start + batch_size]
            effective_batch = [
                f"{self._document_prefix}{text}"
                if include_document_prefix
                else text
                for text in batch
            ]
            encoded = tokenizer_call(
                effective_batch,
                add_special_tokens=self._add_special_tokens,
                truncation=False,
            )
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise ValueError("reference tokenizer did not return input_ids")
            input_batches = encoded["input_ids"]
            if isinstance(input_batches, (str, bytes)) or not isinstance(
                input_batches,
                Sequence,
            ):
                raise ValueError("reference tokenizer returned invalid input_ids")
            if len(input_batches) != len(batch):
                raise ValueError("reference tokenizer returned unexpected batch size")
            for input_ids in input_batches:
                if isinstance(input_ids, (str, bytes)) or not isinstance(
                    input_ids,
                    Sequence,
                ):
                    raise ValueError("reference tokenizer returned invalid input_ids")
                if any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in input_ids
                ):
                    raise ValueError("reference tokenizer returned invalid input_ids")
                counts.append(len(input_ids))
        return counts


def _workloads() -> dict[str, list[str]]:
    short = [f"fila {index}: órbita estable" for index in range(4_096)]
    medium = [
        "La misión analiza datos orbitales y conserva resultados exactos. " * 8
        for _ in range(1_024)
    ]
    long_rows = [
        ";".join(
            f"campo-{field}=valor-{index}-{field}"
            for field in range(120)
        )
        for index in range(256)
    ]
    mixed_seed = [
        "",
        "prueba",
        "The orbit remains stable.",
        "A órbita permanece estável.",
        "1.234,56; 3.14",
        "texto con\nsalto de línea",
        "campo=valor;" * 80,
    ]
    mixed = [mixed_seed[index % len(mixed_seed)] for index in range(2_048)]
    large_synthetic = [
        f"registro {index} valor {index % 97}"
        for index in range(8_192)
    ]
    return {
        "many_short_texts": short,
        "medium_sentences": medium,
        "long_tabular_rows": long_rows,
        "mixed_batch": mixed,
        "large_synthetic_batch": large_synthetic,
    }


def _measure_peak_bytes(operation: Callable[[], list[int]]) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        operation()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def _counts_sha256(counts: Sequence[int]) -> str:
    payload = json.dumps(list(counts), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("nerv", "transformers", "tokenizers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, ensure_ascii=False, indent=2)
            temporary.write("\n")
            temporary.flush()
        temporary_path.replace(path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def run_benchmark(
    *,
    output_path: Path,
    repetitions: int,
    batch_size: int,
) -> dict[str, object]:
    """Measure the previous fallback against validated length output."""
    config = load_encoder_config()
    tokenizer = AutoTokenizer.from_pretrained(
        config["encoder_model_name"],
        local_files_only=True,
    )
    reference_counter = _ReferenceTokenCounter(
        tokenizer,
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
    )
    optimized_counter = TokenCounter(
        config["encoder_model_name"],
        tokenizer=tokenizer,
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
    )
    results: list[dict[str, object]] = []
    benchmark_started = time.perf_counter()

    for name, texts in _workloads().items():
        reference_operation = partial(
            reference_counter.count_many,
            texts,
            include_document_prefix=True,
            batch_size=batch_size,
        )
        optimized_operation = partial(
            optimized_counter.count_many,
            texts,
            include_document_prefix=True,
            batch_size=batch_size,
        )
        reference_counts = reference_operation()
        optimized_counts = optimized_operation()
        if reference_counts != optimized_counts:
            raise AssertionError(f"Count mismatch for workload {name}.")

        reference_durations: list[float] = []
        optimized_durations: list[float] = []
        for repetition in range(repetitions):
            ordered = (
                ((reference_operation, reference_durations),
                 (optimized_operation, optimized_durations))
                if repetition % 2 == 0
                else ((optimized_operation, optimized_durations),
                      (reference_operation, reference_durations))
            )
            for operation, durations in ordered:
                started = time.perf_counter()
                measured_counts = operation()
                durations.append(time.perf_counter() - started)
                if measured_counts != reference_counts:
                    raise AssertionError(
                        f"Non-deterministic counts for workload {name}."
                    )

        reference_median = statistics.median(reference_durations)
        optimized_median = statistics.median(optimized_durations)
        results.append(
            {
                "name": name,
                "text_count": len(texts),
                "total_characters": sum(map(len, texts)),
                "batch_size": batch_size,
                "warmup_runs_per_implementation": 1,
                "measured_repetitions": repetitions,
                "reference_durations_seconds": reference_durations,
                "optimized_durations_seconds": optimized_durations,
                "reference_median_seconds": reference_median,
                "optimized_median_seconds": optimized_median,
                "speedup_ratio": reference_median / optimized_median,
                "exact_count_equality": True,
                "counts_sha256": _counts_sha256(reference_counts),
                "reference_tracemalloc_peak_bytes": _measure_peak_bytes(
                    reference_operation
                ),
                "optimized_tracemalloc_peak_bytes": _measure_peak_bytes(
                    optimized_operation
                ),
            }
        )

    metrics: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "benchmark": "token_counter_return_length_v1",
        "model_name": config["encoder_model_name"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "add_special_tokens": config["add_special_tokens"],
        "document_prefix": config["document_prefix"],
        "truncation": False,
        "batch_size": batch_size,
        "measured_repetitions": repetitions,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_processor_count": os.cpu_count(),
        },
        "dependency_versions": _dependency_versions(),
        "source_fingerprints": {
            "benchmark_sha256": _file_sha256(Path(__file__)),
            "token_counter_sha256": _file_sha256(
                Path(__file__).resolve().parents[2]
                / "src"
                / "nerv"
                / "chunking"
                / "token_counter.py"
            ),
            "encoder_config_sha256": _file_sha256(
                Path(__file__).resolve().parents[2]
                / "config"
                / "encoder_config.json"
            ),
        },
        "workloads": results,
        "total_benchmark_duration_seconds": (
            time.perf_counter() - benchmark_started
        ),
        "memory_measurement_scope": (
            "tracemalloc Python allocations only; tokenizer/native RSS is not measured"
        ),
        "implementation_note": (
            "The real tokenizer returns input_ids together with length; the "
            "optimized adapter avoids Python token-ID validation and scalar ID copying."
        ),
    }
    _write_json_atomic(output_path, metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    metrics = run_benchmark(
        output_path=args.output,
        repetitions=args.repetitions,
        batch_size=args.batch_size,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
