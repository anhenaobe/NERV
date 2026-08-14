"""Focused command-boundary tests for the real-corpus workflow."""

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import nerv.chunking
import nerv.chunking.real_corpus as real_corpus
from nerv.chunking.pipeline import (
    PipelineDocumentProgress,
    PipelineSummary,
    process_document,
)
from nerv.chunking.real_corpus import (
    benchmark_command,
    parse_args,
    produce_command,
    run_cli,
    validate_command,
)
from nerv.chunking.token_counter import TokenCounter

from .fakes import DeterministicFakeTokenizer


def _counter() -> TokenCounter:
    return TokenCounter(
        "intfloat/multilingual-e5-small",
        tokenizer=DeterministicFakeTokenizer(),
        document_prefix="passage: ",
    )


def _write_source(path: Path, document_count: int = 1) -> None:
    records = [
        {
            "doc_id": f"DOC-{index}",
            "fuente": f"source-{index}.pdf",
            "formato": "pdf",
            "fenomeno": index,
            "texto": f"Document {index} contains one complete sentence.",
        }
        for index in range(document_count)
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _summary(
    input_path: Path,
    output_path: Path,
    documents: int = 1,
) -> PipelineSummary:
    return PipelineSummary(
        document_count=documents,
        chunk_count=documents,
        sentence_count=documents,
        oversized_chunk_count=0,
        minimum_tokens_per_chunk=10,
        maximum_tokens_per_chunk=10,
        average_tokens_per_chunk=10.0,
        processing_duration_seconds=0.1,
        documents_per_second=documents * 10.0,
        chunks_per_second=documents * 10.0,
        input_path=input_path,
        output_path=output_path,
        encoder_model_name="intfloat/multilingual-e5-small",
        tokenizer_class="DeterministicFakeTokenizer",
        tokenizer_revision=None,
        max_tokens=256,
        overlap_tokens=32,
        encoder_max_input_tokens=512,
        encoder_special_token_overhead=2,
        effective_content_max_tokens=510,
        hard_split_source_unit_count=1,
        hard_split_generated_chunk_count=2,
        hard_split_by_format={"pdf": 1},
        hard_split_by_strategy={"punctuation": 1},
    )


def _progress(index: int, total: int) -> PipelineDocumentProgress:
    return PipelineDocumentProgress(
        document_index=index,
        total_document_count=total,
        doc_id=f"DOC-{index - 1}",
        document_format="pdf",
        character_count=40,
        line_count=1,
        detected_language="en",
        language_detection_duration_seconds=0.01,
        splitting_duration_seconds=0.02,
        chunking_duration_seconds=0.03,
        record_construction_duration_seconds=0.01,
        writing_duration_seconds=0.01,
        document_duration_seconds=0.08,
        generated_chunk_count=1,
        oversized_chunk_count=0,
        processed_document_count=index,
        total_generated_chunk_count=index,
        elapsed_time_seconds=index * 0.08,
        estimated_remaining_seconds=(total - index) * 0.08,
        hard_split_source_unit_count=1,
        hard_split_generated_chunk_count=2,
        hard_split_by_format={"pdf": 1},
        hard_split_by_strategy={"punctuation": 1},
    )


def test_produce_runs_exactly_once_without_benchmarking(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "chunks.jsonl"
    _write_source(source)
    args = parse_args(
        [
            "produce",
            "--input",
            str(source),
            "--output",
            str(output),
            "--metrics-output",
            str(tmp_path / "production.json"),
            "--progress-output",
            str(tmp_path / "progress.json"),
            "--config-output",
            str(tmp_path / "config.json"),
        ]
    )
    calls = 0

    def fake_runner(
        input_path: Path,
        output_path: Path,
        **options: Any,
    ) -> PipelineSummary:
        nonlocal calls
        calls += 1
        temporary = tmp_path / "work" / ".chunks.jsonl.test.tmp"
        temporary.parent.mkdir()
        temporary.write_text("partial", encoding="utf-8")
        options["temporary_path_callback"](temporary)
        options["progress_callback"](_progress(1, 1))
        output_path.write_text("{}\n", encoding="utf-8")
        temporary.unlink()
        return _summary(input_path, output_path)

    assert (
        produce_command(
            args,
            token_counter=_counter(),
            pipeline_runner=fake_runner,
        )
        == 0
    )
    assert calls == 1
    metrics = json.loads((tmp_path / "production.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "produce"
    assert metrics["completion_status"] == "completed"
    assert metrics["processed_document_count"] == 1
    assert metrics["hard_split_source_unit_count"] == 1
    assert metrics["hard_split_generated_chunk_count"] == 2
    assert metrics["hard_split_by_format"] == {"pdf": 1}
    assert metrics["hard_split_by_strategy"] == {"punctuation": 1}


def test_validate_never_invokes_production_or_overwrites_chunks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_source(source)
    counter = _counter()
    text = "Document 0 contains one complete sentence."
    record = {
        "doc_id": "DOC-0",
        "chunk_id": "DOC-0-chunk-0000",
        "fuente": "source-0.pdf",
        "formato": "pdf",
        "fenomeno": 0,
        "posicion": 0,
        "num_tokens": counter.count(text, include_document_prefix=True),
        "texto": text,
    }
    chunks.write_text(json.dumps(record) + "\n", encoding="utf-8")
    original = chunks.read_bytes()
    metrics_path = tmp_path / "validation.json"
    args = parse_args(
        [
            "validate",
            "--input",
            str(source),
            "--chunks",
            str(chunks),
            "--metrics-output",
            str(metrics_path),
        ]
    )

    assert validate_command(args, token_counter=counter) == 0
    assert chunks.read_bytes() == original
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["completion_status"] == "completed"
    assert metrics["valid_chunk_count"] == 1
    assert metrics["chunks_le_soft_limit"] == 1
    assert metrics["chunks_gt_soft_limit_but_le_hard_limit"] == 0
    assert metrics["chunks_gt_hard_limit"] == 0
    assert metrics["maximum_num_tokens"] == record["num_tokens"]


def test_validate_hard_fails_actual_encoder_count_above_limit(
    tmp_path: Path,
) -> None:
    """Stored counts cannot conceal a chunk beyond the 512-token contract."""
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    metrics_path = tmp_path / "validation.json"
    _write_source(source)
    counter = _counter()
    text = "x" * 600
    record = {
        "doc_id": "DOC-0",
        "chunk_id": "DOC-0-chunk-0000",
        "fuente": "source-0.pdf",
        "formato": "pdf",
        "fenomeno": 0,
        "posicion": 0,
        "num_tokens": counter.count(text, include_document_prefix=True),
        "texto": text,
    }
    chunks.write_text(json.dumps(record) + "\n", encoding="utf-8")
    args = parse_args(
        [
            "validate",
            "--input",
            str(source),
            "--chunks",
            str(chunks),
            "--metrics-output",
            str(metrics_path),
        ]
    )

    assert validate_command(args, token_counter=counter) == 2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["invalid_chunk_count"] == 1
    assert metrics["encoder_hard_limit_exceeded_count"] == 1
    assert metrics["chunks_gt_hard_limit"] == 1
    assert metrics["maximum_num_tokens"] == record["num_tokens"]


def test_validate_requires_one_complete_hard_split_parent_trace(
    tmp_path: Path,
) -> None:
    """Part metadata must reconstruct exactly one complete logical parent."""
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    text = "x" * 900
    document = {
        "doc_id": "DOC-HARD",
        "fuente": "source.csv",
        "formato": "csv",
        "fenomeno": 1,
        "texto": text,
    }
    source.write_text(json.dumps(document) + "\n", encoding="utf-8")
    counter = _counter()
    records = process_document(
        document,
        sentence_splitter=lambda value: [value],
        token_counter=counter,
        max_tokens=256,
        overlap_tokens=32,
        encoder_max_input_tokens=510,
    )
    chunks.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    valid_metrics = tmp_path / "valid.json"
    valid_args = parse_args(
        [
            "validate",
            "--input",
            str(source),
            "--chunks",
            str(chunks),
            "--metrics-output",
            str(valid_metrics),
        ]
    )
    assert validate_command(valid_args, token_counter=counter) == 0

    chunks.write_text(
        "".join(json.dumps(record) + "\n" for record in records[:-1]),
        encoding="utf-8",
    )
    invalid_metrics = tmp_path / "invalid.json"
    invalid_args = parse_args(
        [
            "validate",
            "--input",
            str(source),
            "--chunks",
            str(chunks),
            "--metrics-output",
            str(invalid_metrics),
        ]
    )
    assert validate_command(invalid_args, token_counter=counter) == 2
    metrics = json.loads(invalid_metrics.read_text(encoding="utf-8"))
    assert metrics["hard_split_metadata_error_count"] == 1
    assert metrics["invalid_chunk_count"] == len(records) - 1


def test_validate_rejects_metrics_path_that_aliases_chunks(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_source(source)
    chunks.write_text("production-sentinel\n", encoding="utf-8")
    args = parse_args(
        [
            "validate",
            "--input",
            str(source),
            "--chunks",
            str(chunks),
            "--metrics-output",
            str(chunks),
        ]
    )

    with pytest.raises(ValueError, match="metrics_output must differ"):
        validate_command(args, token_counter=_counter())
    assert chunks.read_text(encoding="utf-8") == "production-sentinel\n"


def test_cli_rejects_log_path_that_aliases_chunks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_source(source)
    chunks.write_text("production-sentinel\n", encoding="utf-8")

    assert (
        run_cli(
            [
                "validate",
                "--input",
                str(source),
                "--chunks",
                str(chunks),
                "--metrics-output",
                str(tmp_path / "validation.json"),
                "--log-path",
                str(chunks),
            ]
        )
        == 2
    )
    assert "command rejected" in capsys.readouterr().err
    assert chunks.read_text(encoding="utf-8") == "production-sentinel\n"


def test_cli_rejects_log_path_hard_linked_to_chunks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    log_alias = tmp_path / "hard-linked.log"
    _write_source(source)
    chunks.write_text("production-sentinel\n", encoding="utf-8")
    try:
        os.link(chunks, log_alias)
    except OSError as error:
        pytest.skip(f"hard links are unavailable: {error}")

    assert (
        run_cli(
            [
                "validate",
                "--input",
                str(source),
                "--chunks",
                str(chunks),
                "--metrics-output",
                str(tmp_path / "validation.json"),
                "--log-path",
                str(log_alias),
            ]
        )
        == 2
    )
    assert "command rejected" in capsys.readouterr().err
    assert chunks.read_text(encoding="utf-8") == "production-sentinel\n"


def test_benchmark_is_isolated_and_repetitions_are_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "doc_ids": ["DOC-0"]}),
        encoding="utf-8",
    )
    production_output = tmp_path / "chunks.jsonl"
    production_output.write_text("production-sentinel\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    metrics_path = tmp_path / "benchmark.json"
    args = parse_args(
        [
            "benchmark",
            "--input",
            str(source),
            "--manifest",
            str(manifest),
            "--metrics-output",
            str(metrics_path),
            "--work-dir",
            str(work_dir),
            "--repetitions",
            "2",
        ]
    )
    calls = 0
    snapshots: list[dict[str, Any]] = []
    write_json_atomic = real_corpus._write_json_atomic

    def recording_write(path: Path, value: object) -> None:
        if path == metrics_path:
            snapshots.append(json.loads(json.dumps(value)))
        write_json_atomic(path, value)

    monkeypatch.setattr(real_corpus, "_write_json_atomic", recording_write)

    def fake_runner(
        input_path: Path,
        output_path: Path,
        **_: Any,
    ) -> PipelineSummary:
        nonlocal calls
        calls += 1
        output_path.write_text('{"stable":true}\n', encoding="utf-8")
        return _summary(input_path, output_path)

    assert benchmark_command(
        args,
        token_counter=_counter(),
        pipeline_runner=fake_runner,
    ) == 0
    assert calls == 3
    assert production_output.read_text(encoding="utf-8") == "production-sentinel\n"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["warmup_completed"] is True
    assert metrics["completed_repetitions"] == 2
    assert metrics["deterministic_output"] is True
    assert not (work_dir / ".benchmark.lock").exists()
    assert any(snapshot["completed_repetitions"] == 1 for snapshot in snapshots)
    assert any(snapshot["completed_repetitions"] == 2 for snapshot in snapshots)


def test_benchmark_resumes_completed_repetitions(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"doc_ids": ["DOC-0"]}), encoding="utf-8")
    metrics_path = tmp_path / "benchmark.json"
    arguments = [
        "benchmark",
        "--input",
        str(source),
        "--manifest",
        str(manifest),
        "--metrics-output",
        str(metrics_path),
        "--work-dir",
        str(tmp_path / "work"),
        "--repetitions",
        "2",
    ]
    first_args = parse_args(arguments)
    calls = 0

    def interrupted_runner(
        input_path: Path,
        output_path: Path,
        **_: Any,
    ) -> PipelineSummary:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        output_path.write_text('{"stable":true}\n', encoding="utf-8")
        return _summary(input_path, output_path)

    assert benchmark_command(
        first_args,
        token_counter=_counter(),
        pipeline_runner=interrupted_runner,
    ) == 130
    assert json.loads(metrics_path.read_text(encoding="utf-8"))[
        "completed_repetitions"
    ] == 1
    assert json.loads(metrics_path.read_text(encoding="utf-8"))[
        "deterministic_output"
    ] is None

    resumed_calls = 0

    def resumed_runner(
        input_path: Path,
        output_path: Path,
        **_: Any,
    ) -> PipelineSummary:
        nonlocal resumed_calls
        resumed_calls += 1
        output_path.write_text('{"stable":true}\n', encoding="utf-8")
        return _summary(input_path, output_path)

    assert benchmark_command(
        parse_args(arguments),
        token_counter=_counter(),
        pipeline_runner=resumed_runner,
    ) == 0
    assert resumed_calls == 1
    resumed_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert resumed_metrics["completion_status"] == "completed"
    assert resumed_metrics["resume_from_repetition"] == 1
    assert resumed_metrics["deterministic_output"] is True

    resumed_metrics["compatibility"]["workflow_code_fingerprint"] = "stale"
    metrics_path.write_text(
        json.dumps(resumed_metrics),
        encoding="utf-8",
    )
    fresh_calls = 0

    def fresh_runner(
        input_path: Path,
        output_path: Path,
        **_: Any,
    ) -> PipelineSummary:
        nonlocal fresh_calls
        fresh_calls += 1
        output_path.write_text('{"stable":true}\n', encoding="utf-8")
        return _summary(input_path, output_path)

    assert benchmark_command(
        parse_args(arguments),
        token_counter=_counter(),
        pipeline_runner=fresh_runner,
    ) == 0
    assert fresh_calls == 3
    fresh_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert fresh_metrics["resume_from_repetition"] == 0


def test_benchmark_rejects_an_already_locked_work_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"doc_ids": ["DOC-0"]}), encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    lock_path = work_dir / ".benchmark.lock"
    lock_path.write_text("existing-owner\n", encoding="utf-8")
    args = parse_args(
        [
            "benchmark",
            "--input",
            str(source),
            "--manifest",
            str(manifest),
            "--metrics-output",
            str(tmp_path / "benchmark.json"),
            "--work-dir",
            str(work_dir),
        ]
    )

    assert benchmark_command(args, token_counter=_counter()) == 1
    assert lock_path.read_text(encoding="utf-8") == "existing-owner\n"


def test_invalid_manifest_persists_failed_benchmark_metrics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("not-json\n", encoding="utf-8")
    metrics_path = tmp_path / "benchmark.json"
    args = parse_args(
        [
            "benchmark",
            "--input",
            str(source),
            "--manifest",
            str(manifest),
            "--metrics-output",
            str(metrics_path),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert benchmark_command(args, token_counter=_counter()) == 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["completion_status"] == "failed"
    assert metrics["failure_stage"] == "preparation"
    assert metrics["failures"][-1]["category"] == "JSONDecodeError"


def test_importing_nerv_chunking_does_not_execute_a_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    imported = importlib.reload(nerv.chunking)

    assert imported.__name__ == "nerv.chunking"
    assert capsys.readouterr() == ("", "")
