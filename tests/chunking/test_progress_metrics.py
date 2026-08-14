"""Progress persistence, interruption, and failure-counter tests."""

import json
from pathlib import Path
from typing import Any

import pytest

import nerv.chunking.real_corpus as real_corpus
from nerv.chunking.real_corpus import parse_args, produce_command, validate_command
from nerv.chunking.token_counter import TokenCounter

from .fakes import DeterministicFakeTokenizer
from .test_real_corpus_commands import _progress, _write_source


def _counter() -> TokenCounter:
    return TokenCounter(
        "intfloat/multilingual-e5-small",
        tokenizer=DeterministicFakeTokenizer(),
        document_prefix="passage: ",
    )


def test_progress_is_persisted_after_each_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source, document_count=2)
    progress_path = tmp_path / "progress.json"
    metrics_path = tmp_path / "metrics.json"
    args = parse_args(
        [
            "produce",
            "--input",
            str(source),
            "--output",
            str(tmp_path / "chunks.jsonl"),
            "--progress-output",
            str(progress_path),
            "--metrics-output",
            str(metrics_path),
            "--config-output",
            str(tmp_path / "config.json"),
        ]
    )
    observed_progress: list[int] = []
    original_write = real_corpus._write_json_atomic

    def observing_write(path: Path, value: object) -> None:
        if path == progress_path and isinstance(value, dict):
            observed_progress.append(int(value["processed_document_count"]))
        original_write(path, value)

    monkeypatch.setattr(real_corpus, "_write_json_atomic", observing_write)

    def runner(input_path: Path, output_path: Path, **options: Any) -> Any:
        options["progress_callback"](_progress(1, 2))
        options["progress_callback"](_progress(2, 2))
        output_path.write_text("{}\n", encoding="utf-8")
        from .test_real_corpus_commands import _summary

        return _summary(input_path, output_path, documents=2)

    assert produce_command(args, token_counter=_counter(), pipeline_runner=runner) == 0
    assert 1 in observed_progress
    assert 2 in observed_progress
    assert not list(tmp_path.glob(".progress.json.*.tmp"))


def test_missing_production_source_persists_failed_state(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    metrics_path = tmp_path / "production.json"
    args = parse_args(
        [
            "produce",
            "--input",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "chunks.jsonl"),
            "--metrics-output",
            str(metrics_path),
            "--progress-output",
            str(progress_path),
            "--config-output",
            str(tmp_path / "config.json"),
        ]
    )

    assert produce_command(args, token_counter=_counter()) == 1
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completion_status"] == "failed"
    assert progress["failure_count"] == 1
    assert progress["output_publication_status"] == "previous_preserved"


def test_missing_chunks_persists_failed_validation_metrics(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    metrics_path = tmp_path / "validation.json"
    args = parse_args(
        [
            "validate",
            "--input",
            str(source),
            "--chunks",
            str(tmp_path / "missing-chunks.jsonl"),
            "--metrics-output",
            str(metrics_path),
        ]
    )

    assert validate_command(args, token_counter=_counter()) == 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["completion_status"] == "failed"
    assert metrics["failure_count"] == 1


def test_keyboard_interrupt_preserves_output_and_reports_temporary_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    output = tmp_path / "chunks.jsonl"
    output.write_text("completed-output\n", encoding="utf-8")
    progress_path = tmp_path / "progress.json"
    metrics_path = tmp_path / "metrics.json"
    args = parse_args(
        [
            "produce",
            "--input",
            str(source),
            "--output",
            str(output),
            "--progress-output",
            str(progress_path),
            "--metrics-output",
            str(metrics_path),
            "--config-output",
            str(tmp_path / "config.json"),
        ]
    )
    incomplete = tmp_path / ".chunks.jsonl.interrupted.tmp"

    def interrupted_runner(_: Path, __: Path, **options: Any) -> Any:
        incomplete.write_text("incomplete", encoding="utf-8")
        options["temporary_path_callback"](incomplete)
        options["progress_callback"](_progress(1, 1))
        raise KeyboardInterrupt

    assert produce_command(
        args,
        token_counter=_counter(),
        pipeline_runner=interrupted_runner,
    ) == 130
    assert output.read_text(encoding="utf-8") == "completed-output\n"
    assert incomplete.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completion_status"] == "interrupted"
    assert progress["last_completed_document_index"] == 1
    assert progress["temporary_file_path"] == str(incomplete)
    assert progress["output_publication_status"] == "previous_preserved"


def test_failure_totals_are_not_truncated_with_examples(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    chunks = tmp_path / "chunks.jsonl"
    records = [
        {
            "doc_id": "DOC-0",
            "chunk_id": f"DOC-0-chunk-{index:04d}",
            "fuente": "source-0.pdf",
            "formato": "pdf",
            "fenomeno": 0,
            "posicion": index,
            "num_tokens": 0,
            "texto": f"passage: invalid {index}",
        }
        for index in range(5)
    ]
    chunks.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
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
            "--failure-example-limit",
            "2",
        ]
    )

    assert validate_command(args, token_counter=_counter()) == 2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["stored_prefix_error_count"] == 5
    assert metrics["token_count_mismatch_count"] == 5
    assert metrics["failure_count"] >= 10
    assert len(metrics["failure_examples"]) == 2


def test_invalid_chunk_id_cannot_report_success(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_source(source)
    counter = _counter()
    text = "Document 0 contains one complete sentence."
    chunks.write_text(
        json.dumps(
            {
                "doc_id": "DOC-0",
                "chunk_id": 7,
                "fuente": "source-0.pdf",
                "formato": "pdf",
                "fenomeno": 0,
                "posicion": 0,
                "num_tokens": counter.count(
                    text,
                    include_document_prefix=True,
                ),
                "texto": text,
            }
        )
        + "\n",
        encoding="utf-8",
    )
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

    assert validate_command(args, token_counter=counter) == 2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["completion_status"] == "completed_with_failures"
    assert metrics["failure_counts"]["invalid_chunk_id"] == 1


def test_missing_source_document_is_a_complete_validation_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_source(source, document_count=2)
    counter = _counter()
    text = "Document 0 contains one complete sentence."
    chunks.write_text(
        json.dumps(
            {
                "doc_id": "DOC-0",
                "chunk_id": "DOC-0-chunk-0000",
                "fuente": "source-0.pdf",
                "formato": "pdf",
                "fenomeno": 0,
                "posicion": 0,
                "num_tokens": counter.count(
                    text,
                    include_document_prefix=True,
                ),
                "texto": text,
            }
        )
        + "\n",
        encoding="utf-8",
    )
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

    assert validate_command(args, token_counter=counter) == 2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["missing_output_document_count"] == 1
    assert metrics["failure_counts"]["missing_output_document"] == 1


def test_validation_rejects_noncanonical_chunk_id_and_boolean_integer(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    chunks = tmp_path / "chunks.jsonl"
    _write_source(source)
    counter = _counter()
    text = "Document 0 contains one complete sentence."
    chunks.write_text(
        json.dumps(
            {
                "doc_id": "DOC-0",
                "chunk_id": "arbitrary-string",
                "fuente": "source-0.pdf",
                "formato": "pdf",
                "fenomeno": True,
                "posicion": 0,
                "num_tokens": counter.count(
                    text,
                    include_document_prefix=True,
                ),
                "texto": text,
            }
        )
        + "\n",
        encoding="utf-8",
    )
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

    assert validate_command(args, token_counter=counter) == 2
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["failure_counts"]["chunk_id_format_error"] == 1
    assert metrics["failure_counts"]["invalid_field_type"] == 1
