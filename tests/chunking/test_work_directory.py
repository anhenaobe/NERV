"""Atomic publication and local work-directory tests."""

import logging
from pathlib import Path

import pytest

from nerv.chunking.pipeline import main as pipeline_main
from nerv.chunking.pipeline import run_pipeline
from nerv.chunking.token_counter import TokenCounter

from .fakes import DeterministicFakeTokenizer
from .test_real_corpus_commands import _write_source


def _counter() -> TokenCounter:
    return TokenCounter(
        "intfloat/multilingual-e5-small",
        tokenizer=DeterministicFakeTokenizer(),
        document_prefix="passage: ",
    )


def test_work_directory_is_used_before_final_publication(tmp_path: Path) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    output = tmp_path / "published" / "chunks.jsonl"
    work_dir = tmp_path / "local-work"
    observed: list[Path] = []

    summary = run_pipeline(
        source,
        output,
        token_counter=_counter(),
        max_tokens=256,
        overlap_tokens=32,
        work_dir=work_dir,
        temporary_path_callback=observed.append,
    )

    assert summary.document_count == 1
    assert output.exists()
    assert observed and observed[0].parent == work_dir
    assert not observed[0].exists()


def test_interruption_preserves_previous_output_and_incomplete_temp(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source, document_count=2)
    output = tmp_path / "published" / "chunks.jsonl"
    output.parent.mkdir()
    output.write_text("previous-complete-output\n", encoding="utf-8")
    work_dir = tmp_path / "local-work"
    observed: list[Path] = []

    def interrupt_after_first_document(_: object) -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_pipeline(
            source,
            output,
            token_counter=_counter(),
            max_tokens=256,
            overlap_tokens=32,
            work_dir=work_dir,
            preserve_incomplete=True,
            temporary_path_callback=observed.append,
            progress_callback=interrupt_after_first_document,
        )

    assert output.read_text(encoding="utf-8") == "previous-complete-output\n"
    assert observed and observed[0].exists()
    assert observed[0].parent == work_dir


def test_logging_contains_document_and_stage_timings(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = tmp_path / "documents.jsonl"
    _write_source(source)
    output = tmp_path / "chunks.jsonl"
    with caplog.at_level(logging.INFO, logger="nerv.chunking.pipeline"):
        run_pipeline(
            source,
            output,
            token_counter=_counter(),
            max_tokens=256,
            overlap_tokens=32,
            total_document_count=1,
            global_progress_interval=1,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "doc_id=DOC-0" in message and "stage=split" in message
        for message in messages
    )
    assert any(
        "stage=chunk" in message and "duration=" in message
        for message in messages
    )
    assert any(
        "Global progress:" in message and "(estimate)" in message
        for message in messages
    )


def test_standalone_pipeline_rejects_metrics_alias_before_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "chunks.jsonl"
    _write_source(source)
    original = source.read_bytes()

    with pytest.raises(ValueError, match="metrics_output must differ"):
        pipeline_main(
            [
                "--input",
                str(source),
                "--output",
                str(output),
                "--metrics-output",
                str(source),
                "--local-files-only",
            ]
        )
    assert source.read_bytes() == original
    assert not output.exists()
