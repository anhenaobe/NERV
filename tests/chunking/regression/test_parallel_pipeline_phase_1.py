"""Phase-1 regression tests for the single spawn-worker architecture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import BinaryIO

import pytest
from tests.chunking.fakes import DeterministicFakeTokenizer

import nerv.chunking.parallel_pipeline as parallel
from nerv.chunking.configuration import EncoderConfig, load_encoder_config
from nerv.chunking.pipeline import process_document, run_pipeline
from nerv.chunking.real_corpus import parse_args
from nerv.chunking.sentence_splitter import split_sentences
from nerv.chunking.token_counter import TokenCounter

_FACTORY_CALL_COUNT = 0


class _FailingTokenizer(DeterministicFakeTokenizer):
    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        **options: object,
    ) -> dict[str, object]:
        values = text if isinstance(text, list) else [text]
        if any("FAIL-WORKER" in value for value in values):
            raise RuntimeError("controlled worker failure")
        return super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
        )


def fake_counter_factory(
    config: EncoderConfig,
    local_files_only: bool,
) -> TokenCounter:
    """Spawn-picklable deterministic counter factory."""
    del local_files_only
    global _FACTORY_CALL_COUNT
    _FACTORY_CALL_COUNT += 1
    return TokenCounter(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
        tokenizer=DeterministicFakeTokenizer(),
    )


def failing_counter_factory(
    config: EncoderConfig,
    local_files_only: bool,
) -> TokenCounter:
    """Spawn-picklable counter that fails only on a controlled document."""
    del local_files_only
    return TokenCounter(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        add_special_tokens=config["add_special_tokens"],
        document_prefix=config["document_prefix"],
        tokenizer=_FailingTokenizer(),
    )


def initializer_failure_factory(
    config: EncoderConfig,
    local_files_only: bool,
) -> TokenCounter:
    """Spawn-picklable controlled initializer failure."""
    del config, local_files_only
    raise RuntimeError("controlled initializer failure")


def _document(
    doc_id: str,
    text: str,
    *,
    document_format: str = "pdf",
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "fuente": f"{doc_id}.{document_format}",
        "formato": document_format,
        "fenomeno": 1,
        "texto": text,
    }


def _write_documents(path: Path, documents: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(document, ensure_ascii=False) + "\n"
            for document in documents
        ),
        encoding="utf-8",
    )


def _fake_counter(config: EncoderConfig) -> TokenCounter:
    return fake_counter_factory(config, False)


def test_initializer_reuses_one_counter_for_multiple_tasks(tmp_path: Path) -> None:
    global _FACTORY_CALL_COUNT
    _FACTORY_CALL_COUNT = 0
    config = load_encoder_config()
    fingerprint = parallel.configuration_fingerprint(config)
    parallel.initialize_worker(config, fingerprint, True, fake_counter_factory)
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()

    first = parallel.process_document_task(
        1,
        _document("DOC-1", "One complete sentence."),
        str(documents_dir),
        fingerprint,
    )
    second = parallel.process_document_task(
        2,
        _document("DOC-2", "Otra oración completa."),
        str(documents_dir),
        fingerprint,
    )

    assert _FACTORY_CALL_COUNT == 1
    assert first.worker_pid == second.worker_pid
    assert parallel.worker_metadata_task().initialization_seconds >= 0.0


def test_worker_manifest_and_bytes_match_existing_document_path(
    tmp_path: Path,
) -> None:
    config = load_encoder_config()
    fingerprint = parallel.configuration_fingerprint(config)
    parallel.initialize_worker(config, fingerprint, True, fake_counter_factory)
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    document = _document("DOC-HARD", "x" * 900, document_format="csv")

    result = parallel.process_document_task(
        1,
        document,
        str(documents_dir),
        fingerprint,
    )
    expected_records = process_document(
        document,
        sentence_splitter=split_sentences,
        token_counter=_fake_counter(config),
        max_tokens=config["chunk_max_tokens"],
        overlap_tokens=config["overlap_tokens"],
        encoder_max_input_tokens=510,
    )
    expected = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in expected_records
    ).encode("utf-8")
    actual = Path(result.temporary_path).read_bytes()

    assert actual == expected
    assert result.output_byte_count == len(actual)
    assert result.output_sha256 == hashlib.sha256(actual).hexdigest()
    assert result.hard_split_source_unit_count > 0
    assert [record["posicion"] for record in expected_records] == list(
        range(len(expected_records))
    )
    assert [record["chunk_id"] for record in expected_records] == [
        f"DOC-HARD-chunk-{index:04d}" for index in range(len(expected_records))
    ]
    forbidden_fields = {"document", "records", "chunks", "sentences", "texto"}
    assert forbidden_fields.isdisjoint(result.__dataclass_fields__)


def _result_for_file(
    index: int,
    doc_id: str,
    path: Path,
) -> parallel.WorkerDocumentResult:
    data = path.read_bytes()
    return parallel.WorkerDocumentResult(
        schema_version=parallel.MANIFEST_SCHEMA_VERSION,
        document_index=index,
        doc_id=doc_id,
        document_format="pdf",
        temporary_path=str(path),
        output_byte_count=len(data),
        output_sha256=hashlib.sha256(data).hexdigest(),
        record_count=1,
        sentence_count=1,
        detected_language="en",
        oversized_chunk_count=0,
        token_total=1,
        minimum_tokens_per_chunk=1,
        maximum_tokens_per_chunk=1,
        hard_split_source_unit_count=0,
        hard_split_generated_chunk_count=0,
        hard_split_by_format={},
        hard_split_by_strategy={},
        language_detection_seconds=0.0,
        splitting_seconds=0.0,
        chunking_seconds=0.0,
        record_construction_seconds=0.0,
        worker_temp_write_seconds=0.0,
        worker_service_seconds=0.0,
        worker_pid=1,
        character_count=1,
        line_count=1,
    )


def test_ordered_merge_waits_for_and_restores_canonical_order(
    tmp_path: Path,
) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    first_path = documents_dir / "first.jsonl.tmp"
    second_path = documents_dir / "second.jsonl.tmp"
    first_path.write_bytes(b"first\n")
    second_path.write_bytes(b"second\n")
    first = _result_for_file(1, "DOC-1", first_path)
    second = _result_for_file(2, "DOC-2", second_path)
    output = tmp_path / "merged.jsonl"
    merged: list[int] = []

    with output.open("wb") as destination:
        pending = {2: second}
        next_index = parallel.merge_pending_results(
            pending,
            next_merge_index=1,
            destination=destination,
            run_documents_dir=documents_dir,
            on_merged=lambda result, _seconds: merged.append(result.document_index),
        )
        assert next_index == 1
        pending[1] = first
        next_index = parallel.merge_pending_results(
            pending,
            next_merge_index=next_index,
            destination=destination,
            run_documents_dir=documents_dir,
            on_merged=lambda result, _seconds: merged.append(result.document_index),
        )

    assert next_index == 3
    assert merged == [1, 2]
    assert output.read_bytes() == b"first\nsecond\n"


def test_manifest_verification_rejects_path_outside_run(tmp_path: Path) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    outside = tmp_path / "outside.jsonl.tmp"
    outside.write_bytes(b"outside\n")
    result = _result_for_file(1, "DOC-1", outside)

    with pytest.raises(ValueError, match="outside the run directory"):
        parallel.verify_worker_result(result, run_documents_dir=documents_dir)


def test_manifest_verification_rejects_wrong_identity_and_record_count(
    tmp_path: Path,
) -> None:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    temporary = documents_dir / "document.jsonl.tmp"
    temporary.write_bytes(b"one\n")
    result = _result_for_file(2, "DOC-2", temporary)

    with pytest.raises(ValueError, match="unexpected document index"):
        parallel.verify_result_identity(
            result,
            expected_document_index=1,
            expected_doc_id="DOC-2",
        )
    corrupted_count = parallel.WorkerDocumentResult(
        **{
            **result.__dict__,
            "document_index": 1,
            "record_count": 2,
        }
    )
    with pytest.raises(ValueError, match="record count"):
        parallel.verify_worker_result(
            corrupted_count,
            run_documents_dir=documents_dir,
        )


def test_spawn_worker_pipeline_is_byte_identical_to_sequential(
    tmp_path: Path,
) -> None:
    config = load_encoder_config()
    documents = [
        _document("DOC-EN", "First sentence. Second sentence."),
        _document("DOC-ES", "Primera oración. Segunda oración."),
        _document("DOC-HARD", "z" * 900, document_format="csv"),
    ]
    source = tmp_path / "documents.jsonl"
    sequential = tmp_path / "sequential.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_documents(source, documents)
    run_pipeline(
        source,
        sequential,
        token_counter=_fake_counter(config),
        max_tokens=config["chunk_max_tokens"],
        overlap_tokens=config["overlap_tokens"],
        encoder_max_input_tokens=510,
    )

    run = parallel.run_parallel_pipeline_phase_1(
        source,
        candidate,
        config=config,
        local_files_only=True,
        total_document_count=len(documents),
        work_dir=tmp_path / "work",
        counter_factory=fake_counter_factory,
    )

    assert candidate.read_bytes() == sequential.read_bytes()
    assert run.metrics.start_method == "spawn"
    assert run.metrics.worker_count == 1
    assert run.metrics.workers_initialized == 1
    assert run.metrics.worker_task_count == len(documents)
    assert run.metrics.output_sha256 == hashlib.sha256(
        candidate.read_bytes()
    ).hexdigest()
    assert not list((tmp_path / "work").glob("produce-*"))


def test_parallel_pipeline_creates_nested_output_parent(tmp_path: Path) -> None:
    config = load_encoder_config()
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "nested" / "result" / "chunks.jsonl"
    _write_documents(source, [_document("DOC-1", "One sentence.")])

    parallel.run_parallel_pipeline_phase_1(
        source,
        output,
        config=config,
        local_files_only=True,
        total_document_count=1,
        work_dir=tmp_path / "work",
        counter_factory=fake_counter_factory,
    )

    assert output.is_file()


@pytest.mark.parametrize("alias_target", ["input", "output"])
def test_parallel_pipeline_rejects_config_artifact_aliases(
    tmp_path: Path,
    alias_target: str,
) -> None:
    config = load_encoder_config()
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "chunks.jsonl"
    _write_documents(source, [_document("DOC-1", "One sentence.")])
    output.write_bytes(b"baseline\n")
    config_path = source if alias_target == "input" else output
    source_before = source.read_bytes()
    output_before = output.read_bytes()

    with pytest.raises(ValueError, match="config_path must differ"):
        parallel.run_parallel_pipeline_phase_1(
            source,
            output,
            config=config,
            local_files_only=True,
            config_path=config_path,
            total_document_count=1,
            work_dir=tmp_path / "work",
            counter_factory=fake_counter_factory,
        )

    assert source.read_bytes() == source_before
    assert output.read_bytes() == output_before


def test_parallel_pipeline_rejects_negative_document_count(tmp_path: Path) -> None:
    config = load_encoder_config()
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "chunks.jsonl"
    _write_documents(source, [])

    with pytest.raises(ValueError, match="cannot be negative"):
        parallel.run_parallel_pipeline_phase_1(
            source,
            output,
            config=config,
            local_files_only=True,
            total_document_count=-1,
            work_dir=tmp_path / "work",
            counter_factory=fake_counter_factory,
        )


def test_worker_failure_preserves_previous_published_output(tmp_path: Path) -> None:
    config = load_encoder_config()
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "chunks.jsonl"
    output.write_bytes(b"validated-baseline\n")
    _write_documents(source, [_document("DOC-FAIL", "FAIL-WORKER")])

    with pytest.raises(RuntimeError, match="controlled worker failure"):
        parallel.run_parallel_pipeline_phase_1(
            source,
            output,
            config=config,
            local_files_only=True,
            total_document_count=1,
            work_dir=tmp_path / "work",
            counter_factory=failing_counter_factory,
        )

    assert output.read_bytes() == b"validated-baseline\n"


def test_initializer_failure_preserves_previous_output(tmp_path: Path) -> None:
    config = load_encoder_config()
    source = tmp_path / "documents.jsonl"
    output = tmp_path / "chunks.jsonl"
    output.write_bytes(b"validated-baseline\n")
    _write_documents(source, [_document("DOC-1", "One sentence.")])

    with pytest.raises(RuntimeError, match="controlled initializer failure"):
        parallel.run_parallel_pipeline_phase_1(
            source,
            output,
            config=config,
            local_files_only=True,
            total_document_count=1,
            work_dir=tmp_path / "work",
            counter_factory=initializer_failure_factory,
        )

    assert output.read_bytes() == b"validated-baseline\n"


def test_phase_1_cli_rejects_more_than_one_worker() -> None:
    with pytest.raises(SystemExit):
        parse_args(["produce", "--workers", "2"])


def test_phase_1_cli_requires_work_directory() -> None:
    with pytest.raises(SystemExit):
        parse_args(["produce", "--workers", "1"])


def test_merge_destination_contract_is_binary() -> None:
    annotations = parallel.merge_pending_results.__annotations__
    assert annotations["destination"] in {"BinaryIO", BinaryIO}
