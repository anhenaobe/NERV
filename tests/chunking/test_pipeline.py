"""Offline tests for the streaming tokenizer-based JSONL pipeline."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import nerv.chunking.pipeline as chunking_pipeline
from nerv.chunking import TokenCounter, split_sentences
from nerv.chunking.pipeline import (
    GenuineOversizedSentenceError,
    PipelineSummary,
    parse_args,
    process_document,
    read_documents,
    run_pipeline,
)

from .fakes import DeterministicFakeTokenizer

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "documentos_pipeline_test.jsonl"
)


def _counter(model_name: str = "fake/offline-tokenizer") -> TokenCounter:
    return TokenCounter(
        model_name,
        tokenizer=DeterministicFakeTokenizer(),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_documents() -> dict[str, dict[str, Any]]:
    return {document["doc_id"]: document for document in _read_jsonl(FIXTURE_PATH)}


def test_read_documents_is_lazy_and_reports_line_numbers(tmp_path: Path) -> None:
    """The generator does not decode later lines before they are requested."""
    source = tmp_path / "documents.jsonl"
    source.write_text(
        '{"doc_id":"first"}\n{invalid json}\n',
        encoding="utf-8",
    )

    documents = read_documents(source)
    line_number, first = next(documents)

    assert line_number == 1
    assert first == {"doc_id": "first"}
    with pytest.raises(ValueError, match="line 2"):
        next(documents)


def test_pipeline_preserves_contract_and_writes_configuration(
    tmp_path: Path,
) -> None:
    """A complete offline run preserves metadata and tokenizer counts."""
    output_path = tmp_path / "nested" / "chunks.jsonl"
    config_path = tmp_path / "nested" / "chunking_config.json"
    counter = _counter()

    summary = run_pipeline(
        FIXTURE_PATH,
        output_path,
        token_counter=counter,
        max_tokens=70,
        overlap_tokens=0,
        config_path=config_path,
    )

    records = _read_jsonl(output_path)
    documents = _source_documents()
    assert isinstance(summary, PipelineSummary)
    assert summary.document_count == 4
    assert summary.chunk_count == len(records)
    assert summary.oversized_chunk_count >= 1
    assert summary.encoder_model_name == "fake/offline-tokenizer"
    assert output_path.exists()
    assert config_path.exists()
    assert len(records) > summary.document_count

    records_by_document: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        records_by_document.setdefault(record["doc_id"], []).append(record)
        source = documents[record["doc_id"]]
        assert record["fuente"] == source["fuente"]
        assert record["formato"] == source["formato"]
        assert record["fenomeno"] == source["fenomeno"]
        assert record["num_tokens"] == counter.count(
            record["texto"],
            include_document_prefix=True,
        )
        assert set(record) == {
            "doc_id",
            "chunk_id",
            "fuente",
            "formato",
            "fenomeno",
            "posicion",
            "num_tokens",
            "texto",
        }

    for doc_id, document_records in records_by_document.items():
        assert [record["posicion"] for record in document_records] == list(
            range(len(document_records))
        )
        assert [record["chunk_id"] for record in document_records] == [
            f"{doc_id}-chunk-{position:04d}"
            for position in range(len(document_records))
        ]
        expected_sentences = split_sentences(documents[doc_id]["texto"])
        actual_text = " ".join(str(record["texto"]) for record in document_records)
        assert actual_text == " ".join(expected_sentences)
        assert all(
            any(sentence in str(record["texto"]) for record in document_records)
            for sentence in expected_sentences
        )

    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    assert configuration["encoder_model_name"] == "fake/offline-tokenizer"
    assert configuration["tokenizer_class"] == "DeterministicFakeTokenizer"
    assert configuration["add_special_tokens"] is False
    assert configuration["trust_remote_code"] is False
    assert configuration["chunk_max_tokens"] == 70
    assert configuration["overlap_tokens"] == 0
    assert configuration["oversized_sentence_policy"] == (
        "structural_records_may_subdivide_but_prose_above_limit_fails_closed"
    )
    assert configuration["encoder_max_input_tokens"] == 512


def test_pipeline_subdivides_only_above_hard_limit_with_trace() -> None:
    """The hard layer preserves safe logical chunks and traces only split ones."""
    counter = _counter()
    safe_text = "a" * 500
    safe_records = process_document(
        {
            "doc_id": "safe",
            "fuente": "fixture",
            "formato": "txt",
            "fenomeno": 0,
            "texto": safe_text,
        },
        sentence_splitter=lambda text: [text],
        token_counter=counter,
        max_tokens=256,
        overlap_tokens=32,
        encoder_max_input_tokens=512,
    )
    assert len(safe_records) == 1
    assert safe_records[0]["texto"] == safe_text
    assert set(safe_records[0]) == {
        "doc_id",
        "chunk_id",
        "fuente",
        "formato",
        "fenomeno",
        "posicion",
        "num_tokens",
        "texto",
    }

    oversized_text = "a" * 900
    records = process_document(
        {
            "doc_id": "split",
            "fuente": "fixture",
            "formato": "json",
            "fenomeno": 0,
            "texto": oversized_text,
        },
        sentence_splitter=lambda text: [text],
        token_counter=counter,
        max_tokens=256,
        overlap_tokens=32,
        encoder_max_input_tokens=512,
    )
    assert len(records) >= 2
    assert [record["posicion"] for record in records] == list(range(len(records)))
    assert [record["chunk_id"] for record in records] == [
        f"split-chunk-{position:04d}" for position in range(len(records))
    ]
    assert all(record["num_tokens"] <= 512 for record in records)
    assert all(record["hard_split"] is True for record in records)
    assert {record["parent_chunk_id"] for record in records} == {
        "split-logical-chunk-0000"
    }
    assert [record["hard_split_part_index"] for record in records] == list(
        range(len(records))
    )
    assert {record["hard_split_part_count"] for record in records} == {len(records)}
    assert (
        "".join(
            oversized_text[
                int(record["hard_split_source_start"]) : int(
                    record["hard_split_source_end"]
                )
            ]
            for record in records
        )
        == oversized_text
    )


def test_genuine_oversized_prose_sentence_fails_closed() -> None:
    """Never silently subdivide a demonstrated indivisible prose sentence."""
    text = "a" * 900

    with pytest.raises(
        GenuineOversizedSentenceError,
        match="GENUINE_OVERSIZED_SENTENCE_BLOCKER.*908 tokens.*512",
    ):
        process_document(
            {
                "doc_id": "oversized-prose",
                "fuente": "report.txt",
                "formato": "txt",
                "fenomeno": 0,
                "texto": text,
            },
            sentence_splitter=lambda _: [text],
            token_counter=_counter(),
            max_tokens=256,
            overlap_tokens=32,
            encoder_max_input_tokens=512,
        )


def test_oversized_pdf_sql_is_classified_as_structural() -> None:
    """Allow traceable hard subdivision for deterministic PDF code structure."""
    text = " ".join(
        "SELECT value FROM records JOIN sources ON records.id = sources.id"
        for _ in range(80)
    )

    records = process_document(
        {
            "doc_id": "pdf-sql",
            "fuente": "appendix.pdf",
            "formato": "pdf",
            "fenomeno": 0,
            "texto": text,
        },
        sentence_splitter=lambda _: [text],
        token_counter=_counter(),
        max_tokens=256,
        overlap_tokens=32,
        encoder_max_input_tokens=512,
    )

    assert len(records) > 1
    assert all(record["hard_split"] is True for record in records)
    assert all(record["num_tokens"] <= 512 for record in records)


@pytest.mark.parametrize(
    "text",
    [
        ", ".join(f"Company {index} Incorporated" for index in range(180)),
        " ".join(f"• structural list item {index}" for index in range(120)),
        " ".join(f"term{index} OR field[{index}]" for index in range(120)),
        " ".join(f"https://example.test/reference/{index}" for index in range(120)),
        " ".join(f"({index}) structural section" for index in range(120)),
        " ".join(("AbCdEf0123456789_%-" * 4) for _ in range(40)),
        "[Pagina 17] • " + "objective " * 700,
    ],
)
def test_oversized_pdf_list_structures_are_hard_split(text: str) -> None:
    """Recognize deterministic lists and queries without weakening prose failure."""
    records = process_document(
        {
            "doc_id": "pdf-structural-list",
            "fuente": "appendix.pdf",
            "formato": "pdf",
            "fenomeno": 0,
            "texto": text,
        },
        sentence_splitter=lambda _: [text],
        token_counter=_counter(),
        max_tokens=256,
        overlap_tokens=32,
        encoder_max_input_tokens=512,
    )

    assert len(records) > 1
    assert all(record["hard_split"] is True for record in records)
    assert all(record["num_tokens"] <= 512 for record in records)


def test_hidden_non_additivity_cannot_bypass_final_exact_recount() -> None:
    """An unsampled tokenizer interaction falls back before hard enforcement."""

    class HiddenInteractionTokenizer:
        model_max_length = 512

        def __call__(
            self,
            text: str | list[str],
            *,
            add_special_tokens: bool,
            truncation: bool,
        ) -> dict[str, object]:
            del add_special_tokens, truncation

            def encode(value: str) -> list[int]:
                token_count = len(value.split())
                if "u10 u11" in value:
                    token_count += 600
                return list(range(token_count))

            return {
                "input_ids": (
                    [encode(value) for value in text]
                    if isinstance(text, list)
                    else encode(text)
                )
            }

    counter = TokenCounter(
        "fake/hidden-interaction",
        tokenizer=HiddenInteractionTokenizer(),
    )
    units = [f"u{index}" for index in range(30)]
    records = process_document(
        {
            "doc_id": "non-additive",
            "fuente": "fixture",
            "formato": "txt",
            "fenomeno": 0,
            "texto": " ".join(units),
        },
        sentence_splitter=lambda text: units,
        token_counter=counter,
        max_tokens=256,
        overlap_tokens=0,
        encoder_max_input_tokens=512,
    )
    assert len(records) > 1
    assert all(
        record["num_tokens"]
        == counter.count(str(record["texto"]), include_document_prefix=True)
        for record in records
    )
    assert all(int(record["num_tokens"]) <= 512 for record in records)


def test_pipeline_output_is_deterministic(tmp_path: Path) -> None:
    """Two runs with the same input and tokenizer produce identical JSONL."""
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"

    first_summary = run_pipeline(
        FIXTURE_PATH,
        first_output,
        token_counter=_counter(),
        max_tokens=70,
    )
    second_summary = run_pipeline(
        FIXTURE_PATH,
        second_output,
        token_counter=_counter(),
        max_tokens=70,
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    assert first_summary.chunk_count == second_summary.chunk_count
    assert first_summary.oversized_chunk_count == (second_summary.oversized_chunk_count)


def test_tabular_documents_preserve_rows_without_linguistic_splitting() -> None:
    """CSV rows remain complete and bypass whole-document PySBD processing."""
    document: dict[str, object] = {
        "doc_id": "DOC-TABLE",
        "fuente": "dataset.csv",
        "formato": "csv",
        "fenomeno": 1,
        "texto": "header,value\nfirst,one\n\nsecond,two",
    }

    def unexpected_splitter(_: str) -> list[str]:
        raise AssertionError("Tabular documents must not use PySBD.")

    records = process_document(
        document,
        sentence_splitter=unexpected_splitter,
        token_counter=_counter(),
        max_tokens=200,
        overlap_tokens=0,
    )

    assert [record["texto"] for record in records] == [
        "header,value first,one second,two"
    ]


def test_narrative_document_uses_bounded_splitter_and_one_detection() -> None:
    """The production narrative path performs document-level work once."""
    document: dict[str, object] = {
        "doc_id": "DOC-NARRATIVE",
        "fuente": "report.pdf",
        "formato": "pdf",
        "fenomeno": 1,
        "texto": "The report is complete. The review continues.",
    }

    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        patch.object(
            chunking_pipeline,
            "detect_language",
            wraps=chunking_pipeline.detect_language,
        ) as detector,
        patch.object(
            chunking_pipeline,
            "_split_sentences_bounded_with_stats",
            wraps=chunking_pipeline._split_sentences_bounded_with_stats,
        ) as bounded_splitter,
    ):
        monkeypatch.setattr(
            chunking_pipeline,
            "_create_additive_chunks",
            lambda units, **options: (
                units,
                options["token_counter"].count_many(
                    units,
                    include_document_prefix=True,
                ),
            ),
        )
        records = process_document(
            document,
            token_counter=_counter(),
            max_tokens=200,
            overlap_tokens=0,
        )

    detector.assert_called_once_with(document["texto"])
    bounded_splitter.assert_called_once()
    assert bounded_splitter.call_args.kwargs["language"] == "en"
    assert [record["texto"] for record in records] == [
        "The report is complete.",
        "The review continues.",
    ]


def test_empty_input_lines_are_ignored(tmp_path: Path) -> None:
    """Blank lines are skipped without changing source line reporting."""
    source = tmp_path / "documents.jsonl"
    source.write_text(
        "\n"
        '{"doc_id":"DOC-1","fuente":"one.pdf","formato":"pdf",'
        '"fenomeno":1,"texto":"One sentence."}\n'
        "\n",
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"

    summary = run_pipeline(
        source,
        output,
        token_counter=_counter(),
        max_tokens=100,
    )

    assert summary.document_count == 1
    assert len(_read_jsonl(output)) == 1


def test_invalid_json_keeps_existing_output_unchanged(tmp_path: Path) -> None:
    """A fatal parse error removes the temporary file, not valid old output."""
    source = tmp_path / "invalid.jsonl"
    source.write_text(
        '{"doc_id":"valid","fuente":"one.pdf","formato":"pdf",'
        '"fenomeno":1,"texto":"Valid text."}\n'
        "{invalid json}\n",
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"
    output.write_text("previous-valid-output\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        run_pipeline(
            source,
            output,
            token_counter=_counter(),
            max_tokens=100,
        )

    assert output.read_text(encoding="utf-8") == "previous-valid-output\n"
    assert not list(tmp_path.glob(".chunks.jsonl.*.tmp"))


def test_missing_field_is_rejected_atomically(tmp_path: Path) -> None:
    """Missing contract fields include the source line in the error."""
    source = tmp_path / "missing.jsonl"
    source.write_text(
        '{"doc_id":"DOC-1","fuente":"one.pdf","formato":"pdf","fenomeno":1}\n',
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"

    with pytest.raises(ValueError, match="line 1.*texto"):
        run_pipeline(
            source,
            output,
            token_counter=_counter(),
            max_tokens=100,
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("doc_id", "", "doc_id"),
        ("fuente", 12, "fuente"),
        ("formato", None, "formato"),
        ("fenomeno", "1", "fenomeno"),
        ("fenomeno", True, "fenomeno"),
        ("texto", "   ", "texto"),
    ],
)
def test_invalid_field_types_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    """Every established input field is validated before chunking."""
    document: dict[str, object] = {
        "doc_id": "DOC-1",
        "fuente": "one.pdf",
        "formato": "pdf",
        "fenomeno": 1,
        "texto": "Valid text.",
    }
    document[field] = value
    source = tmp_path / f"{field}.jsonl"
    source.write_text(
        json.dumps(document, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        run_pipeline(
            source,
            tmp_path / "chunks.jsonl",
            token_counter=_counter(),
            max_tokens=100,
        )


def test_cli_requires_encoder_and_uses_safe_defaults() -> None:
    """CLI parsing requires team configuration and disables remote code."""
    args = parse_args(
        [
            "--input",
            "documents.jsonl",
            "--output",
            "chunks.jsonl",
            "--encoder-model",
            "organization/model-name",
        ]
    )

    assert args.encoder_model == "organization/model-name"
    assert args.max_tokens == 256
    assert args.overlap_tokens == 32
    assert args.encoder_model == "organization/model-name"
    assert args.trust_remote_code is False
    assert args.use_slow_tokenizer is False


def test_optional_real_tokenizer_pipeline(tmp_path: Path) -> None:
    """Run the full flow only with an explicitly configured cached tokenizer."""
    model_name = os.getenv("NERV_TEST_ENCODER_MODEL")
    if not model_name:
        pytest.skip("NERV_TEST_ENCODER_MODEL is not configured.")

    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
    except (OSError, ValueError) as error:
        pytest.skip(f"Tokenizer is not available in the local cache: {error}")

    counter = TokenCounter(model_name, tokenizer=tokenizer)
    output = tmp_path / "real-tokenizer-chunks.jsonl"
    summary = run_pipeline(
        FIXTURE_PATH,
        output,
        token_counter=counter,
        max_tokens=128,
    )
    records = _read_jsonl(output)

    assert summary.document_count == 4
    assert all(
        record["num_tokens"]
        == counter.count(
            record["texto"],
            include_document_prefix=True,
        )
        for record in records
    )
    assert all(
        record["num_tokens"] <= 128 or record["doc_id"] == "PIPE-ES-OVERSIZED"
        for record in records
    )
