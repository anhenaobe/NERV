"""Focused tests for the production Phase-2 E2E orchestrator."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from nerv import pipeline
from nerv.chunking.configuration import EncoderCapacity

from .test_embeddings import FakeEncoder


def _write_tiny_inputs(root: Path) -> tuple[Path, Path]:
    corpus = root / "raw"
    corpus.mkdir(parents=True, exist_ok=True)
    phenomenon = corpus / "F1_synthetic"
    phenomenon.mkdir(exist_ok=True)
    for number in range(12):
        (phenomenon / f"document-{number:02d}.txt").write_text(
            f"Orbital evidence number {number} is stable and measurable.",
            encoding="utf-8",
        )
    queries = root / "queries.jsonl"
    queries.write_text(
        "".join(
            json.dumps(
                {
                    "query_id": f"q{number:03d}",
                    "text": f"orbital evidence {number}",
                }
            )
            + "\n"
            for number in range(1, 51)
        ),
        encoding="utf-8",
    )
    return corpus, queries


def _arguments(root: Path, extra: list[str] | None = None) -> Namespace:
    corpus, queries = _write_tiny_inputs(root)
    values = [
        "--corpus",
        str(corpus),
        "--documents",
        str(root / "documentos.jsonl"),
        "--chunks",
        str(root / "chunks.jsonl"),
        "--chunk-config",
        str(root / "chunking_config.json"),
        "--chunk-metrics",
        str(root / "chunking_metrics.json"),
        "--embeddings",
        str(root / "embeddings.npy"),
        "--embedding-manifest",
        str(root / "embeddings.manifest.json"),
        "--faiss-index",
        str(root / "index.faiss"),
        "--faiss-metadata",
        str(root / "metadata.jsonl"),
        "--queries",
        str(queries),
        "--results",
        str(root / "resultados.jsonl"),
        "--metrics",
        str(root / "run_metrics.json"),
        "--run-manifest",
        str(root / "run_manifest.json"),
        "--embedding-batch-size",
        "4",
    ]
    if extra:
        values.extend(extra)
    return pipeline.build_parser().parse_args(values)


def _fresh_arguments(root: Path, extra: list[str] | None = None) -> Namespace:
    values = [
        "--mode",
        "fresh",
        "--device",
        "cuda",
        "--embedding-batch-size",
        "16",
        "--fixture-validation",
        "--corpus",
        str(pipeline.PROJECT_ROOT / "tests" / "fixtures" / "e2e_tiny" / "raw"),
        "--expected-document-sha256",
        pipeline.EXPECTED_FULL_DOCUMENT_SHA256,
        "--expected-chunk-sha256",
        pipeline.EXPECTED_FULL_CHUNK_SHA256,
        "--historical-document-count",
        str(pipeline.EXPECTED_FULL_DOCUMENT_COUNT),
        "--historical-chunk-count",
        str(pipeline.EXPECTED_FULL_CHUNK_COUNT),
    ]
    if extra:
        values.extend(extra)
    return _arguments(root, values)


def _production_fresh_arguments(root: Path) -> Namespace:
    return _arguments(
        root,
        [
            "--mode",
            "fresh",
            "--device",
            "cuda",
            "--embedding-batch-size",
            "16",
            "--expected-document-count",
            str(pipeline.EXPECTED_FULL_DOCUMENT_COUNT),
            "--expected-chunk-count",
            str(pipeline.EXPECTED_FULL_CHUNK_COUNT),
            "--expected-document-sha256",
            pipeline.EXPECTED_FULL_DOCUMENT_SHA256,
            "--expected-chunk-sha256",
            pipeline.EXPECTED_FULL_CHUNK_SHA256,
            "--historical-document-count",
            str(pipeline.EXPECTED_FULL_DOCUMENT_COUNT),
            "--historical-chunk-count",
            str(pipeline.EXPECTED_FULL_CHUNK_COUNT),
        ],
    )


def _resume_fixture_arguments(
    root: Path,
) -> tuple[Namespace, Path, Path, Path]:
    parent_run_id = "fixture-fresh-parent"
    parent = root / parent_run_id
    artifacts = parent / "artifacts"
    parent_args = _arguments(
        artifacts,
        [
            "--run-id",
            parent_run_id,
            "--to-stage",
            "chunking",
            "--run-manifest",
            str(parent / "run_manifest.json"),
            "--metrics",
            str(parent / "run_metrics.json"),
            "--report",
            str(parent / "e2e_execution_report.txt"),
            "--pipeline-log",
            str(parent / "pipeline.log"),
        ],
    )
    pipeline.run(parent_args)
    parent_manifest_path = parent / "run_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    parent_manifest.update(
        {
            "mode": "fresh",
            "status": "failed",
            "failed_stage": "chunking",
            "error": (
                "ValueError: chunk count does not match the explicitly expected "
                "identity."
            ),
            "stages_executed": ["ingestion", "chunking"],
            "stages_reused": [],
        }
    )
    parent_manifest["stage_records"]["ingestion"]["action"] = "EXECUTED"
    parent_manifest["stage_records"]["chunking"].update(
        {"status": "failed", "action": None}
    )
    parent_manifest_path.write_text(
        json.dumps(parent_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    documents = artifacts / "documentos.jsonl"
    chunks = artifacts / "chunks.jsonl"
    chunk_config = artifacts / "chunking_config.json"
    document_count, document_sha = pipeline._validate_documents(documents)
    chunk_info = pipeline._validate_chunks(chunks, chunk_config, None)
    evidence_report = root / "fixture_postmortem.txt"
    evidence_report.write_text(
        "fixture canonical validation passed\n", encoding="utf-8"
    )
    approval = {
        "schema_version": 1,
        "approval_status": "REVIEWED_VALIDATED",
        "run_id": parent_run_id,
        "parent_mode": "fresh",
        "documents": {"count": document_count, "sha256": document_sha},
        "chunks": {
            "count": chunk_info["rows"],
            "sha256": chunk_info["sha256"],
            "maximum_num_tokens": chunk_info["max_stored_tokens"],
        },
        "chunking_config": {
            "sha256": pipeline.sha256_file(chunk_config),
            "configuration_fingerprint": pipeline._configuration_fingerprint(
                pipeline.load_encoder_config()
            ),
        },
        "canonical_chunk_validation": {
            "completion_status": "completed",
            "valid_chunk_count": chunk_info["rows"],
            "invalid_chunk_count": 0,
            "failure_count": 0,
            "maximum_num_tokens": chunk_info["max_stored_tokens"],
            "checked_document_count": document_count,
            "token_count_mismatch_count": 0,
            "encoder_hard_limit_exceeded_count": 0,
            "hard_split_metadata_error_count": 0,
            "duplicate_chunk_id_count": 0,
            "position_error_count": 0,
            "document_order_error_count": 0,
            "unknown_document_count": 0,
            "missing_output_document_count": 0,
        },
        "inherited_stage_seconds": {"ingestion": 10.25, "chunking": 4.75},
        "evidence_report": str(evidence_report),
    }
    approval_path = root / "resume_approval.json"
    approval_path.write_text(
        json.dumps(approval, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    continuation = root / "continuation"
    resume_args = _arguments(
        continuation,
        [
            "--mode",
            "resume-fresh",
            "--run-id",
            "fixture-resume",
            "--resume-run-id",
            parent_run_id,
            "--resume-run-directory",
            str(parent),
            "--resume-approval",
            str(approval_path),
            "--from-stage",
            "embeddings",
            "--to-stage",
            "validation",
            "--device",
            "cuda",
            "--embedding-batch-size",
            "16",
            "--documents",
            str(documents),
            "--chunks",
            str(chunks),
            "--chunk-config",
            str(chunk_config),
            "--chunk-metrics",
            str(artifacts / "chunking_metrics.json"),
            "--faiss-metadata",
            str(chunks),
        ],
    )
    return resume_args, parent, approval_path, continuation


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_cuda_available", lambda: True)
    monkeypatch.setattr(
        pipeline,
        "load_encoder",
        lambda **_options: FakeEncoder(model_max_length=514),
    )
    monkeypatch.setattr(
        pipeline,
        "resolve_encoder_capacity",
        lambda **_options: EncoderCapacity(
            encoder_max_input_tokens=512,
            encoder_special_token_overhead=2,
            effective_content_max_tokens=510,
            tokenizer_model_max_length=514,
            model_max_position_embeddings=514,
        ),
    )


def test_tiny_e2e_stage_order_metrics_reuse_and_force(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    args = _arguments(tmp_path)
    interrupted = tmp_path / ".embeddings.npy.interrupted.tmp"
    interrupted.write_bytes(b"invalid partial CPU attempt")
    metrics, manifest = pipeline.run(args)

    assert manifest["status"] == "completed"
    assert manifest["stages_requested"] == list(pipeline.STAGES)
    assert manifest["stages_executed"] == list(pipeline.STAGES)
    assert manifest["stages_reused"] == []
    assert interrupted.read_bytes() == b"invalid partial CPU attempt"
    assert metrics["stages"]["ingestion"]["documents_written"] == 12
    assert metrics["stages"]["chunking"]["chunks"] >= 12
    assert metrics["stages"]["embeddings"]["dimension"] == 384
    assert metrics["stages"]["faiss"]["index_type"] == "IndexFlatIP"
    assert metrics["stages"]["retrieval"]["query_count"] == 50
    assert metrics["stages"]["validation"]["valid"] is True
    report = (tmp_path / "e2e_execution_report.txt").read_text(encoding="utf-8")
    assert "2. FINAL VERDICT\nPASS" in report
    assert "37. STAGE RUNTIME TABLE" in report
    assert "43. RECOMMENDED NEXT OPTIMIZATION" in report
    persisted_metrics = json.loads(
        (tmp_path / "run_metrics.json").read_text(encoding="utf-8")
    )
    assert set(persisted_metrics["stages"]) == set(pipeline.STAGES)
    assert all(
        manifest["stage_records"][stage]["status"] == "completed"
        for stage in pipeline.STAGES
    )
    assert all(
        isinstance(metrics[key], float) and metrics[key] >= 0.0
        for key in (
            "T_ingestion",
            "T_chunking",
            "T_embeddings",
            "T_faiss_build",
            "T_query_loading",
            "T_query_encoding",
            "T_retrieval",
            "T_validation",
            "T_total",
        )
    )

    reused_metrics, reused = pipeline.run(
        _arguments(tmp_path, ["--reuse-existing"])
    )
    assert reused["status"] == "completed"
    assert reused["stages_executed"] == ["retrieval", "validation"]
    assert reused["stages_reused"] == list(pipeline.STAGES[:4])
    assert reused_metrics["T_embeddings"] == 0.0
    assert reused_metrics["T_faiss_build"] == 0.0

    _forced_metrics, forced = pipeline.run(
        _arguments(
            tmp_path,
            [
                "--from-stage",
                "retrieval",
                "--to-stage",
                "validation",
                "--reuse-existing",
                "--force-stage",
                "retrieval",
            ],
        )
    )
    assert forced["stages_requested"] == ["retrieval", "validation"]
    assert forced["stages_executed"] == ["retrieval", "validation"]
    assert forced["stages_reused"] == [
        "ingestion",
        "chunking",
        "embeddings",
        "faiss",
    ]


def test_reuse_rejects_chunk_embedding_and_faiss_identity_mismatches(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    pipeline.run(_arguments(tmp_path))
    chunks = tmp_path / "chunks.jsonl"
    original_chunks = chunks.read_bytes()
    chunks.write_bytes(original_chunks + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        pipeline.run(
            _arguments(
                tmp_path,
                [
                    "--from-stage",
                    "embeddings",
                    "--to-stage",
                    "embeddings",
                    "--reuse-existing",
                ],
            )
        )
    failed = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["failed_stage"] == "embeddings"
    failure_report = (tmp_path / "e2e_execution_report.txt").read_text(
        encoding="utf-8"
    )
    assert "2. FINAL VERDICT\nFAIL" in failure_report
    assert "FAILED STAGE: embeddings" in failure_report
    assert "SHA-256" in failure_report
    chunks.write_bytes(original_chunks)

    embedding_manifest = tmp_path / "embeddings.manifest.json"
    original_embedding_manifest = embedding_manifest.read_text(encoding="utf-8")
    payload = json.loads(original_embedding_manifest)
    payload["encoder_model_name"] = "wrong/model"
    embedding_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="encoder_model_name"):
        pipeline.run(
            _arguments(
                tmp_path,
                [
                    "--from-stage",
                    "embeddings",
                    "--to-stage",
                    "embeddings",
                    "--reuse-existing",
                ],
            )
        )
    embedding_manifest.write_text(original_embedding_manifest, encoding="utf-8")

    sidecar = tmp_path / "index.faiss.manifest.json"
    original_sidecar = sidecar.read_text(encoding="utf-8")
    payload = json.loads(original_sidecar)
    payload["embeddings_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="embedding identity"):
        pipeline.run(
            _arguments(
                tmp_path,
                [
                    "--from-stage",
                    "faiss",
                    "--to-stage",
                    "faiss",
                    "--reuse-existing",
                ],
            )
        )
    sidecar.write_text(original_sidecar, encoding="utf-8")


def test_cuda_request_fails_before_reuse_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was explicitly requested"):
        pipeline.run(
            _fresh_arguments(tmp_path)
        )
    failure_report = (tmp_path / "e2e_execution_report.txt").read_text(
        encoding="utf-8"
    )
    assert "FAILED STAGE: initialization" in failure_report
    assert "CUDA was explicitly requested" in failure_report


def test_invalid_stage_range_force_and_output_alias_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not come after"):
        pipeline.run(
            _arguments(
                tmp_path,
                ["--from-stage", "faiss", "--to-stage", "chunking"],
            )
        )
    with pytest.raises(ValueError, match="outside the requested range"):
        pipeline.run(
            _arguments(
                tmp_path,
                [
                    "--from-stage",
                    "embeddings",
                    "--to-stage",
                    "faiss",
                    "--force-stage",
                    "ingestion",
                ],
            )
        )

    embeddings = tmp_path / "embeddings.npy"
    manifest = tmp_path / "embeddings.manifest.json"
    embeddings.write_bytes(b"published matrix")
    manifest.write_bytes(b"published manifest")
    with pytest.raises(ValueError, match="never overwrites published artifacts"):
        pipeline.run(
            _arguments(tmp_path, ["--force-stage", "embeddings"])
        )
    assert embeddings.read_bytes() == b"published matrix"
    assert manifest.read_bytes() == b"published manifest"
    assert not (tmp_path / "run_manifest.json").exists()


def test_invalid_fresh_paths_and_validation_only_do_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "_cuda_available", lambda: True)
    external_log = tmp_path.parent / f"{tmp_path.name}-protected.log"
    external_log.write_bytes(b"protected")
    args = _fresh_arguments(
        tmp_path,
        ["--pipeline-log", str(external_log)],
    )
    with pytest.raises(ValueError, match="must be isolated"):
        pipeline.run(args)
    assert external_log.read_bytes() == b"protected"
    assert not (tmp_path / "run_manifest.json").exists()
    assert not (tmp_path / "run_metrics.json").exists()

    validation_root = tmp_path / "validation-only"
    validation_args = _arguments(
        validation_root,
        ["--from-stage", "validation", "--to-stage", "validation"],
    )
    with pytest.raises(ValueError, match="validation-only E2E runs are prohibited"):
        pipeline.run(validation_args)
    assert not (validation_root / "run_manifest.json").exists()
    assert not (validation_root / "pipeline.log").exists()


def test_retrieval_always_executes_and_fragment_fallback_is_bounded(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    pipeline.run(_arguments(tmp_path))
    results = tmp_path / "resultados.jsonl"
    rerun_args = _arguments(
        tmp_path, ["--from-stage", "retrieval", "--reuse-existing"]
    )
    queries = rerun_args.queries
    records = [
        json.loads(line)
        for line in queries.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["text"] = "materially changed query text"
    queries.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    results.write_bytes(b"stale unbound results")
    _metrics, manifest = pipeline.run(rerun_args)
    assert "retrieval" in manifest["stages_executed"]
    assert "retrieval" not in manifest["stages_reused"]
    assert results.read_bytes() != b"stale unbound results"

    text = " ".join(f"word{number}" for number in range(300))
    bounded = pipeline._truncate_respecting_sentences(text, 250)
    assert len(bounded.split()) == 250


def test_fresh_stops_when_canonical_chunk_validation_fails(
    tmp_path: Path,
    fake_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "validate_chunk_artifact", lambda *_a, **_k: 2)
    monkeypatch.setattr(
        pipeline,
        "generate_embedding_artifact",
        lambda *_a, **_k: pytest.fail("embeddings ran after failed chunk validation"),
    )
    with pytest.raises(ValueError, match="canonical chunk validation failed"):
        pipeline.run(_fresh_arguments(tmp_path))
    same = tmp_path / "same.json"
    with pytest.raises(ValueError, match="must be distinct"):
        pipeline.run(
            _arguments(
                tmp_path,
                ["--metrics", str(same), "--run-manifest", str(same)],
            )
        )


def test_stage_callable_map_covers_the_explicit_contract() -> None:
    assert tuple(pipeline.STAGE_CALLABLES) == pipeline.STAGES
    assert all(len(contract) == 4 for contract in pipeline.STAGE_CALLABLES.values())


def test_fresh_executes_every_stage_in_order_without_reuse(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    metrics, manifest = pipeline.run(_fresh_arguments(tmp_path))

    assert manifest["status"] == "completed"
    assert manifest["stages_requested"] == list(pipeline.STAGES)
    assert manifest["stages_executed"] == list(pipeline.STAGES)
    assert manifest["stages_reused"] == []
    assert [
        manifest["stage_records"][stage]["action"] for stage in pipeline.STAGES
    ] == ["EXECUTED"] * len(pipeline.STAGES)
    assert all(
        metrics["stages"][stage]["input_count"] >= 1
        and metrics["stages"][stage]["output_count"] >= 1
        for stage in pipeline.STAGES
    )
    fresh = manifest["fresh_execution"]
    assert fresh["raw_corpus_file_count"] == 12
    assert fresh["raw_corpus_path"].endswith("raw")
    assert fresh["document_reproducibility"]["classification"] == "DIFFERENT"
    assert fresh["chunk_reproducibility"]["classification"] == "DIFFERENT"
    assert fresh["chunk_identity_policy"]["policy"] == (
        "COMPARISON_ONLY_UPSTREAM_CHANGED"
    )
    report = (tmp_path / "e2e_execution_report.txt").read_text(encoding="utf-8")
    assert "FRESH EXECUTION" in report
    assert "Any reused stages: NONE" in report
    assert "OFFICIAL QUERY RESULTS: NOT AVAILABLE" in report


def test_fresh_rejects_reuse_and_nonisolated_outputs(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    with pytest.raises(ValueError, match="prohibits --reuse-existing"):
        pipeline.run(_fresh_arguments(tmp_path / "reuse", ["--reuse-existing"]))

    outside = tmp_path / "accepted-production" / "embeddings.npy"
    with pytest.raises(ValueError, match="must be isolated"):
        pipeline.run(
            _fresh_arguments(
                tmp_path / "isolation",
                ["--embeddings", str(outside)],
            )
        )
    assert not outside.exists()


def test_fresh_failure_preserves_external_production_sentinels(
    tmp_path: Path,
    fake_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = tmp_path / "accepted-production"
    production.mkdir()
    sentinels = {
        production / "documentos.jsonl": b"accepted documents",
        production / "chunks.jsonl": b"accepted chunks",
        production / "embeddings.npy": b"accepted embeddings",
        production / "index.faiss": b"accepted index",
    }
    for path, content in sentinels.items():
        path.write_bytes(content)

    def fail_chunking(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("directed fresh chunking failure")

    monkeypatch.setattr(pipeline, "run_chunking_pipeline", fail_chunking)
    with pytest.raises(RuntimeError, match="directed fresh chunking failure"):
        pipeline.run(_fresh_arguments(tmp_path / "failed-run"))

    assert all(path.read_bytes() == content for path, content in sentinels.items())
    failed_manifest = json.loads(
        (tmp_path / "failed-run" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["failed_stage"] == "chunking"


def test_production_fresh_fails_before_ingestion_when_ocr_is_unavailable(
    tmp_path: Path,
    fake_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline,
        "_validate_ocr_runtime",
        lambda: (_ for _ in ()).throw(RuntimeError("directed OCR unavailable")),
    )
    with pytest.raises(RuntimeError, match="directed OCR unavailable"):
        pipeline.run(_production_fresh_arguments(tmp_path))

    manifest = json.loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["failed_stage"] == "initialization"
    assert manifest["stages_executed"] == []
    assert not (tmp_path / "documentos.jsonl").exists()


def test_reproducibility_classification_levels(tmp_path: Path) -> None:
    historical = tmp_path / "historical.jsonl"
    fresh = tmp_path / "fresh.jsonl"
    historical.write_text(
        '{"doc_id":"old","fuente":"x.txt","formato":"txt",'
        '"fenomeno":1,"texto":"same"}\n',
        encoding="utf-8",
    )
    fresh.write_text(historical.read_text(encoding="utf-8"), encoding="utf-8")
    identical = pipeline._classify_reproducibility(
        actual_path=fresh,
        actual_count=1,
        actual_sha256=pipeline.sha256_file(fresh),
        historical_path=historical,
        historical_count=1,
        historical_sha256=pipeline.sha256_file(historical),
        artifact_kind="documents",
    )
    assert identical["classification"] == "BYTE_IDENTICAL"

    fresh.write_text(
        '{"doc_id":"new","fuente":"x.txt","formato":"txt",'
        '"fenomeno":1,"texto":"same"}\n',
        encoding="utf-8",
    )
    semantic = pipeline._classify_reproducibility(
        actual_path=fresh,
        actual_count=1,
        actual_sha256=pipeline.sha256_file(fresh),
        historical_path=historical,
        historical_count=1,
        historical_sha256=pipeline.sha256_file(historical),
        artifact_kind="documents",
    )
    assert semantic["classification"] == "SEMANTICALLY_EQUIVALENT"

    fresh.write_text(
        '{"doc_id":"new","fuente":"x.txt","formato":"txt",'
        '"fenomeno":1,"texto":"changed"}\n',
        encoding="utf-8",
    )
    structural = pipeline._classify_reproducibility(
        actual_path=fresh,
        actual_count=1,
        actual_sha256=pipeline.sha256_file(fresh),
        historical_path=historical,
        historical_count=1,
        historical_sha256=pipeline.sha256_file(historical),
        artifact_kind="documents",
    )
    assert structural["classification"] == "STRUCTURALLY_EQUIVALENT"

    different = pipeline._classify_reproducibility(
        actual_path=fresh,
        actual_count=2,
        actual_sha256=pipeline.sha256_file(fresh),
        historical_path=historical,
        historical_count=1,
        historical_sha256=pipeline.sha256_file(historical),
        artifact_kind="documents",
    )
    assert different["classification"] == "DIFFERENT"


def test_dependency_aware_chunk_identity_policy() -> None:
    strict = pipeline._apply_dependency_aware_chunk_policy(
        current_document_sha256="A" * 64,
        historical_document_sha256="A" * 64,
        current_chunk_count=10,
        current_chunk_sha256="B" * 64,
        historical_chunk_count=10,
        historical_chunk_sha256="B" * 64,
    )
    assert strict["policy"] == "STRICT_HISTORICAL_IDENTITY"

    with pytest.raises(ValueError, match="count is strict"):
        pipeline._apply_dependency_aware_chunk_policy(
            current_document_sha256="A" * 64,
            historical_document_sha256="A" * 64,
            current_chunk_count=9,
            current_chunk_sha256="B" * 64,
            historical_chunk_count=10,
            historical_chunk_sha256="B" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256 is strict"):
        pipeline._apply_dependency_aware_chunk_policy(
            current_document_sha256="A" * 64,
            historical_document_sha256="A" * 64,
            current_chunk_count=10,
            current_chunk_sha256="C" * 64,
            historical_chunk_count=10,
            historical_chunk_sha256="B" * 64,
        )

    comparison = pipeline._apply_dependency_aware_chunk_policy(
        current_document_sha256="C" * 64,
        historical_document_sha256="A" * 64,
        current_chunk_count=9,
        current_chunk_sha256="D" * 64,
        historical_chunk_count=10,
        historical_chunk_sha256="B" * 64,
    )
    assert comparison["policy"] == "COMPARISON_ONLY_UPSTREAM_CHANGED"
    assert comparison["chunk_count_matches_historical"] is False
    assert comparison["chunk_sha256_matches_historical"] is False


def test_resume_fresh_executes_only_downstream_and_reports_inherited_timing(
    tmp_path: Path,
    fake_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, parent, _approval, continuation = _resume_fixture_arguments(tmp_path)
    parent_embedding = parent / "artifacts" / "embeddings.npy"
    parent_faiss = parent / "artifacts" / "index.faiss"
    parent_embedding.write_bytes(b"historical incompatible embeddings")
    parent_faiss.write_bytes(b"historical incompatible index")

    monkeypatch.setattr(
        pipeline,
        "procesar_corpus",
        lambda *_args, **_kwargs: pytest.fail("resume-fresh executed ingestion"),
    )
    monkeypatch.setattr(
        pipeline,
        "run_chunking_pipeline",
        lambda *_args, **_kwargs: pytest.fail("resume-fresh executed chunking"),
    )
    metrics, manifest = pipeline.run(args)

    assert manifest["status"] == "completed"
    assert manifest["stages_inherited"] == ["ingestion", "chunking"]
    assert manifest["stages_reused"] == []
    assert manifest["stages_executed"] == list(pipeline.STAGES[2:])
    assert manifest["stage_records"]["ingestion"]["action"] == (
        "INHERITED_VALIDATED"
    )
    assert manifest["stage_records"]["chunking"]["action"] == (
        "INHERITED_VALIDATED"
    )
    assert all(
        manifest["stage_records"][stage]["action"] == "EXECUTED"
        for stage in pipeline.STAGES[2:]
    )
    downstream = sum(
        float(metrics["stages"][stage]["seconds"]) for stage in pipeline.STAGES[2:]
    )
    assert metrics["resumed_fresh_compute_equivalent_e2e_seconds"] == pytest.approx(
        10.25 + 4.75 + downstream
    )
    assert metrics["continuation_wall_clock_seconds"] == metrics["T_total"]
    assert parent_embedding.read_bytes() == b"historical incompatible embeddings"
    assert parent_faiss.read_bytes() == b"historical incompatible index"
    assert (continuation / "embeddings.npy").is_file()
    assert (continuation / "index.faiss").is_file()
    report = (continuation / "e2e_execution_report.txt").read_text(
        encoding="utf-8"
    )
    assert "RESUME-FRESH CONTINUATION" in report
    assert "RESUMED FRESH COMPUTE-EQUIVALENT E2E TIME" in report
    assert "CONTINUATION WALL-CLOCK TIME" in report
    assert "OFFICIAL QUERY RESULTS: NOT AVAILABLE" in report


def test_resume_fresh_rejects_nonfresh_parent_and_altered_artifacts(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    args, parent, _approval, _continuation = _resume_fixture_arguments(tmp_path)
    manifest_path = parent / "run_manifest.json"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    payload = json.loads(original_manifest)
    payload["mode"] = "full"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="FRESH parent"):
        pipeline.run(args)
    manifest_path.write_text(original_manifest, encoding="utf-8")

    documents = parent / "artifacts" / "documentos.jsonl"
    original_documents = documents.read_bytes()
    documents.write_bytes(original_documents + b"\n")
    with pytest.raises(ValueError, match="document SHA-256"):
        pipeline.run(args)
    documents.write_bytes(original_documents)

    chunks = parent / "artifacts" / "chunks.jsonl"
    original_chunks = chunks.read_bytes()
    chunks.write_bytes(original_chunks + b"\n")
    with pytest.raises(ValueError, match="chunk SHA-256"):
        pipeline.run(args)
    chunks.write_bytes(original_chunks)


def test_resume_fresh_rejects_unexpected_approval_and_structural_failure(
    tmp_path: Path,
    fake_runtime: None,
) -> None:
    args, parent, approval_path, _continuation = _resume_fixture_arguments(tmp_path)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    original_approval = approval_path.read_text(encoding="utf-8")
    approval["chunks"]["count"] += 1
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    with pytest.raises(ValueError, match="chunk count differs"):
        pipeline.run(args)

    approval_path.write_text(original_approval, encoding="utf-8")
    chunks_path = parent / "artifacts" / "chunks.jsonl"
    lines = chunks_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["num_tokens"] = 511
    lines[0] = json.dumps(first)
    chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds effective stored capacity"):
        pipeline.run(args)
