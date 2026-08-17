"""Bounded, read-only performance evidence collector for NERV Rev B.

The collector deliberately does not change semantic pipeline code, frozen
configuration, production artifacts, or the full-corpus execution path.  It
writes only isolated profiling evidence supplied with ``--output-dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from nerv.chunking.configuration import load_encoder_config
from nerv.chunking.pipeline import PipelineDocumentProgress, run_pipeline
from nerv.chunking.token_counter import TokenCounter
from nerv.embeddings.embedding_generator import generate_embeddings
from nerv.embeddings.encoder import load_encoder
from nerv.ingestion import lector_corpus as ingestion_module

ingestion: Any = ingestion_module

PDF_CLASSES = ("native_pdf", "mixed_pdf", "ocr_pdf")
SAMPLE_GROUPS = (
    *PDF_CLASSES,
    "image",
    "json",
    "csv",
    "xlsx",
    "pbf",
    "text",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".avif"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects.")
            yield value


def _stable_hash(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(
                record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest().upper()


def _percentile(values: Sequence[float | int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _spread(items: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Pick stable low/central/high representatives without duplicating rows."""
    if len(items) <= count:
        return list(items)
    positions = [
        round(index * (len(items) - 1) / (count - 1)) for index in range(count)
    ]
    return [items[position] for position in dict.fromkeys(positions)]


def _classify_pdf(path: Path, corpus: Path) -> dict[str, Any]:
    started = time.perf_counter()
    native_pages = 0
    empty_pages = 0
    error: str | None = None
    page_count = 0
    try:
        if ingestion.fitz is None:
            raise RuntimeError("PyMuPDF is unavailable")
        with ingestion.fitz.open(path) as document:
            page_count = document.page_count
            for page in document:
                if page.get_text("text").strip():
                    native_pages += 1
                else:
                    empty_pages += 1
    except Exception as exception:  # audit evidence must preserve malformed PDFs
        error = f"{type(exception).__name__}: {exception}"
    if error is not None:
        classification = "unreadable_pdf"
    elif native_pages == page_count:
        classification = "native_pdf"
    elif native_pages == 0:
        classification = "ocr_pdf"
    else:
        classification = "mixed_pdf"
    return {
        "path": path.relative_to(corpus).as_posix(),
        "bytes": path.stat().st_size,
        "pages": page_count,
        "native_pages": native_pages,
        "empty_pages": empty_pages,
        "classification": classification,
        "native_scan_seconds": time.perf_counter() - started,
        "error": error,
    }


def _pdf_ocr_flags(cache_path: Path) -> dict[str, bool]:
    """Load the accepted-run cache without reading PDFs again."""
    flags: dict[str, bool] = {}
    for row in _iter_jsonl(cache_path):
        signature = row.get("firma")
        if not isinstance(signature, dict):
            continue
        relative = signature.get("ruta_relativa")
        if not isinstance(relative, str) or not relative.casefold().endswith(".pdf"):
            continue
        used_ocr = signature.get("usar_ocr")
        if isinstance(used_ocr, bool):
            flags[relative] = used_ocr
    return flags


def audit_corpus(
    corpus: Path,
    cache_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Inventory all files and classify only a bounded PDF audit sample."""
    started = time.perf_counter()
    files = sorted(path for path in corpus.rglob("*") if path.is_file())
    by_extension: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_extension[path.suffix.casefold() or "[no_extension]"].append(path)

    ocr_flags = _pdf_ocr_flags(cache_path)
    pdf_paths = by_extension.get(".pdf", [])
    native_records = [
        {
            "path": path.relative_to(corpus).as_posix(),
            "bytes": path.stat().st_size,
            "classification": "native_pdf",
            "source": "accepted ingestion cache: usar_ocr=false",
        }
        for path in pdf_paths
        if ocr_flags.get(path.relative_to(corpus).as_posix()) is False
    ]
    ocr_candidates = [
        path
        for path in pdf_paths
        if ocr_flags.get(path.relative_to(corpus).as_posix()) is True
    ]
    ocr_candidates.sort(key=lambda path: (path.stat().st_size, str(path)))
    bounded_candidates = _spread(
        [
            {"path": path.relative_to(corpus).as_posix(), "bytes": path.stat().st_size}
            for path in ocr_candidates
        ],
        8,
    )
    pdfs = [
        _classify_pdf(corpus / str(row["path"]), corpus) for row in bounded_candidates
    ]
    pdf_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pdfs:
        pdf_by_class[str(row["classification"])].append(row)
    for rows in pdf_by_class.values():
        rows.sort(
            key=lambda row: (int(row["pages"]), int(row["bytes"]), str(row["path"]))
        )

    groups: dict[str, list[dict[str, Any]]] = {
        "native_pdf": _spread(native_records, 2),
        "mixed_pdf": _spread(pdf_by_class["mixed_pdf"], 2),
        "ocr_pdf": _spread(pdf_by_class["ocr_pdf"], 2),
    }
    extension_groups = {
        "image": [
            path
            for extension in IMAGE_EXTENSIONS
            for path in by_extension.get(extension, [])
        ],
        "json": by_extension.get(".json", []),
        "csv": by_extension.get(".csv", []),
        "xlsx": by_extension.get(".xlsx", []),
        "pbf": by_extension.get(".pbf", []),
        "text": [
            *by_extension.get(".txt", []),
            *by_extension.get(".md", []),
            *by_extension.get(".html", []),
            *by_extension.get(".htm", []),
        ],
    }
    for name, paths in extension_groups.items():
        records = [
            {"path": path.relative_to(corpus).as_posix(), "bytes": path.stat().st_size}
            for path in paths
        ]
        records.sort(key=lambda row: (int(str(row["bytes"])), str(row["path"])))
        groups[name] = _spread(records, 3 if name != "text" else 1)

    selected = [
        {**record, "group": group}
        for group in SAMPLE_GROUPS
        for record in groups.get(group, [])
    ]
    result: dict[str, Any] = {
        "scope": (
            "read-only filesystem inventory, accepted-run cache analysis, and a "
            "bounded native PDF classification sample; no corpus-wide PDF scan or OCR"
        ),
        "corpus": str(corpus.resolve()),
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "format_inventory": {
            extension: {
                "files": len(paths),
                "bytes": sum(path.stat().st_size for path in paths),
            }
            for extension, paths in sorted(by_extension.items())
        },
        "pdf_scan": {
            "total_pdf_files": len(pdf_paths),
            "bounded_native_classification_files": len(pdfs),
            "accepted_cache_pdf_entries": len(ocr_flags),
            "accepted_cache_second_pass_files": sum(ocr_flags.values()),
            "pages": sum(int(row["pages"]) for row in pdfs),
            "native_pages": sum(int(row["native_pages"]) for row in pdfs),
            "empty_pages": sum(int(row["empty_pages"]) for row in pdfs),
            "native_scan_seconds": sum(
                float(row["native_scan_seconds"]) for row in pdfs
            ),
            "classification_counts_in_bounded_sample": {
                name: len(rows) for name, rows in sorted(pdf_by_class.items())
            },
            "records": pdfs,
            "full_pdf_page_count": (
                "not measured: a full page scan was stopped to avoid a material "
                "re-execution of ingestion"
            ),
        },
        "representative_subset": selected,
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(output_dir / "corpus_audit.json", result)
    return result


def profile_sample_pdf_reads(
    corpus: Path, audit: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    """Measure native/OCR/render spans on selected PDFs using production reader."""
    rows: list[dict[str, Any]] = []
    for entry in audit["representative_subset"]:
        if entry["group"] not in PDF_CLASSES:
            continue
        path = corpus / str(entry["path"])
        durations = {"render_seconds": 0.0, "tesseract_seconds": 0.0}
        original_render = ingestion._renderizar_pagina_pdf
        original_ocr = ingestion._ocr_imagen

        def timed_render(
            page: Any,
            original_render: Any = original_render,
            durations: dict[str, float] = durations,
        ) -> Any:
            started = time.perf_counter()
            try:
                return original_render(page)
            finally:
                durations["render_seconds"] += time.perf_counter() - started

        def timed_ocr(
            image: Any,
            original_ocr: Any = original_ocr,
            durations: dict[str, float] = durations,
        ) -> str:
            started = time.perf_counter()
            try:
                return cast(str, original_ocr(image))
            finally:
                durations["tesseract_seconds"] += time.perf_counter() - started

        ingestion._renderizar_pagina_pdf = timed_render
        ingestion._ocr_imagen = timed_ocr
        diagnostics: dict[str, int] = {}
        started = time.perf_counter()
        error: str | None = None
        try:
            ingestion.leer_pdf(path, usar_ocr=True, diagnostico=diagnostics)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        finally:
            ingestion._renderizar_pagina_pdf = original_render
            ingestion._ocr_imagen = original_ocr
        total = time.perf_counter() - started
        rows.append(
            {
                "path": entry["path"],
                "classification": entry["group"],
                "pages": diagnostics.get("paginas", entry.get("pages")),
                "native_pages": diagnostics.get("paginas_nativas"),
                "ocr_pages": diagnostics.get("paginas_ocr"),
                "empty_pages": diagnostics.get("paginas_sin_texto"),
                "total_seconds": total,
                "render_seconds": durations["render_seconds"],
                "tesseract_seconds": durations["tesseract_seconds"],
                "native_plus_postprocess_seconds": total
                - durations["render_seconds"]
                - durations["tesseract_seconds"],
                "error": error,
            }
        )
    result = {
        "configuration": {"dpi": ingestion.OCR_DPI, "languages": ingestion.OCR_IDIOMAS},
        "pdfs": rows,
    }
    _write_json(output_dir / "selected_pdf_ocr_profile.json", result)
    return result


def benchmark_ingestion(
    corpus: Path, audit: dict[str, Any], worker_count: int, output_dir: Path
) -> dict[str, Any]:
    """Run the existing two-pass ingestion logic on the deterministic subset."""
    selected_paths = {str(entry["path"]) for entry in audit["representative_subset"]}
    started = time.perf_counter()
    first_started = time.perf_counter()
    documents, errors, statistics = ingestion.procesar_corpus(
        corpus,
        max_trabajadores=worker_count,
        max_procesos_pesados=worker_count,
        modo_doc_id=ingestion.MODO_DOC_ID_RAPIDO,
        usar_ocr=False,
        ruta_cache=None,
        rutas_incluidas=selected_paths,
    )
    first_seconds = time.perf_counter() - first_started
    pending = {
        str(error["archivo"])
        for error in errors
        if error.get("estado") == "pendiente_ocr"
    }
    second_seconds = 0.0
    if pending:
        second_started = time.perf_counter()
        documents, errors, statistics = ingestion.resolver_pendientes_ocr(
            corpus,
            documents,
            errors,
            pending,
            statistics,
            worker_count,
            worker_count,
            ingestion.MODO_DOC_ID_RAPIDO,
            None,
        )
        second_seconds = time.perf_counter() - second_started
    formats = Counter(str(document["formato"]) for document in documents)
    result = {
        "worker_count": worker_count,
        "sample_input_files": len(selected_paths),
        "first_pass_seconds": first_seconds,
        "pending_ocr_files": len(pending),
        "second_pass_seconds": second_seconds,
        "total_seconds": time.perf_counter() - started,
        "documents": len(documents),
        "errors": len(errors),
        "statistics": statistics,
        "documents_by_format": dict(sorted(formats.items())),
        "document_content_hash": _stable_hash(documents),
        "error_content_hash": _stable_hash(errors),
        "resource_measurement": {
            "cpu_utilization": (
                "not measured: psutil is unavailable in the fixed environment"
            ),
            "peak_ram": "not measured: psutil is unavailable in the fixed environment",
            "tesseract_process_count": (
                "not sampled: no process-monitor dependency is installed"
            ),
        },
    }
    _write_json(output_dir / f"ingestion_workers_{worker_count}.json", result)
    return result


def _sample_documents(path: Path, sample_per_format: int = 2) -> list[dict[str, Any]]:
    by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in _iter_jsonl(path):
        by_format[str(document["formato"])].append(document)
    selected: list[dict[str, Any]] = []
    for format_name in sorted(by_format):
        bounded = [
            item
            for item in by_format[format_name]
            if len(str(item["texto"])) <= 500_000
        ]
        ordered = sorted(
            bounded or by_format[format_name],
            key=lambda item: (len(str(item["texto"])), str(item["doc_id"])),
        )
        selected.extend(_spread(ordered, sample_per_format))
    selected.sort(key=lambda item: str(item["doc_id"]))
    return selected


def profile_chunking(documents_path: Path, output_dir: Path) -> dict[str, Any]:
    """Profile current serial chunking only; current source has no 2/4 worker path."""
    selected = _sample_documents(documents_path)
    input_path = output_dir / "chunking_sample_documents.jsonl"
    input_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    output_path = output_dir / "chunking_sample_output.jsonl"
    config = load_encoder_config()
    counter = TokenCounter(
        config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        document_prefix=config["document_prefix"],
        add_special_tokens=config["add_special_tokens"],
        local_files_only=True,
    )
    progress: list[PipelineDocumentProgress] = []
    summary = run_pipeline(
        input_path,
        output_path,
        token_counter=counter,
        max_tokens=config["chunk_max_tokens"],
        overlap_tokens=config["overlap_tokens"],
        encoder_max_input_tokens=510,
        total_document_count=len(selected),
        progress_callback=progress.append,
        work_dir=output_dir / "chunking_work",
    )
    by_stage = {
        "language_detection_seconds": [
            row.language_detection_duration_seconds for row in progress
        ],
        "sentence_splitting_seconds": [
            row.splitting_duration_seconds for row in progress
        ],
        "chunking_and_tokenizer_seconds": [
            row.chunking_duration_seconds for row in progress
        ],
        "record_construction_seconds": [
            row.record_construction_duration_seconds for row in progress
        ],
        "jsonl_writing_seconds": [row.writing_duration_seconds for row in progress],
        "document_total_seconds": [row.document_duration_seconds for row in progress],
    }
    per_document = [
        {
            "doc_id": row.doc_id,
            "format": row.document_format,
            "characters": row.character_count,
            "chunks": row.generated_chunk_count,
            "seconds": row.document_duration_seconds,
            "language_detection_seconds": row.language_detection_duration_seconds,
            "sentence_splitting_seconds": row.splitting_duration_seconds,
            "chunking_and_tokenizer_seconds": row.chunking_duration_seconds,
            "record_construction_seconds": row.record_construction_duration_seconds,
            "jsonl_writing_seconds": row.writing_duration_seconds,
        }
        for row in progress
    ]
    result = {
        "sample_documents": len(selected),
        "summary": {
            "seconds": summary.processing_duration_seconds,
            "chunks": summary.chunk_count,
        },
        "stage_summaries": {
            name: _summary(values) for name, values in by_stage.items()
        },
        "slowest_documents": sorted(
            per_document, key=lambda row: float(row["seconds"]), reverse=True
        )[:10],
        "parallelism": {
            "workers_1": "measured serial pipeline",
            "workers_2_and_4": (
                "not runnable: current parallel_pipeline.py is Phase 1 and has "
                "PHASE_1_WORKER_COUNT = 1"
            ),
        },
    }
    _write_json(output_dir / "chunking_profile.json", result)
    return result


def _uniform_positions(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        raise ValueError("total and count must be positive")
    if total <= count:
        return list(range(total))
    if count == 1:
        return [total // 2]
    return list(
        dict.fromkeys(
            round(index * (total - 1) / (count - 1)) for index in range(count)
        )
    )


def _encode_ordered(
    texts: Sequence[str],
    counts: Sequence[int],
    order: Sequence[int],
    encoder: Any,
    batch_size: int,
    config: Any,
) -> tuple[np.ndarray, dict[str, int | float]]:
    import torch

    vectors = np.empty((len(texts), config["embedding_dimension"]), dtype=np.float32)
    padding = 0
    processed = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for offset in range(0, len(order), batch_size):
        indices = list(order[offset : offset + batch_size])
        batch_counts = [counts[index] + 2 for index in indices]
        padding += max(batch_counts) * len(indices) - sum(batch_counts)
        batch_vectors = generate_embeddings(
            [texts[index] for index in indices], encoder, len(indices), config=config
        )
        vectors[indices] = batch_vectors
        processed += len(indices)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return vectors, {
        "seconds": seconds,
        "vectors_per_second": processed / seconds if seconds else 0.0,
        "padding_tokens": padding,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def profile_embeddings(
    chunks_path: Path,
    output_dir: Path,
    sample_size: int,
    expected_chunk_count: int,
) -> dict[str, Any]:
    """Compare current ordering with length bucketing while restoring row identity."""
    selected_positions = set(_uniform_positions(expected_chunk_count, sample_size))
    chunks: list[dict[str, Any]] = []
    full_counts: list[int] = []
    for index, chunk in enumerate(_iter_jsonl(chunks_path)):
        full_counts.append(int(chunk["num_tokens"]))
        if index in selected_positions:
            chunks.append(chunk)
    if len(full_counts) != expected_chunk_count:
        raise ValueError(
            f"chunk count changed: expected {expected_chunk_count}, "
            f"observed {len(full_counts)}"
        )
    if len(chunks) != len(selected_positions):
        raise ValueError("uniform sample did not preserve every selected row")
    texts = [str(chunk["texto"]) for chunk in chunks]
    counts = [int(chunk["num_tokens"]) for chunk in chunks]
    config = load_encoder_config()
    encoder = load_encoder(device="cuda", local_files_only=True)
    # Stabilize CUDA lazy initialization outside timed measurements.
    generate_embeddings(
        texts[: min(16, len(texts))], encoder, min(16, len(texts)), config=config
    )
    sequential = list(range(len(texts)))
    bucketed = sorted(sequential, key=lambda index: (counts[index], index))
    variants = [
        ("sequential_batch_16", sequential, 16),
        ("bucketed_batch_16", bucketed, 16),
        ("bucketed_batch_24", bucketed, 24),
        ("bucketed_batch_32", bucketed, 32),
    ]
    matrices: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    for name, order, batch_size in variants:
        matrix, metrics = _encode_ordered(
            texts, counts, order, encoder, batch_size, config
        )
        matrices[name] = matrix
        results[name] = metrics
    baseline = matrices["sequential_batch_16"]
    for name, matrix in matrices.items():
        difference = np.abs(matrix - baseline)
        norms = np.linalg.norm(matrix, axis=1)
        results[name]["shape"] = list(matrix.shape)
        results[name]["finite"] = bool(np.isfinite(matrix).all())
        results[name]["l2_normalized"] = bool(
            np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5)
        )
        results[name]["max_abs_vs_sequential"] = float(difference.max())
        results[name]["allclose_vs_sequential_atol_1e-5"] = bool(
            np.allclose(matrix, baseline, rtol=1e-5, atol=1e-5)
        )
    query_indices = _uniform_positions(len(texts), min(16, len(texts)))
    query_vectors = generate_embeddings(
        [texts[index] for index in query_indices],
        encoder,
        min(16, len(query_indices)),
        kind="query",
        config=config,
    )
    top_k = min(10, len(texts))
    baseline_scores = query_vectors @ baseline.T
    baseline_top_k = np.argsort(-baseline_scores, axis=1)[:, :top_k]
    retrieval: dict[str, Any] = {}
    for name, matrix in matrices.items():
        scores = query_vectors @ matrix.T
        top_indices = np.argsort(-scores, axis=1)[:, :top_k]
        retrieval[name] = {
            "queries": len(query_indices),
            "top_k": top_k,
            "identical_rankings": bool(np.array_equal(top_indices, baseline_top_k)),
            "differing_queries": int(
                np.count_nonzero(np.any(top_indices != baseline_top_k, axis=1))
            ),
            "maximum_score_difference": float(np.max(np.abs(scores - baseline_scores))),
        }
    result = {
        "full_artifact_chunks": len(full_counts),
        "sample_chunks": len(chunks),
        "token_lengths": {
            name: _percentile(full_counts, percentile)
            for name, percentile in (
                ("min", 0),
                ("p25", 0.25),
                ("median", 0.5),
                ("p75", 0.75),
                ("p90", 0.9),
                ("p95", 0.95),
                ("p99", 0.99),
                ("max", 1),
            )
        },
        "padding_definition": (
            "sum(batch_max(num_tokens + 2) * batch_size) - sum(num_tokens + 2); "
            "+2 are model special tokens"
        ),
        "results": results,
        "retrieval_comparison": retrieval,
        "identity": {
            "row_restoration": (
                "every encoded vector assigned to its original sample index"
            ),
            "ordered_chunk_ids": [str(chunk["chunk_id"]) for chunk in chunks],
        },
    }
    _write_json(output_dir / "embedding_profile.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus/raw"))
    parser.add_argument(
        "--documents",
        type=Path,
        default=Path(
            "outputs/runs/20260813-145837-fresh-9187d9a4/artifacts/documentos.jsonl"
        ),
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path(
            "outputs/runs/20260813-145837-fresh-9187d9a4/artifacts/chunks.jsonl"
        ),
    )
    parser.add_argument(
        "--ingestion-cache",
        type=Path,
        default=Path(
            "outputs/runs/20260813-145837-fresh-9187d9a4/artifacts/"
            "documentos.ingestion_cache.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ingestion-workers", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-pdf-read-profile", action="store_true")
    parser.add_argument("--skip-chunking", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--embedding-sample", type=int, default=2048)
    parser.add_argument("--expected-chunk-count", type=int, default=335393)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = args.corpus.resolve()
    output_dir = args.output_dir.resolve()
    if not corpus.is_dir():
        raise FileNotFoundError(corpus)
    audit: dict[str, Any] | None = None
    if not args.skip_audit:
        audit = audit_corpus(corpus, args.ingestion_cache.resolve(), output_dir)
    elif not args.skip_pdf_read_profile or args.ingestion_workers is not None:
        audit_path = output_dir / "corpus_audit.json"
        loaded = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{audit_path} must contain one JSON object")
        audit = loaded
    if not args.skip_pdf_read_profile:
        if audit is None:
            raise AssertionError("PDF profiling requires the corpus audit")
        profile_sample_pdf_reads(corpus, audit, output_dir)
    if args.ingestion_workers is not None:
        if audit is None:
            raise AssertionError("ingestion profiling requires the corpus audit")
        benchmark_ingestion(corpus, audit, args.ingestion_workers, output_dir)
    if not args.skip_chunking:
        profile_chunking(args.documents.resolve(), output_dir)
    if not args.skip_embeddings:
        profile_embeddings(
            args.chunks.resolve(),
            output_dir,
            args.embedding_sample,
            args.expected_chunk_count,
        )


if __name__ == "__main__":
    main()
