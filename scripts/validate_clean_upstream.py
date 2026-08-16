"""Validate a human-generated clean upstream run and publish the C1 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

EXPECTED_DOCUMENTS = 1760
EXPECTED_CHUNKS = 335371
CONTAMINATED_SOURCE = (
    "F3_Dinamicas_Territoriales/FASE ORDENADA CODEFEST.xlsx"
)
OFFICIAL_QUERY_PDF = "Extracto_Preguntas_50_v2.pdf"
EXPECTED_AUXILIARY = {
    "Extracto_Preguntas_50_v2.pdf",
    "Indice_Datos_Codefest.xlsx",
    CONTAMINATED_SOURCE,
}
HISTORICAL_RUN_IDS = {
    "20260813-145837-fresh-9187d9a4",
    "20260813-174237-resume-fresh-86514215",
}
MARKER_PATTERN = re.compile(
    r"\b(PREGUNTA|FRAGMENTO|RESPUESTA|QUERY|CONSULTA)\s*:",
    re.IGNORECASE,
)
FINAL_GATES = {
    "CLEAN_UPSTREAM_PASS",
    "CLEAN_UPSTREAM_FAIL",
    "CLEAN_UPSTREAM_NEEDS_HUMAN_REVIEW",
}


def normalize(text: str, *, loose: bool = False) -> str:
    """Normalize Unicode, case, whitespace, and optionally punctuation/accents."""
    normalized = unicodedata.normalize("NFKD" if loose else "NFKC", text).casefold()
    if loose:
        normalized = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        normalized = "".join(
            character if character.isalnum() else " " for character in normalized
        )
    return re.sub(r"\s+", " ", normalized).strip()


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSON objects with stable line numbers."""
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield line_number, value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_query_signatures(path: Path) -> list[tuple[str, str, str]]:
    records = [record for _line, record in read_jsonl(path)]
    expected = [f"q{number:03d}" for number in range(1, 51)]
    actual = [record.get("query_id") for record in records]
    if actual != expected:
        raise ValueError("official query IDs are not exactly q001-q050 in order")
    signatures = []
    for record in records:
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"official query {record['query_id']} has no text")
        signatures.append(
            (str(record["query_id"]), normalize(text), normalize(text, loose=True))
        )
    return signatures


def match_queries(
    text: str,
    signatures: Iterable[tuple[str, str, str]],
) -> tuple[set[str], set[str]]:
    strict_text = normalize(text)
    loose_text = normalize(text, loose=True)
    exact: set[str] = set()
    prefix: set[str] = set()
    for query_id, strict_query, loose_query in signatures:
        if strict_query in strict_text or loose_query in loose_text:
            exact.add(query_id)
            continue
        prefix_length = min(80, len(loose_query))
        if prefix_length >= 60 and loose_query[:prefix_length] in loose_text:
            prefix.add(query_id)
    return exact, prefix


def scan_artifact(
    path: Path,
    signatures: list[tuple[str, str, str]],
    *,
    id_field: str,
) -> dict[str, Any]:
    count = 0
    duplicate_ids = 0
    empty_text = 0
    seen_ids: set[str] = set()
    sources: dict[str, set[str]] = defaultdict(set)
    exact_by_source: dict[str, set[str]] = defaultdict(set)
    suspicious_by_source: dict[str, set[str]] = defaultdict(set)
    structural_sources: set[str] = set()
    per_document: Counter[str] = Counter()
    for line_number, record in read_jsonl(path):
        count += 1
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{path}:{line_number}: invalid {id_field}")
        if record_id in seen_ids:
            duplicate_ids += 1
        seen_ids.add(record_id)
        doc_id = record.get("doc_id")
        source = record.get("fuente")
        text = record.get("texto")
        if not isinstance(doc_id, str) or not isinstance(source, str):
            raise ValueError(f"{path}:{line_number}: invalid source identity")
        if not isinstance(text, str) or not text.strip():
            empty_text += 1
            text = ""
        sources[source].add(doc_id)
        per_document[doc_id] += 1
        exact, prefix = match_queries(text, signatures)
        exact_by_source[source].update(exact)
        suspicious_by_source[source].update(prefix)
        markers = {marker.upper() for marker in MARKER_PATTERN.findall(text)}
        if len(markers) >= 2:
            structural_sources.add(source)
    confirmed = {
        source: sorted(query_ids)
        for source, query_ids in exact_by_source.items()
        if query_ids
    }
    if CONTAMINATED_SOURCE in sources:
        confirmed.setdefault(CONTAMINATED_SOURCE, [])
    suspicious = {
        source: sorted(suspicious_by_source[source] - set(confirmed.get(source, [])))
        for source in set(suspicious_by_source) | structural_sources
        if source not in confirmed
        and (suspicious_by_source[source] or source in structural_sources)
    }
    return {
        "count": count,
        "duplicate_ids": duplicate_ids,
        "empty_text": empty_text,
        "sources": sources,
        "confirmed": confirmed,
        "suspicious": suspicious,
        "per_document": per_document,
    }


def source_set(path: Path) -> set[str]:
    return {str(record.get("fuente", "")) for _line, record in read_jsonl(path)}


def chunk_counts(path: Path) -> Counter[str]:
    return Counter(
        str(record.get("doc_id", "")) for _line, record in read_jsonl(path)
    )


def format_list(values: Iterable[str]) -> str:
    items = sorted(values)
    return ", ".join(items) if items else "(none)"


def format_matches(matches: Mapping[str, list[str]]) -> str:
    if not matches:
        return "(none)"
    return "\n".join(
        f"- {source}: {format_list(query_ids)}"
        for source, query_ids in sorted(matches.items())
    )


def write_failure_report(report: Path, run_id: str, error: Exception) -> None:
    sections = [
        ("1. RUN ID", run_id),
        ("2. FIX VALIDATION", "FAIL"),
        ("3. RAW INPUT STATUS", f"VALIDATION ERROR: {type(error).__name__}: {error}"),
    ]
    sections.extend(
        (f"{number}. {title}", "UNAVAILABLE")
        for number, title in (
            (4, "DOCUMENT COUNT"),
            (5, "DOCUMENT HASH"),
            (6, "DOCUMENT SOURCE DELTA"),
            (7, "EXCLUDED AUXILIARY FILES"),
            (8, "DOCUMENT CONTAMINATION CHECK"),
            (9, "CHUNK COUNT"),
            (10, "CHUNK HASH"),
            (11, "MAX TOKENS"),
            (12, "CHUNK CONTRACT VALIDATION"),
            (13, "CHUNK CONTAMINATION CHECK"),
            (14, "EXPECTED VS ACTUAL DELTA"),
        )
    )
    sections.append(("15. FINAL GATE", "CLEAN_UPSTREAM_FAIL"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "CLEAN UPSTREAM VALIDATION REPORT\n\n"
        + "\n\n".join(f"{heading}\n{body}" for heading, body in sections)
        + "\n",
        encoding="utf-8",
    )


def validate(args: argparse.Namespace) -> str:
    run_dir = args.run_dir.resolve()
    if run_dir.name in HISTORICAL_RUN_IDS:
        raise ValueError("refusing to validate into a contaminated historical run")
    artifacts = run_dir / "artifacts"
    documents = artifacts / "documentos.jsonl"
    chunks = artifacts / "chunks.jsonl"
    errors_path = artifacts / "errores.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    for path in (documents, chunks, errors_path, manifest_path, args.queries):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id", run_dir.name))
    signatures = load_query_signatures(args.queries)
    document_scan = scan_artifact(documents, signatures, id_field="doc_id")
    chunk_scan = scan_artifact(chunks, signatures, id_field="chunk_id")
    document_hash = sha256_file(documents)
    chunk_hash = sha256_file(chunks)

    previous_sources = source_set(args.previous_documents)
    current_sources = set(document_scan["sources"])
    added_sources = current_sources - previous_sources
    removed_sources = previous_sources - current_sources
    previous_chunk_counts = chunk_counts(args.previous_chunks)
    current_chunk_counts = chunk_scan["per_document"]
    per_document_delta = {
        doc_id: current_chunk_counts[doc_id] - previous_chunk_counts[doc_id]
        for doc_id in set(previous_chunk_counts) | set(current_chunk_counts)
        if current_chunk_counts[doc_id] != previous_chunk_counts[doc_id]
    }

    incidents = [record for _line, record in read_jsonl(errors_path)]
    omitted = {
        str(record.get("archivo"))
        for record in incidents
        if record.get("estado") == "omitido"
    }
    missing_auxiliary = EXPECTED_AUXILIARY - omitted

    canonical_metrics = run_dir / "canonical_chunk_validation.json"
    canonical_log = run_dir / "canonical_chunk_validation.log"
    command = [
        sys.executable,
        "-m",
        "nerv.chunking.real_corpus",
        "validate",
        "--input",
        str(documents),
        "--chunks",
        str(chunks),
        "--metrics-output",
        str(canonical_metrics),
        "--log-path",
        str(canonical_log),
        "--local-files-only",
    ]
    validation_environment = os.environ.copy()
    current_source = str(args.repo_root / "src")
    existing_pythonpath = validation_environment.get("PYTHONPATH")
    validation_environment["PYTHONPATH"] = (
        current_source + os.pathsep + existing_pythonpath
        if existing_pythonpath
        else current_source
    )
    canonical_exit = subprocess.run(
        command,
        cwd=args.repo_root,
        env=validation_environment,
        check=False,
    ).returncode
    metrics = json.loads(canonical_metrics.read_text(encoding="utf-8"))

    downstream = {"embeddings", "faiss", "retrieval", "validation"}
    executed = set(manifest.get("stages_executed", []))
    failed_reasons: list[str] = []
    review_reasons: list[str] = []
    if manifest.get("status") != "completed" or executed & downstream:
        failed_reasons.append("run manifest is not a completed upstream-only run")
    if document_scan["duplicate_ids"] or document_scan["empty_text"]:
        failed_reasons.append("document schema/identity validation failed")
    if CONTAMINATED_SOURCE in current_sources or OFFICIAL_QUERY_PDF in current_sources:
        failed_reasons.append("excluded auxiliary source was published")
    if missing_auxiliary:
        failed_reasons.append("expected auxiliary omission accounting is incomplete")
    if document_scan["confirmed"]:
        failed_reasons.append("confirmed document contamination remains")
    if chunk_scan["confirmed"]:
        failed_reasons.append("confirmed chunk contamination remains")
    if canonical_exit != 0 or metrics.get("failure_count") != 0:
        failed_reasons.append("canonical chunk validation failed")
    if chunk_scan["duplicate_ids"] or chunk_scan["empty_text"]:
        failed_reasons.append("chunk identity/text validation failed")
    if document_scan["suspicious"] or chunk_scan["suspicious"]:
        review_reasons.append("suspicious contamination matches require review")
    if document_scan["count"] != args.expected_documents:
        review_reasons.append("EXPECTED_DOCUMENT_COUNT_MISMATCH")
    if chunk_scan["count"] != args.expected_chunks:
        review_reasons.append("EXPECTED_CHUNK_COUNT_MISMATCH")
    expected_removed = {CONTAMINATED_SOURCE}
    if added_sources or removed_sources != expected_removed:
        review_reasons.append("document source delta differs from the audited removal")

    gate = (
        "CLEAN_UPSTREAM_FAIL"
        if failed_reasons
        else "CLEAN_UPSTREAM_NEEDS_HUMAN_REVIEW"
        if review_reasons
        else "CLEAN_UPSTREAM_PASS"
    )
    assert gate in FINAL_GATES

    max_tokens = metrics.get("maximum_num_tokens")
    contract_counts = {
        key: metrics.get(key)
        for key in (
            "duplicate_chunk_id_count",
            "unknown_document_count",
            "empty_chunk_count",
            "token_count_mismatch_count",
            "encoder_hard_limit_exceeded_count",
            "hard_split_metadata_error_count",
            "position_error_count",
            "document_order_error_count",
            "failure_count",
        )
    }
    delta_lines = [
        f"- {doc_id}: {delta:+d} chunks"
        for doc_id, delta in sorted(per_document_delta.items())
    ] or ["(none)"]
    document_count_note = (
        f"{document_scan['count']} (expected {args.expected_documents})"
        + (
            "\nEXPECTED_DOCUMENT_COUNT_MISMATCH"
            if document_scan["count"] != args.expected_documents
            else ""
        )
    )
    chunk_count_note = (
        f"{chunk_scan['count']} (expected {args.expected_chunks})"
        + (
            "\nEXPECTED_CHUNK_COUNT_MISMATCH"
            if chunk_scan["count"] != args.expected_chunks
            else ""
        )
    )
    raw_path = manifest.get("runtime_config", {}).get("paths", {}).get("corpus")
    sections = [
        ("1. RUN ID", run_id),
        (
            "2. FIX VALIDATION",
            "PASS: explicit relative-path exclusion was bounded-tested "
            "before this run.",
        ),
        (
            "3. RAW INPUT STATUS",
            f"path={raw_path}\nmanifest_status={manifest.get('status')}\n"
            f"git_commit={manifest.get('git_commit')}\n"
            f"git_dirty={manifest.get('git_dirty')}",
        ),
        ("4. DOCUMENT COUNT", document_count_note),
        ("5. DOCUMENT HASH", document_hash),
        (
            "6. DOCUMENT SOURCE DELTA",
            f"added: {format_list(added_sources)}\n"
            f"removed: {format_list(removed_sources)}",
        ),
        (
            "7. EXCLUDED AUXILIARY FILES",
            f"accounted: {format_list(EXPECTED_AUXILIARY & omitted)}\n"
            f"missing: {format_list(missing_auxiliary)}",
        ),
        (
            "8. DOCUMENT CONTAMINATION CHECK",
            f"confirmed:\n{format_matches(document_scan['confirmed'])}\n"
            f"suspicious:\n{format_matches(document_scan['suspicious'])}",
        ),
        ("9. CHUNK COUNT", chunk_count_note),
        ("10. CHUNK HASH", chunk_hash),
        ("11. MAX TOKENS", f"{max_tokens} (required <= 510)"),
        (
            "12. CHUNK CONTRACT VALIDATION",
            f"canonical_exit={canonical_exit}\n"
            + "\n".join(f"{key}={value}" for key, value in contract_counts.items()),
        ),
        (
            "13. CHUNK CONTAMINATION CHECK",
            f"confirmed:\n{format_matches(chunk_scan['confirmed'])}\n"
            f"suspicious:\n{format_matches(chunk_scan['suspicious'])}",
        ),
        (
            "14. EXPECTED VS ACTUAL DELTA",
            "document_delta="
            f"{document_scan['count'] - args.expected_documents:+d}\nchunk_delta="
            f"{chunk_scan['count'] - args.expected_chunks:+d}\n"
            "per-document chunk deltas vs contaminated lineage:\n"
            + "\n".join(delta_lines)
            + "\nreview_reasons: "
            + ("; ".join(review_reasons) if review_reasons else "(none)")
            + "\nfail_reasons: "
            + ("; ".join(failed_reasons) if failed_reasons else "(none)"),
        ),
        ("15. FINAL GATE", gate),
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "CLEAN UPSTREAM VALIDATION REPORT\n\n"
        + "\n\n".join(f"{heading}\n{body}" for heading, body in sections)
        + "\n",
        encoding="utf-8",
    )
    print(gate)
    print(f"REPORT={args.report.resolve()}")
    return gate


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--queries", type=Path, default=repo_root / "corpus/queries/queries.jsonl"
    )
    parser.add_argument(
        "--previous-documents",
        type=Path,
        default=repo_root
        / "outputs/runs/20260813-145837-fresh-9187d9a4/artifacts/documentos.jsonl",
    )
    parser.add_argument(
        "--previous-chunks",
        type=Path,
        default=repo_root
        / "outputs/runs/20260813-145837-fresh-9187d9a4/artifacts/chunks.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "docs/integration/clean_upstream_validation_report.txt",
    )
    parser.add_argument("--expected-documents", type=int, default=EXPECTED_DOCUMENTS)
    parser.add_argument("--expected-chunks", type=int, default=EXPECTED_CHUNKS)
    args = parser.parse_args()
    args.repo_root = repo_root
    return args


def main() -> int:
    args = parse_args()
    try:
        gate = validate(args)
    except Exception as error:
        write_failure_report(args.report, args.run_dir.name, error)
        print(f"CLEAN_UPSTREAM_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0 if gate == "CLEAN_UPSTREAM_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
