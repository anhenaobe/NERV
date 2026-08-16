"""Exporta resultados oficiales de recuperacion para calificacion humana."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REQUIRED_COLUMNS = [
    "query_id",
    "query_text",
    "rank",
    "score",
    "chunk_id",
    "doc_id",
    "source",
    "chunk_text",
    "human_grade",
    "human_note",
]
OFFICIAL_QUERY_IDS = tuple(f"q{number:03d}" for number in range(1, 51))
DEFAULT_QUERIES = Path("corpus/queries/queries.jsonl")
DEFAULT_RESULTS = Path(
    "outputs/runs/20260813-174237-resume-fresh-86514215/official_results.jsonl"
)
DEFAULT_CHUNKS = Path(
    "outputs/runs/20260813-145837-fresh-9187d9a4/artifacts/chunks.jsonl"
)
DEFAULT_OUTPUT = Path("outputs/evaluation/retrieval_manual_eval.csv")


class ValidationError(ValueError):
    """Indica que un artefacto de entrada no cumple el contrato de evaluacion."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lee objetos JSONL y falla con contexto si una linea no es un objeto."""
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationError(f"No se pudo leer {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValidationError(f"{path}:{line_number}: no se permiten lineas vacias")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"{path}:{line_number}: JSON invalido: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise ValidationError(f"{path}:{line_number}: se esperaba un objeto JSON")
        records.append(record)
    return records


def require_string(record: Mapping[str, Any], field: str, context: str) -> str:
    """Obtiene una cadena no vacia de un registro de entrada."""
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{context}: {field} debe ser una cadena no vacia")
    return value


def optional_finite_score(fragment: Mapping[str, Any], context: str) -> float | str:
    """Obtiene una puntuacion finita o una celda vacia cuando no fue serializada."""
    if "score" not in fragment:
        return ""
    value = fragment["score"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{context}: score debe ser numerico")
    score = float(value)
    if not math.isfinite(score):
        raise ValidationError(f"{context}: score debe ser finito")
    return score


def index_queries(
    records: Iterable[Mapping[str, Any]], expected_query_ids: tuple[str, ...]
) -> dict[str, str]:
    """Valida el adaptador de consultas y lo indexa por identificador."""
    queries: dict[str, str] = {}
    for position, record in enumerate(records, start=1):
        query_id = require_string(record, "query_id", f"consulta {position}")
        if query_id in queries:
            raise ValidationError(f"consulta {position}: query_id duplicado {query_id}")
        queries[query_id] = require_string(record, "text", f"consulta {query_id}")
    valid_query_ids = (
        len(queries) == len(expected_query_ids)
        and set(queries) == set(expected_query_ids)
    )
    if not valid_query_ids:
        raise ValidationError(
            "los query_id no son exactamente los esperados: "
            f"{', '.join(expected_query_ids)}"
        )
    return queries


def index_chunks(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Indexa chunks aceptados, preservando su metadata como fuente de verdad."""
    chunks: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        chunk_id = require_string(record, "chunk_id", f"chunk {position}")
        if chunk_id in chunks:
            raise ValidationError(f"chunk {position}: chunk_id duplicado {chunk_id}")
        require_string(record, "doc_id", f"chunk {chunk_id}")
        require_string(record, "fuente", f"chunk {chunk_id}")
        require_string(record, "texto", f"chunk {chunk_id}")
        chunks[chunk_id] = record
    return chunks


def export_manual_eval(
    query_records: Iterable[Mapping[str, Any]],
    result_records: Iterable[Mapping[str, Any]],
    chunk_records: Iterable[Mapping[str, Any]],
    output_path: Path,
    *,
    expected_query_ids: tuple[str, ...] = OFFICIAL_QUERY_IDS,
) -> int:
    """Valida y exporta exactamente los cinco mejores fragmentos por consulta."""
    queries = index_queries(query_records, expected_query_ids)
    chunks = index_chunks(chunk_records)
    results: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(result_records, start=1):
        query_id = require_string(record, "query_id", f"resultado {position}")
        if query_id in results:
            raise ValidationError(
                f"resultado {position}: query_id duplicado {query_id}"
            )
        results[query_id] = record
    valid_result_ids = (
        len(results) == len(expected_query_ids)
        and set(results) == set(expected_query_ids)
    )
    if not valid_result_ids:
        raise ValidationError(
            "los resultados no tienen exactamente los query_id esperados: "
            f"{', '.join(expected_query_ids)}"
        )

    rows: list[dict[str, object]] = []
    for query_id in expected_query_ids:
        fragments = results[query_id].get("fragments")
        if not isinstance(fragments, list) or len(fragments) < 5:
            raise ValidationError(
                f"resultado {query_id}: se requieren al menos 5 fragments"
            )
        top_five = fragments[:5]
        ranks: list[int] = []
        for fragment in top_five:
            if not isinstance(fragment, Mapping):
                raise ValidationError(
                    f"resultado {query_id}: fragment debe ser un objeto"
                )
            rank = fragment.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise ValidationError(f"resultado {query_id}: rank debe ser un entero")
            ranks.append(rank)
        if ranks != [1, 2, 3, 4, 5]:
            raise ValidationError(
                f"resultado {query_id}: los ranks top 5 deben ser 1, 2, 3, 4, 5; "
                f"se recibio {ranks}"
            )

        for fragment in top_five:
            assert isinstance(fragment, Mapping)
            rank = fragment["rank"]
            context = f"resultado {query_id}, rank {rank}"
            chunk_id = require_string(fragment, "chunk_id", context)
            doc_id = require_string(fragment, "doc_id", context)
            score = optional_finite_score(fragment, context)
            chunk = chunks.get(chunk_id)
            if chunk is None:
                raise ValidationError(
                    f"{context}: chunk_id no existe en chunks aceptados: {chunk_id}"
                )
            chunk_doc_id = require_string(chunk, "doc_id", f"chunk {chunk_id}")
            if doc_id != chunk_doc_id:
                raise ValidationError(
                    f"{context}: doc_id {doc_id} no coincide con chunk {chunk_id} "
                    f"({chunk_doc_id})"
                )
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": queries[query_id],
                    "rank": rank,
                    "score": score,
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source": require_string(chunk, "fuente", f"chunk {chunk_id}"),
                    "chunk_text": require_string(chunk, "texto", f"chunk {chunk_id}"),
                    "human_grade": "",
                    "human_note": "",
                }
            )

    expected_rows = len(expected_query_ids) * 5
    if len(rows) != expected_rows:
        raise ValidationError(
            f"se esperaban {expected_rows} filas, se obtuvieron {len(rows)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    """Construye la interfaz de linea de comandos del exportador."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Ejecuta la exportacion sin modificar los artefactos de entrada."""
    args = parse_args()
    try:
        exported = export_manual_eval(
            read_jsonl(args.queries),
            read_jsonl(args.results),
            read_jsonl(args.chunks),
            args.output,
        )
    except ValidationError as exc:
        raise SystemExit(f"ERROR de validacion: {exc}") from exc
    print(f"CSV de evaluacion preparado: {args.output} ({exported} filas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
