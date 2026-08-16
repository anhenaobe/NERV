"""Calcula metricas internas desde un CSV calificado por una persona."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from prepare_retrieval_manual_eval import (
    OFFICIAL_QUERY_IDS,
    REQUIRED_COLUMNS,
    ValidationError,
)

DEFAULT_INPUT = Path("outputs/evaluation/retrieval_manual_eval.csv")
DEFAULT_REPORT = Path("outputs/evaluation/retrieval_manual_report.txt")
VALID_GRADES = frozenset({"0", "1", "2", "3"})


@dataclass(frozen=True)
class Judgment:
    """Una calificacion humana de un fragmento recuperado."""

    query_id: str
    rank: int
    grade: int


@dataclass(frozen=True)
class ManualMetrics:
    """Metricas agregadas y clasificacion por consulta."""

    total_queries: int
    total_judgments: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    precision_at_5: float
    mrr: float
    mean_grade: float
    grade_distribution: Counter[int]
    classifications: dict[str, str]


def read_judgments(
    csv_path: Path, *, expected_query_ids: tuple[str, ...] = OFFICIAL_QUERY_IDS
) -> list[Judgment]:
    """Lee un CSV completo y rechaza grados faltantes, invalidos o filas alteradas."""
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as input_file:
            reader = csv.DictReader(input_file)
            if reader.fieldnames != REQUIRED_COLUMNS:
                raise ValidationError(
                    f"{csv_path}: columnas invalidas; se esperaban {REQUIRED_COLUMNS}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ValidationError(f"No se pudo leer {csv_path}: {exc}") from exc

    expected_rows = len(expected_query_ids) * 5
    if len(rows) != expected_rows:
        raise ValidationError(
            f"{csv_path}: se esperaban {expected_rows} filas, hay {len(rows)}"
        )

    judgments: list[Judgment] = []
    grouped: dict[str, list[Judgment]] = {}
    for row_number, row in enumerate(rows, start=2):
        query_id = row["query_id"]
        grade_text = row["human_grade"]
        if grade_text not in VALID_GRADES:
            display_value = "vacio" if grade_text == "" else repr(grade_text)
            raise ValidationError(
                f"{csv_path}:{row_number}: human_grade invalido ({display_value}); "
                "se acepta solo 0, 1, 2 o 3"
            )
        try:
            rank = int(row["rank"])
        except ValueError as exc:
            raise ValidationError(
                f"{csv_path}:{row_number}: rank invalido {row['rank']!r}"
            ) from exc
        judgment = Judgment(query_id=query_id, rank=rank, grade=int(grade_text))
        judgments.append(judgment)
        grouped.setdefault(query_id, []).append(judgment)

    if tuple(grouped) != expected_query_ids:
        raise ValidationError(
            f"{csv_path}: query_id invalidos; se esperaban "
            f"{', '.join(expected_query_ids)}"
        )
    for query_id, query_judgments in grouped.items():
        ranks = sorted(judgment.rank for judgment in query_judgments)
        if ranks != [1, 2, 3, 4, 5]:
            raise ValidationError(
                f"{csv_path}: {query_id} debe tener ranks 1 a 5; tiene {ranks}"
            )
    return judgments


def calculate_metrics(judgments: list[Judgment]) -> ManualMetrics:
    """Calcula metricas con relevante definido como grado mayor o igual a dos."""
    by_query: dict[str, list[Judgment]] = {}
    for judgment in judgments:
        by_query.setdefault(judgment.query_id, []).append(judgment)
    if not by_query:
        raise ValidationError("no hay juicios humanos para calcular metricas")

    hit_counts = {1: 0, 3: 0, 5: 0}
    relevant_at_five = 0
    reciprocal_rank_sum = 0.0
    classifications: dict[str, str] = {}
    distribution: Counter[int] = Counter()
    grade_sum = 0
    for query_id, query_judgments in by_query.items():
        ordered = sorted(query_judgments, key=lambda judgment: judgment.rank)
        grades = [judgment.grade for judgment in ordered]
        distribution.update(grades)
        grade_sum += sum(grades)
        relevant_ranks = [judgment.rank for judgment in ordered if judgment.grade >= 2]
        for cutoff in hit_counts:
            if any(rank <= cutoff for rank in relevant_ranks):
                hit_counts[cutoff] += 1
        relevant_at_five += sum(grade >= 2 for grade in grades)
        if relevant_ranks:
            reciprocal_rank_sum += 1 / min(relevant_ranks)
        if any(judgment.rank <= 3 and judgment.grade == 3 for judgment in ordered):
            classifications[query_id] = "GOOD"
        elif relevant_ranks:
            classifications[query_id] = "ACCEPTABLE"
        else:
            classifications[query_id] = "BAD"

    total_queries = len(by_query)
    total_judgments = len(judgments)
    return ManualMetrics(
        total_queries=total_queries,
        total_judgments=total_judgments,
        hit_at_1=hit_counts[1] / total_queries,
        hit_at_3=hit_counts[3] / total_queries,
        hit_at_5=hit_counts[5] / total_queries,
        precision_at_5=relevant_at_five / total_judgments,
        mrr=reciprocal_rank_sum / total_queries,
        mean_grade=grade_sum / total_judgments,
        grade_distribution=distribution,
        classifications=classifications,
    )


def render_report(metrics: ManualMetrics, judgments: list[Judgment]) -> str:
    """Genera un informe legible que declara su naturaleza de juicio interno."""
    by_query: dict[str, list[Judgment]] = {}
    for judgment in judgments:
        by_query.setdefault(judgment.query_id, []).append(judgment)
    classification_counts = Counter(metrics.classifications.values())
    bad_ids = [
        query_id
        for query_id, classification in metrics.classifications.items()
        if classification == "BAD"
    ]
    lines = [
        "REPORTE DE EVALUACION HUMANA INTERNA DE RECUPERACION",
        "No representa una precision oficial de competencia.",
        "",
        f"TOTAL QUERIES: {metrics.total_queries}",
        f"TOTAL HUMAN JUDGMENTS: {metrics.total_judgments}",
        f"HIT@1: {metrics.hit_at_1:.4f}",
        f"HIT@3: {metrics.hit_at_3:.4f}",
        f"HIT@5: {metrics.hit_at_5:.4f}",
        f"PRECISION@5: {metrics.precision_at_5:.4f}",
        f"MRR: {metrics.mrr:.4f}",
        f"MEAN RELEVANCE GRADE: {metrics.mean_grade:.4f}",
        "GRADE DISTRIBUTION: "
        + ", ".join(
            f"{grade}={metrics.grade_distribution[grade]}" for grade in range(4)
        ),
        f"GOOD QUERY COUNT: {classification_counts['GOOD']}",
        f"ACCEPTABLE QUERY COUNT: {classification_counts['ACCEPTABLE']}",
        f"BAD QUERY COUNT: {classification_counts['BAD']}",
        f"BAD QUERY IDS: {', '.join(bad_ids) if bad_ids else '(none)'}",
        "",
        "PER-QUERY SUMMARY",
    ]
    for query_id, query_judgments in by_query.items():
        grades = ", ".join(
            str(judgment.grade)
            for judgment in sorted(query_judgments, key=lambda judgment: judgment.rank)
        )
        lines.append(
            f"{query_id}: {metrics.classifications[query_id]} | grades: {grades}"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    """Construye la interfaz de linea de comandos del calculador."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Calcula y publica el informe solo cuando todas las celdas tienen grado."""
    args = parse_args()
    try:
        judgments = read_judgments(args.input)
        metrics = calculate_metrics(judgments)
    except ValidationError as exc:
        raise SystemExit(f"ERROR de validacion: {exc}") from exc
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(metrics, judgments), encoding="utf-8")
    print(f"Reporte de evaluacion humana escrito: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
