"""Pruebas unitarias del flujo de evaluacion humana de recuperacion."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from evaluate_retrieval_manual import Judgment, calculate_metrics, read_judgments
from prepare_retrieval_manual_eval import (
    OFFICIAL_QUERY_IDS,
    REQUIRED_COLUMNS,
    ValidationError,
    export_manual_eval,
)

QUERY_IDS = ("q001", "q002", "q003")


def sample_inputs(
    query_ids: tuple[str, ...] = QUERY_IDS,
) -> tuple[
    list[dict[str, str]], list[dict[str, object]], list[dict[str, str]]
]:
    """Crea consultas y cinco resultados validos para cada una."""
    queries = [
        {"query_id": query_id, "text": f"Pregunta {query_id}"} for query_id in query_ids
    ]
    chunks: list[dict[str, str]] = []
    results: list[dict[str, object]] = []
    for query_id in query_ids:
        fragments = []
        for rank in range(1, 6):
            chunk_id = f"DOC-{query_id}-chunk-{rank:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": f"DOC-{query_id}",
                    "fuente": f"source/{query_id}.txt",
                    "texto": f'Texto {query_id}, linea {rank}\ncon comillas: "si".',
                }
            )
            fragments.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "doc_id": f"DOC-{query_id}",
                    "score": 1.0 / rank,
                }
            )
        results.append({"query_id": query_id, "fragments": fragments})
    return queries, results, chunks


def test_export_writes_exactly_top_five_and_preserves_csv_text(tmp_path: Path) -> None:
    queries, results, chunks = sample_inputs()
    results[0]["fragments"].append(
        {"rank": 6, "chunk_id": "not-exported", "doc_id": "DOC-q001", "score": 0.1}
    )
    output_path = tmp_path / "manual.csv"

    exported = export_manual_eval(
        queries, results, chunks, output_path, expected_query_ids=QUERY_IDS
    )

    assert exported == 15
    with output_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert len(rows) == 15
    assert [row["rank"] for row in rows[:5]] == ["1", "2", "3", "4", "5"]
    assert all(row["human_grade"] == "" and row["human_note"] == "" for row in rows)
    assert rows[0]["chunk_text"] == 'Texto q001, linea 1\ncon comillas: "si".'


def test_export_accepts_missing_score_as_blank(tmp_path: Path) -> None:
    queries, results, chunks = sample_inputs()
    fragments = results[0]["fragments"]
    assert isinstance(fragments, list)
    del fragments[0]["score"]
    output_path = tmp_path / "manual.csv"

    export_manual_eval(
        queries, results, chunks, output_path, expected_query_ids=QUERY_IDS
    )

    with output_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert rows[0]["score"] == ""
    assert rows[1]["score"] == "0.5"


@pytest.mark.parametrize("invalid_score", ["invalid", float("nan"), float("inf")])
def test_export_rejects_invalid_present_score(
    tmp_path: Path, invalid_score: object
) -> None:
    queries, results, chunks = sample_inputs()
    fragments = results[0]["fragments"]
    assert isinstance(fragments, list)
    fragments[0]["score"] = invalid_score

    with pytest.raises(ValidationError, match="score debe"):
        export_manual_eval(
            queries,
            results,
            chunks,
            tmp_path / "manual.csv",
            expected_query_ids=QUERY_IDS,
        )


def test_export_official_shape_writes_250_blank_judgments(tmp_path: Path) -> None:
    queries, results, chunks = sample_inputs(OFFICIAL_QUERY_IDS)
    for result in results:
        fragments = result["fragments"]
        assert isinstance(fragments, list)
        for fragment in fragments:
            del fragment["score"]
    output_path = tmp_path / "manual.csv"

    exported = export_manual_eval(queries, results, chunks, output_path)

    with output_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    assert exported == 250
    assert len(rows) == 250
    assert {row["query_id"] for row in rows} == set(OFFICIAL_QUERY_IDS)
    assert all(row["score"] == "" for row in rows)
    assert all(row["human_grade"] == "" for row in rows)
    assert all(row["human_note"] == "" for row in rows)


def write_grades(path: Path, grades: list[str]) -> None:
    """Escribe un CSV completo y pequeno para validar el lector humano."""
    rows = []
    for index, grade in enumerate(grades):
        query_id = QUERY_IDS[index // 5]
        rows.append(
            {
                "query_id": query_id,
                "query_text": query_id,
                "rank": str(index % 5 + 1),
                "score": "0.5",
                "chunk_id": f"chunk-{index}",
                "doc_id": f"doc-{index}",
                "source": "source.txt",
                "chunk_text": "text",
                "human_grade": grade,
                "human_note": "",
            }
        )
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("invalid_grade", ["", "4", "x"])
def test_read_judgments_rejects_missing_or_invalid_grade(
    tmp_path: Path, invalid_grade: str
) -> None:
    path = tmp_path / "graded.csv"
    write_grades(path, [invalid_grade] + ["0"] * (len(QUERY_IDS) * 5 - 1))

    with pytest.raises(ValidationError, match="human_grade invalido"):
        read_judgments(path, expected_query_ids=QUERY_IDS)


def test_metrics_and_classification() -> None:
    judgments = [
        Judgment("q001", 1, 3),
        Judgment("q001", 2, 0),
        Judgment("q001", 3, 1),
        Judgment("q001", 4, 0),
        Judgment("q001", 5, 0),
        Judgment("q002", 1, 0),
        Judgment("q002", 2, 1),
        Judgment("q002", 3, 0),
        Judgment("q002", 4, 2),
        Judgment("q002", 5, 0),
        Judgment("q003", 1, 0),
        Judgment("q003", 2, 1),
        Judgment("q003", 3, 0),
        Judgment("q003", 4, 1),
        Judgment("q003", 5, 0),
    ]

    metrics = calculate_metrics(judgments)

    assert metrics.hit_at_1 == pytest.approx(1 / 3)
    assert metrics.hit_at_3 == pytest.approx(1 / 3)
    assert metrics.hit_at_5 == pytest.approx(2 / 3)
    assert metrics.precision_at_5 == pytest.approx(2 / 15)
    assert metrics.mrr == pytest.approx(5 / 12)
    assert metrics.mean_grade == pytest.approx(9 / 15)
    assert metrics.grade_distribution == {0: 9, 1: 4, 2: 1, 3: 1}
    assert metrics.classifications == {
        "q001": "GOOD",
        "q002": "ACCEPTABLE",
        "q003": "BAD",
    }
