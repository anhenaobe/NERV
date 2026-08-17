"""Focused tests for clean-upstream contamination and report preparation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from validate_clean_upstream import (
    CONTAMINATED_SOURCE,
    match_queries,
    normalize,
    scan_artifact,
    write_failure_report,
)


class CleanUpstreamValidationTests(unittest.TestCase):
    def test_normalized_exact_and_substantial_prefix_do_not_expose_text(self) -> None:
        query = "¿Cómo cambia una pregunta sintética con espacios y puntuación?"
        signatures = [("q001", normalize(query), normalize(query, loose=True))]

        exact, prefix = match_queries(
            "COMO cambia una pregunta sintetica con espacios y puntuacion",
            signatures,
        )

        self.assertEqual(exact, {"q001"})
        self.assertEqual(prefix, set())

    def test_scan_separates_confirmed_and_structural_suspicious_sources(self) -> None:
        query = (
            "consulta sintetica suficientemente extensa para superar el minimo "
            "de coincidencia establecido"
        )
        signatures = [("q001", normalize(query), normalize(query, loose=True))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "documents.jsonl"
            records = [
                {
                    "doc_id": "DOC-1",
                    "fuente": CONTAMINATED_SOURCE,
                    "texto": query,
                },
                {
                    "doc_id": "DOC-2",
                    "fuente": "legitimo.xlsx",
                    "texto": "PREGUNTA: ejemplo FRAGMENTO: evidencia sintetica",
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            scan = scan_artifact(path, signatures, id_field="doc_id")

        self.assertEqual(scan["confirmed"], {CONTAMINATED_SOURCE: ["q001"]})
        self.assertEqual(scan["suspicious"], {"legitimo.xlsx": []})
        self.assertEqual(scan["duplicate_ids"], 0)
        self.assertEqual(scan["empty_text"], 0)

    def test_failure_report_has_all_sections_and_exact_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.txt"
            write_failure_report(report, "run-test", ValueError("synthetic"))
            text = report.read_text(encoding="utf-8")

        for number in range(1, 16):
            self.assertIn(f"{number}. ", text)
        self.assertTrue(text.rstrip().endswith("CLEAN_UPSTREAM_FAIL"))


if __name__ == "__main__":
    unittest.main()
