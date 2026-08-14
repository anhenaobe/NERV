"""Contract tests for the frozen representative real-corpus manifest."""

import json
from pathlib import Path
from typing import Any

MANIFEST_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "real_corpus_benchmark_manifest_v2.json"
)
EXPECTED_DOC_IDS = [
    "DOC-953b7532f1b5",
    "DOC-6891a760adbe",
    "DOC-83e80067468b",
    "DOC-a50c6b9e443d",
    "DOC-32d649e20ea2",
    "DOC-f180383827bf",
]


def _manifest() -> dict[str, Any]:
    loaded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_representative_manifest_has_frozen_unique_selection() -> None:
    manifest = _manifest()
    documents = manifest["documents"]
    doc_ids = manifest["doc_ids"]

    assert manifest["schema_version"] == 2
    assert doc_ids == EXPECTED_DOC_IDS
    assert len(doc_ids) == len(set(doc_ids)) == 6
    assert [document["doc_id"] for document in documents] == doc_ids


def test_representative_manifest_covers_required_variation() -> None:
    manifest = _manifest()
    documents = manifest["documents"]
    format_sizes = {
        (document["format"], document["size_class"])
        for document in documents
    }
    languages = {document["language"] for document in documents}

    assert {("pdf", "small"), ("pdf", "medium"), ("pdf", "large")} <= (
        format_sizes
    )
    assert {("csv", "small"), ("csv", "large")} <= format_sizes
    assert {"es", "en", "pt"} <= languages
    assert any(language.startswith("ambiguous_") for language in languages)
    assert any(
        document.get("oversized_unit", {}).get("confirmed") is True
        for document in documents
    )


def test_representative_manifest_excludes_known_pathological_extremes() -> None:
    manifest = _manifest()
    doc_ids = set(manifest["doc_ids"])
    excluded = set(manifest["excluded_documents"])

    assert {"DOC-2d7c54534744", "DOC-9b7f304a0986"} <= excluded
    assert doc_ids.isdisjoint(excluded)
