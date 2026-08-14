"""Strict loader for query-only JSONL artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TypedDict


class QueryRecord(TypedDict):
    """Validated query input kept separate from document ingestion."""

    query_id: str
    text: str


_QUERY_ID = re.compile(r"q[0-9]{3}\Z")


def load_queries(
    path: Path,
    *,
    expected_ids: list[str] | None = None,
) -> list[QueryRecord]:
    """Load unique qNNN records and optionally require one exact order."""
    queries: list[QueryRecord] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid query JSON at line {line_number}."
                ) from error
            if not isinstance(raw, dict) or set(raw) != {"query_id", "text"}:
                raise ValueError(
                    f"query line {line_number} must contain only query_id and text."
                )
            query_id = raw["query_id"]
            text = raw["text"]
            if not isinstance(query_id, str) or _QUERY_ID.fullmatch(query_id) is None:
                raise ValueError(f"query line {line_number} has an invalid query_id.")
            if query_id in seen:
                raise ValueError(f"duplicate query_id: {query_id!r}.")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"query {query_id!r} has empty text.")
            seen.add(query_id)
            queries.append({"query_id": query_id, "text": text.strip()})
    if (
        expected_ids is not None
        and [item["query_id"] for item in queries] != expected_ids
    ):
        raise ValueError("query IDs or their order do not match the expected contract.")
    return queries
