"""Generate the deterministic CODEFEST Stage-1 delivery results.

The program deliberately loads the submitted FAISS index and metadata directly.
It never rebuilds embeddings, changes retrieval order, or calls a generative
provider. Fragment presentation uses only source text already present in the
delivered metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from nerv.chunking.configuration import load_encoder_config
from nerv.embeddings.encoder import load_encoder
from nerv.evaluation.validator import validate_results
from nerv.retrieval.queries import load_queries
from nerv.retrieval.query_encoder import encode_query
from nerv.retrieval.ranking import (
    aggregate_scores_by_document,
    deduplicate_by_chunk,
    rank_results,
)
from nerv.retrieval.retriever import retrieve
from nerv.vector_database.load_index import load_index

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = (
    PROJECT_ROOT.parent
    if PROJECT_ROOT.name == "entrega" and (PROJECT_ROOT.parent / "config").is_dir()
    else PROJECT_ROOT
)
DEFAULT_DELIVERY_DIR = (
    PROJECT_ROOT if PROJECT_ROOT.name == "entrega" else PROJECT_ROOT / "entrega"
)
CANDIDATE_POOL_SIZE = 50
MAX_WORDS_PER_FRAGMENT = 250
DELIVERY_ENCODER_DIRECTORY = "encoder_intfloat_multilingual-e5-small"
_REQUIRED_METADATA_FIELDS = frozenset(
    {
        "doc_id",
        "chunk_id",
        "fuente",
        "formato",
        "fenomeno",
        "posicion",
        "num_tokens",
        "texto",
    }
)
_TERMINAL_BOUNDARY = re.compile(r"[.!?…](?=\s|$)")
_JSON_STRUCTURAL_FIELD = re.compile(
    r"(?:keywords\[\d+\]:|doi:|issue:|body_paragraphs\[\d+\]:|"
    r"lists\[\d+\]:|alerta_meta\.)"
)
_PDF_CAPTION_MARKERS = (
    "[Pagina ",
    "Gráfico ",
    "Tabla ",
    "Fuente:",
    "FUENTE:",
    "Figura ",
    "CUADRO ",
    "Capítulo ",
    "CAPÍTULO ",
    "Anexo ",
    "ANEXO ",
)


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Load one metadata record per FAISS row and validate delivery fields."""
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not _REQUIRED_METADATA_FIELDS.issubset(
                record
            ):
                raise ValueError(
                    f"metadata line {line_number} lacks required delivery fields."
                )
            records.append(record)
    return records


def _has_terminal_boundary_at_end(text: str) -> bool:
    return bool(text.rstrip()) and text.rstrip()[-1] in ".?!…"


def _is_structural_non_sentence(record: dict[str, Any], text: str) -> bool:
    """Recognize deterministic non-prose endings represented by source data."""
    document_format = record["formato"].casefold()
    if document_format == "json":
        return bool(_JSON_STRUCTURAL_FIELD.search(text))
    if document_format != "pdf":
        return False
    if len(re.findall(r"\d+[,.]\d+%", text)) >= 4:
        return True
    if len(re.findall(r"(?<!\w)[a-z]\)", text)) >= 2 and ";" in text:
        return True

    tail = text[-220:]
    words = text.split()
    last_word = words[-1].rstrip(").]") if words else ""
    return any(marker in tail for marker in _PDF_CAPTION_MARKERS) and (
        last_word.isdigit()
        or last_word.isupper()
        or "Pagina" in tail
        or "Gráfico" in tail
        or "Tabla" in tail
    )


def _bounded_fragment_text(record: dict[str, Any], max_words: int) -> str:
    """Return at most ``max_words`` from the exact delivered source text.

    Corrected chunks that already fit are preserved byte-for-byte. Longer prose
    is shortened only to an existing terminal boundary within the word limit;
    longer structural data uses its deterministic first ``max_words`` words.
    """
    text = record["texto"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"chunk {record.get('chunk_id')!r} has empty source text.")
    words = text.split()
    if len(words) <= max_words:
        return text

    bounded_text = " ".join(words[:max_words])
    if _has_terminal_boundary_at_end(bounded_text) or _is_structural_non_sentence(
        record, text
    ):
        return bounded_text

    boundaries = list(_TERMINAL_BOUNDARY.finditer(bounded_text))
    if not boundaries:
        raise ValueError(
            "no complete source sentence is available within the permitted "
            f"fragment boundary for chunk {record['chunk_id']!r}."
        )
    return bounded_text[: boundaries[-1].end()].strip()


def _build_fragment_candidates(
    hits: list[tuple[int, float]], metadata: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": metadata[position]["doc_id"],
            "chunk_id": metadata[position]["chunk_id"],
            "text": metadata[position]["texto"],
            "score": score,
            "metadata": metadata[position],
        }
        for position, score in hits
    ]


def _resolve_query_result(
    query_id: str,
    hits: list[tuple[int, float]],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Format frozen numeric retrieval/ranking results in the official schema."""
    fragment_candidates = deduplicate_by_chunk(
        _build_fragment_candidates(hits, metadata)
    )
    top_fragments = rank_results(fragment_candidates, top_k=10)
    top_documents = rank_results(
        aggregate_scores_by_document(fragment_candidates), top_k=3
    )
    return {
        "query_id": query_id,
        "documents": [
            {"rank": rank, "doc_id": document["doc_id"]}
            for rank, document in enumerate(top_documents, start=1)
        ],
        "fragments": [
            {
                "rank": rank,
                "chunk_id": fragment["chunk_id"],
                "doc_id": fragment["doc_id"],
                "text": _bounded_fragment_text(
                    fragment["metadata"], MAX_WORDS_PER_FRAGMENT
                ),
            }
            for rank, fragment in enumerate(top_fragments, start=1)
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--delivery-dir", type=Path, default=DEFAULT_DELIVERY_DIR)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate and validate the 50 official deterministic retrieval records."""
    args = _parse_args()
    project_root = args.project_root.resolve()
    delivery_dir = args.delivery_dir.resolve()
    vector_dir = delivery_dir / "base_vectorial" / DELIVERY_ENCODER_DIRECTORY
    metadata_path = vector_dir / "metadata.jsonl"
    index_path = vector_dir / "index.faiss"
    config_path = project_root / "config" / "encoder_config.json"
    output_path = (args.output or delivery_dir / "resultados.jsonl").resolve()
    queries_path = (
        args.queries or project_root / "corpus" / "queries" / "queries.jsonl"
    ).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"missing public encoder config: {config_path}")
    metadata = _load_metadata(metadata_path)
    index = load_index(index_path, metadata_path=metadata_path, config_path=config_path)
    if index.ntotal != len(metadata):
        raise ValueError("FAISS row count does not match delivery metadata.")

    expected_ids = [f"q{number:03d}" for number in range(1, 51)]
    queries = load_queries(queries_path, expected_ids=expected_ids)
    encoder_config = load_encoder_config(config_path)
    encoder = load_encoder(
        device=args.device,
        local_files_only=args.local_files_only,
        config_path=config_path,
    )

    results = []
    for query_item in queries:
        query_embedding = encode_query(
            query_item["text"], encoder, config=encoder_config
        )
        hits = retrieve(query_embedding, index, top_k=CANDIDATE_POOL_SIZE)
        results.append(_resolve_query_result(query_item["query_id"], hits, metadata))

    validate_results(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in results:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Generated {len(results)} records: {output_path}")


if __name__ == "__main__":
    main()
