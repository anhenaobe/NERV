import json
from pathlib import Path
from typing import Any

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
from nerv.utils.config import load_config
from nerv.utils.paths import resolve_project_path
from nerv.vector_database.load_index import load_index

PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_POOL_SIZE = 50  # candidatos crudos por consulta, antes de agregar/deduplicar
MAX_WORDS_PER_FRAGMENT = 250


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Carga metadata.jsonl; la fila i debe corresponder al id interno i de FAISS."""
    records = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _truncate_respecting_sentences(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])
    last_boundary = max(
        truncated.rfind(". "), truncated.rfind("? "), truncated.rfind("! ")
    )
    if last_boundary == -1:
        return text
    return truncated[: last_boundary + 1].strip()


def _build_fragment_candidates(
    hits: list[tuple[int, float]],
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates = []
    for position, score in hits:
        record = metadata[position]
        candidates.append(
            {
                "doc_id": record["doc_id"],
                "chunk_id": record["chunk_id"],
                "text": record["texto"],
                "score": score,
            }
        )
    return candidates


def _resolve_query_result(
    query_id: str,
    hits: list[tuple[int, float]],
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    fragment_candidates = _build_fragment_candidates(hits, metadata)
    fragment_candidates = deduplicate_by_chunk(fragment_candidates)

    top_fragments = rank_results(fragment_candidates, top_k=10)
    document_candidates = aggregate_scores_by_document(fragment_candidates)
    top_documents = rank_results(document_candidates, top_k=3)

    formatted_documents = [
        {"rank": rank, "doc_id": doc["doc_id"]}
        for rank, doc in enumerate(top_documents, start=1)
    ]
    formatted_fragments = [
        {
            "rank": rank,
            "chunk_id": fragment["chunk_id"],
            "doc_id": fragment["doc_id"],
            "text": _truncate_respecting_sentences(
                fragment["text"], MAX_WORDS_PER_FRAGMENT
            ),
        }
        for rank, fragment in enumerate(top_fragments, start=1)
    ]

    return {
        "query_id": query_id,
        "documents": formatted_documents,
        "fragments": formatted_fragments,
    }


def main() -> None:
    """Coordinar la generación de ``resultados.jsonl``."""
    print("[Iniciando] Generador maestro de recuperación NERV...")

    config_path = resolve_project_path(PROJECT_ROOT, Path("config/config.yaml"))
    config = load_config(config_path)

    print("[Cargando] Encoder desde config/encoder_config.json")
    encoder = load_encoder(device=config["runtime"].get("device", "cpu"))

    index_path = resolve_project_path(PROJECT_ROOT, Path(config["faiss"]["index_path"]))
    metadata_path = resolve_project_path(
        PROJECT_ROOT, Path(config["faiss"]["metadata_path"])
    )
    print(f"[Cargando] Índice FAISS: {index_path}")
    index = load_index(index_path, metadata_path=metadata_path)
    metadata = _load_metadata(metadata_path)
    print(f"[Cargando] Metadata: {len(metadata)} fragmentos indexados")

    queries_path = resolve_project_path(PROJECT_ROOT, Path(config["paths"]["queries"]))
    queries = load_queries(queries_path)
    print(f"[Procesando] {len(queries)} consultas")

    results = []
    for query_item in queries:
        query_id = query_item["query_id"]
        query_text = query_item["text"]

        query_embedding = encode_query(query_text, encoder)
        hits = retrieve(query_embedding, index, top_k=CANDIDATE_POOL_SIZE)
        results.append(_resolve_query_result(query_id, hits, metadata))

    print("[Validando] Verificando resultados.jsonl contra el esquema oficial...")
    validate_results(results)

    output_path = resolve_project_path(PROJECT_ROOT, Path("resultados.jsonl"))
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in results:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Completado] {output_path}")


if __name__ == "__main__":
    main()
