"""Contracts for ranking retrieval candidates.

Ademas de `rank_results` (parte del contrato original), este modulo
incluye dos utilidades de agregacion que necesita la recuperacion a dos
niveles descrita en la Seccion 8.6 del PDF del reto.
"""

from collections.abc import Sequence
from typing import Any


def rank_results(
    candidates: Sequence[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Order candidates by descending score and return the requested count."""
    if top_k <= 0:
        raise ValueError(f"top_k debe ser positivo, se recibio {top_k}.")
    ordered = sorted(candidates, key=lambda item: item["score"], reverse=True)
    return ordered[:top_k]


def deduplicate_by_chunk(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove repeated fragments, keeping the highest-scoring occurrence."""
    best_by_chunk: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        current_best = best_by_chunk.get(chunk_id)
        if current_best is None or candidate["score"] > current_best["score"]:
            best_by_chunk[chunk_id] = candidate
    return list(best_by_chunk.values())


def aggregate_scores_by_document(
    fragment_candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate fragment-level scores into document-level scores (max pooling).

    CAMBIO respecto a la version anterior (CombSUM -> max pooling, Seccion
    8.6 del PDF): el score de un documento es el de su MEJOR fragmento
    individual, no la suma de todos. Esto evita que documentos con muchos
    chunks (por ejemplo, filas de CSV convertidas una por una) ganen por
    volumen en vez de por relevancia real de un fragmento concreto.
    """
    best_score_by_doc: dict[str, float] = {}
    for fragment in fragment_candidates:
        doc_id = fragment["doc_id"]
        score = fragment["score"]
        if doc_id not in best_score_by_doc or score > best_score_by_doc[doc_id]:
            best_score_by_doc[doc_id] = score
    return [
        {"doc_id": doc_id, "score": score}
        for doc_id, score in best_score_by_doc.items()
    ]
