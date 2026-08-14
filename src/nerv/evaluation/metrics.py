from collections.abc import Sequence


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if k <= 0:
        raise ValueError(f"k debe ser positivo, se recibió {k}.")
    top_k_retrieved = retrieved_ids[:k]
    if not top_k_retrieved:
        return 0.0
    relevant_count = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_ids)
    return relevant_count / k
