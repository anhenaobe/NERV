"""Contracts for retrieving vector candidates."""

from collections.abc import Sequence
from typing import Any

import numpy as np


def retrieve(
    query_embedding: Sequence[float],
    index: Any,
    top_k: int,
) -> list[tuple[int, float]]:
    if top_k <= 0:
        raise ValueError(f"top_k debe ser positivo, se recibió {top_k}.")

    query_vector = np.asarray([query_embedding], dtype="float32")
    scores, positions = index.search(query_vector, top_k)

    results: list[tuple[int, float]] = []
    for position, score in zip(positions[0], scores[0], strict=True):
        if position == -1:
            continue  # FAISS rellena con -1 si el índice tiene menos de top_k vectores
        results.append((int(position), float(score)))
    return results
