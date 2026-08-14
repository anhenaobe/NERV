"""Generate normalized embeddings under the frozen semantic contract."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from nerv.chunking.configuration import EncoderConfig, load_encoder_config


def generate_embeddings(
    texts: Sequence[str],
    encoder: Any,
    batch_size: int,
    *,
    kind: Literal["document", "query"] = "document",
    config: EncoderConfig | None = None,
) -> np.ndarray:
    """Encode raw texts, owning prefix application in exactly one location."""
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings.")
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("every text must be a string.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")
    semantic = config or load_encoder_config()
    prefix = semantic["document_prefix" if kind == "document" else "query_prefix"]
    prefixed = [f"{prefix}{text}" for text in texts]
    vectors = encoder.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=semantic["normalize_embeddings"],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(vectors)
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32, copy=False)
    expected_shape = (len(prefixed), semantic["embedding_dimension"])
    if matrix.shape != expected_shape:
        raise ValueError(
            f"encoder returned shape={matrix.shape}; expected {expected_shape}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("encoder returned NaN or infinite values.")
    if semantic["normalize_embeddings"] and len(matrix):
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
            raise ValueError("encoder returned vectors that are not L2-normalized.")
    return matrix
