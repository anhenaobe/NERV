"""Encode queries with the canonical prefix and normalization contract."""

from typing import Any, cast

from nerv.chunking.configuration import EncoderConfig, load_encoder_config
from nerv.embeddings.embedding_generator import generate_embeddings


def encode_query(
    query: str,
    encoder: Any,
    *,
    config: EncoderConfig | None = None,
) -> list[float]:
    """Encode one raw query; prefix ownership remains inside the generator."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    semantic = config or load_encoder_config()
    matrix = generate_embeddings(
        [query.strip()],
        encoder,
        batch_size=1,
        kind="query",
        config=semantic,
    )
    return cast(list[float], matrix[0].tolist())
