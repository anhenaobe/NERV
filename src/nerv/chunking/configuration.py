"""Load the canonical frozen encoder and chunking configuration."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast


class EncoderConfig(TypedDict):
    """Validated fields required by chunking and future embeddings."""

    schema_version: int
    encoder_model_name: str
    tokenizer_revision: str | None
    embedding_dimension: int
    chunk_max_tokens: int
    encoder_max_input_tokens: int
    overlap_tokens: int
    document_prefix: str
    query_prefix: str
    add_special_tokens: bool
    normalize_embeddings: bool
    normalization_method: str
    future_similarity_metric: str
    future_faiss_index: str
    oversized_sentence_policy: str


DEFAULT_ENCODER_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "encoder_config.json"
)


@dataclass(frozen=True)
class EncoderCapacity:
    """Verified total, special-token, and stored-content token capacities."""

    encoder_max_input_tokens: int
    encoder_special_token_overhead: int
    effective_content_max_tokens: int
    tokenizer_model_max_length: int
    model_max_position_embeddings: int


def load_encoder_config(
    path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> EncoderConfig:
    """Read the single canonical frozen configuration."""
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("encoder configuration must be a JSON object.")
    required = set(EncoderConfig.__required_keys__)
    if set(data) != required:
        raise ValueError("encoder configuration fields do not match the schema.")
    config = cast(EncoderConfig, data)
    if config["schema_version"] != 2:
        raise ValueError("encoder configuration schema_version must be 2.")
    soft_limit = config["chunk_max_tokens"]
    hard_limit = config["encoder_max_input_tokens"]
    overlap = config["overlap_tokens"]
    if (
        isinstance(soft_limit, bool)
        or not isinstance(soft_limit, int)
        or soft_limit <= 0
    ):
        raise ValueError("chunk_max_tokens must be a positive integer.")
    if (
        isinstance(hard_limit, bool)
        or not isinstance(hard_limit, int)
        or hard_limit < soft_limit
    ):
        raise ValueError(
            "encoder_max_input_tokens must be an integer at least chunk_max_tokens."
        )
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or overlap < 0
        or overlap >= soft_limit
    ):
        raise ValueError(
            "overlap_tokens must be non-negative and smaller than chunk_max_tokens."
        )
    approved_values: dict[str, object] = {
        "encoder_model_name": "intfloat/multilingual-e5-small",
        "embedding_dimension": 384,
        "chunk_max_tokens": 256,
        "encoder_max_input_tokens": 512,
        "overlap_tokens": 32,
        "document_prefix": "passage: ",
        "query_prefix": "query: ",
        "add_special_tokens": False,
        "normalize_embeddings": True,
        "normalization_method": "l2",
        "future_similarity_metric": "inner_product",
        "future_faiss_index": "IndexFlatIP",
        "oversized_sentence_policy": (
            "preserve_through_encoder_limit_then_hierarchical_subdivide"
        ),
    }
    for field, approved in approved_values.items():
        if config[field] != approved:  # type: ignore[literal-required]
            raise ValueError(f"encoder configuration field {field!r} is not approved.")
    revision = config["tokenizer_revision"]
    if revision is not None and (not isinstance(revision, str) or not revision.strip()):
        raise ValueError("tokenizer_revision must be a non-empty string or null.")
    return config


def verify_encoder_capacity(
    *,
    model_name: str,
    revision: str | None,
    tokenizer_model_max_length: object,
    configured_hard_limit: int,
    local_files_only: bool,
) -> int:
    """Verify the versioned hard limit against tokenizer and model metadata."""
    if (
        isinstance(tokenizer_model_max_length, int)
        and tokenizer_model_max_length < configured_hard_limit
    ):
        raise ValueError(
            "the configured encoder hard limit exceeds tokenizer.model_max_length."
        )
    try:
        from transformers import AutoConfig

        options: dict[str, object] = {"local_files_only": local_files_only}
        if revision is not None:
            options["revision"] = revision
        model_config = AutoConfig.from_pretrained(model_name, **options)
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "could not verify the encoder model position limit."
        ) from error
    model_limit = getattr(model_config, "max_position_embeddings", None)
    if not isinstance(model_limit, int):
        raise ValueError("the encoder model does not declare max_position_embeddings.")
    if model_limit < configured_hard_limit:
        raise ValueError(
            "the configured encoder hard limit exceeds model.max_position_embeddings."
        )
    return model_limit


def derive_effective_content_max_tokens(
    *,
    encoder_max_input_tokens: int,
    encoder_special_token_overhead: int,
    tokenizer_model_max_length: object,
    model_max_position_embeddings: object,
) -> EncoderCapacity:
    """Derive the safe no-special-token budget and fail closed on drift."""
    if (
        isinstance(encoder_max_input_tokens, bool)
        or not isinstance(encoder_max_input_tokens, int)
        or encoder_max_input_tokens <= 0
    ):
        raise ValueError("encoder_max_input_tokens must be a positive integer.")
    if (
        isinstance(encoder_special_token_overhead, bool)
        or not isinstance(encoder_special_token_overhead, int)
        or encoder_special_token_overhead <= 0
    ):
        raise ValueError("encoder_special_token_overhead must be a positive integer.")
    if (
        isinstance(tokenizer_model_max_length, bool)
        or not isinstance(tokenizer_model_max_length, int)
        or tokenizer_model_max_length <= 0
    ):
        raise ValueError("tokenizer_model_max_length must be a positive integer.")
    if (
        isinstance(model_max_position_embeddings, bool)
        or not isinstance(model_max_position_embeddings, int)
        or model_max_position_embeddings <= 0
    ):
        raise ValueError(
            "model_max_position_embeddings must be a positive integer."
        )
    if tokenizer_model_max_length < encoder_max_input_tokens:
        raise ValueError(
            "configured total capacity exceeds tokenizer.model_max_length."
        )
    if model_max_position_embeddings < encoder_max_input_tokens:
        raise ValueError(
            "configured total capacity exceeds model.max_position_embeddings."
        )
    effective = encoder_max_input_tokens - encoder_special_token_overhead
    if effective <= 0:
        raise ValueError("special-token overhead leaves no content capacity.")
    return EncoderCapacity(
        encoder_max_input_tokens=encoder_max_input_tokens,
        encoder_special_token_overhead=encoder_special_token_overhead,
        effective_content_max_tokens=effective,
        tokenizer_model_max_length=tokenizer_model_max_length,
        model_max_position_embeddings=model_max_position_embeddings,
    )


def resolve_encoder_capacity(
    *,
    config: EncoderConfig,
    tokenizer_model_max_length: object,
    encoder_special_token_overhead: int,
    local_files_only: bool,
) -> EncoderCapacity:
    """Verify real model metadata and derive the one effective chunk budget."""
    model_limit = verify_encoder_capacity(
        model_name=config["encoder_model_name"],
        revision=config["tokenizer_revision"],
        tokenizer_model_max_length=tokenizer_model_max_length,
        configured_hard_limit=config["encoder_max_input_tokens"],
        local_files_only=local_files_only,
    )
    return derive_effective_content_max_tokens(
        encoder_max_input_tokens=config["encoder_max_input_tokens"],
        encoder_special_token_overhead=encoder_special_token_overhead,
        tokenizer_model_max_length=tokenizer_model_max_length,
        model_max_position_embeddings=model_limit,
    )
