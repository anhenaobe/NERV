"""Load and verify the encoder named by the frozen semantic contract."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

from nerv.chunking.configuration import (
    DEFAULT_ENCODER_CONFIG_PATH,
    EncoderConfig,
    derive_effective_content_max_tokens,
    load_encoder_config,
    resolve_encoder_capacity,
)


def _special_token_count(tokenizer: Any) -> int:
    """Return the tokenizer's single-sequence special-token overhead."""
    counter = getattr(tokenizer, "num_special_tokens_to_add", None)
    if not callable(counter):
        raise ValueError("the encoder tokenizer cannot report special-token overhead.")
    count = counter(pair=False)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError(
            "the encoder tokenizer reported invalid special-token overhead."
        )
    return count


def encoder_content_capacity(encoder: Any, config: EncoderConfig) -> int:
    """Return the no-special-token content capacity of an encoder.

    Chunk ``num_tokens`` deliberately excludes model-added special tokens.  A
    SentenceTransformer tokenizer normally adds those tokens during encoding,
    so they must be reserved from the model's total sequence length.
    """
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("the encoder does not expose its tokenizer.")
    encoder_limit = getattr(encoder, "max_seq_length", None)
    if not isinstance(encoder_limit, int) or encoder_limit <= 0:
        raise ValueError("the encoder does not expose a valid max_seq_length.")
    capacity = derive_effective_content_max_tokens(
        encoder_max_input_tokens=config["encoder_max_input_tokens"],
        encoder_special_token_overhead=_special_token_count(tokenizer),
        tokenizer_model_max_length=getattr(tokenizer, "model_max_length", None),
        model_max_position_embeddings=encoder_limit,
    )
    return capacity.effective_content_max_tokens


def validate_encoder_inputs(
    texts: list[str],
    expected_content_counts: list[int],
    encoder: Any,
    config: EncoderConfig,
) -> None:
    """Prove that inputs fit without SentenceTransformer truncation."""
    if len(texts) != len(expected_content_counts):
        raise ValueError("text/count alignment is invalid.")
    tokenizer = getattr(encoder, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("the encoder does not expose its tokenizer.")
    capacity = encoder_content_capacity(encoder, config)
    for text, expected_count in zip(texts, expected_content_counts, strict=True):
        encoded = tokenizer(
            text,
            add_special_tokens=config["add_special_tokens"],
            truncation=False,
        )
        input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if (
            isinstance(input_ids, (str, bytes))
            or not isinstance(input_ids, Sequence)
            or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in input_ids
            )
        ):
            raise ValueError("the encoder tokenizer returned invalid input_ids.")
        if len(input_ids) != expected_count:
            raise ValueError(
                "chunk num_tokens does not match the encoder tokenizer without "
                "special tokens."
            )
        if expected_count > capacity:
            raise ValueError(
                "input would be silently truncated after model-specific special "
                f"tokens: content={expected_count}, safe_capacity={capacity}."
            )


def load_encoder(
    *,
    device: str = "cpu",
    local_files_only: bool = False,
    config_path: Path = DEFAULT_ENCODER_CONFIG_PATH,
) -> Any:
    """Load the configured SentenceTransformer without duplicated constants."""
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'.")
    try:
        torch = import_module("torch")
        sentence_transformers = import_module("sentence_transformers")
    except ImportError as error:
        raise ImportError(
            "torch and sentence-transformers are required to load the encoder."
        ) from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is not available.")

    config = load_encoder_config(config_path)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    options: dict[str, object] = {
        "device": device,
        "local_files_only": local_files_only,
    }
    revision = config["tokenizer_revision"]
    if revision is not None:
        options["revision"] = revision
    sentence_transformer = sentence_transformers.SentenceTransformer
    model = sentence_transformer(config["encoder_model_name"], **options)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("SentenceTransformer did not expose a tokenizer.")
    capacity = resolve_encoder_capacity(
        config=config,
        tokenizer_model_max_length=getattr(tokenizer, "model_max_length", None),
        encoder_special_token_overhead=_special_token_count(tokenizer),
        local_files_only=local_files_only,
    )
    current_limit = getattr(model, "max_seq_length", None)
    if (
        not isinstance(current_limit, int)
        or current_limit < capacity.encoder_max_input_tokens
    ):
        raise ValueError(
            "SentenceTransformer max_seq_length is below the configured total capacity."
        )
    model.max_seq_length = capacity.encoder_max_input_tokens
    return model
