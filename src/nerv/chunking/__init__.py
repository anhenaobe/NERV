"""Stable public interfaces for language-aware document chunking."""

from typing import TYPE_CHECKING, Any

from .chunker import create_chunks
from .language_detector import LanguageDetectionResult, detect_language
from .sentence_splitter import split_sentences
from .token_counter import (
    TokenCounter,
    count_tokens,
    count_words_as_tokens,
    get_token_counter,
)

if TYPE_CHECKING:
    from .pipeline import PipelineSummary


def __getattr__(name: str) -> Any:
    """Load pipeline interfaces lazily so module CLI execution stays clean."""
    if name in {"PipelineSummary", "run_pipeline"}:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LanguageDetectionResult",
    "PipelineSummary",
    "TokenCounter",
    "count_tokens",
    "count_words_as_tokens",
    "create_chunks",
    "detect_language",
    "get_token_counter",
    "run_pipeline",
    "split_sentences",
]
