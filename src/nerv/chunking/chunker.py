"""Deterministic grouping of complete sentences into chunks."""

from collections.abc import Callable, Sequence

from .token_counter import count_words_as_tokens

TokenCountFunction = Callable[[str], int]


def _validate_configuration(max_tokens: int, overlap_tokens: int) -> None:
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError("max_tokens must be an integer.")
    if isinstance(overlap_tokens, bool) or not isinstance(overlap_tokens, int):
        raise TypeError("overlap_tokens must be an integer.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens.")


def _count_checked(text: str, count_tokens: TokenCountFunction) -> int:
    count = count_tokens(text)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count_tokens must return a non-negative integer.")
    return count


def _overlap_suffix(
    sentences: list[str],
    overlap_tokens: int,
    count_tokens: TokenCountFunction,
) -> list[str]:
    """Return the longest complete-sentence suffix within the overlap budget."""
    if overlap_tokens == 0:
        return []

    suffix: list[str] = []
    for sentence in reversed(sentences):
        candidate = [sentence, *suffix]
        if _count_checked(" ".join(candidate), count_tokens) > overlap_tokens:
            break
        suffix = candidate
    return suffix


def create_chunks(
    sentences: Sequence[str],
    max_tokens: int,
    overlap_tokens: int = 0,
    *,
    count_tokens: TokenCountFunction | None = None,
) -> list[str]:
    """Group complete sentences under a tokenizer-based content limit.

    ``count_tokens`` must be the counter for the future embedding encoder in
    production. Omitting it preserves the previous whitespace-based baseline
    only for backward compatibility and historical evaluation.

    A sentence that exceeds ``max_tokens`` is emitted unchanged as a standalone
    oversized chunk. Candidate counts always use the fully joined candidate
    text because tokenizer behavior can change at whitespace boundaries.
    """
    _validate_configuration(max_tokens, overlap_tokens)
    if isinstance(sentences, (str, bytes)) or not isinstance(sentences, Sequence):
        raise TypeError("sentences must be a sequence of strings.")

    counter = count_tokens or count_words_as_tokens
    normalized_sentences: list[str] = []
    for sentence in sentences:
        if not isinstance(sentence, str):
            raise TypeError("every sentence must be a string.")
        stripped_sentence = sentence.strip()
        if stripped_sentence:
            normalized_sentences.append(stripped_sentence)

    chunks: list[str] = []
    current: list[str] = []

    for sentence in normalized_sentences:
        sentence_size = _count_checked(sentence, counter)
        if sentence_size > max_tokens:
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(sentence)
            continue

        candidate = [*current, sentence]
        if current and _count_checked(" ".join(candidate), counter) > max_tokens:
            completed = current
            chunks.append(" ".join(completed))
            current = _overlap_suffix(
                completed,
                overlap_tokens,
                counter,
            )

            while (
                current
                and _count_checked(
                    " ".join([*current, sentence]),
                    counter,
                )
                > max_tokens
            ):
                current.pop(0)

        current.append(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks
