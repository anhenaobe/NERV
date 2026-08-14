"""Offline unit tests for the Hugging Face tokenizer adapter."""

import os
import sys
from types import SimpleNamespace

import pytest

from nerv.chunking.token_counter import TokenCounter

from .fakes import DeterministicFakeTokenizer


class LengthAwareFakeTokenizer:
    """Return deterministic IDs and optional lengths like Hugging Face."""

    model_max_length = 1_000_000_000_000_000_000

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
        return_length: bool = False,
        return_attention_mask: bool = True,
        return_token_type_ids: bool = True,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
                "return_length": return_length,
                "return_attention_mask": return_attention_mask,
                "return_token_type_ids": return_token_type_ids,
            }
        )

        def encode(value: str) -> list[int]:
            token_ids = [
                (ord(character) % 10_000) + 1
                for character in value
                if not character.isspace()
            ]
            if add_special_tokens:
                return [10_001, *token_ids, 10_002]
            return token_ids

        input_ids = (
            [encode(value) for value in text]
            if isinstance(text, list)
            else encode(text)
        )
        response: dict[str, object] = {"input_ids": input_ids}
        if return_length:
            response["length"] = (
                [len(value) for value in input_ids]
                if isinstance(text, list)
                else [len(input_ids)]
            )
        return response


def test_count_and_encode_return_validated_ids() -> None:
    """The count always equals the length of the returned token IDs."""
    tokenizer = DeterministicFakeTokenizer()
    counter = TokenCounter("team/future-encoder", tokenizer=tokenizer)

    token_ids = counter.encode("Clear text.")

    assert token_ids
    assert counter.count("Clear text.") == len(token_ids)


def test_special_token_policy_is_forwarded() -> None:
    """Special tokens are disabled by default and can be enabled explicitly."""
    default_tokenizer = DeterministicFakeTokenizer()
    default_counter = TokenCounter(
        "team/future-encoder",
        tokenizer=default_tokenizer,
    )
    special_tokenizer = DeterministicFakeTokenizer()
    special_counter = TokenCounter(
        "team/future-encoder",
        tokenizer=special_tokenizer,
        add_special_tokens=True,
    )

    default_count = default_counter.count("abc")
    special_count = special_counter.count("abc")

    assert default_count == 3
    assert special_count == 5
    assert default_tokenizer.calls[-1]["add_special_tokens"] is False
    assert special_tokenizer.calls[-1]["add_special_tokens"] is True


def test_injected_tokenizer_is_reused_without_loading() -> None:
    """One counter keeps and reuses the exact injected tokenizer instance."""
    tokenizer = DeterministicFakeTokenizer()
    counter = TokenCounter("", tokenizer=tokenizer)

    counter.count("first")
    counter.count("second")

    assert len(tokenizer.calls) == 2
    assert counter.tokenizer_class == "DeterministicFakeTokenizer"


def test_empty_text_is_valid_and_deterministic() -> None:
    """Empty content encodes to an empty sequence without artificial tokens."""
    counter = TokenCounter(
        "team/future-encoder",
        tokenizer=DeterministicFakeTokenizer(),
    )

    assert counter.encode("") == []
    assert counter.count("") == 0
    assert counter.encode("") == counter.encode("")


def test_document_prefix_is_counted_only_when_requested() -> None:
    """Prefix counting changes IDs without modifying the caller's text."""
    counter = TokenCounter(
        "intfloat/multilingual-e5-small",
        tokenizer=DeterministicFakeTokenizer(),
        document_prefix="passage: ",
    )
    text = "Original chunk text."
    plain = counter.encode(text)
    prefixed = counter.encode(text, include_document_prefix=True)

    assert len(prefixed) > len(plain)
    assert text == "Original chunk text."


def test_count_many_matches_individual_counts() -> None:
    """Batch counting preserves the exact scalar count contract."""
    counter = TokenCounter(
        "intfloat/multilingual-e5-small",
        tokenizer=DeterministicFakeTokenizer(),
        document_prefix="passage: ",
    )
    texts = ["first row", "segunda fila", ""]

    assert counter.count_many(texts, include_document_prefix=True, batch_size=2) == [
        counter.count(text, include_document_prefix=True) for text in texts
    ]


@pytest.mark.parametrize("include_document_prefix", [False, True])
def test_length_output_matches_reference_matrix(
    include_document_prefix: bool,
) -> None:
    """Length output remains exact for representative text semantics."""
    tokenizer = LengthAwareFakeTokenizer()
    counter = TokenCounter(
        "intfloat/multilingual-e5-small",
        tokenizer=tokenizer,
        document_prefix="passage: ",
    )
    texts = [
        "",
        "prueba",
        "La órbita permanece estable.",
        "The orbit remains stable.",
        "A órbita permanece estável.",
        "acentos: áéíóú ñ ç ã",
        "¿Puntuación?! -- yes...",
        "1.234,56 y 3.14",
        "x" * 5_000,
        "espacios     repetidos",
        "primera línea\nsegunda línea",
        "texto sintético " * 2_000,
    ]

    assert counter.count_many(
        texts,
        include_document_prefix=include_document_prefix,
        batch_size=4,
    ) == [
        len(
            counter.encode(
                text,
                include_document_prefix=include_document_prefix,
            )
        )
        for text in texts
    ]


@pytest.mark.parametrize("item_count", [1, 255, 256, 257, 512])
def test_length_output_preserves_batch_boundaries(item_count: int) -> None:
    """Optimized batches preserve size, order, and the default boundary."""
    counter = TokenCounter(
        "intfloat/multilingual-e5-small",
        tokenizer=LengthAwareFakeTokenizer(),
    )
    texts = [f"item {index}" for index in range(item_count)]

    assert counter.count_many(texts) == [len(counter.encode(text)) for text in texts]


def test_count_uses_length_output_without_changing_encode() -> None:
    """Counting requests length metadata while encoding still returns IDs."""
    tokenizer = LengthAwareFakeTokenizer()
    counter = TokenCounter("team/future-encoder", tokenizer=tokenizer)

    assert counter.count("abc") == 3
    assert tokenizer.calls[-1]["return_length"] is True
    assert tokenizer.calls[-1]["return_attention_mask"] is False
    assert tokenizer.calls[-1]["return_token_type_ids"] is False
    assert counter.encode("abc") == [98, 99, 100]
    assert tokenizer.calls[-1]["return_length"] is False


def test_missing_length_output_falls_back_and_caches_capability() -> None:
    """Input-ID-only tokenizers keep exact behavior without repeated probes."""
    tokenizer = DeterministicFakeTokenizer()
    counter = TokenCounter("team/future-encoder", tokenizer=tokenizer)

    assert counter.count("first") == len(counter.encode("first"))
    call_count = len(tokenizer.calls)
    assert counter.count("second") == len(counter.encode("second"))
    assert len(tokenizer.calls) == call_count + 2


@pytest.mark.parametrize(
    "length",
    [None, "1", [-1], [True], ["1"], [1, 2]],
)
def test_invalid_length_output_uses_exact_input_id_fallback(length: object) -> None:
    """Malformed optional metadata cannot become an incorrect count."""

    class InvalidLengthTokenizer(LengthAwareFakeTokenizer):
        def __call__(
            self,
            text: str | list[str],
            *,
            add_special_tokens: bool,
            truncation: bool,
            return_length: bool = False,
            return_attention_mask: bool = True,
            return_token_type_ids: bool = True,
        ) -> dict[str, object]:
            response = super().__call__(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_length=return_length,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
            )
            if return_length:
                response["length"] = length
            return response

    counter = TokenCounter(
        "team/future-encoder",
        tokenizer=InvalidLengthTokenizer(),
    )

    assert counter.count("abc") == 3


def test_plausible_but_inexact_length_uses_input_id_fallback() -> None:
    """A well-typed false length cannot silently change chunk boundaries."""

    class InexactLengthTokenizer(LengthAwareFakeTokenizer):
        def __call__(
            self,
            text: str | list[str],
            *,
            add_special_tokens: bool,
            truncation: bool,
            return_length: bool = False,
            return_attention_mask: bool = True,
            return_token_type_ids: bool = True,
        ) -> dict[str, object]:
            response = super().__call__(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_length=return_length,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
            )
            if return_length:
                response["length"] = [999]
            return response

    counter = TokenCounter(
        "team/future-encoder",
        tokenizer=InexactLengthTokenizer(),
    )

    assert counter.count("abc") == 3


def test_internal_type_error_from_optimized_call_is_not_swallowed() -> None:
    """Only an explicitly unsupported keyword can activate fallback."""

    class BrokenLengthTokenizer(LengthAwareFakeTokenizer):
        def __call__(
            self,
            text: str | list[str],
            *,
            add_special_tokens: bool,
            truncation: bool,
            return_length: bool = False,
            return_attention_mask: bool = True,
            return_token_type_ids: bool = True,
        ) -> dict[str, object]:
            if return_length:
                raise TypeError("internal normalization failure")
            return super().__call__(
                text,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_length=return_length,
                return_attention_mask=return_attention_mask,
                return_token_type_ids=return_token_type_ids,
            )

    counter = TokenCounter(
        "team/future-encoder",
        tokenizer=BrokenLengthTokenizer(),
    )

    with pytest.raises(TypeError, match="internal normalization failure"):
        counter.count("abc")


def test_non_string_text_is_rejected() -> None:
    """The public encoding boundary accepts strings only."""
    counter = TokenCounter(
        "team/future-encoder",
        tokenizer=DeterministicFakeTokenizer(),
    )
    with pytest.raises(TypeError, match="text must be a string"):
        counter.count(42)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"input_ids": None},
        {"input_ids": "123"},
        {"input_ids": [1, -1]},
        {"input_ids": [1, True]},
        {"input_ids": [1, "2"]},
    ],
)
def test_missing_or_invalid_input_ids_are_rejected(response: object) -> None:
    """Malformed tokenizer results never become misleading token counts."""

    class InvalidTokenizer:
        def __call__(
            self,
            text: str,
            *,
            add_special_tokens: bool,
            truncation: bool,
        ) -> object:
            return response

    counter = TokenCounter("team/future-encoder", tokenizer=InvalidTokenizer())
    with pytest.raises(ValueError, match="input_ids"):
        counter.count("text")


def test_empty_model_name_requires_an_injected_tokenizer() -> None:
    """Real loading always requires an exact non-empty model identifier."""
    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        TokenCounter("")


@pytest.mark.parametrize(
    "text",
    [
        "análisis lingüístico",
        "informação pública",
        "don't truncate this",
    ],
)
def test_unicode_and_contractions_are_deterministic(text: str) -> None:
    """The adapter preserves multilingual text and English contractions."""
    counter = TokenCounter(
        "team/future-encoder",
        tokenizer=DeterministicFakeTokenizer(),
    )
    assert counter.encode(text) == counter.encode(text)
    assert counter.count(text) == len(counter.encode(text))


def test_encoding_never_requests_truncation() -> None:
    """Long content is forwarded in full with truncation disabled."""
    tokenizer = DeterministicFakeTokenizer()
    counter = TokenCounter("team/future-encoder", tokenizer=tokenizer)
    text = "x" * 5_000

    assert counter.count(text) == 5_000
    assert tokenizer.calls[-1] == {
        "text": text,
        "add_special_tokens": False,
        "truncation": False,
    }
    assert counter.reported_model_max_length == tokenizer.model_max_length


def test_real_loader_is_called_once_per_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal construction delegates once to AutoTokenizer.from_pretrained."""
    tokenizer = DeterministicFakeTokenizer()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(
            model_name: str,
            **options: object,
        ) -> DeterministicFakeTokenizer:
            calls.append((model_name, options))
            return tokenizer

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    counter = TokenCounter(
        "organization/model-name",
        revision="abc123",
        use_fast=False,
        trust_remote_code=False,
    )

    counter.count("first")
    counter.count("second")

    assert calls == [
        (
            "organization/model-name",
            {
                "use_fast": False,
                "trust_remote_code": False,
                "local_files_only": False,
                "revision": "abc123",
            },
        )
    ]


def test_optional_locally_cached_real_tokenizer() -> None:
    """Exercise a real tokenizer only when a local model is explicitly selected."""
    model_name = os.getenv("NERV_TEST_ENCODER_MODEL")
    if not model_name:
        pytest.skip("NERV_TEST_ENCODER_MODEL is not configured.")

    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )
    except (OSError, ValueError) as error:
        pytest.skip(f"Tokenizer is not available in the local cache: {error}")

    counter = TokenCounter(model_name, tokenizer=tokenizer)
    token_ids = counter.encode("Tokenizer integration check.")
    assert counter.count("Tokenizer integration check.") == len(token_ids)
