"""Token counting with the tokenizer configured for the embedding encoder."""

from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from inspect import Parameter, signature
from typing import Protocol, cast


class _TokenizerProtocol(Protocol):
    """Minimal standard tokenizer behavior required by ``TokenCounter``."""

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> Mapping[str, object]: ...


class TokenCounter:
    """Count tokens with the tokenizer associated with the embedding encoder.

    Special tokens are disabled by default so chunk limits represent content
    tokens. Encoding never requests truncation or applies ``model_max_length``.
    """

    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        use_fast: bool = True,
        trust_remote_code: bool = False,
        add_special_tokens: bool = False,
        document_prefix: str = "passage: ",
        local_files_only: bool = False,
        tokenizer: object | None = None,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if tokenizer is None and not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if revision is not None and (
            not isinstance(revision, str) or not revision.strip()
        ):
            raise ValueError("revision must be a non-empty string or None.")
        if not isinstance(document_prefix, str):
            raise TypeError("document_prefix must be a string.")

        self._model_name = model_name.strip()
        self._revision = revision
        self._use_fast = use_fast
        self._trust_remote_code = trust_remote_code
        self._add_special_tokens = add_special_tokens
        self._document_prefix = document_prefix
        self._local_files_only = local_files_only
        loaded_tokenizer = tokenizer or self._load_tokenizer()
        self._tokenizer = cast(_TokenizerProtocol, loaded_tokenizer)
        self._return_length_supported: bool | None = None
        try:
            tokenizer_parameters = signature(loaded_tokenizer.__call__).parameters
        except (TypeError, ValueError):
            self._verbose_supported = False
        else:
            self._verbose_supported = "verbose" in tokenizer_parameters or any(
                parameter.kind is Parameter.VAR_KEYWORD
                for parameter in tokenizer_parameters.values()
            )

    def _try_returned_lengths(
        self,
        text: str | list[str],
        *,
        expected_count: int,
    ) -> list[int] | None:
        """Request validated lengths or return ``None`` for exact fallback.

        Hugging Face tokenizers may accept ``return_length`` while injected
        tokenizers expose only the smaller standard protocol. Capability is
        detected once per counter. Only a ``TypeError`` that explicitly names
        an unexpected optimized keyword is treated as unsupported. Malformed,
        absent, or input-ID-inconsistent length output selects the existing
        validated input-ID path.
        """
        if self._return_length_supported is False:
            return None

        tokenizer_call = cast(Callable[..., object], self._tokenizer)
        try:
            options: dict[str, object] = {
                "add_special_tokens": self._add_special_tokens,
                "truncation": False,
                "return_length": True,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            }
            if self._verbose_supported:
                options["verbose"] = False
            encoded = tokenizer_call(text, **options)
        except TypeError as error:
            message = str(error)
            optimized_options = (
                "return_length",
                "return_attention_mask",
                "return_token_type_ids",
            )
            if "unexpected keyword argument" in message and any(
                option in message for option in optimized_options
            ):
                self._return_length_supported = False
                return None
            raise

        if not isinstance(encoded, Mapping) or "length" not in encoded:
            self._return_length_supported = False
            return None
        raw_lengths = encoded["length"]
        if isinstance(raw_lengths, (str, bytes)) or not isinstance(
            raw_lengths,
            Sequence,
        ):
            self._return_length_supported = False
            return None
        if len(raw_lengths) != expected_count:
            self._return_length_supported = False
            return None

        lengths: list[int] = []
        for length in raw_lengths:
            if isinstance(length, bool) or not isinstance(length, int) or length < 0:
                self._return_length_supported = False
                return None
            lengths.append(length)

        input_ids = encoded.get("input_ids")
        if isinstance(input_ids, (str, bytes)) or not isinstance(
            input_ids,
            Sequence,
        ):
            self._return_length_supported = False
            return None
        if expected_count == 1 and (
            not input_ids or isinstance(input_ids[0], int)
        ):
            input_batches: Sequence[object] = [input_ids]
        else:
            input_batches = input_ids
        if len(input_batches) != expected_count:
            self._return_length_supported = False
            return None

        validate_token_ids = self._return_length_supported is None
        input_id_lengths: list[int] = []
        for token_ids in input_batches:
            if isinstance(token_ids, (str, bytes)) or not isinstance(
                token_ids,
                Sequence,
            ):
                self._return_length_supported = False
                return None
            if validate_token_ids and any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in token_ids
            ):
                self._return_length_supported = False
                return None
            input_id_lengths.append(len(token_ids))
        if input_id_lengths != lengths:
            self._return_length_supported = False
            return None

        self._return_length_supported = True
        return lengths

    def _load_tokenizer(self) -> object:
        """Load the configured Hugging Face tokenizer once."""
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "transformers is required to load a real tokenizer. "
                "Install the project dependencies or inject a tokenizer."
            ) from error

        options: dict[str, object] = {
            "use_fast": self._use_fast,
            "trust_remote_code": self._trust_remote_code,
            "local_files_only": self._local_files_only,
        }
        if self._revision is not None:
            options["revision"] = self._revision
        try:
            return AutoTokenizer.from_pretrained(self._model_name, **options)
        except OSError as error:
            if self._local_files_only:
                raise RuntimeError(
                    f"The tokenizer for {self._model_name!r} is not available "
                    "locally."
                ) from error
            raise RuntimeError(
                f"Could not load the tokenizer for {self._model_name!r}."
            ) from error

    def encode(
        self,
        text: str,
        *,
        include_document_prefix: bool = False,
    ) -> list[int]:
        """Encode text without truncation and return validated token IDs."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        effective_text = (
            f"{self._document_prefix}{text}"
            if include_document_prefix
            else text
        )
        encoded = self._tokenizer(
            effective_text,
            add_special_tokens=self._add_special_tokens,
            truncation=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("the tokenizer did not return input_ids.")

        input_ids = encoded["input_ids"]
        if isinstance(input_ids, (str, bytes)) or not isinstance(
            input_ids,
            Sequence,
        ):
            raise ValueError("the tokenizer returned invalid input_ids.")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in input_ids
        ):
            raise ValueError("the tokenizer returned invalid input_ids.")

        return list(input_ids)

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Return validated content-token character offsets without truncation.

        Offset mappings are used only by the hard-limit splitter. They refer to
        the supplied text itself: the document prefix is deliberately excluded.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        tokenizer_call = cast(Callable[..., object], self._tokenizer)
        try:
            encoded = tokenizer_call(
                text,
                add_special_tokens=self._add_special_tokens,
                truncation=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except TypeError as error:
            message = str(error)
            if not (
                "unexpected keyword argument" in message
                and any(
                    option in message
                    for option in (
                        "return_offsets_mapping",
                        "return_attention_mask",
                        "return_token_type_ids",
                    )
                )
            ):
                raise
            raise ValueError(
                "the tokenizer does not provide the offset mapping required "
                "for hard-limit fallback splitting."
            ) from error
        except NotImplementedError as error:
            raise ValueError(
                "the tokenizer does not provide the offset mapping required "
                "for hard-limit fallback splitting."
            ) from error
        if not isinstance(encoded, Mapping) or "offset_mapping" not in encoded:
            raise ValueError("the tokenizer did not return offset_mapping.")
        raw_offsets = encoded["offset_mapping"]
        if isinstance(raw_offsets, (str, bytes)) or not isinstance(
            raw_offsets,
            Sequence,
        ):
            raise ValueError("the tokenizer returned invalid offset_mapping.")

        offsets: list[tuple[int, int]] = []
        previous_start = 0
        for raw_offset in raw_offsets:
            if isinstance(raw_offset, (str, bytes)) or not isinstance(
                raw_offset,
                Sequence,
            ) or len(raw_offset) != 2:
                raise ValueError("the tokenizer returned invalid offset_mapping.")
            start, end = raw_offset
            if (
                isinstance(start, bool)
                or not isinstance(start, int)
                or isinstance(end, bool)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end > len(text)
                or start < previous_start
            ):
                raise ValueError("the tokenizer returned invalid offset_mapping.")
            previous_start = start
            if end > start:
                offsets.append((start, end))
        return offsets

    def count(
        self,
        text: str,
        *,
        include_document_prefix: bool = False,
    ) -> int:
        """Return the number of encoded content tokens."""
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        effective_text = (
            f"{self._document_prefix}{text}"
            if include_document_prefix
            else text
        )
        returned_lengths = self._try_returned_lengths(
            effective_text,
            expected_count=1,
        )
        if returned_lengths is not None:
            return returned_lengths[0]

        token_count = len(
            self.encode(
                text,
                include_document_prefix=include_document_prefix,
            )
        )
        if token_count < 0:
            raise ValueError("the tokenizer returned a negative token count.")
        return token_count

    def count_many(
        self,
        texts: Sequence[str],
        *,
        include_document_prefix: bool = False,
        batch_size: int = 256,
    ) -> list[int]:
        """Count many texts in tokenizer batches without truncation."""
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("every text must be a string.")

        counts: list[int] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            effective_batch = [
                f"{self._document_prefix}{text}"
                if include_document_prefix
                else text
                for text in batch
            ]
            returned_lengths = self._try_returned_lengths(
                effective_batch,
                expected_count=len(batch),
            )
            if returned_lengths is not None:
                counts.extend(returned_lengths)
                continue

            encoded = self._tokenizer(
                effective_batch,
                add_special_tokens=self._add_special_tokens,
                truncation=False,
            )
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise ValueError("the tokenizer did not return input_ids.")
            input_batches = encoded["input_ids"]
            if isinstance(input_batches, (str, bytes)) or not isinstance(
                input_batches,
                Sequence,
            ):
                raise ValueError("the tokenizer returned invalid input_ids.")
            if len(input_batches) != len(batch):
                raise ValueError("the tokenizer returned an unexpected batch size.")
            for input_ids in input_batches:
                if isinstance(input_ids, (str, bytes)) or not isinstance(
                    input_ids,
                    Sequence,
                ):
                    raise ValueError("the tokenizer returned invalid input_ids.")
                if any(
                    isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    for token_id in input_ids
                ):
                    raise ValueError("the tokenizer returned invalid input_ids.")
                counts.append(len(input_ids))
        return counts

    @property
    def model_name(self) -> str:
        """Return the configured encoder model identifier."""
        return self._model_name

    @property
    def revision(self) -> str | None:
        """Return the optional pinned tokenizer revision."""
        return self._revision

    @property
    def use_fast(self) -> bool:
        """Return whether fast tokenizer loading was requested."""
        return self._use_fast

    @property
    def trust_remote_code(self) -> bool:
        """Return whether remote tokenizer code is allowed."""
        return self._trust_remote_code

    @property
    def add_special_tokens(self) -> bool:
        """Return whether special tokens are included in counts."""
        return self._add_special_tokens

    @property
    def tokenizer_class(self) -> str:
        """Return the concrete tokenizer class name."""
        return type(self._tokenizer).__name__

    @property
    def reported_model_max_length(self) -> object:
        """Return the tokenizer-reported limit without treating it as safe."""
        return getattr(self._tokenizer, "model_max_length", None)

    @property
    def encoder_special_token_overhead(self) -> int:
        """Return required single-sequence model tokens excluded from counts."""
        function = getattr(self._tokenizer, "num_special_tokens_to_add", None)
        if not callable(function):
            raise ValueError(
                "the tokenizer cannot report required special-token overhead."
            )
        overhead = function(pair=False)
        if isinstance(overhead, bool) or not isinstance(overhead, int) or overhead < 0:
            raise ValueError("the tokenizer reported invalid special-token overhead.")
        return overhead

    @property
    def document_prefix(self) -> str:
        """Return the prefix counted for encoder-ready document chunks."""
        return self._document_prefix

    @property
    def local_files_only(self) -> bool:
        """Return whether tokenizer loading is restricted to the local cache."""
        return self._local_files_only


@lru_cache(maxsize=4)
def get_token_counter(
    model_name: str,
    revision: str | None = None,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    add_special_tokens: bool = False,
    document_prefix: str = "passage: ",
    local_files_only: bool = False,
) -> TokenCounter:
    """Return a cached counter for reusable non-injected configurations."""
    return TokenCounter(
        model_name,
        revision=revision,
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        add_special_tokens=add_special_tokens,
        document_prefix=document_prefix,
        local_files_only=local_files_only,
    )


def count_tokens(text: str, model_name: str) -> int:
    """Count text with a cached tokenizer for the configured encoder."""
    return get_token_counter(model_name).count(text)


def count_words_as_tokens(text: str) -> int:
    """Count whitespace units only for the preserved baseline evaluation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return len(text.split())
