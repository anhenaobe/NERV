"""Small deterministic test doubles for tokenizer-based chunking tests."""


class DeterministicFakeTokenizer:
    """Encode each non-whitespace Unicode code point as one fake token."""

    model_max_length = 1_000_000_000_000_000_000

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """Mirror the configured encoder's two single-sequence model tokens."""
        return 2

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        text: str | list[str],
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, object]:
        """Return deterministic IDs while recording standard tokenizer options."""
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "truncation": truncation,
            }
        )

        def encode(value: str) -> list[int]:
            input_ids = [
                (ord(character) % 10_000) + 1
                for character in value
                if not character.isspace()
            ]
            if add_special_tokens:
                input_ids = [10_001, *input_ids, 10_002]
            return input_ids

        input_ids = (
            [encode(value) for value in text]
            if isinstance(text, list)
            else encode(text)
        )
        return {"input_ids": input_ids}
