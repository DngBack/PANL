"""The narrow tokenizer surface the rest of the package depends on.

Only a fast (Rust-backed) tokenizer can return `offset_mapping`, and offsets are what let us
resolve semantic positions from real tokenization instead of hard-coded indices. Everything
downstream therefore talks to this protocol, and `load_tokenizer` is the single place that
touches `transformers`.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable


@runtime_checkable
class FastTokenizer(Protocol):
    """Structural type for a HF fast tokenizer (satisfied by the test double as well)."""

    is_fast: bool

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = ...,
        return_offsets_mapping: bool = ...,
    ) -> Any: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> Any: ...

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]: ...

    def decode(self, ids: list[int], *, skip_special_tokens: bool = ...) -> str: ...


def load_tokenizer(model_id: str, *, revision: str | None = None) -> FastTokenizer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        msg = (
            f"{model_id} did not load a fast tokenizer; offset mappings are unavailable and "
            f"semantic positions cannot be resolved"
        )
        raise RuntimeError(msg)
    return cast(FastTokenizer, tokenizer)


def encode_with_offsets(
    tokenizer: FastTokenizer, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `text` verbatim, returning ids and per-token character spans.

    `add_special_tokens=False` because the prompt string already carries whatever special
    tokens the chat template inserted; letting the tokenizer add more would shift every
    position and change what the model actually sees.
    """
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = [int(i) for i in encoding["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]
    if len(input_ids) != len(offsets):
        msg = f"tokenizer returned {len(input_ids)} ids but {len(offsets)} offsets"
        raise RuntimeError(msg)
    return input_ids, offsets
