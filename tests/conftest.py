"""Shared fixtures.

`FakeTokenizer` is a byte-level-BPE-ish test double: it reproduces the boundary behaviour
that matters (a leading space joins the following word; a newline is its own token; special
tokens are atomic) without any network access, and it can be told to *misbehave* in exactly
the ways a real tokenizer might. That second part is the point -- the resolver's job is to
refuse a tokenizer that destroys a semantic boundary, and the only way to test refusal is to
hand it one that does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from panl.config import DataBuildConfig, SplitRatioConfig
from panl.models.tokenizer import FastTokenizer

SPECIAL_TOKENS = ("<|im_start|>", "<|im_end|>")


@dataclass
class FakeTokenizer:
    """A deterministic offline stand-in for a Hugging Face fast tokenizer.

    Args:
        merge_newline_into_next: emit "\\nWord" as a single token, like a tokenizer whose
            pre-tokenizer does not split on newlines. Destroys PANL.
        merge_answer_into_newline: emit "Answer\\n" as a single token. Destroys LAT.
        merge_confidence_prefix: emit "Confidence:" as one token. Collapses PANL+1 onto CC.
        special_token_offsets: HF fast tokenizers disagree on whether added tokens get real
            offsets or (0, 0); default to (0, 0), the harder case.
    """

    merge_newline_into_next: bool = False
    merge_answer_into_newline: bool = False
    merge_confidence_prefix: bool = False
    zero_width_special_offsets: bool = True
    is_fast: bool = True
    _vocab: dict[str, int] = field(default_factory=dict)

    def _id_for(self, piece: str) -> int:
        return self._vocab.setdefault(piece, len(self._vocab))

    def _pieces(self, text: str) -> list[tuple[str, int, int]]:
        pieces: list[tuple[str, int, int]] = []
        i = 0
        n = len(text)
        while i < n:
            special = next((s for s in SPECIAL_TOKENS if text.startswith(s, i)), None)
            if special is not None:
                pieces.append((special, i, i + len(special)))
                i += len(special)
                continue

            if self.merge_confidence_prefix and text.startswith("Confidence:", i):
                pieces.append(("Confidence:", i, i + len("Confidence:")))
                i += len("Confidence:")
                continue

            char = text[i]

            if char == "\n":
                start = i
                i += 1
                if self.merge_newline_into_next:
                    while i < n and (text[i].isalnum() or text[i] == "_"):
                        i += 1
                pieces.append((text[start:i], start, i))
                continue

            if char == " " and i + 1 < n and text[i + 1].isalnum():
                start = i
                i += 1
                while i < n and text[i].isalnum():
                    i += 1
                if self.merge_answer_into_newline and i < n and text[i] == "\n":
                    i += 1
                pieces.append((text[start:i], start, i))
                continue

            if char.isalnum():
                start = i
                while i < n and text[i].isalnum():
                    i += 1
                if self.merge_answer_into_newline and i < n and text[i] == "\n":
                    i += 1
                pieces.append((text[start:i], start, i))
                continue

            pieces.append((char, i, i + 1))
            i += 1

        return pieces

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        pieces = self._pieces(text)
        encoding: dict[str, Any] = {"input_ids": [self._id_for(p) for p, _, _ in pieces]}
        if return_offsets_mapping:
            encoding["offset_mapping"] = [
                (0, 0) if (self.zero_width_special_offsets and p in SPECIAL_TOKENS) else (s, e)
                for p, s, e in pieces
            ]
        return encoding

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        inverse = {v: k for k, v in self._vocab.items()}
        return [inverse[i] for i in ids]

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        pieces = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            pieces = [p for p in pieces if p not in SPECIAL_TOKENS]
        return "".join(pieces)

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> str:
        if tokenize:
            raise NotImplementedError("the double only renders templates to text")
        parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conversation]
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


@pytest.fixture
def fake_tokenizer() -> FastTokenizer:
    return FakeTokenizer()


@pytest.fixture
def tiny_config() -> DataBuildConfig:
    """A small but structurally complete build: two families, one held out."""
    return DataBuildConfig(
        tier=1,
        seed=7,
        families=("capital_of", "currency_of", "author_of", "inventor_of"),
        pairings_per_fact=2,
        templates_per_family=3,
        family_holdout=("inventor_of",),
        split_ratios=SplitRatioConfig(train=0.7, validation=0.15, test=0.15),
    )
