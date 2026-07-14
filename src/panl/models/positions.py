"""Resolution of the semantic positions AC, LAT, PANL, PANL+1 and CC.

Positions are derived from the tokenizer's own character offsets, never from hard-coded
indices, because BPE merges are tokenizer-specific: whether "\\nConfidence" is one token or
two is a property of the pre-tokenizer regex, not something a position constant can encode.

Every semantic boundary is checked rather than assumed. A tokenizer that glues the answer to
the following newline, or the newline to the word after it, destroys the very distinction the
experiment rests on -- PANL would no longer be a position that sees the completed answer and
nothing else. In that case we raise instead of silently returning a plausible index. This is
the "stop if the target prompts cannot be reproduced" gate of the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from panl.models.prompts import RenderedPrompt
from panl.models.tokenizer import FastTokenizer, encode_with_offsets

#: Ordered, and the order is asserted: AC precedes the answer, PANL follows it, CC ends it.
POSITION_NAMES: Final[tuple[str, ...]] = ("AC", "LAT", "PANL", "PANL1", "CC")


class PositionResolutionError(RuntimeError):
    """A semantic position could not be resolved for this tokenizer and prompt."""


@dataclass(frozen=True, slots=True)
class ResolvedPositions:
    indices: dict[str, int]
    token_ids: dict[str, int]
    tokens: dict[str, str]
    input_ids: tuple[int, ...]
    #: [start, end) token span of the answer itself. The conditional-likelihood controls of
    #: plan section 4.3 are computed over exactly these tokens.
    answer_span: tuple[int, int]

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def n_answer_tokens(self) -> int:
        return self.answer_span[1] - self.answer_span[0]

    def as_record(self) -> dict[str, object]:
        """Flat form for a result row: indices and token ids per plan section 4.1."""
        record: dict[str, object] = {"n_tokens": self.n_tokens}
        for name in POSITION_NAMES:
            record[f"pos_{name}"] = self.indices[name]
            record[f"tok_{name}"] = self.token_ids[name]
        return record


def _char_to_token(offsets: list[tuple[int, int]], n_chars: int) -> list[int]:
    """Map each character index to the token covering it, or -1 if none.

    Zero-width spans are skipped: fast tokenizers report `(0, 0)` for special tokens such as
    `<|im_start|>`, and treating those as covering character 0 would misattribute anchors.
    """
    mapping = [-1] * n_chars
    for index, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        for char in range(start, min(end, n_chars)):
            mapping[char] = index
    return mapping


def _describe(tokens: list[str], index: int, radius: int = 3) -> str:
    low = max(0, index - radius)
    high = min(len(tokens), index + radius + 1)
    pieces = [f"[{i}]{tokens[i]!r}" + ("*" if i == index else "") for i in range(low, high)]
    return " ".join(pieces)


def resolve_positions(tokenizer: FastTokenizer, prompt: RenderedPrompt) -> ResolvedPositions:
    input_ids, offsets = encode_with_offsets(tokenizer, prompt.text)
    tokens = tokenizer.convert_ids_to_tokens(list(input_ids))
    anchors = prompt.anchors
    char_map = _char_to_token(offsets, len(prompt.text))

    def token_at(char: int, what: str) -> int:
        index = char_map[char]
        if index < 0:
            msg = f"{what}: character {char} ({prompt.text[char]!r}) is not covered by any token"
            raise PositionResolutionError(msg)
        return index

    ac = token_at(anchors.answer_colon, "AC")
    if offsets[ac][1] != anchors.answer_boundary:
        raise PositionResolutionError(
            f"AC: the token holding the answer colon spans "
            f"{offsets[ac]} and runs past the colon into the answer -- the model would see "
            f"answer text at the position that is supposed to precede it. "
            f"Tokens: {_describe(tokens, ac)}"
        )

    first_answer = token_at(anchors.answer_start, "answer start")
    if offsets[first_answer][0] < anchors.answer_boundary:
        raise PositionResolutionError(
            f"answer start: the first answer token spans {offsets[first_answer]} and swallows "
            f"the 'Answer:' colon. Tokens: {_describe(tokens, first_answer)}"
        )

    lat = token_at(anchors.answer_end - 1, "LAT")
    if offsets[lat][1] != anchors.answer_end:
        raise PositionResolutionError(
            f"LAT: the last answer token spans {offsets[lat]} and continues past the end of "
            f"the answer (char {anchors.answer_end}) -- it is merged with the newline. "
            f"Tokens: {_describe(tokens, lat)}"
        )

    panl = token_at(anchors.newline, "PANL")
    if offsets[panl] != (anchors.newline, anchors.newline + 1):
        raise PositionResolutionError(
            f"PANL: the post-answer newline is not its own token; it is part of "
            f"{tokens[panl]!r} spanning {offsets[panl]}. PANL must see the completed answer "
            f"and nothing beyond it. Tokens: {_describe(tokens, panl)}"
        )

    panl1 = panl + 1
    if panl1 >= len(input_ids):
        raise PositionResolutionError("PANL+1: the prompt ends at PANL; no control position exists")

    cc = token_at(anchors.confidence_colon, "CC")
    if offsets[cc][1] != anchors.confidence_colon + 1:
        raise PositionResolutionError(
            f"CC: the token holding the confidence colon spans {offsets[cc]} and does not end "
            f"at the colon. Tokens: {_describe(tokens, cc)}"
        )
    if cc != len(input_ids) - 1:
        raise PositionResolutionError(
            f"CC: the confidence colon is token {cc} but the prompt has {len(input_ids)} "
            f"tokens; the prompt must end at CC so that the next-token distribution is the "
            f"confidence distribution. Tokens: {_describe(tokens, cc)}"
        )

    # PANL+1 must be a usable control: distinct from PANL and from the read-out position.
    if not ac < lat < panl < panl1 < cc:
        raise PositionResolutionError(
            f"positions are not strictly ordered: AC={ac} LAT={lat} PANL={panl} "
            f"PANL1={panl1} CC={cc}. PANL+1 must fall strictly between PANL and CC to serve "
            f"as a control position."
        )

    indices = {"AC": ac, "LAT": lat, "PANL": panl, "PANL1": panl1, "CC": cc}
    return ResolvedPositions(
        indices=indices,
        token_ids={name: input_ids[i] for name, i in indices.items()},
        tokens={name: tokens[i] for name, i in indices.items()},
        input_ids=tuple(input_ids),
        answer_span=(first_answer, lat + 1),
    )
