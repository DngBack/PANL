"""Confidence class tokens and the pre-softmax margin.

The primary behavioural outcome is a *pre-softmax* class-logit margin, not the reported
confidence number: plan section 4.2 notes that output nonlinearities can manufacture a
non-zero difference-in-differences even when the underlying computation is additive.

The class-token mapping is frozen before any experiment runs, so it has to be verified: if
" high" were not a single token, the logit at CC would score a word fragment and the margin
would silently mean something else.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from panl.models.tokenizer import FastTokenizer


class ConfidenceTokenError(RuntimeError):
    """A confidence class does not map onto exactly one token for this tokenizer."""


@dataclass(frozen=True, slots=True)
class ConfidenceClasses:
    high_text: str
    low_text: str
    high_token_id: int
    low_token_id: int
    high_token: str
    low_token: str

    def margin(self, logits: Sequence[float]) -> float:
        """z = logit(high) - logit(low) at the CC position. Higher means more confident."""
        return float(logits[self.high_token_id]) - float(logits[self.low_token_id])


def resolve_confidence_classes(
    tokenizer: FastTokenizer,
    *,
    high: str = "high",
    low: str = "low",
    leading_space: bool = True,
) -> ConfidenceClasses:
    """Resolve the single-token continuations of the prompt for each confidence class.

    The prompt ends at "Confidence:", so the continuation carries the separating space and
    the class words are looked up as " high" / " low".
    """
    prefix = " " if leading_space else ""
    resolved: dict[str, tuple[int, str]] = {}

    for label, word in (("high", high), ("low", low)):
        text = f"{prefix}{word}"
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=False)
        ids = [int(i) for i in encoding["input_ids"]]
        if len(ids) != 1:
            pieces = tokenizer.convert_ids_to_tokens(ids)
            msg = (
                f"the {label} confidence class {text!r} tokenizes to {len(ids)} tokens "
                f"({pieces}), not 1. A multi-token class cannot be scored by a single logit "
                f"at CC; pick a different class word or drop the leading space."
            )
            raise ConfidenceTokenError(msg)
        resolved[label] = (ids[0], tokenizer.convert_ids_to_tokens(ids)[0])

    (high_id, high_token) = resolved["high"]
    (low_id, low_token) = resolved["low"]
    if high_id == low_id:
        msg = f"the two confidence classes map to the same token id {high_id}"
        raise ConfidenceTokenError(msg)

    return ConfidenceClasses(
        high_text=f"{prefix}{high}",
        low_text=f"{prefix}{low}",
        high_token_id=high_id,
        low_token_id=low_id,
        high_token=high_token,
        low_token=low_token,
    )
