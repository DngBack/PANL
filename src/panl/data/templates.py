"""Question paraphrase templates, one set per relation family.

A template is assigned per *block*, not per question. Both questions of a block therefore
share a template, which keeps the template out of the within-block Q x A contrast: it
becomes a block-level covariate instead of something confounded with the question factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from panl.data.facts import Fact


@dataclass(frozen=True, slots=True)
class QuestionTemplate:
    template_id: str
    #: Format string with a single `{s}` placeholder for the fact's `subject_phrase`.
    text: str

    def render(self, fact: Fact) -> str:
        return self.text.format(s=fact.subject_phrase)


TEMPLATES: Final[dict[str, tuple[QuestionTemplate, ...]]] = {
    "capital_of": (
        QuestionTemplate("capital_of/t0", "What is the capital of {s}?"),
        QuestionTemplate("capital_of/t1", "Which city is the capital of {s}?"),
        QuestionTemplate("capital_of/t2", "Name the capital city of {s}."),
    ),
    "currency_of": (
        QuestionTemplate("currency_of/t0", "What is the official currency of {s}?"),
        QuestionTemplate("currency_of/t1", "Which currency is used in {s}?"),
        QuestionTemplate("currency_of/t2", "Name the official currency of {s}."),
    ),
    "author_of": (
        QuestionTemplate("author_of/t0", "Who wrote {s}?"),
        QuestionTemplate("author_of/t1", "Which author wrote {s}?"),
        QuestionTemplate("author_of/t2", "Name the author of {s}."),
    ),
    "inventor_of": (
        QuestionTemplate("inventor_of/t0", "Who invented {s}?"),
        QuestionTemplate("inventor_of/t1", "Which inventor created {s}?"),
        QuestionTemplate("inventor_of/t2", "Name the inventor of {s}."),
    ),
}


def templates_for(family: str, limit: int | None = None) -> tuple[QuestionTemplate, ...]:
    try:
        available = TEMPLATES[family]
    except KeyError as exc:
        msg = f"no question templates for relation family {family!r}"
        raise KeyError(msg) from exc
    if limit is None:
        return available
    if not 1 <= limit <= len(available):
        msg = f"{family}: requested {limit} templates, only {len(available)} exist"
        raise ValueError(msg)
    return available[:limit]
