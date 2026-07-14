"""Prompt rendering with character-level anchors for the semantic positions.

The prompt ends exactly at the confidence colon, so the next-token distribution at the final
position *is* the verbal-confidence distribution. The answer is teacher-forced into the
assistant turn, which is what puts AC, LAT, PANL, PANL+1 and CC all inside a single
contiguous span we control.

Rendering records character spans as it builds the string; `positions.py` then maps those
spans onto real tokens. Nothing here assumes anything about how the text will be tokenized.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from panl.models.tokenizer import FastTokenizer

#: Chosen by measurement, not by taste. Under a looser system prompt Qwen2.5-7B answers
#: "The capital of France is Paris." -- so teacher-forcing " Paris" produced an *off-policy*
#: trajectory, and the answer's log-probability then measured surprise at the format rather
#: than knowledge of the fact, which silently invalidates the fluency controls of plan section
#: 4.3. Under this prompt the bare entity is what greedy decoding already emits.
#:
#: Any change here must be re-checked with `panl score`, which reports how often the model's
#: own greedy answer is the gold entity in the gold surface form.
DEFAULT_SYSTEM: Final[str] = (
    "Answer factual questions in exactly this format, and nothing else:\n"
    "Answer: <the answer entity only -- no sentence, no punctuation, no explanation>\n"
    "Confidence: <high or low>\n"
    "Never restate the question. Never write a full sentence."
)
DEFAULT_INSTRUCTION: Final[str] = (
    "Answer each question with the answer entity only -- no sentence, no explanation. "
    "Then state your confidence in the answer as either high or low."
)


class PromptStyle(StrEnum):
    #: Use the model's chat template; the answer is a teacher-forced assistant turn.
    CHAT = "chat"
    #: Plain-text completion, for base models without a chat template.
    RAW = "raw"


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    system: str = DEFAULT_SYSTEM
    instruction: str = DEFAULT_INSTRUCTION
    question_prefix: str = "Question:"
    answer_prefix: str = "Answer:"
    confidence_prefix: str = "Confidence:"
    separator: str = " "
    newline: str = "\n"

    def __post_init__(self) -> None:
        # The anchors are defined as "the colon of X", so the colon has to be there.
        if not self.answer_prefix.endswith(":"):
            msg = f"answer_prefix must end with ':' (the AC anchor), got {self.answer_prefix!r}"
            raise ValueError(msg)
        if not self.confidence_prefix.endswith(":"):
            msg = (
                f"confidence_prefix must end with ':' (the CC anchor), "
                f"got {self.confidence_prefix!r}"
            )
            raise ValueError(msg)
        if self.newline != "\n":
            msg = f"newline must be a single '\\n' (the PANL anchor), got {self.newline!r}"
            raise ValueError(msg)

    @property
    def template_hash(self) -> str:
        payload = "\x1f".join(
            (
                self.system,
                self.instruction,
                self.question_prefix,
                self.answer_prefix,
                self.confidence_prefix,
                self.separator,
                self.newline,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class PromptAnchors:
    """Character offsets into the rendered prompt. All are absolute, not relative."""

    #: The ':' of "Answer:". The model's next token after this is the answer.
    answer_colon: int
    #: End of "Answer:" -- no token may span this boundary into the answer.
    answer_boundary: int
    answer_start: int
    answer_end: int
    #: The '\n' that follows the answer. This is PANL.
    newline: int
    #: The ':' of "Confidence:", i.e. the final character of the prompt.
    confidence_colon: int


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    anchors: PromptAnchors
    question: str
    answer: str
    style: PromptStyle
    template_hash: str

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class _Builder:
    """Appends text while tracking the character span of each piece."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0

    def add(self, piece: str) -> tuple[int, int]:
        start = self._length
        self._parts.append(piece)
        self._length += len(piece)
        return start, self._length

    def text(self) -> str:
        return "".join(self._parts)


class PromptRenderer:
    def __init__(
        self,
        tokenizer: FastTokenizer | None = None,
        *,
        template: PromptTemplate | None = None,
        style: PromptStyle = PromptStyle.CHAT,
    ) -> None:
        if style is PromptStyle.CHAT and tokenizer is None:
            msg = "chat style needs a tokenizer to apply the model's chat template"
            raise ValueError(msg)
        self.tokenizer = tokenizer
        self.template = template or PromptTemplate()
        self.style = style

    def _prefix(self, question: str) -> str:
        template = self.template
        if self.style is PromptStyle.RAW:
            return (
                f"{template.instruction}{template.newline}{template.newline}"
                f"{template.question_prefix}{template.separator}{question}{template.newline}"
            )

        assert self.tokenizer is not None  # guaranteed by __init__
        messages = [
            {"role": "system", "content": template.system},
            {
                "role": "user",
                "content": f"{template.question_prefix}{template.separator}{question}",
            },
        ]
        # add_generation_prompt=True stops the template at the start of the assistant turn,
        # so we can append the forced answer without a stray end-of-turn token in between.
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(prefix, str):
            msg = f"apply_chat_template returned {type(prefix).__name__}, expected str"
            raise TypeError(msg)
        return prefix

    def render(self, question: str, answer: str) -> RenderedPrompt:
        if not question.strip() or not answer.strip():
            msg = "question and answer must both be non-empty"
            raise ValueError(msg)

        template = self.template
        builder = _Builder()
        builder.add(self._prefix(question))

        _, answer_boundary = builder.add(template.answer_prefix)
        builder.add(template.separator)
        answer_start, answer_end = builder.add(answer)
        newline_start, _ = builder.add(template.newline)
        _, confidence_end = builder.add(template.confidence_prefix)

        anchors = PromptAnchors(
            answer_colon=answer_boundary - 1,
            answer_boundary=answer_boundary,
            answer_start=answer_start,
            answer_end=answer_end,
            newline=newline_start,
            confidence_colon=confidence_end - 1,
        )
        return RenderedPrompt(
            text=builder.text(),
            anchors=anchors,
            question=question,
            answer=answer,
            style=self.style,
            template_hash=template.template_hash,
        )
