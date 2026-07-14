"""Frozen position snapshots, one per tokenizer.

Plan section 7 asks CI to check that "prompt rendering and semantic token positions match
frozen snapshots per tokenizer". A snapshot pins the resolved indices and token ids for a
fixed set of cases, so that a transformers upgrade or a tokenizer revision bump that shifts
tokenization fails a cheap CPU test instead of quietly moving PANL under a GPU run.

The cases are chosen to stress the boundaries that byte-level BPE actually gets wrong:
accented characters (multi-byte, split across byte tokens), digits, apostrophes, multi-word
answers, and a crossed cell whose answer has nothing to do with the question.
"""

from __future__ import annotations

from typing import Any, Final

from panl.models.confidence import resolve_confidence_classes
from panl.models.positions import POSITION_NAMES, resolve_positions
from panl.models.prompts import PromptRenderer, PromptStyle, PromptTemplate
from panl.models.tokenizer import FastTokenizer

#: (question, answer). The last two are crossed cells: the answer is another block's gold.
SNAPSHOT_CASES: Final[tuple[tuple[str, str], ...]] = (
    ("What is the capital of France?", "Paris"),
    ("Which city is the capital of Australia?", "Canberra"),
    ("Name the capital city of Iceland.", "Reykjavík"),
    ("What is the official currency of Japan?", "Japanese yen"),
    ("Who wrote Nineteen Eighty-Four?", "George Orwell"),
    ("Who wrote Catch-22?", "Joseph Heller"),
    ("Who invented the Rubik's Cube?", "Ernő Rubik"),
    ("What is the capital of France?", "Tokyo"),
)


def build_snapshot(
    tokenizer: FastTokenizer,
    *,
    model_id: str,
    style: PromptStyle = PromptStyle.CHAT,
    template: PromptTemplate | None = None,
    cases: tuple[tuple[str, str], ...] = SNAPSHOT_CASES,
) -> dict[str, Any]:
    template = template or PromptTemplate()
    renderer = PromptRenderer(tokenizer, template=template, style=style)
    classes = resolve_confidence_classes(tokenizer)

    records: list[dict[str, Any]] = []
    for question, answer in cases:
        prompt = renderer.render(question, answer)
        resolved = resolve_positions(tokenizer, prompt)
        records.append(
            {
                "question": question,
                "answer": answer,
                "prompt_sha256": prompt.prompt_sha256,
                "n_tokens": resolved.n_tokens,
                "indices": {name: resolved.indices[name] for name in POSITION_NAMES},
                "token_ids": {name: resolved.token_ids[name] for name in POSITION_NAMES},
                "tokens": {name: resolved.tokens[name] for name in POSITION_NAMES},
            }
        )

    return {
        "model_id": model_id,
        "style": style.value,
        "template_hash": template.template_hash,
        "confidence_classes": {
            "high_token_id": classes.high_token_id,
            "low_token_id": classes.low_token_id,
            "high_token": classes.high_token,
            "low_token": classes.low_token,
        },
        "cases": records,
    }
