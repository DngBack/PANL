"""Scoring every candidate fact against the model, before any block is built.

Plan section 3.2 asks for the stress subsets to be created "after scoring all candidates".
This is that pass, and E0 showed why it cannot be skipped. Two things are measured per fact:

**Is the item on-policy?** What does the model answer when we let it answer? If its own greedy
answer is not the gold entity in the gold surface form, then teacher-forcing our gold is an
off-policy trajectory, and the answer's log-probability -- the nuisance variable the whole
fluency control rests on -- measures surprise at our wording instead of knowledge of the fact.

**Is the item saturated?** The confidence margin the model reports for the *correct* answer.
E0 on Qwen2.5-7B ran into a 30-logit clean gap because every fact was one the model knew cold;
in that regime the read-out is a step function and no single-position patch can move it. Blocks
have to be built from items that span the confidence range, not from items that are all easy.

The greedy answer also yields the naturalistic error set: where the model answers something
other than gold, that answer is by construction a high-likelihood wrong answer.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rich.progress import Progress

from panl.config import ExperimentConfig
from panl.data.facts import RELATION_FAMILIES, Fact, facts_by_family
from panl.data.templates import templates_for
from panl.models.adapter import HookedModelAdapter
from panl.models.batching import make_batches
from panl.models.positions import resolve_positions
from panl.models.prompts import PromptRenderer, PromptTemplate


def normalize_entity(text: str) -> str:
    """Casefold and strip accents, so "Reykjavik" and "Reykjavík" are one entity.

    Surface-form differences are not knowledge differences. Conflating them would file a
    correct answer in a rare spelling as a model error, and the low-likelihood-correct stress
    subset is precisely the set of items where that distinction matters most.
    """
    decomposed = unicodedata.normalize("NFKD", text.strip())
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().split())


@dataclass(slots=True)
class ScoringResult:
    scores: pd.DataFrame

    def summary(self) -> dict[str, Any]:
        frame = self.scores
        return {
            "n_facts": len(frame),
            "on_policy_exact": float(frame["greedy_exact"].mean()),
            "on_policy_entity": float(frame["greedy_entity"].mean()),
            "model_errors": int((~frame["greedy_entity"]).sum()),
            "median_gold_margin": float(frame["gold_margin"].median()),
            "unsaturated_fraction": float((frame["gold_margin"].abs() < 10).mean()),
        }


def score_facts(
    model: HookedModelAdapter,
    config: ExperimentConfig,
    *,
    template_index: int = 0,
    progress: Progress | None = None,
) -> ScoringResult:
    grouped = facts_by_family(config.families or RELATION_FAMILIES)
    facts: list[Fact] = [f for group in grouped.values() for f in group]

    renderer = PromptRenderer(
        model.tokenizer, template=PromptTemplate(), style=model.spec.prompt_style
    )
    questions = [templates_for(fact.relation_family)[template_index].render(fact) for fact in facts]

    # 1. What would the model say on its own?
    prefixes = [
        renderer.render(question, "X").text[
            : renderer.render(question, "X").anchors.answer_boundary
        ]
        for question in questions
    ]
    task = progress.add_task("greedy answers", total=1) if progress else None
    greedy = model.greedy_answers(prefixes)
    if progress and task is not None:
        progress.advance(task)

    # 2. How confident is it in the *gold* answer, and how likely does it find that answer?
    prompts = [renderer.render(q, f.answer) for q, f in zip(questions, facts, strict=True)]
    resolved = [resolve_positions(model.tokenizer, p) for p in prompts]

    margins = np.full(len(facts), np.nan)
    nll = np.full(len(facts), np.nan)
    n_answer_tokens = np.zeros(len(facts), dtype=int)

    batches = list(make_batches(resolved, max_batch_size=config.batch_size))
    task = progress.add_task("gold margins", total=len(batches)) if progress else None
    for batch in batches:
        out = model.run(batch)
        for offset, row in enumerate(batch.row_indices):
            margins[row] = float(out.confidence_margin[offset])
            logprobs = out.answer_logprobs[offset]
            nll[row] = float(-logprobs.float().mean())
            n_answer_tokens[row] = len(logprobs)
        if progress and task is not None:
            progress.advance(task)

    scores = pd.DataFrame(
        {
            "fact_id": [f.fact_id for f in facts],
            "relation_family": [f.relation_family for f in facts],
            "subject": [f.subject for f in facts],
            "gold": [f.answer for f in facts],
            "question": questions,
            "greedy_answer": greedy,
            "gold_margin": margins,
            "gold_nll_per_token": nll,
            "gold_n_tokens": n_answer_tokens,
        }
    )
    scores["greedy_exact"] = scores["greedy_answer"] == scores["gold"]
    scores["greedy_entity"] = [
        normalize_entity(g) == normalize_entity(a)
        for g, a in zip(scores["greedy_answer"], scores["gold"], strict=True)
    ]
    # An answer the model produces itself but that is not the gold entity is, by construction,
    # a high-likelihood wrong answer -- the stress subset of plan section 3.2.
    scores["is_model_error"] = ~scores["greedy_entity"]
    # Correct entity, wrong surface form: the alias case, and the one lever that decouples
    # `matched` from `correct` in the block design.
    scores["is_alias"] = scores["greedy_entity"] & ~scores["greedy_exact"]

    return ScoringResult(scores=scores)
