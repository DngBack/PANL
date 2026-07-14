"""Construction of crossed 2x2 blocks.

Each block holds two facts from the same relation family:

    (Q1, A1) matched      (Q1, A2) crossed
    (Q2, A1) crossed      (Q2, A2) matched

so every question identity and every answer identity occurs exactly once in each
condition. Two facts may only be paired when neither's answer is correct for the other's
subject -- otherwise a "crossed" cell would silently be a correct answer and the
interaction contrast would be measuring nothing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from panl.data.facts import Fact
from panl.data.schema import CELLS, AnswerSource, Cell, PolicyStatus, Quadruple, Split
from panl.data.templates import QuestionTemplate, templates_for


def can_pair(a: Fact, b: Fact) -> bool:
    """Whether two facts form a valid crossed block.

    Rejects a fact against itself, and any pair where swapping the answers would produce a
    crossed cell that is actually correct (identical answer entities, or an answer listed
    in the other subject's `also_correct`).
    """
    if a.relation_family != b.relation_family:
        return False
    if a.fact_id == b.fact_id:
        return False
    if a.answer_id in b.correct_answer_ids():
        return False
    return b.answer_id not in a.correct_answer_ids()


@dataclass(frozen=True, slots=True)
class Block:
    block_id: str
    relation_family: str
    fact_q1: Fact
    fact_q2: Fact
    template: QuestionTemplate

    def fact_for(self, index: int) -> Fact:
        return self.fact_q1 if index == 1 else self.fact_q2

    def rows(self, *, dataset_tier: int, split: Split) -> list[Quadruple]:
        rows: list[Quadruple] = []
        for cell in CELLS:
            q_fact = self.fact_for(cell.q_index)
            a_fact = self.fact_for(cell.a_index)
            # Derived from the fact base rather than from `cell`, so that the validator's
            # matched-vs-correct cross-check tests the data instead of restating it.
            correct = a_fact.answer_id in q_fact.correct_answer_ids()
            rows.append(
                Quadruple(
                    block_id=self.block_id,
                    relation_family=self.relation_family,
                    dataset_tier=dataset_tier,
                    cell=cell,
                    q_index=cell.q_index,
                    a_index=cell.a_index,
                    question_id=q_fact.question_id,
                    answer_id=a_fact.answer_id,
                    subject=q_fact.subject,
                    question=self.template.render(q_fact),
                    answer=a_fact.answer,
                    matched=cell.matched,
                    correct=correct,
                    answer_source=AnswerSource.GOLD if cell.matched else AnswerSource.DISTRACTOR,
                    policy_status=(
                        PolicyStatus.NATURAL if cell.matched else PolicyStatus.OFF_POLICY
                    ),
                    template_id=self.template.template_id,
                    split=split,
                )
            )
        return rows


def _make_block_id(family: str, a: Fact, b: Fact, template: QuestionTemplate) -> str:
    suffix = template.template_id.rsplit("/", maxsplit=1)[-1]
    return f"{family}/{a.subject_slug}__{b.subject_slug}/{suffix}"


def pair_facts(
    facts: list[Fact],
    *,
    pairings_per_fact: int,
    rng: random.Random,
) -> list[tuple[Fact, Fact]]:
    """Pair facts within one family over `pairings_per_fact` greedy matching rounds.

    Each round shuffles the facts and greedily matches them, skipping pairs already used
    and pairs rejected by `can_pair`. A fact that finds no partner in a round is simply
    unpaired that round, so the yield can be slightly below the nominal
    `len(facts) * pairings_per_fact / 2`.
    """
    if pairings_per_fact < 1:
        msg = f"pairings_per_fact must be >= 1, got {pairings_per_fact}"
        raise ValueError(msg)

    used: set[tuple[str, str]] = set()
    pairs: list[tuple[Fact, Fact]] = []

    for _ in range(pairings_per_fact):
        pool = list(facts)
        rng.shuffle(pool)
        while len(pool) >= 2:
            first = pool.pop(0)
            partner_at: int | None = None
            for index, candidate in enumerate(pool):
                key = tuple(sorted((first.fact_id, candidate.fact_id)))
                if can_pair(first, candidate) and key not in used:
                    partner_at = index
                    break
            if partner_at is None:
                continue
            second = pool.pop(partner_at)
            left, right = sorted((first, second), key=lambda f: f.fact_id)
            used.add((left.fact_id, right.fact_id))
            pairs.append((left, right))

    return pairs


def build_blocks(
    facts_by_family: dict[str, list[Fact]],
    *,
    pairings_per_fact: int,
    templates_per_family: int,
    seed: int | str,
) -> list[Block]:
    """Build every block, assigning templates round-robin within each family.

    Round-robin (rather than random) template assignment keeps the templates balanced
    across blocks, which matters because the template is a covariate in the E1 model.

    Callers pass one pool at a time (a family restricted to a single split), so pairs never
    straddle a split boundary -- see `panl.data.splits`.
    """
    blocks: list[Block] = []
    for family, facts in facts_by_family.items():
        # A per-family stream keeps a change in one family from reshuffling the others.
        rng = random.Random(f"{seed}:{family}")
        templates = templates_for(family, limit=templates_per_family)
        pairs = pair_facts(facts, pairings_per_fact=pairings_per_fact, rng=rng)
        for index, (left, right) in enumerate(pairs):
            template = templates[index % len(templates)]
            blocks.append(
                Block(
                    block_id=_make_block_id(family, left, right, template),
                    relation_family=family,
                    fact_q1=left,
                    fact_q2=right,
                    template=template,
                )
            )

    seen: set[str] = set()
    for block in blocks:
        if block.block_id in seen:
            msg = f"duplicate block_id generated: {block.block_id}"
            raise ValueError(msg)
        seen.add(block.block_id)

    return blocks


__all__ = ["Block", "Cell", "build_blocks", "can_pair", "pair_facts"]
