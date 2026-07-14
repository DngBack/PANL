"""Split assignment, performed at the level of *facts* and stratified by relation family.

Two design decisions, both learned the hard way:

**Split facts, not blocks.** Blocks are not independent -- a fact reused across pairing rounds
ties its blocks together. Splitting blocks and then trying to untangle the identity graph does
not work: with more than one pairing round the graph fuses into a handful of giant components,
and any component-level assignment drops whole families into a single split. Assigning facts
first makes question and answer identities disjoint across splits *by construction*, and blocks
are then built inside each pool.

**Stratify per family.** Otherwise a split can end up with no `author_of` blocks at all, which
silently turns the "unseen identities, known family" evaluation into a second held-out-family
evaluation. Every fittable family must appear in every fittable split.

Facts are clustered by answer identity before assignment, because two facts may share an answer
(two novels by the same author). Splitting such a pair would leak the answer identity even
though the questions differ.

Blocks inside one split still share facts when `pairings_per_fact > 1`, so the block bootstrap
of plan section 4.2 is approximately, not exactly, independent. Set `pairings_per_fact: 1` for
strictly independent blocks, at the cost of halving the block count.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from panl.data.facts import Fact
from panl.data.schema import Split

FITTABLE_SPLITS: tuple[Split, Split, Split] = (Split.TRAIN, Split.VALIDATION, Split.TEST)


@dataclass(frozen=True, slots=True)
class SplitRatios:
    train: float = 0.7
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        for name, value in (
            ("train", self.train),
            ("validation", self.validation),
            ("test", self.test),
        ):
            if value <= 0:
                msg = f"split ratio {name} must be > 0, got {value}"
                raise ValueError(msg)
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            msg = f"split ratios must sum to 1.0, got {total}"
            raise ValueError(msg)

    def target(self, split: Split, n_items: int) -> float:
        return {
            Split.TRAIN: self.train,
            Split.VALIDATION: self.validation,
            Split.TEST: self.test,
        }[split] * n_items


def answer_clusters(facts: Sequence[Fact]) -> list[list[Fact]]:
    """Group facts that share an answer identity; such facts cannot be split apart."""
    clusters: dict[str, list[Fact]] = {}
    for fact in facts:
        clusters.setdefault(fact.answer_id, []).append(fact)
    return [clusters[key] for key in sorted(clusters)]


def _distinct_answers(facts: Sequence[Fact]) -> int:
    return len({fact.answer_id for fact in facts})


def assign_facts(
    facts_by_family: dict[str, list[Fact]],
    *,
    family_holdout: tuple[str, ...] = (),
    ratios: SplitRatios | None = None,
    seed: int | str = 0,
) -> dict[str, Split]:
    """Map each `fact_id` to its split, stratified within each relation family."""
    ratios = ratios or SplitRatios()
    assignment: dict[str, Split] = {}

    fittable_families = [f for f in facts_by_family if f not in family_holdout]
    if not fittable_families:
        msg = "every relation family was held out; nothing left to fit on"
        raise ValueError(msg)

    for family, facts in facts_by_family.items():
        if family in family_holdout:
            for fact in facts:
                assignment[fact.fact_id] = Split.FAMILY_HOLDOUT
            continue

        clusters = answer_clusters(facts)
        # Shuffle first, then sort by size: the shuffle randomizes ties, the sort keeps the
        # packing balanced. Both are seeded, so the result is reproducible.
        rng = random.Random(f"{seed}:{family}")
        rng.shuffle(clusters)
        clusters.sort(key=len, reverse=True)

        counts: dict[Split, int] = dict.fromkeys(FITTABLE_SPLITS, 0)
        pools: dict[Split, list[Fact]] = {split: [] for split in FITTABLE_SPLITS}
        for cluster in clusters:
            deficits = [
                (ratios.target(split, len(facts)) - counts[split], -order, split)
                for order, split in enumerate(FITTABLE_SPLITS)
            ]
            _, _, chosen = max(deficits)
            for fact in cluster:
                assignment[fact.fact_id] = chosen
                pools[chosen].append(fact)
            counts[chosen] += len(cluster)

        # A pool needs two facts with two distinct answers, or no block can be built from it
        # and the family silently vanishes from that split.
        for split in FITTABLE_SPLITS:
            pool = pools[split]
            if len(pool) < 2 or _distinct_answers(pool) < 2:
                msg = (
                    f"{family}: the {split.value} pool holds {len(pool)} fact(s) with "
                    f"{_distinct_answers(pool)} distinct answer(s) and cannot form a block. "
                    f"The family has {len(facts)} facts; add more, or widen the split ratio."
                )
                raise ValueError(msg)

    return assignment
