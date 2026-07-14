"""Block construction: the 2x2 crossed design and the pairing rules that protect it."""

from __future__ import annotations

import random

from panl.config import DataBuildConfig
from panl.data.blocks import build_blocks, can_pair, pair_facts
from panl.data.build import build_rows
from panl.data.facts import Fact, facts_by_family
from panl.data.schema import CELLS, AnswerSource, PolicyStatus, Split


def _fact(subject: str, answer: str, family: str = "capital_of", **kwargs: object) -> Fact:
    return Fact(
        relation_family=family,
        subject=subject,
        subject_phrase=subject,
        answer=answer,
        **kwargs,  # type: ignore[arg-type]
    )


class TestPairingRules:
    def test_rejects_a_fact_against_itself(self) -> None:
        fact = _fact("France", "Paris")
        assert not can_pair(fact, fact)

    def test_rejects_facts_that_share_an_answer(self) -> None:
        """Two Dickens novels cannot form a block: the crossed cells would both be correct."""
        a = _fact("Great Expectations", "Charles Dickens", family="author_of")
        b = _fact("Oliver Twist", "Charles Dickens", family="author_of")
        assert not can_pair(a, b)

    def test_rejects_a_pair_when_one_answer_is_also_correct_for_the_other(self) -> None:
        a = _fact("Country A", "Alpha", also_correct=("beta",))
        b = _fact("Country B", "Beta")
        assert not can_pair(a, b)
        assert not can_pair(b, a)

    def test_rejects_cross_family_pairs(self) -> None:
        a = _fact("France", "Paris", family="capital_of")
        b = _fact("Japan", "Japanese yen", family="currency_of")
        assert not can_pair(a, b)

    def test_accepts_a_clean_pair(self) -> None:
        assert can_pair(_fact("France", "Paris"), _fact("Japan", "Tokyo"))

    def test_pairings_are_deterministic_given_the_seed(self) -> None:
        facts = facts_by_family(("capital_of",))["capital_of"]
        first = pair_facts(facts, pairings_per_fact=2, rng=random.Random("seed:capital_of"))
        second = pair_facts(facts, pairings_per_fact=2, rng=random.Random("seed:capital_of"))
        assert [(a.fact_id, b.fact_id) for a, b in first] == [
            (a.fact_id, b.fact_id) for a, b in second
        ]

    def test_no_pair_is_produced_twice(self) -> None:
        facts = facts_by_family(("currency_of",))["currency_of"]
        pairs = pair_facts(facts, pairings_per_fact=3, rng=random.Random(0))
        keys = [(a.fact_id, b.fact_id) for a, b in pairs]
        assert len(keys) == len(set(keys))


class TestBlockStructure:
    def test_a_block_expands_to_the_full_2x2(self) -> None:
        grouped = {"capital_of": facts_by_family(("capital_of",))["capital_of"][:2]}
        block = build_blocks(grouped, pairings_per_fact=1, templates_per_family=1, seed=1)[0]
        rows = block.rows(dataset_tier=1, split=Split.TRAIN)

        assert len(rows) == 4
        assert {r.cell for r in rows} == set(CELLS)
        assert len({(r.question_id, r.answer_id) for r in rows}) == 4
        assert len({r.question_id for r in rows}) == 2
        assert len({r.answer_id for r in rows}) == 2

    def test_matched_cells_are_the_diagonal_and_are_correct(self) -> None:
        grouped = {"capital_of": facts_by_family(("capital_of",))["capital_of"][:2]}
        block = build_blocks(grouped, pairings_per_fact=1, templates_per_family=1, seed=1)[0]
        rows = {r.cell.value: r for r in block.rows(dataset_tier=1, split=Split.TRAIN)}

        for cell in ("q1a1", "q2a2"):
            assert rows[cell].matched
            assert rows[cell].correct
            assert rows[cell].answer_source is AnswerSource.GOLD
            assert rows[cell].policy_status is PolicyStatus.NATURAL

        for cell in ("q1a2", "q2a1"):
            assert not rows[cell].matched
            # The load-bearing property: a crossed cell is genuinely wrong.
            assert not rows[cell].correct
            assert rows[cell].answer_source is AnswerSource.DISTRACTOR
            assert rows[cell].policy_status is PolicyStatus.OFF_POLICY

    def test_each_identity_appears_once_matched_and_once_crossed(self) -> None:
        grouped = {"author_of": facts_by_family(("author_of",))["author_of"][:2]}
        block = build_blocks(grouped, pairings_per_fact=1, templates_per_family=1, seed=1)[0]
        rows = block.rows(dataset_tier=1, split=Split.TRAIN)

        for key in ("question_id", "answer_id"):
            by_identity: dict[str, list[bool]] = {}
            for row in rows:
                by_identity.setdefault(getattr(row, key), []).append(row.matched)
            for identity, flags in by_identity.items():
                assert sorted(flags) == [False, True], identity

    def test_both_questions_of_a_block_share_a_template(self) -> None:
        blocks = build_blocks(
            facts_by_family(("capital_of",)),
            pairings_per_fact=2,
            templates_per_family=3,
            seed=11,
        )
        for block in blocks:
            template_ids = {r.template_id for r in block.rows(dataset_tier=1, split=Split.TRAIN)}
            assert len(template_ids) == 1

    def test_templates_are_balanced_across_blocks(self) -> None:
        blocks = build_blocks(
            facts_by_family(("capital_of",)),
            pairings_per_fact=2,
            templates_per_family=3,
            seed=11,
        )
        counts: dict[str, int] = {}
        for block in blocks:
            counts[block.template.template_id] = counts.get(block.template.template_id, 0) + 1
        assert max(counts.values()) - min(counts.values()) <= 1


class TestFullBuild:
    def test_build_is_deterministic(self, tiny_config: DataBuildConfig) -> None:
        first, _ = build_rows(tiny_config)
        second, _ = build_rows(tiny_config)
        assert [r.model_dump() for r in first] == [r.model_dump() for r in second]

    def test_changing_the_seed_changes_the_pairing(self, tiny_config: DataBuildConfig) -> None:
        _, blocks_a = build_rows(tiny_config)
        _, blocks_b = build_rows(tiny_config.model_copy(update={"seed": 999}))
        assert {b.block_id for b in blocks_a} != {b.block_id for b in blocks_b}

    def test_row_count_is_four_per_block(self, tiny_config: DataBuildConfig) -> None:
        rows, blocks = build_rows(tiny_config)
        assert len(rows) == 4 * len(blocks)
