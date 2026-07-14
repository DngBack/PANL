"""Splits are assigned to facts, stratified by family, so that identities cannot leak."""

from __future__ import annotations

import pytest

from panl.config import DataBuildConfig
from panl.data.build import build_rows
from panl.data.facts import Fact, facts_by_family
from panl.data.schema import Split
from panl.data.splits import FITTABLE_SPLITS, SplitRatios, answer_clusters, assign_facts


def _fact(subject: str, answer: str, family: str = "author_of") -> Fact:
    return Fact(relation_family=family, subject=subject, subject_phrase=subject, answer=answer)


class TestAnswerClusters:
    def test_facts_sharing_an_answer_cluster_together(self) -> None:
        """Two Dickens novels must move between splits as a unit, or the author identity
        would appear in both."""
        facts = [
            _fact("Great Expectations", "Charles Dickens"),
            _fact("Oliver Twist", "Charles Dickens"),
            _fact("Ulysses", "James Joyce"),
        ]
        clusters = answer_clusters(facts)
        assert sorted(len(c) for c in clusters) == [1, 2]

    def test_distinct_answers_stay_separate(self) -> None:
        facts = [_fact("A", "Alpha"), _fact("B", "Beta")]
        assert len(answer_clusters(facts)) == 2


class TestFactAssignment:
    def test_held_out_family_goes_entirely_to_holdout(self) -> None:
        grouped = facts_by_family()
        assignment = assign_facts(grouped, family_holdout=("inventor_of",), seed=1)
        for fact in grouped["inventor_of"]:
            assert assignment[fact.fact_id] is Split.FAMILY_HOLDOUT
        for fact in grouped["capital_of"]:
            assert assignment[fact.fact_id] is not Split.FAMILY_HOLDOUT

    def test_every_family_reaches_every_fittable_split(self) -> None:
        """The bug this guards: a family confined to one split is a held-out family in
        disguise, and train would never see it."""
        grouped = facts_by_family()
        assignment = assign_facts(grouped, family_holdout=("inventor_of",), seed=1)
        for family, facts in grouped.items():
            if family == "inventor_of":
                continue
            splits = {assignment[f.fact_id] for f in facts}
            assert set(FITTABLE_SPLITS) <= splits, f"{family} is missing from a split"

    def test_an_answer_identity_never_straddles_a_split(self) -> None:
        grouped = facts_by_family()
        assignment = assign_facts(grouped, family_holdout=("inventor_of",), seed=3)
        seen: dict[str, Split] = {}
        for facts in grouped.values():
            for fact in facts:
                split = assignment[fact.fact_id]
                if fact.answer_id in seen:
                    assert seen[fact.answer_id] is split, fact.answer_id
                else:
                    seen[fact.answer_id] = split

    def test_assignment_is_deterministic(self) -> None:
        grouped = facts_by_family()
        first = assign_facts(grouped, family_holdout=("inventor_of",), seed=5)
        second = assign_facts(grouped, family_holdout=("inventor_of",), seed=5)
        assert first == second

    def test_ratios_are_respected_within_each_family(self) -> None:
        grouped = facts_by_family()
        assignment = assign_facts(grouped, family_holdout=("inventor_of",), seed=1)
        for family, facts in grouped.items():
            if family == "inventor_of":
                continue
            train = sum(assignment[f.fact_id] is Split.TRAIN for f in facts)
            share = train / len(facts)
            assert 0.60 <= share <= 0.80, f"{family}: train share {share:.2f}"

    def test_a_family_too_small_to_split_fails_loudly(self) -> None:
        grouped = {"author_of": [_fact("A", "Alpha"), _fact("B", "Beta")]}
        with pytest.raises(ValueError, match="cannot form a block"):
            assign_facts(grouped, seed=1)

    def test_holding_out_every_family_is_rejected(self) -> None:
        grouped = facts_by_family(("capital_of",))
        with pytest.raises(ValueError, match="nothing left to fit on"):
            assign_facts(grouped, family_holdout=("capital_of",))


class TestBuiltRows:
    def test_no_identity_crosses_a_split(self, tiny_config: DataBuildConfig) -> None:
        rows, _ = build_rows(tiny_config)
        for key in ("question_id", "answer_id", "block_id"):
            seen: dict[str, Split] = {}
            for row in rows:
                identity = getattr(row, key)
                if identity in seen:
                    assert seen[identity] is row.split, f"{key} {identity} leaks across splits"
                else:
                    seen[identity] = row.split

    def test_every_split_is_populated(self, tiny_config: DataBuildConfig) -> None:
        rows, _ = build_rows(tiny_config)
        assert {row.split for row in rows} == {
            Split.TRAIN,
            Split.VALIDATION,
            Split.TEST,
            Split.FAMILY_HOLDOUT,
        }

    def test_every_fittable_family_appears_in_train(self, tiny_config: DataBuildConfig) -> None:
        rows, _ = build_rows(tiny_config)
        train_families = {r.relation_family for r in rows if r.split is Split.TRAIN}
        expected = set(tiny_config.families) - set(tiny_config.family_holdout)
        assert train_families == expected

    def test_a_block_never_straddles_two_splits(self, tiny_config: DataBuildConfig) -> None:
        rows, _ = build_rows(tiny_config)
        by_block: dict[str, set[Split]] = {}
        for row in rows:
            by_block.setdefault(row.block_id, set()).add(row.split)
        assert all(len(splits) == 1 for splits in by_block.values())


class TestSplitRatios:
    def test_ratios_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            SplitRatios(train=0.5, validation=0.2, test=0.2)

    def test_ratios_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="must be > 0"):
            SplitRatios(train=1.0, validation=0.0, test=0.0)
