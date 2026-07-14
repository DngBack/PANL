"""The fact base must be functional: one correct answer entity per subject per family."""

from __future__ import annotations

from collections import Counter

import pytest

from panl.data.facts import (
    RELATION_FAMILIES,
    TIER1_FACTS,
    check_fact_base,
    facts_by_family,
    slugify,
)


def test_fact_base_is_internally_consistent() -> None:
    assert check_fact_base() == []


def test_every_family_is_populated() -> None:
    grouped = facts_by_family()
    assert set(grouped) == set(RELATION_FAMILIES)
    for family, facts in grouped.items():
        # Fewer than ~20 facts per family cannot support two pairing rounds and three splits.
        assert len(facts) >= 20, f"{family} has only {len(facts)} facts"


def test_subjects_are_unique_within_a_family() -> None:
    counts = Counter((f.relation_family, f.subject_slug) for f in TIER1_FACTS)
    duplicates = [key for key, count in counts.items() if count > 1]
    assert duplicates == []


def test_answer_reuse_stays_inside_a_family() -> None:
    """Two facts may share an answer (two Dickens novels); the block builder must then
    refuse to pair them. Here we just record that ids remain family-scoped."""
    for fact in TIER1_FACTS:
        assert fact.answer_id.startswith(f"a:{fact.relation_family}:")
        assert fact.question_id.startswith(f"q:{fact.relation_family}:")


def test_ids_are_unique() -> None:
    assert len({f.fact_id for f in TIER1_FACTS}) == len(TIER1_FACTS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("France", "france"),
        ("Saudi Arabia", "saudi-arabia"),
        ("Reykjavík", "reykjavik"),
        ("Catch-22", "catch-22"),
        ("Rubik's Cube", "rubik-s-cube"),
        ("Ernő Rubik", "erno-rubik"),
        # "+" carries identity: folding it away would collide with "C programming language".
        ("C++ programming language", "c-plus-plus-programming-language"),
    ],
)
def test_slugify(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_slugify_keeps_c_and_cpp_apart() -> None:
    assert slugify("C programming language") != slugify("C++ programming language")


def test_slugs_do_not_collide_after_ascii_folding() -> None:
    """Accent folding could silently merge two distinct entities into one identity."""
    for family in RELATION_FAMILIES:
        facts = facts_by_family((family,))[family]
        subject_slugs = Counter(f.subject_slug for f in facts)
        assert [s for s, c in subject_slugs.items() if c > 1] == []
