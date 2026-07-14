"""The validator is only worth anything if it fails on bad data.

Every test here corrupts a valid table in one specific way and asserts the corresponding
violation is reported. A validator that only ever sees clean data is untested.
"""

from __future__ import annotations

import pandas as pd
import pytest

from panl.config import DataBuildConfig
from panl.data.build import build_rows
from panl.data.schema import COLUMNS, Split, to_table
from panl.data.validate import validate_frame


@pytest.fixture
def frame(tiny_config: DataBuildConfig) -> pd.DataFrame:
    rows, _ = build_rows(tiny_config)
    return to_table(rows).to_pandas()


def _first_block(frame: pd.DataFrame) -> str:
    return str(frame.loc[frame["split"] == Split.TRAIN.value, "block_id"].iloc[0])


def test_a_freshly_built_table_is_clean(frame: pd.DataFrame) -> None:
    report = validate_frame(frame)
    assert report.ok, report.summary()
    assert report.n_rows == 4 * report.n_blocks


def test_empty_table_is_rejected() -> None:
    report = validate_frame(pd.DataFrame(columns=list(COLUMNS)))
    assert not report.ok
    assert "table is empty" in report.violations


def test_missing_cell_is_caught(frame: pd.DataFrame) -> None:
    block = _first_block(frame)
    corrupted = frame.drop(frame[(frame["block_id"] == block) & (frame["cell"] == "q1a2")].index)
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("has 3 rows, expected exactly 4" in v for v in report.violations)


def test_duplicated_cell_is_caught(frame: pd.DataFrame) -> None:
    block = _first_block(frame)
    rows = frame[frame["block_id"] == block]
    duplicate = rows[rows["cell"] == "q1a1"]
    corrupted = pd.concat([frame.drop(rows[rows["cell"] == "q2a2"].index), duplicate])
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("expected the 2x2 set" in v for v in report.violations)


def test_flipped_matched_label_is_caught(frame: pd.DataFrame) -> None:
    corrupted = frame.copy()
    target = corrupted[
        (corrupted["block_id"] == _first_block(frame)) & (corrupted["cell"] == "q1a1")
    ]
    corrupted.loc[target.index, "matched"] = False
    report = validate_frame(corrupted)
    assert not report.ok
    # The row-level model catches the cell/label contradiction before the block check does.
    assert any("cell" in v and "matched" in v for v in report.violations)


def test_crossed_cell_labelled_correct_is_caught(frame: pd.DataFrame) -> None:
    """The failure mode that silently voids the experiment: a 'wrong' answer that is right."""
    corrupted = frame.copy()
    target = corrupted[corrupted["cell"] == "q1a2"].index[:1]
    corrupted.loc[target, "correct"] = True
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("alias" in v for v in report.violations)


def test_block_straddling_two_splits_is_caught(frame: pd.DataFrame) -> None:
    corrupted = frame.copy()
    block = _first_block(frame)
    target = corrupted[(corrupted["block_id"] == block) & (corrupted["cell"] == "q2a2")].index
    corrupted.loc[target, "split"] = Split.TEST.value
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("leaks across splits" in v for v in report.violations)


def test_question_identity_leaking_across_splits_is_caught(frame: pd.DataFrame) -> None:
    """Move one whole block to test; if it shares a question identity with a train block,
    the identity now spans both splits and must be reported."""
    corrupted = frame.copy()
    train = corrupted[corrupted["split"] == Split.TRAIN.value]
    counts = train.groupby("question_id")["block_id"].nunique()
    shared = counts[counts > 1]
    if shared.empty:
        pytest.skip("no question identity is reused inside train in this build")
    question_id = str(shared.index[0])
    block = str(train.loc[train["question_id"] == question_id, "block_id"].iloc[0])
    corrupted.loc[corrupted["block_id"] == block, "split"] = Split.TEST.value

    report = validate_frame(corrupted)
    assert not report.ok
    assert any(f"question identity {question_id} leaks" in v for v in report.violations)


def test_held_out_family_appearing_in_train_is_caught(frame: pd.DataFrame) -> None:
    corrupted = frame.copy()
    held = corrupted[corrupted["split"] == Split.FAMILY_HOLDOUT.value]
    block = str(held["block_id"].iloc[0])
    corrupted.loc[corrupted["block_id"] == block, "split"] = Split.TRAIN.value
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("is held out but also appears" in v for v in report.violations)


def test_family_missing_from_a_split_is_caught(frame: pd.DataFrame) -> None:
    """A family confined to validation and test is a held-out family in disguise: train
    would never see it. An earlier block-level splitter did exactly this to author_of."""
    corrupted = frame.copy()
    train_authors = (corrupted["relation_family"] == "author_of") & (
        corrupted["split"] == Split.TRAIN.value
    )
    corrupted.loc[train_authors, "split"] = Split.TEST.value
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("author_of is absent from ['train']" in v for v in report.violations)


def test_template_varying_inside_a_block_is_caught(frame: pd.DataFrame) -> None:
    corrupted = frame.copy()
    target = corrupted[corrupted["block_id"] == _first_block(frame)].index[:1]
    corrupted.loc[target, "template_id"] = "capital_of/t2"
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("template_id is not constant" in v for v in report.violations)


def test_unfilled_placeholder_is_caught(frame: pd.DataFrame) -> None:
    corrupted = frame.copy()
    corrupted.loc[corrupted.index[:1], "question"] = "What is the capital of {s}?"
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("unfilled placeholder" in v for v in report.violations)


def test_answer_identity_with_two_surface_forms_is_caught(frame: pd.DataFrame) -> None:
    corrupted = frame.copy()
    answer_id = str(corrupted["answer_id"].iloc[0])
    rows = corrupted[corrupted["answer_id"] == answer_id].index
    corrupted.loc[rows[:1], "answer"] = "Something Else"
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("more than one answer string" in v for v in report.violations)


def test_unknown_enum_value_is_caught(frame: pd.DataFrame) -> None:
    corrupted = frame.copy()
    corrupted.loc[corrupted.index[:1], "policy_status"] = "wishful_thinking"
    report = validate_frame(corrupted)
    assert not report.ok
    assert any("policy_status" in v for v in report.violations)
