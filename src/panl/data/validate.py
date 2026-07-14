"""Invariants the quadruple table must satisfy before any GPU time is spent on it.

These are the CI gates listed in plan section 7. They exist because each failure mode here
is silent: a block with a duplicated cell, a question identity straddling train and test, or
a "crossed" answer that happens to be correct will all produce a perfectly well-formed table
and a meaningless interaction estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pydantic import ValidationError

from panl.data.schema import CELLS, AnswerSource, Quadruple, Split, read_table


@dataclass(slots=True)
class ValidationReport:
    n_rows: int = 0
    n_blocks: int = 0
    counts_by_split: dict[str, int] = field(default_factory=dict)
    counts_by_family: dict[str, int] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def summary(self) -> str:
        head = (
            f"{self.n_rows} rows / {self.n_blocks} blocks | "
            f"splits={self.counts_by_split} | families={self.counts_by_family}"
        )
        if self.ok:
            return f"{head}\nOK: all invariants hold."
        lines = "\n".join(f"  - {v}" for v in self.violations)
        return f"{head}\nFAILED with {len(self.violations)} violation(s):\n{lines}"


_EXPECTED_CELLS = {cell.value for cell in CELLS}
_BLOCK_CONSTANT_COLUMNS = ("relation_family", "template_id", "split", "dataset_tier")


def _native_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Rows as plain Python values.

    pandas hands back numpy scalars (`numpy.int8`, `numpy.bool_`), which pydantic rejects
    because they are not `int`/`bool` subclasses -- without this the model check would flag
    every row of a perfectly valid table.
    """
    return [
        {key: (value.item() if hasattr(value, "item") else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _check_rows_against_model(
    frame: pd.DataFrame, violations: list[str], *, limit: int = 20
) -> None:
    """Re-validate every row through the pydantic model: enum domains, cell consistency."""
    reported = 0
    for record in _native_records(frame):
        try:
            Quadruple.model_validate(record)
        except ValidationError as exc:
            reported += 1
            if reported <= limit:
                block = record.get("block_id", "?")
                cell = record.get("cell", "?")
                first = exc.errors()[0]
                loc = ".".join(str(p) for p in first["loc"]) or "row"
                violations.append(f"row {block}/{cell}: {loc}: {first['msg']}")
    if reported > limit:
        violations.append(f"... and {reported - limit} further row-level schema violations")


def _check_block_structure(frame: pd.DataFrame, violations: list[str]) -> None:
    for block_id, block in frame.groupby("block_id", sort=True):
        if len(block) != 4:
            violations.append(f"block {block_id}: has {len(block)} rows, expected exactly 4")
            continue

        cells = set(block["cell"])
        if cells != _EXPECTED_CELLS:
            violations.append(f"block {block_id}: cells are {sorted(cells)}, expected the 2x2 set")

        pairs = list(zip(block["question_id"], block["answer_id"], strict=True))
        if len(set(pairs)) != 4:
            violations.append(f"block {block_id}: (question_id, answer_id) pairs are not unique")

        question_ids = set(block["question_id"])
        answer_ids = set(block["answer_id"])
        if len(question_ids) != 2:
            violations.append(
                f"block {block_id}: {len(question_ids)} distinct question identities, expected 2"
            )
        if len(answer_ids) != 2:
            violations.append(
                f"block {block_id}: {len(answer_ids)} distinct answer identities, expected 2"
            )

        n_matched = int(block["matched"].sum())
        if n_matched != 2:
            violations.append(f"block {block_id}: {n_matched} matched cells, expected 2")

        # Each identity must appear once matched and once crossed, otherwise the identity is
        # confounded with the condition and the interaction contrast is not identity-balanced.
        for column, label in (("question_id", "question"), ("answer_id", "answer")):
            for identity, rows in block.groupby(column, sort=True):
                matched_flags = sorted(bool(m) for m in rows["matched"])
                if matched_flags != [False, True]:
                    violations.append(
                        f"block {block_id}: {label} identity {identity} is not balanced "
                        f"across matched/crossed (matched flags: {matched_flags})"
                    )

        for column in _BLOCK_CONSTANT_COLUMNS:
            distinct = set(block[column])
            if len(distinct) != 1:
                violations.append(
                    f"block {block_id}: {column} is not constant within the block: "
                    f"{sorted(distinct)}"
                )


def _check_correctness_labels(frame: pd.DataFrame, violations: list[str]) -> None:
    crossed_correct = frame[
        (~frame["matched"])
        & frame["correct"]
        & (frame["answer_source"] != AnswerSource.ALIAS.value)
    ]
    for row in crossed_correct.itertuples():
        violations.append(
            f"block {row.block_id}/{row.cell}: crossed cell is labelled correct but the answer "
            f"is not an alias -- the fact pair is not functional"
        )

    matched_wrong = frame[frame["matched"] & ~frame["correct"]]
    for row in matched_wrong.itertuples():
        violations.append(
            f"block {row.block_id}/{row.cell}: matched cell is labelled incorrect "
            f"(answer_source={row.answer_source})"
        )


def _check_split_leakage(frame: pd.DataFrame, violations: list[str], *, limit: int = 10) -> None:
    for column, label in (
        ("block_id", "block"),
        ("question_id", "question identity"),
        ("answer_id", "answer identity"),
    ):
        spread = frame.groupby(column)["split"].nunique()
        leaked = sorted(spread[spread > 1].index)
        for identity in leaked[:limit]:
            splits = sorted(set(frame.loc[frame[column] == identity, "split"]))
            violations.append(f"{label} {identity} leaks across splits {splits}")
        if len(leaked) > limit:
            violations.append(f"... and {len(leaked) - limit} further {label} leaks")


def _check_family_holdout(frame: pd.DataFrame, violations: list[str]) -> None:
    held_out = frame[frame["split"] == Split.FAMILY_HOLDOUT.value]
    if held_out.empty:
        return
    holdout_families = set(held_out["relation_family"])
    elsewhere = frame[frame["split"] != Split.FAMILY_HOLDOUT.value]
    overlap = holdout_families & set(elsewhere["relation_family"])
    for family in sorted(overlap):
        violations.append(
            f"relation family {family} is held out but also appears in a fittable split"
        )


def _check_family_coverage(frame: pd.DataFrame, violations: list[str]) -> None:
    """Every fittable family must reach every fittable split.

    A family confined to one split is not a split -- it is a second held-out family wearing
    the wrong label, and it turns the "unseen identities, known family" evaluation into
    something it is not. This check exists because an earlier block-level splitter did exactly
    that: it put all of `author_of` into validation and test, leaving train without a single
    author block.
    """
    fittable = frame[frame["split"] != Split.FAMILY_HOLDOUT.value]
    if fittable.empty:
        return
    expected = {Split.TRAIN.value, Split.VALIDATION.value, Split.TEST.value}
    for family, rows in fittable.groupby("relation_family", sort=True):
        missing = expected - set(rows["split"])
        if missing:
            violations.append(
                f"relation family {family} is absent from {sorted(missing)}; a fittable family "
                f"must appear in every fittable split"
            )


def _check_text_fields(frame: pd.DataFrame, violations: list[str]) -> None:
    for row in frame.itertuples():
        if "{" in row.question or "}" in row.question:
            violations.append(
                f"block {row.block_id}/{row.cell}: question has an unfilled placeholder: "
                f"{row.question!r}"
            )
        if not str(row.question).strip() or not str(row.answer).strip():
            violations.append(f"block {row.block_id}/{row.cell}: empty question or answer text")

    # The rendered question depends on the template, so identity alone need not fix the text;
    # identity plus template must.
    per_template = frame.groupby(["question_id", "template_id"])["question"].nunique()
    for question_id, template_id in per_template[per_template > 1].index:
        violations.append(
            f"question identity {question_id} renders differently under one template "
            f"({template_id}) -- identity and text are out of sync"
        )

    per_answer = frame.groupby("answer_id")["answer"].nunique()
    for answer_id in per_answer[per_answer > 1].index:
        violations.append(f"answer identity {answer_id} maps to more than one answer string")


def validate_frame(frame: pd.DataFrame) -> ValidationReport:
    report = ValidationReport(
        n_rows=len(frame),
        n_blocks=int(frame["block_id"].nunique()) if len(frame) else 0,
        counts_by_split={str(k): int(v) for k, v in frame["split"].value_counts().items()}
        if len(frame)
        else {},
        counts_by_family={
            str(k): int(v) for k, v in frame["relation_family"].value_counts().items()
        }
        if len(frame)
        else {},
    )
    if frame.empty:
        report.violations.append("table is empty")
        return report

    _check_rows_against_model(frame, report.violations)
    _check_block_structure(frame, report.violations)
    _check_correctness_labels(frame, report.violations)
    _check_split_leakage(frame, report.violations)
    _check_family_holdout(frame, report.violations)
    _check_family_coverage(frame, report.violations)
    _check_text_fields(frame, report.violations)
    return report


def validate_table(table: pa.Table) -> ValidationReport:
    return validate_frame(table.to_pandas())


def validate_path(path: Path) -> ValidationReport:
    """Read the Parquet file (asserting the frozen schema) and check every invariant."""
    return validate_table(read_table(path))
