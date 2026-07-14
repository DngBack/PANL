"""Data contract for the crossed-block quadruple table.

The table is the single source of truth for every downstream experiment, so the
column set here mirrors `docs/experiment-plan.md` section 3.1 exactly. Extra columns
(`cell`, `q_index`, `a_index`, `dataset_tier`, `subject`) are additions that make the
factorial contrast of section 4.2 computable without re-deriving cell membership.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerSource(StrEnum):
    """Where the answer text in a cell came from."""

    GOLD = "gold"
    MODEL_ERROR = "model_error"
    ALIAS = "alias"
    DISTRACTOR = "distractor"


class PolicyStatus(StrEnum):
    """Relationship between the cell and what the model would generate on-policy.

    Tier-1 blocks are teacher-forced, so only `natural` (a gold answer the model would
    plausibly emit) and `off_policy` (a swapped in-family gold used as a distractor) are
    assigned at build time. The likelihood scorer promotes rows to `high_lp_wrong` /
    `low_lp_correct` once conditional log-probabilities exist.
    """

    NATURAL = "natural"
    HIGH_LP_WRONG = "high_lp_wrong"
    LOW_LP_CORRECT = "low_lp_correct"
    OFF_POLICY = "off_policy"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    FAMILY_HOLDOUT = "family_holdout"


class Cell(StrEnum):
    """The four cells of a crossed block, named by (question index, answer index)."""

    Q1A1 = "q1a1"
    Q1A2 = "q1a2"
    Q2A1 = "q2a1"
    Q2A2 = "q2a2"

    @property
    def q_index(self) -> int:
        return int(self.value[1])

    @property
    def a_index(self) -> int:
        return int(self.value[3])

    @property
    def matched(self) -> bool:
        return self.q_index == self.a_index


CELLS: Final[tuple[Cell, ...]] = (Cell.Q1A1, Cell.Q1A2, Cell.Q2A1, Cell.Q2A2)


class Quadruple(BaseModel):
    """One cell of one crossed block: a (question, answer) pair under teacher forcing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    block_id: str = Field(min_length=1)
    relation_family: str = Field(min_length=1)
    dataset_tier: int = Field(ge=1, le=3)

    cell: Cell
    q_index: int = Field(ge=1, le=2)
    a_index: int = Field(ge=1, le=2)

    question_id: str = Field(min_length=1)
    answer_id: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)

    matched: bool
    correct: bool

    answer_source: AnswerSource
    policy_status: PolicyStatus
    template_id: str = Field(min_length=1)
    split: Split

    @model_validator(mode="after")
    def _check_cell_consistency(self) -> Quadruple:
        if (self.q_index, self.a_index) != (self.cell.q_index, self.cell.a_index):
            msg = f"cell {self.cell} disagrees with q_index={self.q_index} a_index={self.a_index}"
            raise ValueError(msg)
        if self.matched is not self.cell.matched:
            msg = f"cell {self.cell} implies matched={self.cell.matched}, got {self.matched}"
            raise ValueError(msg)
        if self.correct and not self.matched and self.answer_source is not AnswerSource.ALIAS:
            msg = "a crossed cell may only be correct when the answer is an alias"
            raise ValueError(msg)
        return self


#: Explicit Arrow schema. Written and re-read on every build so a drift in column
#: names, order, or types fails loudly instead of silently producing an odd Parquet file.
QUADRUPLE_SCHEMA: Final[pa.Schema] = pa.schema(
    [
        pa.field("block_id", pa.string(), nullable=False),
        pa.field("relation_family", pa.string(), nullable=False),
        pa.field("dataset_tier", pa.int8(), nullable=False),
        pa.field("cell", pa.string(), nullable=False),
        pa.field("q_index", pa.int8(), nullable=False),
        pa.field("a_index", pa.int8(), nullable=False),
        pa.field("question_id", pa.string(), nullable=False),
        pa.field("answer_id", pa.string(), nullable=False),
        pa.field("subject", pa.string(), nullable=False),
        pa.field("question", pa.string(), nullable=False),
        pa.field("answer", pa.string(), nullable=False),
        pa.field("matched", pa.bool_(), nullable=False),
        pa.field("correct", pa.bool_(), nullable=False),
        pa.field("answer_source", pa.string(), nullable=False),
        pa.field("policy_status", pa.string(), nullable=False),
        pa.field("template_id", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
    ]
)

COLUMNS: Final[tuple[str, ...]] = tuple(QUADRUPLE_SCHEMA.names)


def to_table(rows: list[Quadruple]) -> pa.Table:
    """Materialize validated rows into an Arrow table with the frozen schema."""
    columns: dict[str, list[object]] = {name: [] for name in COLUMNS}
    for row in rows:
        dumped = row.model_dump(mode="json")
        for name in COLUMNS:
            columns[name].append(dumped[name])
    return pa.Table.from_pydict(columns, schema=QUADRUPLE_SCHEMA)


def write_table(rows: list[Quadruple], path: Path) -> pa.Table:
    table = to_table(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return table


def read_table(path: Path) -> pa.Table:
    """Read the table and assert it still matches the frozen contract."""
    table = pq.read_table(path)
    if table.schema.names != list(COLUMNS):
        msg = f"column set drifted: expected {list(COLUMNS)}, found {table.schema.names}"
        raise ValueError(msg)
    if not table.schema.equals(QUADRUPLE_SCHEMA):
        msg = f"schema drifted from the frozen contract:\n{table.schema}"
        raise ValueError(msg)
    return table
