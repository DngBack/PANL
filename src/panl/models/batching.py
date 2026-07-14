"""Prompt batches, grouped by exact token length.

Batches hold prompts of *identical* length, so no padding is ever introduced. This is a
deliberate choice over left-padding: a padded batch only produces correct activations if the
model's rotary offsets and attention mask are threaded through exactly right, and a silent
error there would corrupt every activation and every patch without failing anything. Prompts
here differ only in the answer and the question, so their lengths cluster tightly and
length-grouping costs almost nothing.

It also has a convenient consequence. The prompt ends with a fixed suffix -- newline,
"Confidence", ":" -- so within a length group CC, PANL+1 and PANL sit at the same index for
every row. AC and LAT still vary with the question and answer, so positions are carried
per row regardless.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from panl.models.positions import POSITION_NAMES, ResolvedPositions


@dataclass(frozen=True, slots=True)
class PromptBatch:
    """A group of equal-length tokenized prompts, ready for a forward pass."""

    #: [batch, seq]
    input_ids: torch.Tensor
    #: Row index into the caller's original list, so results can be scattered back.
    row_indices: tuple[int, ...]
    #: position name -> [batch] token index
    positions: dict[str, torch.Tensor]
    #: [batch, 2] answer token span, for the likelihood controls.
    answer_spans: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    def to(self, device: torch.device | str) -> PromptBatch:
        return PromptBatch(
            input_ids=self.input_ids.to(device),
            row_indices=self.row_indices,
            positions={k: v.to(device) for k, v in self.positions.items()},
            answer_spans=self.answer_spans.to(device),
        )


def make_batches(
    resolved: Sequence[ResolvedPositions],
    *,
    max_batch_size: int = 32,
    device: torch.device | str = "cpu",
) -> Iterator[PromptBatch]:
    """Group prompts by token length and yield batches of at most `max_batch_size`.

    Rows are grouped by length, then chunked. Order within a length group follows the input
    order, so the batching is deterministic.
    """
    by_length: dict[int, list[int]] = {}
    for index, item in enumerate(resolved):
        by_length.setdefault(item.n_tokens, []).append(index)

    for length in sorted(by_length):
        rows = by_length[length]
        for start in range(0, len(rows), max_batch_size):
            chunk = rows[start : start + max_batch_size]
            items = [resolved[i] for i in chunk]
            yield PromptBatch(
                input_ids=torch.tensor(
                    [list(item.input_ids) for item in items], dtype=torch.long, device=device
                ),
                row_indices=tuple(chunk),
                positions={
                    name: torch.tensor(
                        [item.indices[name] for item in items], dtype=torch.long, device=device
                    )
                    for name in POSITION_NAMES
                },
                answer_spans=torch.tensor(
                    [list(item.answer_span) for item in items], dtype=torch.long, device=device
                ),
            )
