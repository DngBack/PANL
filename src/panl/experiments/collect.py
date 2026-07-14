"""Activation and behaviour collection.

One forward pass per cell produces three things: the pre-softmax confidence margin at CC,
the teacher-forced likelihood statistics of the answer (the nuisance controls of plan section
4.3), and the residual stream at the five semantic positions across every layer.

Nothing here interprets any of it. Collection is separated from analysis so that the test set
can be opened exactly once, by the analysis, and so that a re-analysis never needs a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.progress import Progress

from panl.activations.store import ActivationWriter, StoreSpec
from panl.config import ExperimentConfig
from panl.models.adapter import HookedModelAdapter
from panl.models.batching import make_batches
from panl.models.positions import POSITION_NAMES, ResolvedPositions, resolve_positions
from panl.models.prompts import PromptRenderer, PromptTemplate, RenderedPrompt


@dataclass(slots=True)
class CollectionResult:
    behavior: pd.DataFrame
    #: Positions per row, in the same order as `behavior`. Handed to E0 so the patching sweep
    #: re-uses the exact tokenization that produced the behaviour, rather than redoing it.
    resolved: list[ResolvedPositions]
    activations_path: Path | None
    activations_sha256: str | None
    n_rows: int
    n_layers: int
    d_model: int


def select_rows(table: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Take the first `n_blocks` blocks of the requested splits, keeping blocks whole.

    Blocks are kept whole because a partial block has no interaction contrast at all -- the
    2x2 is the unit of measurement, not the cell.
    """
    frame = table
    if config.splits:
        frame = frame[frame["split"].isin(list(config.splits))]
    if config.families:
        frame = frame[frame["relation_family"].isin(list(config.families))]

    blocks = sorted(frame["block_id"].unique())
    if config.n_blocks is not None:
        if len(blocks) < config.n_blocks:
            msg = (
                f"asked for {config.n_blocks} blocks but only {len(blocks)} match "
                f"splits={config.splits} families={config.families}"
            )
            raise ValueError(msg)
        blocks = blocks[: config.n_blocks]

    selected = frame[frame["block_id"].isin(blocks)].copy()
    # Deterministic order: the row index is the join key into the Zarr array.
    return selected.sort_values(["block_id", "cell"]).reset_index(drop=True)


def _likelihood_stats(logprobs: torch.Tensor) -> dict[str, float]:
    """The conditional-likelihood controls of plan section 4.3, per answer."""
    values = logprobs.float().cpu().numpy()
    n = len(values)
    return {
        "lp_mean": float(values.mean()),
        "lp_min": float(values.min()),
        "lp_max": float(values.max()),
        "lp_var": float(values.var()) if n > 1 else 0.0,
        "lp_first": float(values[0]),
        "lp_last": float(values[-1]),
        "lp_sum": float(values.sum()),
        "nll": float(-values.sum()),
        "nll_per_token": float(-values.mean()),
        "n_answer_tokens": n,
    }


def collect(
    model: HookedModelAdapter,
    quadruples: pd.DataFrame,
    config: ExperimentConfig,
    *,
    activations_path: Path | None = None,
    progress: Progress | None = None,
) -> CollectionResult:
    rows = select_rows(quadruples, config)
    renderer = PromptRenderer(
        model.tokenizer, template=PromptTemplate(), style=model.spec.prompt_style
    )

    prompts: list[RenderedPrompt] = []
    resolved: list[ResolvedPositions] = []
    for row in rows.itertuples():
        prompt = renderer.render(str(row.question), str(row.answer))
        prompts.append(prompt)
        resolved.append(resolve_positions(model.tokenizer, prompt))

    layers = list(range(model.n_layers))
    writer: ActivationWriter | None = None
    if activations_path is not None:
        writer = ActivationWriter(
            activations_path,
            StoreSpec(n_rows=len(rows), n_layers=len(layers), d_model=model.d_model),
            metadata={
                "model_id": model.spec.model_id,
                "revision": model.spec.revision,
                "compute_dtype": model.spec.dtype,
                "fold_ln": model.spec.fold_ln,
                "center_writing_weights": model.spec.center_writing_weights,
                "center_unembed": model.spec.center_unembed,
                "prompt_template_hash": prompts[0].template_hash if prompts else "",
                "block_ids": sorted(rows["block_id"].unique().tolist()),
            },
        )

    margins = np.full(len(rows), np.nan, dtype=np.float64)
    likelihood: list[dict[str, float]] = [{} for _ in range(len(rows))]

    batches = list(make_batches(resolved, max_batch_size=config.batch_size))
    task = progress.add_task("forward", total=len(batches)) if progress else None

    for batch in batches:
        out = model.run(batch, cache_layers=layers if writer is not None else None)
        for offset, row_index in enumerate(batch.row_indices):
            margins[row_index] = float(out.confidence_margin[offset])
            likelihood[row_index] = _likelihood_stats(out.answer_logprobs[offset])
        if writer is not None:
            writer.write(batch.row_indices, out.activations)
        if progress and task is not None:
            progress.advance(task)

    behavior = rows.copy()
    behavior["row"] = np.arange(len(rows))
    behavior["confidence_margin"] = margins
    for key in (
        "lp_mean", "lp_min", "lp_max", "lp_var", "lp_first", "lp_last",
        "lp_sum", "nll", "nll_per_token", "n_answer_tokens",
    ):  # fmt: skip
        behavior[key] = [item[key] for item in likelihood]

    behavior["n_tokens"] = [item.n_tokens for item in resolved]
    behavior["prompt_sha256"] = [item.prompt_sha256 for item in prompts]
    for name in POSITION_NAMES:
        behavior[f"pos_{name}"] = [item.indices[name] for item in resolved]
        behavior[f"tok_{name}"] = [item.token_ids[name] for item in resolved]

    checksum = writer.finalize() if writer is not None else None

    return CollectionResult(
        behavior=behavior,
        resolved=resolved,
        activations_path=activations_path,
        activations_sha256=checksum,
        n_rows=len(rows),
        n_layers=len(layers),
        d_model=model.d_model,
    )


def store_metadata(result: CollectionResult) -> dict[str, Any]:
    return {
        "n_rows": result.n_rows,
        "n_blocks": int(result.behavior["block_id"].nunique()),
        "n_layers": result.n_layers,
        "d_model": result.d_model,
        "activations_sha256": result.activations_sha256,
    }
