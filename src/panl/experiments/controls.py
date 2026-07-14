"""Controls for the isolated PANL patch.

The headline result is that freezing PANL to a partner cell's trajectory transplants the
confidence decision. On its own that sentence does not exclude two much duller explanations,
and plan §E3 requires both to be ruled out before the word "causal" is used:

  **"any vector of that size would do it."** Patching PANL writes a large bf16 vector into a
  live residual stream. Maybe *any* such write derails the read-out and the partner's content
  is irrelevant. → `GAUSSIAN`, a random direction with the partner's norm, and `MEAN`, the
  dataset-mean PANL. Both are perturbations of the right magnitude carrying no answer.

  **"any other answer would do it."** Maybe the effect is not about the *partner's* answer but
  about replacing PANL with any well-formed PANL from any answer. → `RANDOM_CELL`, the PANL of
  a cell from a different block. Same distribution, same norm scale, wrong content.

The real patch must beat all three. If `GAUSSIAN` also flips decisions, we have shown that
breaking PANL breaks confidence — which is not news, and is not what the paper claims. If
`RANDOM_CELL` also flips them, the effect is "PANL carries *an* answer" rather than "PANL
carries *this* answer", and the Q x A story collapses.

Direction matters too, and only `restore` was reported before. `ablate` runs the same patch
backwards: push a crossed cell's PANL into a matched run and the model should *lose*
confidence. Plan §E3: causal evidence requires the effect to be bidirectional.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.progress import Progress

from panl.analysis.stats import block_bootstrap_ratio
from panl.config import ExperimentConfig
from panl.experiments.e0 import patch_pairs
from panl.experiments.routes import ISOLATE_PANL
from panl.models.adapter import HookedModelAdapter
from panl.models.batching import make_batches
from panl.models.positions import ResolvedPositions


class SourceKind(StrEnum):
    #: The intervention under test: the partner cell, same question, other answer.
    PARTNER = "partner"
    #: A cell from a different block. Right distribution, right norm, wrong answer.
    RANDOM_CELL = "random_cell"
    #: A random direction scaled to the partner's norm. A perturbation carrying no answer.
    GAUSSIAN = "gaussian"
    #: The dataset-mean PANL. An ablation towards the average, not a transplant.
    MEAN = "mean"
    #: A *matched* cell from a different block: an unrelated question with its own right answer.
    #: Carries "this answer fits" but about a different pair.
    MATCHED_DONOR = "matched_donor"
    #: A *crossed* cell from a different block: an unrelated question with a wrong answer.
    #: Carries "this answer does not fit" but about a different pair.
    CROSSED_DONOR = "crossed_donor"


def build_source(
    kind: SourceKind,
    cached: torch.Tensor,
    source_rows: np.ndarray,
    layers: list[int],
    *,
    generator: torch.Generator,
    donor_pool: np.ndarray | None = None,
    pair_blocks: np.ndarray | None = None,
    row_blocks: np.ndarray | None = None,
) -> torch.Tensor:
    """Replacement vectors for one control. Returns [n_pairs, len(layers), d_model].

    Args:
        cached: [n_rows, n_layers, d_model] activations at this position, from clean runs.
        source_rows: [n_pairs] the partner cell's row index, per pair.
        donor_pool: row indices a donor may be drawn from, for the donor controls.
        pair_blocks: [n_pairs] block index of each pair's target, so a donor is never drawn
            from the target's own block.
        row_blocks: [n_rows] block index of each row.
    """
    partner = cached[source_rows][:, layers, :]

    if kind is SourceKind.PARTNER:
        return partner

    if kind in (SourceKind.RANDOM_CELL, SourceKind.MATCHED_DONOR, SourceKind.CROSSED_DONOR):
        pool = (
            torch.arange(cached.shape[0], device=cached.device)
            if donor_pool is None
            else torch.as_tensor(donor_pool, device=cached.device)
        )
        if len(pool) == 0:
            msg = f"{kind.value}: the donor pool is empty"
            raise ValueError(msg)

        # A donor is drawn once per pair and reused across layers, so the replacement is a
        # coherent trajectory rather than a different cell stitched in at every layer.
        picks = torch.randint(
            0, len(pool), (len(source_rows),), generator=generator, device=cached.device
        )
        donors = pool[picks]

        # Never donate a cell to itself, or the "wrong content" control silently becomes the
        # intervention for that row. (The block rejection below subsumes this when block
        # information is available, but this must hold regardless.)
        rows_t = torch.as_tensor(source_rows, device=cached.device)
        collides = donors == rows_t
        donors[collides] = pool[(picks[collides] + 1) % len(pool)]

        if pair_blocks is not None and row_blocks is not None:
            # A donor from the target's own block is not an unrelated cell: it shares a question
            # or an answer with the target, which is precisely the content under test.
            target_blocks = torch.as_tensor(pair_blocks, device=cached.device)
            blocks = torch.as_tensor(row_blocks, device=cached.device)
            for _ in range(16):
                bad = blocks[donors] == target_blocks
                if not bad.any():
                    break
                redraw = torch.randint(
                    0, len(pool), (int(bad.sum()),), generator=generator, device=cached.device
                )
                donors[bad] = pool[redraw]
            if (blocks[donors] == target_blocks).any():
                msg = f"{kind.value}: could not draw an out-of-block donor for every pair"
                raise RuntimeError(msg)

        return cached[donors][:, layers, :]

    if kind is SourceKind.GAUSSIAN:
        noise = torch.randn(
            partner.shape, generator=generator, device=cached.device, dtype=torch.float32
        )
        # Norm-matched per (pair, layer): the perturbation is exactly as large as the real one.
        scale = partner.float().norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True)
        scaled: torch.Tensor = noise * scale
        return scaled.to(partner.dtype)

    mean = cached.float().mean(dim=0)[layers, :]  # [len(layers), d_model]
    expanded: torch.Tensor = mean.unsqueeze(0).expand(len(source_rows), -1, -1)
    return expanded.to(partner.dtype)


def _stable_seed(base: int, *parts: str) -> int:
    """A reproducible seed from strings.

    Python's `hash()` of a str is salted per process, so seeding an RNG with it would give a
    different set of random donors on every run -- and a control that cannot be reproduced is
    not a control.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2**31)


def isolated_controls(
    model: HookedModelAdapter,
    behavior: pd.DataFrame,
    resolved: list[ResolvedPositions],
    config: ExperimentConfig,
    *,
    start_layer: int,
    positions: tuple[str, ...] = ("PANL", "PANL1", "AC"),
    kinds: tuple[SourceKind, ...] = tuple(SourceKind),
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Run the cumulative isolated patch for every (position, source kind, direction).

    `start_layer` should be the read cliff found by the layer sweep -- the last start layer at
    which freezing the position still transplants the decision.
    """
    layers = list(range(start_layer, model.n_layers))
    edges = list(ISOLATE_PANL)

    clean = np.full(len(behavior), np.nan)
    for batch in make_batches(resolved, max_batch_size=config.batch_size):
        out = model.run_with_knockout(batch, edges=edges)
        for offset, row in enumerate(batch.row_indices):
            clean[row] = float(out[offset])

    # The baseline is the margin under the knockout: the knockout is the regime, not the effect.
    isolated = behavior.copy()
    isolated["confidence_margin"] = clean
    pairs = patch_pairs(isolated)
    source_rows = pairs["source_row"].to_numpy()

    cached: dict[str, torch.Tensor] = {
        position: torch.zeros(
            (len(behavior), model.n_layers, model.d_model),
            dtype=torch.bfloat16,
            device=model.device,
        )
        for position in positions
    }
    all_layers = list(range(model.n_layers))
    for batch in make_batches(resolved, max_batch_size=config.batch_size):
        result = model.run(batch, cache_layers=all_layers)
        index = torch.tensor(batch.row_indices, device=model.device)
        for position in positions:
            cached[position][index] = torch.stack(
                [result.activations[layer][position] for layer in all_layers], dim=1
            ).to(torch.bfloat16)

    target_resolved = [resolved[int(row)] for row in pairs["target_row"]]
    batches = list(make_batches(target_resolved, max_batch_size=config.batch_size))

    task = progress.add_task("controls", total=len(positions) * len(kinds)) if progress else None
    records: list[pd.DataFrame] = []

    # Donor pools for the "is it *this* answer, or *an* answer?" controls.
    block_codes = behavior["block_id"].astype("category").cat.codes.to_numpy()
    matched_flags = behavior["matched"].to_numpy().astype(bool)
    pools: dict[SourceKind, np.ndarray | None] = {
        SourceKind.RANDOM_CELL: np.arange(len(behavior)),
        SourceKind.MATCHED_DONOR: np.flatnonzero(matched_flags),
        SourceKind.CROSSED_DONOR: np.flatnonzero(~matched_flags),
    }
    pair_blocks = block_codes[pairs["target_row"].to_numpy()]

    for position in positions:
        for kind in kinds:
            # Seeded per (position, kind) so a rerun reproduces the same random donors.
            generator = torch.Generator(device=model.device)
            generator.manual_seed(_stable_seed(config.seed, position, kind.value))
            source = build_source(
                kind,
                cached[position],
                source_rows,
                layers,
                generator=generator,
                donor_pool=pools.get(kind),
                pair_blocks=pair_blocks,
                row_blocks=block_codes,
            )

            patched = np.full(len(pairs), np.nan)
            for batch in batches:
                margins = model.run_with_patch(
                    batch,
                    layer=layers,
                    position=position,
                    source=source[list(batch.row_indices)],
                    edges=edges,
                )
                for offset, pair_index in enumerate(batch.row_indices):
                    patched[pair_index] = float(margins[offset])

            frame = pairs.copy()
            frame["position"] = position
            frame["source_kind"] = kind.value
            frame["start_layer"] = start_layer
            frame["patched"] = patched
            records.append(frame)
            if progress and task is not None:
                progress.advance(task)

    out = pd.concat(records, ignore_index=True)
    out["moved"] = out["patched"] - out["target_clean"]
    out["gap"] = out["source_clean"] - out["target_clean"]
    out["flipped"] = np.sign(out["patched"]) != np.sign(out["target_clean"])
    return out


def summarize_controls(controls: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Per (position, source kind, direction): effect, absolute shift, flip rate."""
    records: list[dict[str, Any]] = []
    for (position, kind, direction), group in controls.groupby(
        ["position", "source_kind", "direction"], sort=True
    ):
        per_block = group.groupby("block_id")[["moved", "gap"]].mean()
        effect = block_bootstrap_ratio(
            per_block["moved"].to_numpy(),
            per_block["gap"].to_numpy(),
            n_boot=config.n_boot,
            seed=config.seed,
        )
        records.append(
            {
                "position": str(position),
                "source_kind": str(kind),
                "direction": str(direction),
                "effect": effect.mean,
                "ci_low": effect.ci_low,
                "ci_high": effect.ci_high,
                "mean_moved": float(per_block["moved"].mean()),
                "flip_rate": float(group["flipped"].mean()),
                "n_blocks": effect.n_blocks,
            }
        )
    return pd.DataFrame(records)


#: A control is "beaten" if the real patch flips at least this many times more often.
CONTROL_MARGIN = 3.0

#: Controls are compared to the intervention *within a direction*, never pooled across them.
#:
#: This is not a stylistic choice, it is what the data forced. A norm-matched random vector
#: written into PANL flips 96% of decisions in the `restore` direction and 1% in `ablate`. The
#: reason is that the model is confident *by default*: PANL carries doubt, not confidence, so
#: destroying PANL -- with anything -- makes a diffident model confident, while nothing about a
#: random vector makes a confident model doubt. The `restore` direction is therefore confounded
#: for every destructive control, and a gate that pooled the two directions (an earlier version
#: did) would have quietly credited the intervention for an effect that noise reproduces.
CONFOUNDED_DIRECTIONS: tuple[str, ...] = ("restore",)


def evaluate_control_gates(summary: pd.DataFrame) -> dict[str, Any]:
    panl = summary[summary["position"] == "PANL"]

    def flip(kind: str, direction: str) -> float:
        rows = panl[(panl["source_kind"] == kind) & (panl["direction"] == direction)]
        return float(rows["flip_rate"].iloc[0]) if len(rows) else float("nan")

    def effect(kind: str, direction: str) -> float:
        rows = panl[(panl["source_kind"] == kind) & (panl["direction"] == direction)]
        return float(rows["effect"].iloc[0]) if len(rows) else float("nan")

    def beats(kind: str, direction: str) -> bool:
        control, real = flip(kind, direction), flip("partner", direction)
        if np.isnan(control):
            return True
        return bool(real >= CONTROL_MARGIN * max(control, 0.01))

    partner_restore = flip("partner", "restore")
    partner_ablate = flip("partner", "ablate")
    bidirectional = bool(partner_restore >= 0.5 and partner_ablate >= 0.5)

    # `ablate` is the uncontaminated direction: only a source that actually carries "this answer
    # does not fit" can make a confident model doubt.
    destructive_ok = all(beats(kind, "ablate") for kind in ("gaussian", "mean"))

    # The decisive test of the Q x A claim. A crossed cell from *another* block carries "the
    # answer does not fit" about a different question and a different answer. If transplanting
    # it removes confidence as well as the true partner does, then PANL holds a generic doubt
    # signal, not information about *this* pair -- and the relational story does not survive.
    pair_specific = beats("crossed_donor", "ablate")

    return {
        "bidirectional": bidirectional,
        "beats_destructive_controls": destructive_ok,
        "effect_is_pair_specific": pair_specific,
        "overall": bidirectional and destructive_ok and pair_specific,
        "partner_flip_restore": partner_restore,
        "partner_flip_ablate": partner_ablate,
        "partner_effect_restore": effect("partner", "restore"),
        "partner_effect_ablate": effect("partner", "ablate"),
        "control_flips": {
            kind: {d: flip(kind, d) for d in ("restore", "ablate")}
            for kind in ("random_cell", "gaussian", "mean", "matched_donor", "crossed_donor")
        },
        "restore_is_confounded": bool(flip("gaussian", "restore") >= 0.5),
        "control_margin": CONTROL_MARGIN,
    }
