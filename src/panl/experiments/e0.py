"""E0 -- pipeline reproduction and calibration.

E0 asks one question before any mechanism is proposed: does this implementation recover the
known PANL confidence-cache pattern at all? It has two parts.

**Behavioural signal.** Within a block, `I_k = z11 - z12 - z21 + z22` on the pre-softmax
confidence scale. Note that in this design `I_k` is exactly twice the matched-minus-crossed
contrast, because the matched cells *are* the diagonal -- the difference-in-differences and
the "does confidence track relational fit" question are the same number. If that contrast is
not reliably positive, nothing downstream is interpretable.

**Positional localization.** Take two cells that share a question and differ only in the
answer, and move the residual stream at one semantic position from one run into the other.
If PANL caches the information the confidence read-out consumes, patching PANL should carry
the source's confidence across, while patching AC -- a position that precedes the answer
entirely -- should not.

CC is a sanity check, not evidence: it is the read-out position itself, so a patch there
transplants the answer's whole final state and *must* score near 1. A CC effect below the
PANL effect would mean the harness is broken, not that PANL is special.

Gate E0: do not interpret E1-E4 if the confidence contrast is unreliable or the PANL effect
fails to separate from its controls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.progress import Progress

from panl.analysis.stats import (
    Estimate,
    auc,
    block_bootstrap,
    block_bootstrap_ratio,
    paired_effect_size,
)
from panl.config import ExperimentConfig
from panl.models.adapter import HookedModelAdapter
from panl.models.batching import make_batches
from panl.models.positions import POSITION_NAMES, ResolvedPositions

#: Minimum normalized PANL effect, and the factor by which it must beat its controls, for the
#: gate to pass. Fixed here rather than chosen after looking at the sweep.
PANL_EFFECT_FLOOR = 0.30
CONTROL_MARGIN = 2.0


@dataclass(slots=True)
class E0Result:
    signal: dict[str, Any]
    patching: pd.DataFrame
    summary: pd.DataFrame
    gates: dict[str, Any]

    @property
    def passed(self) -> bool:
        return bool(self.gates["overall"])


# -- behavioural signal -------------------------------------------------------------------


def block_contrasts(behavior: pd.DataFrame) -> pd.DataFrame:
    """One row per block: the interaction contrast on the confidence margin."""
    wide = behavior.pivot_table(
        index="block_id", columns="cell", values="confidence_margin", aggfunc="first"
    )
    missing = {"q1a1", "q1a2", "q2a1", "q2a2"} - set(wide.columns)
    if missing:
        msg = f"blocks are missing cells {sorted(missing)}; cannot form the 2x2 contrast"
        raise ValueError(msg)

    interaction = wide["q1a1"] - wide["q1a2"] - wide["q2a1"] + wide["q2a2"]
    return pd.DataFrame(
        {
            "block_id": wide.index,
            "interaction": interaction.to_numpy(),
            # Identical up to the factor of two, but reported separately because Delta_fit is
            # the quantity E3 intervenes on.
            "delta_fit": (interaction / 2).to_numpy(),
        }
    )


def confidence_signal(behavior: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    contrasts = block_contrasts(behavior)
    values = contrasts["interaction"].to_numpy()

    return {
        "contrasts": contrasts,
        "interaction": block_bootstrap(values, n_boot=config.n_boot, seed=config.seed),
        "delta_fit": block_bootstrap(values / 2, n_boot=config.n_boot, seed=config.seed),
        "effect_size": paired_effect_size(values),
        # Does the margin rank correct answers above incorrect ones at all?
        "calibration_auc": auc(
            behavior["confidence_margin"].to_numpy(), behavior["correct"].to_numpy()
        ),
        "mean_margin_matched": float(behavior.loc[behavior["matched"], "confidence_margin"].mean()),
        "mean_margin_crossed": float(
            behavior.loc[~behavior["matched"], "confidence_margin"].mean()
        ),
    }


# -- patching -----------------------------------------------------------------------------


def patch_pairs(behavior: pd.DataFrame) -> pd.DataFrame:
    """Every (target, source) pair that holds the question fixed and swaps only the answer.

    Within a block that is four pairs: for each of the two questions, the matched cell and the
    crossed cell act as each other's source once. Because the question is shared, the two
    prompts differ only in the answer -- so whatever the patch does cannot be a question
    effect in disguise.
    """
    partner = {"q1a1": "q1a2", "q1a2": "q1a1", "q2a2": "q2a1", "q2a1": "q2a2"}
    by_key = behavior.set_index(["block_id", "cell"])

    records: list[dict[str, Any]] = []
    for row in behavior.itertuples():
        source_cell = partner[str(row.cell)]
        source = by_key.loc[(row.block_id, source_cell)]
        records.append(
            {
                "block_id": row.block_id,
                "target_row": int(row.row),
                "source_row": int(source["row"]),
                "target_cell": str(row.cell),
                "source_cell": source_cell,
                # A crossed target receiving a matched source is a restoration.
                "direction": "ablate" if row.matched else "restore",
                "target_clean": float(row.confidence_margin),
                "source_clean": float(source["confidence_margin"]),
            }
        )
    return pd.DataFrame(records)


@torch.no_grad()
def cache_source_activations(
    model: HookedModelAdapter,
    resolved: list[ResolvedPositions],
    config: ExperimentConfig,
    *,
    progress: Progress | None = None,
) -> dict[str, torch.Tensor]:
    """Residual stream at every layer and position, kept in the model's own dtype.

    Deliberately *not* read back from the Zarr store: that store is float16, and a causal
    patch must carry the values the model actually computed, not a lossy round trip of them.

    Returns position -> [n_rows, n_layers, d_model] on the model's device.
    """
    layers = list(range(model.n_layers))
    out = {
        position: torch.zeros(
            (len(resolved), model.n_layers, model.d_model),
            dtype=torch.bfloat16,
            device=model.device,
        )
        for position in POSITION_NAMES
    }

    batches = list(make_batches(resolved, max_batch_size=config.batch_size))
    task = progress.add_task("cache sources", total=len(batches)) if progress else None
    for batch in batches:
        result = model.run(batch, cache_layers=layers)
        rows = torch.tensor(batch.row_indices, device=model.device)
        for position in POSITION_NAMES:
            stacked = torch.stack(
                [result.activations[layer][position] for layer in layers], dim=1
            )  # [batch, n_layers, d_model]
            out[position][rows] = stacked.to(torch.bfloat16)
        if progress and task is not None:
            progress.advance(task)
    return out


def patching_sweep(
    model: HookedModelAdapter,
    behavior: pd.DataFrame,
    resolved: list[ResolvedPositions],
    config: ExperimentConfig,
    *,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Patch every (layer, position) and record the confidence margin that results."""
    pairs = patch_pairs(behavior)
    sources = cache_source_activations(model, resolved, config, progress=progress)

    layers = list(config.patch_layers) if config.patch_layers else list(range(model.n_layers))
    positions = list(config.patch_positions)

    # Targets are batched by prompt length exactly as in a clean pass. `make_batches` indexes
    # into this list, so its indices are pair indices, not behaviour row indices.
    target_resolved = [resolved[int(row)] for row in pairs["target_row"]]
    source_rows = pairs["source_row"].to_numpy()
    batches = list(make_batches(target_resolved, max_batch_size=config.batch_size))

    records: list[pd.DataFrame] = []
    task = progress.add_task("patching", total=len(layers) * len(positions)) if progress else None

    for position in positions:
        for layer in layers:
            patched = np.full(len(pairs), np.nan, dtype=np.float64)
            for batch in batches:
                rows = [int(source_rows[i]) for i in batch.row_indices]
                margins = model.run_with_patch(
                    batch,
                    layer=layer,
                    position=position,
                    source=sources[position][rows, layer, :],
                )
                for offset, pair_index in enumerate(batch.row_indices):
                    patched[pair_index] = float(margins[offset])

            frame = pairs.copy()
            frame["layer"] = layer
            frame["position"] = position
            frame["patched"] = patched
            records.append(frame)
            if progress and task is not None:
                progress.advance(task)

    out = pd.concat(records, ignore_index=True)
    # How far the patch moved the margin, over how far it could have moved it.
    out["moved"] = out["patched"] - out["target_clean"]
    out["gap"] = out["source_clean"] - out["target_clean"]
    # Did the patch actually change the model's high/low decision at CC?
    out["flipped"] = np.sign(out["patched"]) != np.sign(out["target_clean"])
    return out


def summarize_patching(patching: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Per (direction, position, layer): three readouts of the same patch.

    `effect` normalizes by the clean source-target gap, which is the standard patching metric
    and is the right one when the model is not saturated. It is *not* sufficient on its own:
    a model whose clean gap is 30 logits can be moved 3 logits by a patch and still score
    0.10, while a model with a 1.3-logit gap moved 1.2 logits scores 0.92 -- the same patch
    strength, opposite conclusions. So `mean_moved` (absolute logits) and `flip_rate` (did the
    high/low decision actually change) are reported alongside it, and neither divides by the
    gap.
    """
    patching = patching.copy()
    # Derived rather than required, so this also summarizes a run recorded before the column
    # existed -- a metric change must never force a GPU re-run of a finished sweep.
    patching["flipped"] = np.sign(patching["patched"]) != np.sign(patching["target_clean"])

    records: list[dict[str, Any]] = []
    for (direction, position, layer), group in patching.groupby(
        ["direction", "position", "layer"], sort=True
    ):
        per_block = group.groupby("block_id")[["moved", "gap"]].mean()
        estimate = block_bootstrap_ratio(
            per_block["moved"].to_numpy(),
            per_block["gap"].to_numpy(),
            n_boot=config.n_boot,
            seed=config.seed,
        )
        absolute = block_bootstrap(
            per_block["moved"].to_numpy(),
            n_boot=config.n_boot,
            seed=config.seed,
            permutation=False,
        )
        records.append(
            {
                "direction": str(direction),
                "position": str(position),
                "layer": int(layer),
                "effect": estimate.mean,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "sign_consistency": estimate.sign_consistency,
                "mean_moved": absolute.mean,
                "moved_ci_low": absolute.ci_low,
                "moved_ci_high": absolute.ci_high,
                "flip_rate": float(group["flipped"].mean()),
                "mean_gap": float(per_block["gap"].mean()),
                "n_blocks": estimate.n_blocks,
            }
        )
    return pd.DataFrame(records)


# -- gates --------------------------------------------------------------------------------


#: Above this clean source-target gap the confidence read-out is effectively a step function,
#: and a normalized patching effect stops meaning what it usually means: the denominator is so
#: large that even a strong patch scores near zero. A gate that fails *here* is a statement
#: about the stimuli, not about the mechanism, and must not be read as one.
SATURATION_GAP_LOGITS = 10.0


def evaluate_gates(signal: dict[str, Any], summary: pd.DataFrame) -> dict[str, Any]:
    interaction: Estimate = signal["interaction"]

    restore = summary[summary["direction"] == "restore"]
    peaks = restore.groupby("position")["effect"].max()

    def peak(name: str) -> float:
        return float(peaks.get(name, float("nan")))

    panl, panl1, ac, cc = peak("PANL"), peak("PANL1"), peak("AC"), peak("CC")

    best = restore[restore["position"] == "PANL"].sort_values("effect", ascending=False)
    best_layer = int(best["layer"].iloc[0]) if len(best) else -1

    mean_gap = float(restore["mean_gap"].mean()) if len(restore) else float("nan")
    saturated = bool(mean_gap > SATURATION_GAP_LOGITS)
    panl_moved = float(best["mean_moved"].iloc[0]) if len(best) else float("nan")
    panl_flip = float(best["flip_rate"].iloc[0]) if len(best) else float("nan")

    confidence_ok = bool(interaction.mean > 0 and interaction.ci_low > 0)
    floor_ok = bool(panl >= PANL_EFFECT_FLOOR)
    beats_panl1 = bool(panl >= CONTROL_MARGIN * max(panl1, 1e-6))
    beats_ac = bool(panl >= CONTROL_MARGIN * max(ac, 1e-6))
    # If the read-out position itself is not at least as strong a patch as PANL, the harness
    # is wrong -- this is a check on us, not on the model.
    harness_ok = bool(np.isnan(cc) or cc >= 0.9 * panl)

    return {
        "confidence_signal": confidence_ok,
        "panl_effect_floor": floor_ok,
        "panl_beats_panl1": beats_panl1,
        "panl_beats_ac": beats_ac,
        "cc_sanity": harness_ok,
        "overall": confidence_ok and floor_ok and beats_panl1 and beats_ac and harness_ok,
        "peak_effect": {name: peak(name) for name in POSITION_NAMES},
        "panl_best_layer": best_layer,
        "thresholds": {
            "floor": PANL_EFFECT_FLOOR,
            "control_margin": CONTROL_MARGIN,
            "saturation_gap": SATURATION_GAP_LOGITS,
        },
        # Not a gate: a diagnosis of what a failing gate means. If the read-out is saturated,
        # `panl_effect_floor` is uninterpretable and the right response is to make the
        # distractors harder (plan section 3.2, Tier 2), not to abandon the hypothesis.
        "saturated": saturated,
        "mean_clean_gap": mean_gap,
        "panl_absolute_logits_moved": panl_moved,
        "panl_flip_rate": panl_flip,
    }


def run_e0(
    model: HookedModelAdapter,
    behavior: pd.DataFrame,
    resolved: list[ResolvedPositions],
    config: ExperimentConfig,
    *,
    progress: Progress | None = None,
) -> E0Result:
    signal = confidence_signal(behavior, config)
    patching = patching_sweep(model, behavior, resolved, config, progress=progress)
    summary = summarize_patching(patching, config)
    gates = evaluate_gates(signal, summary)
    return E0Result(signal=signal, patching=patching, summary=summary, gates=gates)
