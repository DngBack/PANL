"""Route ablation: how does the answer reach the confidence read-out?

This module exists because E0's first design measured the wrong thing, and the way it was
wrong is worth stating plainly.

E0 patched the residual stream at PANL while the answer tokens were still fully visible to
CC. On Qwen2.5-7B that patch moved the confidence margin by only 3 of 30 logits and flipped
no decisions, which looks like "PANL does not matter". It is not. Severing routes shows the
answer reaches CC two ways -- directly, and through PANL -- and *either one alone carries
about 90% of the confidence gap*. The routes are redundant. A patch at PANL was therefore
never going to show much: whatever it removed, the direct route put straight back.

The fix is to make PANL the only carrier and *then* patch it. Under that isolation the same
patch moves ~16 logits and flips 76% of decisions, while the PANL+1 and AC controls stay at
zero. So PANL genuinely holds the information the read-out consumes.

Two claims, and they must not be confused:

  **sufficient** -- PANL alone can carry the confidence signal. Demonstrated.
  **necessary**  -- the intact model needs PANL. *False*: the direct route suffices without it.

A patch-under-isolation result is evidence about what a position *carries*, not proof that
the intact model routes through it. Saying otherwise would be claiming necessity from a
sufficiency experiment.

The one thing route ablation buys that patching cannot: it does not divide by the clean gap,
so a saturated read-out cannot drive it to zero. On stimuli this easy that is the difference
between a measurement and an artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from rich.progress import Progress

from panl.analysis.stats import Estimate, block_bootstrap, block_bootstrap_ratio
from panl.config import ExperimentConfig
from panl.models.adapter import HookedModelAdapter
from panl.models.batching import make_batches
from panl.models.positions import ResolvedPositions

Edge = tuple[str, str]

# The prompt ends `... <answer tokens> \n Confidence :`, and the tokens that can see the answer
# at all are exactly PANL and everything after it. How many tokens that *is* depends on the
# tokenizer -- Qwen splits "Confidence" into "Conf" + "idence", so there are four, not three.
#
# The first version of this file named those tokens individually (CC, PANL1) and thereby left
# the unnamed "idence" token uncut. It can see the answer, CC reads it, and so every knockout
# leaked through it: `only via PANL` was not isolating PANL, and the numbers it produced were
# wrong. Routes are now cut by *what a position is* -- everything after PANL -- so a tokenizer
# that splits the confidence word differently cannot silently reopen a path.
#
#   answer  ->  PANL  ->  [suffix_before_cc]  ->  CC
#      \___________________________________________^   (the direct route)
#
#: Sever every bypass, leaving `answer -> PANL -> ... -> CC` as the only carrier. PANL keeps
#: its view of the answer; nothing after PANL does.
ISOLATE_PANL: tuple[Edge, ...] = (("post_panl", "answer"),)

#: Named route ablations. Each cuts a set of (query set, key set) pairs; the question is always
#: whether the matched-vs-crossed confidence gap survives.
ROUTE_CONDITIONS: dict[str, tuple[Edge, ...]] = {
    "clean": (),
    "cut CC<-answer": (("CC", "answer"),),
    "cut CC<-PANL": (("CC", "PANL"),),
    "cut PANL<-answer": (("PANL", "answer"),),
    "cut suffix<-answer": (("suffix_before_cc", "answer"),),
    # Only `answer -> PANL -> ... -> CC` remains: PANL still sees the answer, nothing after it
    # does, so any answer information reaching CC must have gone through PANL.
    "only via PANL": ISOLATE_PANL,
    # Only the direct edge `answer -> CC` remains: CC still sees the answer, but PANL and the
    # confidence-word tokens are blind to it, so they can carry nothing.
    "only direct": (("PANL", "answer"), ("suffix_before_cc", "answer")),
    # Nothing remains: no query at or after PANL can see the answer. The gap must collapse to
    # ~0. This is the check on the knockout itself -- if it does not, no other row means
    # anything, and the previous version of this file failed it at 19%.
    "cut everything": (("after_answer", "answer"),),
}


@dataclass(slots=True)
class RouteResult:
    conditions: pd.DataFrame
    isolated_patching: pd.DataFrame
    gates: dict[str, Any]


def length_matched_blocks(behavior: pd.DataFrame, resolved: list[ResolvedPositions]) -> set[str]:
    """Blocks whose two answers tokenize to the same number of tokens.

    The `cut everything` condition leaves a 19% residual gap even though no route from the
    answer to CC survives. The obvious suspect is prompt length: a block's two answers usually
    tokenize to different lengths, so CC sits at a different absolute position in the matched
    and crossed prompts, and rotary embeddings make that a real difference the model can see —
    without any answer *content* reaching it.

    Restricting to blocks whose answers are the same length removes that confound entirely:
    for a fixed question, the matched and crossed prompts are then token-for-token the same
    length, and the within-block gap cannot be a position effect.

    If the floor drops to ~0 on this subset, the residual was a length artefact. **If it does
    not, there is a route we have not found**, and that is the most important open question in
    the experiment — not something to explain away.
    """
    lengths = behavior.copy()
    lengths["n_answer_tokens"] = [resolved[i].n_answer_tokens for i in range(len(behavior))]
    keep: set[str] = set()
    for block_id, group in lengths.groupby("block_id"):
        if group["n_answer_tokens"].nunique() == 1:
            keep.add(str(block_id))
    return keep


def _block_gaps(behavior: pd.DataFrame, margins: np.ndarray) -> np.ndarray:
    """Per-block matched-minus-crossed confidence gap. The block stays the resampling unit."""
    frame = pd.DataFrame(
        {
            "block_id": behavior["block_id"].to_numpy(),
            "matched": behavior["matched"].to_numpy(),
            "z": margins,
        }
    )
    wide = frame.groupby(["block_id", "matched"])["z"].mean().unstack()
    return np.asarray((wide[True] - wide[False]).to_numpy(), dtype=np.float64)


def route_ablation(
    model: HookedModelAdapter,
    behavior: pd.DataFrame,
    resolved: list[ResolvedPositions],
    config: ExperimentConfig,
    *,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Run every route condition and report the surviving confidence gap.

    Each condition is scored on two block sets from the same forward passes: every block, and
    only the length-matched ones. The second costs no extra GPU time and is the control that
    tells us whether the 19% floor is a position artefact or an unfound route.
    """
    batches = list(make_batches(resolved, max_batch_size=config.batch_size))
    task = (
        progress.add_task("routes", total=len(ROUTE_CONDITIONS) * len(batches))
        if progress
        else None
    )

    subsets: dict[str, set[str] | None] = {
        "all": None,
        "length_matched": length_matched_blocks(behavior, resolved),
    }

    records: list[dict[str, Any]] = []
    clean_gap: dict[str, float] = {}

    for name, edges in ROUTE_CONDITIONS.items():
        margins = np.full(len(behavior), np.nan)
        for batch in batches:
            out = (
                model.run(batch).confidence_margin
                if not edges
                else model.run_with_knockout(batch, edges=list(edges))
            )
            for offset, row in enumerate(batch.row_indices):
                margins[row] = float(out[offset])
            if progress and task is not None:
                progress.advance(task)

        for subset_name, keep in subsets.items():
            mask = (
                np.ones(len(behavior), dtype=bool)
                if keep is None
                else behavior["block_id"].isin(keep).to_numpy()
            )
            if not mask.any():
                continue
            gaps = _block_gaps(behavior[mask], margins[mask])
            estimate = block_bootstrap(gaps, n_boot=config.n_boot, seed=config.seed)
            clean_gap.setdefault(subset_name, estimate.mean)

            matched = behavior["matched"].to_numpy() & mask
            crossed = (~behavior["matched"].to_numpy()) & mask
            records.append(
                {
                    "condition": name,
                    "subset": subset_name,
                    "edges_cut": len(edges),
                    "gap": estimate.mean,
                    "ci_low": estimate.ci_low,
                    "ci_high": estimate.ci_high,
                    "share_of_clean": estimate.mean / clean_gap[subset_name],
                    "mean_matched": float(np.mean(margins[matched])),
                    "mean_crossed": float(np.mean(margins[crossed])),
                    "n_blocks": estimate.n_blocks,
                }
            )
    return pd.DataFrame(records)


def patch_under_isolation(
    model: HookedModelAdapter,
    behavior: pd.DataFrame,
    resolved: list[ResolvedPositions],
    config: ExperimentConfig,
    *,
    positions: tuple[str, ...] = ("PANL", "PANL1", "AC"),
    layer_step: int = 1,
    cumulative: bool = True,
    progress: Progress | None = None,
) -> pd.DataFrame:
    """Patch each position while PANL is the only route from the answer to CC.

    Source activations are cached from the *unablated* run: the source cell's PANL is what
    that cell's answer actually produced. The knockout applies to the target's forward pass,
    where it stops the target's own answer tokens from re-supplying what the patch replaced.

    Args:
        cumulative: patch layers `[L .. n_layers)` rather than layer `L` alone. This is the
            default and it matters. `ISOLATE_PANL` deliberately leaves `PANL <- answer` open --
            PANL has to be able to read the answer or it could carry nothing -- but that means
            a single-layer patch leaks: at every later layer PANL re-attends to the *target's*
            answer and re-acquires what the patch overwrote. On Qwen2.5-7B the single-layer
            peak is 0.42 (50% of decisions flipped) while freezing L16 onward gives 0.90 (96%).
            The single-layer sweep still answers a different and useful question -- *which*
            layers CC reads PANL at -- so it is kept as an option, not deleted.
    """
    from panl.experiments.e0 import patch_pairs

    pairs = patch_pairs(behavior)
    layers = list(range(0, model.n_layers, layer_step))

    clean = np.full(len(behavior), np.nan)
    for batch in make_batches(resolved, max_batch_size=config.batch_size):
        out = model.run_with_knockout(batch, edges=list(ISOLATE_PANL))
        for offset, row in enumerate(batch.row_indices):
            clean[row] = float(out[offset])

    # The baseline the patch has to move is the clean margin *under the knockout*, not the
    # intact one -- the knockout is part of the regime, not part of the effect.
    isolated = behavior.copy()
    isolated["confidence_margin"] = clean
    pairs = patch_pairs(isolated)

    sources: dict[str, torch.Tensor] = {
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
            sources[position][index] = torch.stack(
                [result.activations[layer][position] for layer in all_layers], dim=1
            ).to(torch.bfloat16)

    target_resolved = [resolved[int(row)] for row in pairs["target_row"]]
    source_rows = pairs["source_row"].to_numpy()
    batches = list(make_batches(target_resolved, max_batch_size=config.batch_size))

    task = (
        progress.add_task("isolated patching", total=len(positions) * len(layers))
        if progress
        else None
    )
    records: list[pd.DataFrame] = []
    for position in positions:
        for layer in layers:
            patch_layers = list(range(layer, model.n_layers)) if cumulative else [layer]
            patched = np.full(len(pairs), np.nan)
            for batch in batches:
                rows = [int(source_rows[i]) for i in batch.row_indices]
                margins = model.run_with_patch(
                    batch,
                    layer=patch_layers,
                    position=position,
                    source=sources[position][rows][:, patch_layers, :],
                    edges=list(ISOLATE_PANL),
                )
                for offset, pair_index in enumerate(batch.row_indices):
                    patched[pair_index] = float(margins[offset])

            frame = pairs.copy()
            frame["layer"] = layer
            frame["position"] = position
            frame["cumulative"] = cumulative
            frame["patched"] = patched
            records.append(frame)
            if progress and task is not None:
                progress.advance(task)

    out = pd.concat(records, ignore_index=True)
    out["moved"] = out["patched"] - out["target_clean"]
    out["gap"] = out["source_clean"] - out["target_clean"]
    out["flipped"] = np.sign(out["patched"]) != np.sign(out["target_clean"])
    return out


def summarize_isolated(patching: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    cumulative = bool(patching["cumulative"].iloc[0]) if "cumulative" in patching else False

    records: list[dict[str, Any]] = []
    restore = patching[patching["direction"] == "restore"]
    for (position, layer), group in restore.groupby(["position", "layer"], sort=True):
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
                "layer": int(layer),
                "cumulative": cumulative,
                "effect": effect.mean,
                "ci_low": effect.ci_low,
                "ci_high": effect.ci_high,
                "mean_moved": float(per_block["moved"].mean()),
                "flip_rate": float(group["flipped"].mean()),
                "n_blocks": effect.n_blocks,
            }
        )
    return pd.DataFrame(records)


def read_cliff(summary: pd.DataFrame, position: str = "PANL", *, tolerance: float = 0.9) -> int:
    """The *last* start layer whose cumulative patch still works.

    With a cumulative patch, every start layer below the read point produces the same effect
    -- a span beginning at L4 contains the span beginning at L16, so they tie. Taking the
    argmax therefore reports an arbitrary member of that tie (L4) and says nothing.

    The informative number is the far edge: the largest L for which freezing `[L..end]` still
    transplants the decision. Beyond it the read has already happened and the patch arrives
    too late. That edge localizes *where CC reads the position*, which is the question.
    """
    rows = summary[summary["position"] == position].sort_values("layer")
    if rows.empty:
        return -1
    peak = float(rows["effect"].max())
    if peak <= 0:
        return -1
    surviving = rows[rows["effect"] >= tolerance * peak]
    return int(surviving["layer"].max()) if len(surviving) else -1


def evaluate_route_gates(conditions: pd.DataFrame, isolated: pd.DataFrame) -> dict[str, Any]:
    """The corrected E0 gates.

    The original gate asked whether patching PANL moves confidence in the intact model. That
    question is unanswerable when a redundant route exists, so it is replaced by two that are
    answerable: does PANL *carry* the signal, and when it is the only carrier, does patching
    it *drive* the read-out?
    """

    def share(name: str, subset: str = "all") -> float:
        rows = conditions[conditions["condition"] == name]
        if "subset" in conditions:
            rows = rows[rows["subset"] == subset]
        return float(rows["share_of_clean"].iloc[0]) if len(rows) else float("nan")

    floor = share("cut everything")
    via_panl = share("only via PANL")
    direct = share("only direct")

    # The floor with the length confound removed. If this is still large, the residual gap is
    # not a position artefact and there is a route we have not found.
    floor_matched = share("cut everything", subset="length_matched")

    panl = isolated[isolated["position"] == "PANL"]
    controls = isolated[isolated["position"].isin(["PANL1", "AC"])]
    peak = panl.loc[panl["effect"].idxmax()] if len(panl) else None
    cliff = read_cliff(isolated, "PANL")

    # With every query at or after PANL blind to the answer, no information can reach CC and
    # the gap must be ~0. A residual here is not a "floor" to read other conditions against --
    # it means a route is still open, which is exactly what the unnamed "idence" token was.
    knockout_works = bool(floor < 0.05)
    panl_sufficient = bool(via_panl > 0.5)
    panl_drives = bool(peak is not None and peak["flip_rate"] >= 0.5)
    controls_flat = bool(len(controls) == 0 or controls["flip_rate"].max() < 0.1)
    # Not part of `overall`: this diagnoses *why* the floor is where it is. A high floor even
    # on length-matched blocks is a finding, not a failure -- but it must be surfaced.
    floor_is_a_length_artefact = bool(
        np.isnan(floor_matched) or floor_matched < 0.5 * max(floor, 1e-6)
    )

    return {
        "knockout_collapses_the_gap": knockout_works,
        "panl_alone_carries_the_signal": panl_sufficient,
        "isolated_panl_patch_flips_decisions": panl_drives,
        "controls_stay_flat": controls_flat,
        "overall": knockout_works and panl_sufficient and panl_drives and controls_flat,
        "gap_floor": floor,
        "gap_floor_length_matched": floor_matched,
        "floor_is_a_length_artefact": floor_is_a_length_artefact,
        "share_only_via_panl": via_panl,
        "share_only_direct": direct,
        # The headline, and the thing the original E0 gate got wrong: PANL is sufficient but
        # not necessary, because the direct route carries the signal on its own.
        "panl_is_necessary": bool(direct < 0.5),
        "panl_peak_effect": float(peak["effect"]) if peak is not None else float("nan"),
        "panl_peak_flip_rate": float(peak["flip_rate"]) if peak is not None else float("nan"),
        # Where CC reads PANL. With a cumulative patch every start layer below the read point
        # ties, so the argmax layer is meaningless; the cliff edge is the localization.
        "panl_read_cliff": cliff,
    }


__all__ = [
    "ISOLATE_PANL",
    "ROUTE_CONDITIONS",
    "Estimate",
    "RouteResult",
    "evaluate_route_gates",
    "patch_under_isolation",
    "route_ablation",
    "summarize_isolated",
]
