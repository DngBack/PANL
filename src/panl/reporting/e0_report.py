"""Rendering of the E0 result. Presentation only -- no statistics are computed here."""

from __future__ import annotations

import numpy as np
from rich.console import Console
from rich.table import Table

from panl.experiments.collect import CollectionResult
from panl.experiments.e0 import E0Result
from panl.models.positions import POSITION_NAMES

GATE_LABELS = {
    "confidence_signal": "confidence contrast is positive and its CI excludes zero",
    "panl_effect_floor": "PANL patching effect clears the floor",
    "panl_beats_panl1": "PANL beats the PANL+1 control",
    "panl_beats_ac": "PANL beats the AC control",
    "cc_sanity": "CC (the read-out position) is at least as strong as PANL",
}


def render_e0(console: Console, result: E0Result, collected: CollectionResult) -> None:
    signal = result.signal

    console.print(
        f"\n[bold]Behavioural signal[/] "
        f"({collected.n_rows} cells / {signal['contrasts'].shape[0]} blocks)"
    )
    console.print(
        f"  mean confidence margin: matched {signal['mean_margin_matched']:+.3f}, "
        f"crossed {signal['mean_margin_crossed']:+.3f}"
    )
    console.print(f"  interaction I_k = z11-z12-z21+z22: {signal['interaction']}")
    console.print(f"  Delta_fit (= I_k / 2):             {signal['delta_fit']}")
    console.print(f"  paired effect size (d_z):          {signal['effect_size']:.2f}")
    console.print(f"  calibration AUC (margin vs correct): {signal['calibration_auc']:.3f}")

    restore = result.summary[result.summary["direction"] == "restore"]
    console.print("\n[bold]Patching by position[/] (restore direction, peak layer)")
    console.print(
        "  [dim]effect: 1.0 = the patch fully transplanted the source's confidence. "
        "It divides by the clean gap,\n  so read it next to the absolute shift and the flip "
        "rate, which do not.[/]"
    )

    table = Table("position", "effect", "95% CI", "logits moved", "flip rate", "best layer")
    for position in POSITION_NAMES:
        rows = restore[restore["position"] == position]
        if rows.empty:
            continue
        best = rows.loc[rows["effect"].idxmax()]
        table.add_row(
            position,
            f"{best['effect']:.3f}",
            f"[{best['ci_low']:.3f}, {best['ci_high']:.3f}]",
            f"{best['mean_moved']:+.2f}",
            f"{best['flip_rate']:.0%}",
            str(int(best["layer"])),
        )
    console.print(table)

    _render_layer_profile(console, restore)

    if result.gates["saturated"]:
        gap = result.gates["mean_clean_gap"]
        moved = result.gates["panl_absolute_logits_moved"]
        console.print(
            f"\n[yellow]SATURATED READ-OUT[/] the mean clean gap is {gap:.1f} logits, so the "
            f"confidence\n  read-out is effectively a step function and the normalized effect "
            f"is misleading:\n  the PANL patch moves {moved:+.2f} logits but that is only "
            f"{moved / gap:.0%} of a gap this wide.\n  The distractors are too easy. Build the "
            f"Tier-2 hard-confusion set (plan section 3.2)\n  before reading the PANL gate as "
            f"a statement about the mechanism."
        )

    console.print("\n[bold]Gate E0[/]")
    for key, label in GATE_LABELS.items():
        ok = bool(result.gates[key])
        mark = "[green]PASS[/]" if ok else "[red]FAIL[/]"
        console.print(f"  {mark}  {label}")


def _render_layer_profile(console: Console, restore) -> None:  # type: ignore[no-untyped-def]
    """A coarse per-layer sparkline for PANL and its two controls.

    The shape matters as much as the peak: a real cache should switch on somewhere in the
    middle of the network and stay on, not spike at a single layer.
    """
    console.print("\n[bold]PANL vs controls, by layer[/]")
    for position in ("PANL", "PANL1", "AC"):
        rows = restore[restore["position"] == position].sort_values("layer")
        if rows.empty:
            continue
        effects = rows["effect"].to_numpy()
        console.print(f"  {position:5} {_sparkline(effects)}  max={effects.max():.2f}")


def _sparkline(values: np.ndarray) -> str:
    """Map the fixed range [0, 1] to blocks, so the three rows are directly comparable."""
    blocks = " ▁▂▃▄▅▆▇█"
    clipped = np.clip(values, 0.0, 1.0)
    indices = np.rint(clipped * (len(blocks) - 1)).astype(int)
    return "".join(blocks[i] for i in indices)
