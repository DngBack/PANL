"""Rendering of the route-ablation result. Presentation only."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from panl.experiments.routes import read_cliff

GATE_LABELS = {
    "knockout_collapses_the_gap": "severing every route collapses the confidence gap",
    "panl_alone_carries_the_signal": "PANL alone carries the confidence gap",
    "isolated_panl_patch_flips_decisions": (
        "patching PANL, when it is the only route, flips decisions"
    ),
    "controls_stay_flat": "the PANL+1 and AC controls stay flat",
}


def render_routes(
    console: Console,
    conditions: pd.DataFrame,
    summary: pd.DataFrame,
    gates: dict[str, Any],
    *,
    n_blocks: int,
) -> None:
    console.print(f"\n[bold]Route ablation[/] ({n_blocks} blocks)")
    console.print(
        "  [dim]the matched-minus-crossed confidence gap surviving each cut. "
        "Unlike a patching effect,\n  this never divides by the clean gap, so a saturated "
        "read-out cannot drive it to zero.[/]"
    )

    has_subsets = "subset" in conditions
    full = conditions[conditions["subset"] == "all"] if has_subsets else conditions
    matched = conditions[conditions["subset"] == "length_matched"] if has_subsets else None

    table = Table("condition", "gap (logits)", "95% CI", "% of clean")
    if matched is not None and len(matched):
        table.add_column("% of clean (length-matched)")

    for row in full.itertuples():
        cells = [
            str(row.condition),
            f"{row.gap:+.2f}",
            f"[{row.ci_low:+.2f}, {row.ci_high:+.2f}]",
            f"{row.share_of_clean:.0%}",
        ]
        if matched is not None and len(matched):
            peer = matched[matched["condition"] == row.condition]
            cells.append(f"{float(peer['share_of_clean'].iloc[0]):.0%}" if len(peer) else "-")
        table.add_row(*cells)
    console.print(table)

    if matched is not None and len(matched):
        n_matched = int(matched["n_blocks"].iloc[0])
        console.print(
            f"  [dim]length-matched = the {n_matched} blocks whose two answers tokenize to the "
            f"same length, so\n  matched and crossed prompts are the same length and CC sits at "
            f"the same absolute position.[/]"
        )
        artefact = bool(gates["floor_is_a_length_artefact"])
        floor, floor_lm = gates["gap_floor"], gates["gap_floor_length_matched"]
        if artefact:
            console.print(
                f"\n  [green]The {floor:.0%} floor is a length artefact.[/] With answer lengths "
                f"matched it falls to {floor_lm:.0%}:\n  the residual gap was rotary position, "
                f"not answer content reaching CC by some other path."
            )
        else:
            console.print(
                f"\n  [yellow]The floor survives length matching[/] ({floor:.0%} -> "
                f"{floor_lm:.0%}). Answer information is still\n  reaching CC after every known "
                f"route is severed. There is a path we have not found, and\n  finding it is now "
                f"the most important open question in this experiment."
            )

    cumulative = bool(summary["cumulative"].iloc[0]) if "cumulative" in summary else False
    span = "L..end" if cumulative else "one layer"
    console.print(
        f"\n[bold]Patching with PANL isolated[/] "
        f"(answer -> PANL -> CC is the only route; patch spans {span})"
    )
    if cumulative:
        console.print(
            "  [dim]PANL still reads the answer at every layer, so patching one layer leaks: "
            "PANL simply\n  re-acquires what was overwritten. Freezing the whole trajectory "
            "from L onward closes that.[/]"
        )

    patch_table = Table("position", "peak effect", "logits moved", "flip rate", "widest span")
    for position in ("PANL", "PANL1", "AC"):
        rows = summary[summary["position"] == position]
        if rows.empty:
            continue
        # With a cumulative patch, every start layer below the read point ties at the peak, so
        # the argmax layer is arbitrary. Report the *last* start layer that still works.
        edge = (
            read_cliff(summary, position)
            if cumulative
            else int(rows.loc[rows["effect"].idxmax(), "layer"])
        )
        best = rows.loc[rows["effect"].idxmax()]
        patch_table.add_row(
            position,
            f"{best['effect']:.3f}",
            f"{best['mean_moved']:+.2f}",
            f"{best['flip_rate']:.0%}",
            f"L{edge}..end" if cumulative else f"L{edge}",
        )
    console.print(patch_table)

    panl = summary[summary["position"] == "PANL"].sort_values("layer")
    if not panl.empty:
        header = "from each start layer" if cumulative else "at each layer"
        console.print(f"\n[bold]PANL effect {header}[/] [dim](when it is the only route)[/]")
        for row in panl.itertuples():
            bar = "█" * int(max(0.0, min(1.0, row.effect)) * 34)
            label = f"L{int(row.layer)}..end" if cumulative else f"L{int(row.layer)}"
            console.print(f"  {label:<9} {row.effect:>6.2f}  flip={row.flip_rate:>4.0%}  {bar}")

        if cumulative:
            edge = int(gates["panl_read_cliff"])
            console.print(
                f"\n  [dim]The cliff is the result. Freezing PANL from L{edge} onward still "
                f"transplants the decision;\n  starting a few layers later does nothing -- by "
                f"then CC has already read it. So CC reads PANL\n  at roughly layers "
                f"{edge + 1}-{edge + 4}.[/]"
            )

    console.print("\n[bold]Gate[/]")
    for key, label in GATE_LABELS.items():
        mark = "[green]PASS[/]" if gates[key] else "[red]FAIL[/]"
        console.print(f"  {mark}  {label}")

    console.print("\n[bold]What this licenses you to say[/]")
    console.print(
        f"  PANL is [bold]sufficient[/]: on its own it carries "
        f"{gates['share_only_via_panl']:.0%} of the confidence gap, and freezing it under "
        f"isolation\n  flips {gates['panl_peak_flip_rate']:.0%} of decisions."
    )
    if not gates["panl_is_necessary"]:
        console.print(
            f"  PANL is [bold]not necessary[/]: the direct route answer->CC carries "
            f"{gates['share_only_direct']:.0%} of the gap without it.\n"
            f"  So the intact model is [bold]not[/] shown to route through PANL. A patch-under-"
            f"isolation result is\n  evidence about what a position carries, not proof that the "
            f"unablated model uses it."
        )
