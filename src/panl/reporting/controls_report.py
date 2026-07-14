"""Rendering of the source-control result. Presentation only."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

KINDS = ("partner", "gaussian", "mean", "random_cell", "matched_donor", "crossed_donor")

KIND_LABELS = {
    "partner": "partner cell (the intervention)",
    "gaussian": "random direction, norm-matched",
    "mean": "dataset-mean PANL",
    "random_cell": "any cell, another block",
    "matched_donor": "a MATCHED cell, another block",
    "crossed_donor": "a CROSSED cell, another block",
}

KIND_QUESTIONS = {
    "gaussian": "content, or just a large write?",
    "mean": "transplant, or just ablation?",
    "random_cell": "this answer, or any answer?",
    "matched_donor": "'it fits' -- about THIS pair?",
    "crossed_donor": "'it does not fit' -- about THIS pair?",
}

GATE_LABELS = {
    "bidirectional": "the effect runs both ways (restore and ablate)",
    "beats_destructive_controls": "in ABLATE, the partner beats noise and mean-ablation",
    "effect_is_pair_specific": "in ABLATE, the partner beats a crossed cell from another block",
}


def render_controls(
    console: Console, summary: pd.DataFrame, gates: dict[str, Any], *, start_layer: int
) -> None:
    console.print(f"\n[bold]Source controls[/] (PANL isolated, frozen from L{start_layer} onward)")
    console.print(
        "  [dim]Each control replaces PANL with something of the right size but the wrong "
        "content.\n  The intervention has to beat all of them, or it is not about the "
        "partner's answer.[/]"
    )

    panl = summary[summary["position"] == "PANL"]
    table = Table("source", "asks", "restore: effect / flip", "ablate: effect / flip")
    for kind in KINDS:
        rows = panl[panl["source_kind"] == kind]
        if rows.empty:
            continue

        def cell(direction: str, rows: pd.DataFrame = rows) -> str:
            row = rows[rows["direction"] == direction]
            if row.empty:
                return "-"
            return f"{float(row['effect'].iloc[0]):+.3f} / {float(row['flip_rate'].iloc[0]):.0%}"

        table.add_row(
            KIND_LABELS[kind],
            KIND_QUESTIONS.get(kind, ""),
            cell("restore"),
            cell("ablate"),
        )
    console.print(table)

    console.print("\n[bold]Position controls[/] [dim](partner source, same frozen span)[/]")
    position_table = Table("position", "restore: effect / flip", "ablate: effect / flip")
    for position in ("PANL", "PANL1", "AC"):
        rows = summary[(summary["position"] == position) & (summary["source_kind"] == "partner")]
        if rows.empty:
            continue

        def cell(direction: str, rows: pd.DataFrame = rows) -> str:
            row = rows[rows["direction"] == direction]
            if row.empty:
                return "-"
            return f"{float(row['effect'].iloc[0]):+.3f} / {float(row['flip_rate'].iloc[0]):.0%}"

        position_table.add_row(position, cell("restore"), cell("ablate"))
    console.print(position_table)

    console.print("\n[bold]Gate[/]")
    for key, label in GATE_LABELS.items():
        mark = "[green]PASS[/]" if gates[key] else "[red]FAIL[/]"
        console.print(f"  {mark}  {label}")

    if gates["restore_is_confounded"]:
        console.print(
            "\n  [yellow]The RESTORE direction is confounded.[/] A norm-matched random vector "
            f"flips {gates['control_flips']['gaussian']['restore']:.0%} of restore decisions -- "
            "as many as the real patch.\n  The model is confident *by default*: PANL carries "
            "doubt, not confidence, so destroying it with\n  anything makes a diffident model "
            "confident. Only ABLATE is interpretable."
        )
    if not gates["effect_is_pair_specific"]:
        crossed = gates["control_flips"]["crossed_donor"]["ablate"]
        console.print(
            f"\n  [red]The effect is NOT pair-specific.[/] A crossed cell from an unrelated "
            f"block -- a different\n  question, a different answer -- removes confidence in "
            f"{crossed:.0%} of cases, against "
            f"{gates['partner_flip_ablate']:.0%} for\n  the true partner. PANL holds a generic "
            "doubt signal, not information about *this* pair.\n  The relational Q x A component "
            "is not what is cached here."
        )
