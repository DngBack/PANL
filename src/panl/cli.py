"""`panl` command line entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from panl.config import load_data_config
from panl.data.build import build_dataset
from panl.data.validate import validate_path
from panl.models.confidence import ConfidenceTokenError, resolve_confidence_classes
from panl.models.positions import POSITION_NAMES, PositionResolutionError, resolve_positions
from panl.models.prompts import PromptRenderer, PromptStyle, PromptTemplate
from panl.models.snapshot import SNAPSHOT_CASES, build_snapshot
from panl.models.tokenizer import load_tokenizer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="PANL experiment tooling.")
data_app = typer.Typer(no_args_is_help=True, help="Build and validate the quadruple table.")
positions_app = typer.Typer(no_args_is_help=True, help="Resolve and freeze semantic positions.")
e0_app = typer.Typer(no_args_is_help=True, help="E0: pipeline reproduction and calibration.")
app.add_typer(data_app, name="data")
app.add_typer(positions_app, name="positions")
app.add_typer(e0_app, name="e0")

console = Console()
err = Console(stderr=True)


@data_app.command("build")
def data_build(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path | None, typer.Option("--out", dir_okay=False)] = None,
) -> None:
    """Build data/processed/quadruples.parquet plus its run manifest."""
    cfg = load_data_config(config)
    result = build_dataset(cfg, config_path=config, output=out)

    console.print(f"[bold]{len(result.blocks)}[/] blocks, [bold]{len(result.rows)}[/] rows")
    table = Table("split", "blocks", "rows")
    for split, rows in sorted(result.report.counts_by_split.items()):
        table.add_row(split, str(rows // 4), str(rows))
    console.print(table)
    console.print(f"wrote {result.parquet_path}")
    console.print(f"wrote {result.manifest_path}")

    if not result.report.ok:
        err.print("[red]the freshly built table violates its own invariants[/]")
        err.print(result.report.summary())
        raise typer.Exit(code=1)
    console.print("[green]validation OK[/]")


@data_app.command("validate")
def data_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Check every invariant of plan section 7 against an existing table."""
    report = validate_path(path)
    console.print(report.summary())
    if not report.ok:
        raise typer.Exit(code=1)


@positions_app.command("check")
def positions_check(
    model: Annotated[str, typer.Option("--model")] = "Qwen/Qwen2.5-0.5B-Instruct",
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    style: Annotated[PromptStyle, typer.Option("--style")] = PromptStyle.CHAT,
    show_prompt: Annotated[bool, typer.Option("--show-prompt")] = False,
) -> None:
    """Resolve AC/LAT/PANL/PANL+1/CC for this tokenizer and report what it found.

    Exits non-zero if any semantic boundary is destroyed by tokenization -- the Jul 14-15
    kill gate: "stop if target prompts cannot be reproduced".
    """
    tokenizer = load_tokenizer(model, revision=revision)
    renderer = PromptRenderer(tokenizer, template=PromptTemplate(), style=style)

    try:
        classes = resolve_confidence_classes(tokenizer)
    except ConfidenceTokenError as exc:
        err.print(f"[red]confidence classes unusable:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[bold]{model}[/] | style={style.value} | "
        f"confidence classes: {classes.high_token!r}={classes.high_token_id} "
        f"{classes.low_token!r}={classes.low_token_id}"
    )

    table = Table("question", "answer", "n_tok", *POSITION_NAMES)
    failures = 0
    for question, answer in SNAPSHOT_CASES:
        prompt = renderer.render(question, answer)
        if show_prompt:
            console.print(f"\n[dim]{prompt.text!r}[/]")
        try:
            resolved = resolve_positions(tokenizer, prompt)
        except PositionResolutionError as exc:
            failures += 1
            err.print(f"[red]FAIL[/] {question!r} / {answer!r}: {exc}")
            continue
        table.add_row(
            question if len(question) <= 34 else question[:31] + "...",
            answer,
            str(resolved.n_tokens),
            *(f"{resolved.indices[name]}:{resolved.tokens[name]!r}" for name in POSITION_NAMES),
        )

    console.print(table)
    if failures:
        err.print(f"[red]{failures}/{len(SNAPSHOT_CASES)} prompts failed position resolution[/]")
        raise typer.Exit(code=1)
    console.print(f"[green]all {len(SNAPSHOT_CASES)} prompts resolved[/]")


@positions_app.command("snapshot")
def positions_snapshot(
    out: Annotated[Path, typer.Option("--out", dir_okay=False)],
    model: Annotated[str, typer.Option("--model")] = "Qwen/Qwen2.5-0.5B-Instruct",
    revision: Annotated[str | None, typer.Option("--revision")] = None,
    style: Annotated[PromptStyle, typer.Option("--style")] = PromptStyle.CHAT,
) -> None:
    """Freeze the resolved positions for this tokenizer into a JSON fixture."""
    tokenizer = load_tokenizer(model, revision=revision)
    snapshot = build_snapshot(tokenizer, model_id=model, style=style)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    console.print(f"wrote {out} ({len(snapshot['cases'])} cases)")


@e0_app.command("run")
def e0_run(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    n_blocks: Annotated[int | None, typer.Option("--n-blocks")] = None,
    skip_activations: Annotated[bool, typer.Option("--skip-activations")] = False,
) -> None:
    """Collect behaviour and activations, sweep the patching grid, and check the E0 gates.

    Everything E0 needs happens in one process because the patching sweep must patch with the
    model's own bf16 activations, not with the float16 values written to the store.
    """
    from rich.progress import (
        BarColumn,
        Progress,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    from panl.artifacts.manifest import build_manifest, write_manifest
    from panl.config import load_experiment_config
    from panl.data.schema import read_table
    from panl.experiments.collect import collect, store_metadata
    from panl.experiments.e0 import run_e0
    from panl.models.adapter import HookedModelAdapter
    from panl.reporting.e0_report import render_e0

    cfg = load_experiment_config(config)
    if n_blocks is not None:
        cfg = cfg.model_copy(update={"n_blocks": n_blocks})

    run = run_id or f"e0-{cfg.model.role}-{cfg.n_blocks}b"
    out_dir = cfg.output_root / run
    out_dir.mkdir(parents=True, exist_ok=True)

    quadruples = read_table(cfg.quadruples).to_pandas()
    console.print(f"[bold]{cfg.model.model_id}[/] | run [bold]{run}[/] -> {out_dir}")

    model = HookedModelAdapter.load(cfg.model)
    console.print(
        f"loaded: {model.n_layers} layers, d_model={model.d_model}, device={model.device} | "
        f"confidence classes {model.classes.high_token!r}/{model.classes.low_token!r}"
    )

    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console) as progress:
        collected = collect(
            model,
            quadruples,
            cfg,
            activations_path=None if skip_activations else out_dir / "activations.zarr",
            progress=progress,
        )
        result = run_e0(model, collected.behavior, collected.resolved, cfg, progress=progress)

    behavior_path = out_dir / "behavior.parquet"
    patching_path = out_dir / "patching.parquet"
    summary_path = out_dir / "patching_summary.parquet"
    collected.behavior.to_parquet(behavior_path, index=False)
    result.patching.to_parquet(patching_path, index=False)
    result.summary.to_parquet(summary_path, index=False)

    render_e0(console, result, collected)

    manifest = build_manifest(
        command="e0 run",
        config=cfg,
        config_path=config,
        artifacts={
            "behavior": behavior_path,
            "patching": patching_path,
            "patching_summary": summary_path,
        },
        payload={
            **store_metadata(collected),
            "gates": result.gates,
            "interaction": str(result.signal["interaction"]),
            "calibration_auc": result.signal["calibration_auc"],
        },
    )
    write_manifest(manifest, out_dir / "manifest.json")
    console.print(
        f"\nwrote {out_dir}/{{behavior,patching,patching_summary}}.parquet, manifest.json"
    )

    if not result.passed:
        err.print(
            "\n[red]GATE E0 FAILED[/] -- do not interpret E1-E4. "
            "Debug prompt fidelity and position resolution first."
        )
        raise typer.Exit(code=1)
    console.print("\n[green]GATE E0 PASSED[/]")


@app.command("score")
def score(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)] = Path("outputs/scores.parquet"),
) -> None:
    """Score every candidate fact against the model before any block is built.

    Answers two questions E0 proved we cannot assume: does the model produce our answer format
    on its own, and does it actually find any of these facts hard?
    """
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

    from panl.config import load_experiment_config
    from panl.experiments.scoring import score_facts
    from panl.models.adapter import HookedModelAdapter

    cfg = load_experiment_config(config)
    model = HookedModelAdapter.load(cfg.model)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        result = score_facts(model, cfg, progress=progress)

    out.parent.mkdir(parents=True, exist_ok=True)
    result.scores.to_parquet(out, index=False)

    stats = result.summary()
    console.print(f"\n[bold]{stats['n_facts']}[/] facts scored on {cfg.model.model_id}")
    console.print("\n[bold]Is the format on-policy?[/]")
    console.print(f"  greedy answer == gold string: {stats['on_policy_exact']:.0%}")
    console.print(
        f"  greedy answer == gold entity: {stats['on_policy_entity']:.0%}  "
        f"[dim](accent/case-insensitive)[/]"
    )
    console.print(f"  model errors (a different entity): {stats['model_errors']}")

    console.print("\n[bold]Is the read-out saturated?[/]")
    console.print(
        f"  median confidence margin on the gold answer: {stats['median_gold_margin']:+.1f} logits"
    )
    console.print(f"  facts with |margin| < 10 logits: {stats['unsaturated_fraction']:.0%}")

    table = Table("family", "facts", "on-policy", "model errors", "median gold margin")
    for family, group in result.scores.groupby("relation_family", sort=True):
        table.add_row(
            str(family),
            str(len(group)),
            f"{group['greedy_entity'].mean():.0%}",
            str(int(group["is_model_error"].sum())),
            f"{group['gold_margin'].median():+.1f}",
        )
    console.print(table)
    console.print(f"\nwrote {out}")


@e0_app.command("routes")
def e0_routes(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    layer_step: Annotated[int, typer.Option("--layer-step")] = 2,
) -> None:
    """Sever the routes from the answer to CC, then patch PANL when it is the only one left.

    This supersedes the patching sweep in `e0 run` as the localization evidence. That sweep
    patches PANL with the answer tokens still visible to CC, and a redundant direct route
    simply re-supplies whatever the patch removed -- so it reports a null whatever PANL
    contains. Route ablation does not divide by the clean gap and so is not distorted by a
    saturated read-out either.
    """
    from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

    from panl.artifacts.manifest import build_manifest, write_manifest
    from panl.config import load_experiment_config
    from panl.data.schema import read_table
    from panl.experiments.collect import collect
    from panl.experiments.routes import (
        evaluate_route_gates,
        patch_under_isolation,
        route_ablation,
        summarize_isolated,
    )
    from panl.models.adapter import HookedModelAdapter
    from panl.reporting.routes_report import render_routes

    cfg = load_experiment_config(config)
    run = run_id or f"routes-{cfg.model.role}"
    out_dir = cfg.output_root / run
    out_dir.mkdir(parents=True, exist_ok=True)

    quadruples = read_table(cfg.quadruples).to_pandas()
    model = HookedModelAdapter.load(cfg.model)
    console.print(f"[bold]{cfg.model.model_id}[/] | run [bold]{run}[/] -> {out_dir}")

    columns = (
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console) as progress:
        collected = collect(model, quadruples, cfg, activations_path=None, progress=progress)
        conditions = route_ablation(
            model, collected.behavior, collected.resolved, cfg, progress=progress
        )
        isolated = patch_under_isolation(
            model,
            collected.behavior,
            collected.resolved,
            cfg,
            layer_step=layer_step,
            progress=progress,
        )

    summary = summarize_isolated(isolated, cfg)
    gates = evaluate_route_gates(conditions, summary)

    conditions.to_parquet(out_dir / "route_conditions.parquet", index=False)
    isolated.to_parquet(out_dir / "isolated_patching.parquet", index=False)
    summary.to_parquet(out_dir / "isolated_summary.parquet", index=False)

    render_routes(
        console, conditions, summary, gates, n_blocks=int(collected.behavior["block_id"].nunique())
    )

    write_manifest(
        build_manifest(
            command="e0 routes",
            config=cfg,
            config_path=config,
            artifacts={"conditions": out_dir / "route_conditions.parquet"},
            payload={"gates": gates, "n_blocks": int(collected.behavior["block_id"].nunique())},
        ),
        out_dir / "manifest.json",
    )
    console.print(f"\nwrote {out_dir}/")

    if not gates["overall"]:
        raise typer.Exit(code=1)


@e0_app.command("routes-report")
def e0_routes_report(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Re-render the route-ablation report from a finished run. No GPU, no model.

    A change to how the result is summarized must never cost GPU time twice.
    """
    import pandas as pd

    from panl.config import load_experiment_config
    from panl.experiments.routes import evaluate_route_gates, summarize_isolated
    from panl.reporting.routes_report import render_routes

    cfg = load_experiment_config(config)
    conditions = pd.read_parquet(run_dir / "route_conditions.parquet")
    isolated = pd.read_parquet(run_dir / "isolated_patching.parquet")

    summary = summarize_isolated(isolated, cfg)
    gates = evaluate_route_gates(conditions, summary)
    summary.to_parquet(run_dir / "isolated_summary.parquet", index=False)

    render_routes(console, conditions, summary, gates, n_blocks=int(isolated["block_id"].nunique()))
    if not gates["overall"]:
        raise typer.Exit(code=1)


@e0_app.command("report")
def e0_report(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
) -> None:
    """Re-render the E0 report from a finished run. No GPU, no model, no re-collection.

    Exists so that a change to the summary statistics or the gates can be re-applied to a run
    that already cost GPU time, rather than re-running it.
    """
    import pandas as pd

    from panl.config import load_experiment_config
    from panl.experiments.collect import CollectionResult
    from panl.experiments.e0 import E0Result, confidence_signal, evaluate_gates, summarize_patching
    from panl.reporting.e0_report import render_e0

    cfg = load_experiment_config(config)
    behavior = pd.read_parquet(run_dir / "behavior.parquet")
    patching = pd.read_parquet(run_dir / "patching.parquet")

    summary = summarize_patching(patching, cfg)
    signal = confidence_signal(behavior, cfg)
    result = E0Result(
        signal=signal,
        patching=patching,
        summary=summary,
        gates=evaluate_gates(signal, summary),
    )
    summary.to_parquet(run_dir / "patching_summary.parquet", index=False)

    collected = CollectionResult(
        behavior=behavior,
        resolved=[],
        activations_path=None,
        activations_sha256=None,
        n_rows=len(behavior),
        n_layers=int(patching["layer"].max()) + 1,
        d_model=0,
    )
    render_e0(console, result, collected)
    if not result.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
