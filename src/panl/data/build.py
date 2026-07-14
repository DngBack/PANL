"""End-to-end Tier-1 build: facts -> blocks -> splits -> Parquet + manifest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from panl.artifacts.manifest import build_manifest, write_manifest
from panl.config import DataBuildConfig
from panl.data.blocks import Block, build_blocks
from panl.data.facts import check_fact_base, facts_by_family
from panl.data.schema import Quadruple, Split, to_table, write_table
from panl.data.splits import assign_facts
from panl.data.validate import ValidationReport, validate_table


@dataclass(slots=True)
class BuildResult:
    rows: list[Quadruple]
    blocks: list[Block]
    table: pa.Table
    report: ValidationReport
    parquet_path: Path | None
    manifest_path: Path | None


def build_rows(config: DataBuildConfig) -> tuple[list[Quadruple], list[Block]]:
    """Build the in-memory rows. No file system side effects; used directly by tests.

    Facts are split first, then paired into blocks *within* each (family, split) pool. Pairing
    across the whole family and splitting afterwards would let a block's two facts land in
    different splits, and reusing a fact across pairing rounds would then leak its identity.
    """
    fact_violations = check_fact_base()
    if fact_violations:
        joined = "\n  - ".join(fact_violations)
        msg = f"the Tier-1 fact base is inconsistent:\n  - {joined}"
        raise ValueError(msg)

    grouped = facts_by_family(config.families)
    fact_split = assign_facts(
        grouped,
        family_holdout=config.family_holdout,
        ratios=config.split_ratios.to_ratios(),
        seed=config.seed,
    )

    rows: list[Quadruple] = []
    blocks: list[Block] = []
    for split in Split:
        pools = {
            family: [f for f in facts if fact_split[f.fact_id] is split]
            for family, facts in grouped.items()
        }
        pools = {family: facts for family, facts in pools.items() if len(facts) >= 2}
        if not pools:
            continue
        pool_blocks = build_blocks(
            pools,
            pairings_per_fact=config.pairings_per_fact,
            templates_per_family=config.templates_per_family,
            seed=f"{config.seed}:{split.value}",
        )
        blocks.extend(pool_blocks)
        for block in pool_blocks:
            rows.extend(block.rows(dataset_tier=config.tier, split=split))

    return rows, blocks


def build_dataset(
    config: DataBuildConfig,
    *,
    config_path: Path | None = None,
    output: Path | None = None,
    write: bool = True,
) -> BuildResult:
    rows, blocks = build_rows(config)

    parquet_path = (output or config.output) if write else None
    manifest_path: Path | None = None

    table = write_table(rows, parquet_path) if parquet_path is not None else to_table(rows)
    report = validate_table(table)

    if parquet_path is not None:
        manifest_path = parquet_path.with_suffix(".manifest.json")
        manifest = build_manifest(
            command="data build",
            config=config,
            config_path=config_path,
            artifacts={"quadruples": parquet_path},
            payload={
                "n_rows": len(rows),
                "n_blocks": len(blocks),
                "counts_by_split": report.counts_by_split,
                "counts_by_family": report.counts_by_family,
                "validation_ok": report.ok,
                "violations": report.violations,
            },
        )
        write_manifest(manifest, manifest_path)

    return BuildResult(
        rows=rows,
        blocks=blocks,
        table=table,
        report=report,
        parquet_path=parquet_path,
        manifest_path=manifest_path,
    )
