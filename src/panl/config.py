"""Typed configuration objects, loaded from YAML.

Configs are hashed into the run manifest, so they must serialize deterministically. That is
why `extra="forbid"` is set everywhere: a typo in a YAML key must fail the run rather than
be silently dropped and then recorded in the manifest as if it had taken effect.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from panl.data.facts import RELATION_FAMILIES
from panl.data.splits import SplitRatios
from panl.models.positions import POSITION_NAMES
from panl.models.spec import ModelSpec


class SplitRatioConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def to_ratios(self) -> SplitRatios:
        return SplitRatios(train=self.train, validation=self.validation, test=self.test)


class DataBuildConfig(BaseModel):
    """Everything needed to reproduce `data/processed/quadruples.parquet`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: int = Field(default=1, ge=1, le=3)
    seed: int = 20260714
    families: tuple[str, ...] = RELATION_FAMILIES
    #: How many blocks each fact takes part in. Raising this multiplies the block count but
    #: also fuses the identity graph, which can collapse the splits (see splits.py).
    pairings_per_fact: int = Field(default=2, ge=1)
    templates_per_family: int = Field(default=3, ge=1)
    family_holdout: tuple[str, ...] = ("inventor_of",)
    split_ratios: SplitRatioConfig = SplitRatioConfig()
    output: Path = Path("data/processed/quadruples.parquet")

    @model_validator(mode="after")
    def _check_families(self) -> Self:
        unknown = set(self.families) - set(RELATION_FAMILIES)
        if unknown:
            msg = f"unknown relation families: {sorted(unknown)}"
            raise ValueError(msg)
        stray_holdout = set(self.family_holdout) - set(self.families)
        if stray_holdout:
            msg = f"family_holdout names families that are not built: {sorted(stray_holdout)}"
            raise ValueError(msg)
        if set(self.family_holdout) == set(self.families):
            msg = "every family is held out; nothing would remain to fit on"
            raise ValueError(msg)
        # Ratios are validated by SplitRatios itself; surface the error at config load time.
        self.split_ratios.to_ratios()
        return self


class ExperimentConfig(BaseModel):
    """One E0 run: which model, which cells, how much compute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment: Literal["e0"] = "e0"
    model: ModelSpec = ModelSpec()
    quadruples: Path = Path("data/processed/quadruples.parquet")

    #: E0 is a pipeline check, so it runs on train. The test set is opened once, by E1-E3.
    splits: tuple[str, ...] = ("train",)
    families: tuple[str, ...] = ()
    #: 32 blocks for CI/smoke, 200 for the pilot (plan section E0).
    n_blocks: int | None = 32
    batch_size: int = 32

    #: Layers to sweep in the patching experiment. Empty means every layer.
    patch_layers: tuple[int, ...] = ()
    patch_positions: tuple[str, ...] = POSITION_NAMES

    seed: int = 20260714
    n_boot: int = 10_000
    output_root: Path = Path("outputs")

    @model_validator(mode="after")
    def _check_positions(self) -> Self:
        unknown = set(self.patch_positions) - set(POSITION_NAMES)
        if unknown:
            msg = f"unknown semantic positions: {sorted(unknown)}"
            raise ValueError(msg)
        if self.n_blocks is not None and self.n_blocks < 2:
            msg = f"n_blocks must be at least 2 to bootstrap, got {self.n_blocks}"
            raise ValueError(msg)
        return self


def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path}: expected a YAML mapping, got {type(raw).__name__}"
        raise TypeError(msg)
    return ExperimentConfig.model_validate(raw)


def load_data_config(path: Path) -> DataBuildConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"{path}: expected a YAML mapping, got {type(raw).__name__}"
        raise TypeError(msg)
    return DataBuildConfig.model_validate(raw)


def config_hash(config: BaseModel) -> str:
    """Stable sha256 of a config, for the run manifest."""
    payload: dict[str, Any] = config.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
