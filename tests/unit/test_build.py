"""End-to-end build: Parquet round-trip, frozen schema, and the run manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from panl.config import DataBuildConfig, config_hash, load_data_config
from panl.data.build import build_dataset
from panl.data.schema import COLUMNS, QUADRUPLE_SCHEMA, read_table


def test_build_writes_a_table_and_a_manifest(tiny_config: DataBuildConfig, tmp_path: Path) -> None:
    out = tmp_path / "quadruples.parquet"
    result = build_dataset(tiny_config, output=out)

    assert result.report.ok, result.report.summary()
    assert out.exists()
    assert result.manifest_path is not None and result.manifest_path.exists()

    table = read_table(out)
    assert table.num_rows == len(result.rows)
    assert table.schema.equals(QUADRUPLE_SCHEMA)


def test_written_schema_is_the_frozen_one(tiny_config: DataBuildConfig, tmp_path: Path) -> None:
    out = tmp_path / "q.parquet"
    build_dataset(tiny_config, output=out)
    assert pq.read_schema(out).names == list(COLUMNS)


def test_read_table_rejects_a_drifted_schema(tmp_path: Path) -> None:
    import pyarrow as pa

    path = tmp_path / "wrong.parquet"
    pq.write_table(pa.table({"block_id": ["a"]}), path)
    with pytest.raises(ValueError, match="column set drifted"):
        read_table(path)


def test_manifest_pins_the_inputs(tiny_config: DataBuildConfig, tmp_path: Path) -> None:
    out = tmp_path / "q.parquet"
    result = build_dataset(tiny_config, output=out)
    assert result.manifest_path is not None

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["command"] == "data build"
    assert manifest["config_sha256"] == config_hash(tiny_config)
    assert manifest["config"]["seed"] == tiny_config.seed
    assert manifest["payload"]["n_blocks"] == len(result.blocks)
    assert manifest["payload"]["validation_ok"] is True
    # Reproducibility needs the lock and the commit, per plan section 7.
    assert manifest["uv_lock_sha256"]
    assert "git" in manifest
    assert manifest["artifacts"]["quadruples"]["sha256"]


def test_build_without_writing_touches_no_disk(tiny_config: DataBuildConfig) -> None:
    result = build_dataset(tiny_config, write=False)
    assert result.parquet_path is None
    assert result.manifest_path is None
    assert result.report.ok


def test_the_shipped_config_builds_and_validates(tmp_path: Path) -> None:
    """The config we will actually run must survive its own invariants."""
    config = load_data_config(Path("configs/data/tier1.yaml"))
    result = build_dataset(config, output=tmp_path / "q.parquet")
    assert result.report.ok, result.report.summary()
    # The pilot needs 200 blocks (plan section E0/E1); make sure the fact base can reach it.
    assert len(result.blocks) >= 200, f"only {len(result.blocks)} blocks available"


class TestConfigValidation:
    def test_unknown_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown relation families"):
            DataBuildConfig(families=("capital_of", "phlogiston_of"))

    def test_holding_out_an_unbuilt_family_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not built"):
            DataBuildConfig(families=("capital_of",), family_holdout=("author_of",))

    def test_holding_out_everything_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="nothing would remain"):
            DataBuildConfig(families=("capital_of",), family_holdout=("capital_of",))

    def test_a_typo_in_a_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Extra inputs"):
            DataBuildConfig.model_validate({"pairings_per_facts": 2})

    def test_config_hash_is_stable_and_sensitive(self, tiny_config: DataBuildConfig) -> None:
        assert config_hash(tiny_config) == config_hash(tiny_config.model_copy())
        assert config_hash(tiny_config) != config_hash(
            tiny_config.model_copy(update={"seed": tiny_config.seed + 1})
        )
