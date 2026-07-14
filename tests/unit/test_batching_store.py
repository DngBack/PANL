"""Length-grouped batching and the Zarr activation store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from panl.activations.store import ActivationWriter, StoreSpec, read_activations
from panl.models.batching import make_batches
from panl.models.positions import POSITION_NAMES, resolve_positions
from panl.models.prompts import PromptRenderer, PromptStyle, PromptTemplate
from panl.models.tokenizer import FastTokenizer


def _resolved(tokenizer: FastTokenizer, cases: list[tuple[str, str]]) -> list:  # type: ignore[type-arg]
    renderer = PromptRenderer(tokenizer, template=PromptTemplate(), style=PromptStyle.CHAT)
    return [resolve_positions(tokenizer, renderer.render(q, a)) for q, a in cases]


class TestBatching:
    def test_every_batch_holds_one_token_length(self, fake_tokenizer: FastTokenizer) -> None:
        """The whole point: no padding is ever introduced, so no attention-mask or rotary
        offset can be silently wrong."""
        resolved = _resolved(
            fake_tokenizer,
            [
                ("What is the capital of France?", "Paris"),
                ("What is the capital of Japan?", "Tokyo"),
                ("What is the official currency of Japan?", "Japanese yen"),
                ("Who wrote Nineteen Eighty-Four?", "George Orwell"),
            ],
        )
        for batch in make_batches(resolved, max_batch_size=8):
            lengths = {len(resolved[i].input_ids) for i in batch.row_indices}
            assert len(lengths) == 1
            assert batch.seq_len == lengths.pop()

    def test_every_row_appears_exactly_once(self, fake_tokenizer: FastTokenizer) -> None:
        resolved = _resolved(
            fake_tokenizer,
            [("What is the capital of France?", a) for a in ("Paris", "Tokyo", "Lima", "Oslo")]
            + [("Who wrote Ulysses?", "James Joyce")],
        )
        seen = [i for batch in make_batches(resolved, max_batch_size=2) for i in batch.row_indices]
        assert sorted(seen) == list(range(len(resolved)))

    def test_max_batch_size_is_respected(self, fake_tokenizer: FastTokenizer) -> None:
        resolved = _resolved(
            fake_tokenizer,
            [("What is the capital of France?", a) for a in ("Paris", "Tokyo", "Lima", "Oslo")],
        )
        for batch in make_batches(resolved, max_batch_size=2):
            assert batch.size <= 2

    def test_positions_travel_with_their_rows(self, fake_tokenizer: FastTokenizer) -> None:
        resolved = _resolved(
            fake_tokenizer,
            [
                ("What is the capital of France?", "Paris"),
                ("What is the official currency of Japan?", "Japanese yen"),
            ],
        )
        for batch in make_batches(resolved, max_batch_size=8):
            for offset, row in enumerate(batch.row_indices):
                for name in POSITION_NAMES:
                    assert int(batch.positions[name][offset]) == resolved[row].indices[name]

    def test_the_token_at_each_position_matches_the_prompt(
        self, fake_tokenizer: FastTokenizer
    ) -> None:
        """Guards the index arithmetic: a position must point at the token it claims to."""
        resolved = _resolved(fake_tokenizer, [("What is the capital of France?", "Paris")])
        batch = next(iter(make_batches(resolved, max_batch_size=1)))
        for name in POSITION_NAMES:
            index = int(batch.positions[name][0])
            assert int(batch.input_ids[0, index]) == resolved[0].token_ids[name]

    def test_cc_is_the_last_index_in_every_batch(self, fake_tokenizer: FastTokenizer) -> None:
        resolved = _resolved(
            fake_tokenizer,
            [("What is the capital of France?", "Paris"), ("Who wrote Ulysses?", "James Joyce")],
        )
        for batch in make_batches(resolved, max_batch_size=8):
            assert (batch.positions["CC"] == batch.seq_len - 1).all()

    def test_batching_is_deterministic(self, fake_tokenizer: FastTokenizer) -> None:
        resolved = _resolved(
            fake_tokenizer,
            [("What is the capital of France?", a) for a in ("Paris", "Tokyo", "Lima")],
        )
        first = [b.row_indices for b in make_batches(resolved, max_batch_size=2)]
        second = [b.row_indices for b in make_batches(resolved, max_batch_size=2)]
        assert first == second


class TestActivationStore:
    def test_round_trip_preserves_values_and_layout(self, tmp_path: Path) -> None:
        spec = StoreSpec(n_rows=4, n_layers=3, d_model=8)
        writer = ActivationWriter(tmp_path / "a.zarr", spec, metadata={"model_id": "test"})

        expected = np.zeros(spec.shape, dtype=np.float16)
        rng = np.random.default_rng(0)
        for rows in [(0, 1), (2, 3)]:
            activations = {
                layer: {
                    position: torch.tensor(
                        rng.normal(size=(len(rows), spec.d_model)), dtype=torch.float32
                    )
                    for position in POSITION_NAMES
                }
                for layer in range(spec.n_layers)
            }
            writer.write(rows, activations)
            for offset, row in enumerate(rows):
                for layer in range(spec.n_layers):
                    for p_index, position in enumerate(POSITION_NAMES):
                        expected[row, layer, p_index] = (
                            activations[layer][position][offset].to(torch.float16).numpy()
                        )
        writer.finalize()

        stored, attrs = read_activations(tmp_path / "a.zarr")
        assert stored.shape == spec.shape
        np.testing.assert_array_equal(stored, expected)
        assert attrs["positions"] == list(POSITION_NAMES)
        assert attrs["model_id"] == "test"

    def test_finalize_records_a_checksum(self, tmp_path: Path) -> None:
        spec = StoreSpec(n_rows=2, n_layers=1, d_model=4)
        writer = ActivationWriter(tmp_path / "a.zarr", spec, metadata={})
        writer.write(
            (0, 1),
            {0: {p: torch.zeros(2, 4) for p in POSITION_NAMES}},
        )
        checksum = writer.finalize()
        assert len(checksum) == 64
        _, attrs = read_activations(tmp_path / "a.zarr")
        assert attrs["sha256"] == checksum

    def test_the_store_is_float16_and_therefore_not_patch_material(self, tmp_path: Path) -> None:
        """Documents why interventions re-run the model instead of reading this back: the
        store is lossy, and a causal patch must carry the values the model computed."""
        spec = StoreSpec(n_rows=1, n_layers=1, d_model=4)
        writer = ActivationWriter(tmp_path / "a.zarr", spec, metadata={})
        precise = torch.tensor([[1.0000001, 2.0000002, 3.0000003, 4.0000004]])
        writer.write((0,), {0: {p: precise for p in POSITION_NAMES}})
        writer.finalize()

        stored, attrs = read_activations(tmp_path / "a.zarr")
        assert attrs["dtype"] == "float16"
        assert stored.dtype == np.float16
        assert not np.array_equal(stored[0, 0, 0], precise[0].numpy())


@pytest.mark.parametrize("n_rows", [1, 5])
def test_store_shape_matches_spec(tmp_path: Path, n_rows: int) -> None:
    spec = StoreSpec(n_rows=n_rows, n_layers=2, d_model=6)
    assert spec.shape == (n_rows, 2, len(POSITION_NAMES), 6)
    ActivationWriter(tmp_path / "a.zarr", spec, metadata={}).finalize()
    stored, _ = read_activations(tmp_path / "a.zarr")
    assert stored.shape == spec.shape
