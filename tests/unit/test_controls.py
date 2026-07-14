"""Source controls for the isolated PANL patch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from panl.experiments.controls import (
    SourceKind,
    _stable_seed,
    build_source,
    evaluate_control_gates,
)
from panl.experiments.routes import length_matched_blocks


class TestBuildSource:
    def _cached(self, n_rows: int = 8, n_layers: int = 4, d_model: int = 6) -> torch.Tensor:
        rng = torch.Generator().manual_seed(0)
        return torch.randn((n_rows, n_layers, d_model), generator=rng, dtype=torch.float32)

    def _gen(self) -> torch.Generator:
        return torch.Generator().manual_seed(1)

    def test_partner_returns_the_partner_rows_unchanged(self) -> None:
        cached = self._cached()
        rows = np.array([3, 1, 7])
        layers = [2, 3]
        source = build_source(SourceKind.PARTNER, cached, rows, layers, generator=self._gen())
        assert torch.equal(source, cached[[3, 1, 7]][:, [2, 3], :])

    def test_gaussian_matches_the_partner_norm_layer_by_layer(self) -> None:
        """A perturbation that is *smaller* than the real patch would beat nothing; the point
        of this control is that it is exactly as large."""
        cached = self._cached()
        rows = np.array([0, 4])
        layers = [1, 2, 3]
        partner = cached[rows][:, layers, :]
        source = build_source(SourceKind.GAUSSIAN, cached, rows, layers, generator=self._gen())

        assert source.shape == partner.shape
        torch.testing.assert_close(source.norm(dim=-1), partner.norm(dim=-1), rtol=1e-3, atol=1e-3)
        # And it must not accidentally *be* the partner.
        assert not torch.allclose(source, partner)

    def test_random_cell_never_donates_a_cell_to_itself(self) -> None:
        """If the donor happened to be the partner, the 'wrong content' control would silently
        become the intervention for that row."""
        cached = self._cached(n_rows=3)
        rows = np.array([0, 1, 2] * 20)
        layers = [0, 1]
        source = build_source(SourceKind.RANDOM_CELL, cached, rows, layers, generator=self._gen())
        partner = cached[rows][:, layers, :]
        # No row may be identical to its own partner.
        identical = (source == partner).all(dim=-1).all(dim=-1)
        assert not identical.any()

    def test_random_cell_keeps_a_coherent_trajectory(self) -> None:
        """The donor is drawn once per pair, not once per layer: a replacement stitched from a
        different cell at every layer is not a residual stream any model ever produced."""
        cached = self._cached()
        rows = np.array([0, 1])
        layers = [0, 1, 2, 3]
        source = build_source(SourceKind.RANDOM_CELL, cached, rows, layers, generator=self._gen())
        for pair in range(len(rows)):
            # Each pair's replacement must equal some single cached row across all layers.
            matches = [
                torch.equal(source[pair], cached[donor][:, :][layers, :])
                for donor in range(cached.shape[0])
            ]
            assert any(matches)

    def test_mean_is_the_same_vector_for_every_pair(self) -> None:
        cached = self._cached()
        rows = np.array([0, 3, 5])
        layers = [1, 2]
        source = build_source(SourceKind.MEAN, cached, rows, layers, generator=self._gen())
        assert torch.allclose(source[0], source[1])
        torch.testing.assert_close(
            source[0].float(), cached.mean(dim=0)[layers, :], rtol=1e-4, atol=1e-4
        )

    def test_every_kind_yields_the_same_shape(self) -> None:
        cached = self._cached()
        rows = np.array([0, 2, 4])
        layers = [0, 2]
        shapes = {
            kind: build_source(kind, cached, rows, layers, generator=self._gen()).shape
            for kind in SourceKind
        }
        assert len(set(shapes.values())) == 1


def _summary(
    partner: tuple[float, float],
    gaussian: tuple[float, float] = (0.96, 0.01),
    mean: tuple[float, float] = (0.82, 0.05),
    crossed_donor: tuple[float, float] = (0.49, 0.20),
    matched_donor: tuple[float, float] = (0.90, 0.05),
    random_cell: tuple[float, float] = (0.50, 0.49),
) -> pd.DataFrame:
    """Each kind is (restore flip, ablate flip). Defaults are the values actually observed on
    Qwen2.5-7B, including the confound: a norm-matched random vector flips 96% of RESTORE
    decisions and 1% of ABLATE ones."""
    kinds = {
        "partner": partner,
        "gaussian": gaussian,
        "mean": mean,
        "crossed_donor": crossed_donor,
        "matched_donor": matched_donor,
        "random_cell": random_cell,
    }
    records = []
    for kind, (restore, ablate) in kinds.items():
        for direction, flip in (("restore", restore), ("ablate", ablate)):
            records.append(
                {
                    "position": "PANL",
                    "source_kind": kind,
                    "direction": direction,
                    "effect": 0.99 if kind == "partner" else 0.5,
                    "flip_rate": flip,
                }
            )
    return pd.DataFrame(records)


class TestControlGates:
    def test_a_clean_result_passes(self) -> None:
        gates = evaluate_control_gates(_summary((0.95, 0.96)))
        assert gates["overall"]
        assert gates["bidirectional"]

    def test_the_restore_direction_is_flagged_as_confounded(self) -> None:
        """The finding that forced the gate to be rewritten: writing *noise* into PANL flips
        96% of restore decisions. The model is confident by default -- PANL carries doubt --
        so destroying PANL with anything makes a diffident model confident."""
        gates = evaluate_control_gates(_summary((0.95, 0.96)))
        assert gates["restore_is_confounded"]

    def test_a_gate_that_pooled_directions_would_have_passed_noise(self) -> None:
        """Guards the actual bug. Under the old rule the control's worst direction was compared
        with the intervention's best, so gaussian at 96%-restore / 1%-ablate looked beaten. It
        is not beaten -- it is a confound, and it must only be judged in ablate."""
        gates = evaluate_control_gates(_summary((0.95, 0.96), gaussian=(0.96, 0.01)))
        assert gates["beats_destructive_controls"]  # judged in ablate: 96% vs 1%
        # But in restore the "control" matches the intervention, which is exactly the confound.
        assert gates["control_flips"]["gaussian"]["restore"] >= gates["partner_flip_restore"] * 0.9

    def test_noise_that_also_ablates_fails(self) -> None:
        """If a random vector *removed* confidence too, the result would be 'breaking PANL
        breaks confidence' -- true, uninteresting, and not the claim."""
        gates = evaluate_control_gates(_summary((0.95, 0.96), gaussian=(0.96, 0.90)))
        assert not gates["beats_destructive_controls"]
        assert not gates["overall"]

    def test_a_generic_doubt_signal_fails_the_pair_specificity_gate(self) -> None:
        """THE decisive test. A crossed cell from another block carries 'this answer does not
        fit' about a different question and a different answer. If transplanting it removes
        confidence as well as the true partner does, PANL holds a generic doubt bit, not
        information about this pair -- and the Q x A story does not survive."""
        gates = evaluate_control_gates(_summary((0.95, 0.96), crossed_donor=(0.49, 0.95)))
        assert not gates["effect_is_pair_specific"]
        assert not gates["overall"]

    def test_a_one_way_effect_fails(self) -> None:
        gates = evaluate_control_gates(_summary((0.95, 0.10)))
        assert not gates["bidirectional"]
        assert not gates["overall"]

    def test_control_flips_are_reported_for_every_kind_and_direction(self) -> None:
        gates = evaluate_control_gates(_summary((0.95, 0.96)))
        assert set(gates["control_flips"]) == {
            "random_cell",
            "gaussian",
            "mean",
            "matched_donor",
            "crossed_donor",
        }
        assert set(gates["control_flips"]["crossed_donor"]) == {"restore", "ablate"}


class TestLengthMatchedBlocks:
    def test_only_blocks_whose_answers_tokenize_alike_are_kept(self) -> None:
        """The length confound: if a block's two answers differ in token count, the matched and
        crossed prompts differ in length, CC sits at a different absolute position, and rotary
        embeddings let the model see that without any answer content reaching it."""

        class FakeResolved:
            def __init__(self, n: int) -> None:
                self.n_answer_tokens = n

        behavior = pd.DataFrame(
            {
                "block_id": ["b1"] * 4 + ["b2"] * 4,
                "matched": [True, False, False, True] * 2,
            }
        )
        # b1: both answers are 2 tokens. b2: 1 token vs 3 tokens.
        resolved = [FakeResolved(n) for n in (2, 2, 2, 2, 1, 3, 3, 1)]
        keep = length_matched_blocks(behavior, resolved)  # type: ignore[arg-type]
        assert keep == {"b1"}

    def test_all_blocks_kept_when_every_answer_is_the_same_length(self) -> None:
        class FakeResolved:
            def __init__(self, n: int) -> None:
                self.n_answer_tokens = n

        behavior = pd.DataFrame({"block_id": ["b1"] * 4, "matched": [True, False, False, True]})
        resolved = [FakeResolved(2)] * 4
        assert length_matched_blocks(behavior, resolved) == {"b1"}  # type: ignore[arg-type]


class TestStableSeed:
    """The random donors must be reproducible across processes.

    Python salts `hash()` of a str per process, so seeding the generator with it would draw a
    different donor set on every run -- and a control you cannot reproduce is not a control.
    This is a real bug that was in the first version of this module.
    """

    def test_the_seed_does_not_depend_on_python_hash_randomization(self) -> None:
        import subprocess
        import sys

        script = (
            "from panl.experiments.controls import _stable_seed; "
            "print(_stable_seed(20260714, 'PANL', 'gaussian'))"
        )
        seeds = set()
        for salt in ("0", "1"):
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"},
            )
            seeds.add(out.stdout.strip())
        assert len(seeds) == 1, f"seed changed with PYTHONHASHSEED: {seeds}"

    def test_different_positions_and_kinds_get_different_seeds(self) -> None:
        seeds = {
            _stable_seed(7, position, kind)
            for position in ("PANL", "PANL1", "AC")
            for kind in ("gaussian", "random_cell")
        }
        assert len(seeds) == 6


@pytest.mark.parametrize("kind", list(SourceKind))
def test_source_kinds_are_stable_strings(kind: SourceKind) -> None:
    """These land in a Parquet column and in the manifest; renaming one silently invalidates
    every stored result."""
    assert kind.value in {
        "partner",
        "random_cell",
        "gaussian",
        "mean",
        "matched_donor",
        "crossed_donor",
    }
