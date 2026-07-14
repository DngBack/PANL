"""E0 logic that runs without a GPU: contrasts, patch pairing, and the gates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panl.config import ExperimentConfig
from panl.experiments.collect import select_rows
from panl.experiments.e0 import (
    CONTROL_MARGIN,
    PANL_EFFECT_FLOOR,
    block_contrasts,
    confidence_signal,
    evaluate_gates,
    patch_pairs,
)
from panl.models.spec import ModelSpec


def _behavior(margins: dict[str, float], block: str = "b1") -> pd.DataFrame:
    """One block; `margins` maps cell -> confidence margin."""
    cells = ["q1a1", "q1a2", "q2a1", "q2a2"]
    return pd.DataFrame(
        {
            "block_id": [block] * 4,
            "cell": cells,
            "row": list(range(4)),
            "matched": [c in ("q1a1", "q2a2") for c in cells],
            "correct": [c in ("q1a1", "q2a2") for c in cells],
            "confidence_margin": [margins[c] for c in cells],
        }
    )


class TestBlockContrasts:
    def test_interaction_is_the_difference_in_differences(self) -> None:
        frame = _behavior({"q1a1": 3.0, "q1a2": -1.0, "q2a1": -2.0, "q2a2": 4.0})
        contrasts = block_contrasts(frame)
        assert contrasts["interaction"].iloc[0] == pytest.approx(3.0 - (-1.0) - (-2.0) + 4.0)
        assert contrasts["delta_fit"].iloc[0] == pytest.approx(10.0 / 2)

    def test_a_purely_additive_block_has_zero_interaction(self) -> None:
        """If confidence were question-effect plus answer-effect with no interaction, the
        contrast must vanish. This is the null the whole design tests against."""
        q = {"q1": 1.0, "q2": -0.5}
        a = {"a1": 0.25, "a2": 2.0}
        frame = _behavior(
            {f"{qi}{ai}": q[qi] + a[ai] for qi in ("q1", "q2") for ai in ("a1", "a2")}
        )
        assert block_contrasts(frame)["interaction"].iloc[0] == pytest.approx(0.0)

    def test_swapping_the_q_and_a_labels_leaves_the_contrast_unchanged(self) -> None:
        """Q1/Q2 are arbitrary names. Relabelling both factors must not move the estimate."""
        original = _behavior({"q1a1": 3.0, "q1a2": -1.0, "q2a1": -2.0, "q2a2": 4.0})
        relabelled = _behavior({"q1a1": 4.0, "q1a2": -2.0, "q2a1": -1.0, "q2a2": 3.0})
        assert block_contrasts(original)["interaction"].iloc[0] == pytest.approx(
            block_contrasts(relabelled)["interaction"].iloc[0]
        )

    def test_a_missing_cell_is_refused(self) -> None:
        frame = _behavior({"q1a1": 1.0, "q1a2": 0.0, "q2a1": 0.0, "q2a2": 1.0})
        with pytest.raises(ValueError, match="missing cells"):
            block_contrasts(frame[frame["cell"] != "q2a2"])


class TestPatchPairs:
    def test_every_cell_is_paired_with_its_same_question_partner(self) -> None:
        frame = _behavior({"q1a1": 3.0, "q1a2": -1.0, "q2a1": -2.0, "q2a2": 4.0})
        pairs = patch_pairs(frame).set_index("target_cell")

        # The partner shares the question and differs only in the answer.
        assert pairs.loc["q1a2", "source_cell"] == "q1a1"
        assert pairs.loc["q1a1", "source_cell"] == "q1a2"
        assert pairs.loc["q2a1", "source_cell"] == "q2a2"
        assert pairs.loc["q2a2", "source_cell"] == "q2a1"

    def test_direction_follows_the_target(self) -> None:
        frame = _behavior({"q1a1": 3.0, "q1a2": -1.0, "q2a1": -2.0, "q2a2": 4.0})
        pairs = patch_pairs(frame).set_index("target_cell")
        # A crossed target receiving a matched source is a restoration.
        assert pairs.loc["q1a2", "direction"] == "restore"
        assert pairs.loc["q1a1", "direction"] == "ablate"

    def test_the_clean_gap_points_the_right_way_for_a_restore(self) -> None:
        frame = _behavior({"q1a1": 3.0, "q1a2": -1.0, "q2a1": -2.0, "q2a2": 4.0})
        pairs = patch_pairs(frame)
        restore = pairs[pairs["direction"] == "restore"]
        # Source (matched) is more confident than target (crossed), so there is room to move.
        assert (restore["source_clean"] > restore["target_clean"]).all()


class TestGates:
    def _summary(
        self,
        panl: float,
        panl1: float,
        ac: float,
        cc: float,
        *,
        gap: float = 3.0,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "direction": "restore",
                    "position": p,
                    "layer": 5,
                    "effect": e,
                    "mean_moved": e * gap,
                    "flip_rate": e,
                    "mean_gap": gap,
                }
                for p, e in (("PANL", panl), ("PANL1", panl1), ("AC", ac), ("CC", cc))
            ]
        )

    def _signal(self, mean: float, ci_low: float) -> dict[str, object]:
        from panl.analysis.stats import Estimate

        return {
            "interaction": Estimate(
                mean=mean, ci_low=ci_low, ci_high=mean + 1, sign_consistency=0.9, n_blocks=32
            )
        }

    def test_a_clean_panl_result_passes(self) -> None:
        gates = evaluate_gates(self._signal(2.0, 1.0), self._summary(0.9, 0.02, 0.01, 1.0))
        assert gates["overall"]
        assert gates["panl_best_layer"] == 5

    def test_no_confidence_signal_fails(self) -> None:
        gates = evaluate_gates(self._signal(0.1, -0.5), self._summary(0.9, 0.02, 0.01, 1.0))
        assert not gates["confidence_signal"]
        assert not gates["overall"]

    def test_a_weak_panl_effect_fails_the_floor(self) -> None:
        weak = PANL_EFFECT_FLOOR - 0.01
        gates = evaluate_gates(self._signal(2.0, 1.0), self._summary(weak, 0.0, 0.0, 1.0))
        assert not gates["panl_effect_floor"]
        assert not gates["overall"]

    def test_a_panl_effect_that_the_control_matches_fails(self) -> None:
        """If PANL+1 does the same thing PANL does, the effect is not localized to PANL and
        nothing about a post-answer cache has been shown."""
        gates = evaluate_gates(
            self._signal(2.0, 1.0),
            self._summary(0.9, 0.9 / CONTROL_MARGIN + 0.01, 0.01, 1.0),
        )
        assert not gates["panl_beats_panl1"]
        assert not gates["overall"]

    def test_a_weak_cc_fails_the_harness_check(self) -> None:
        """CC is the read-out position. If patching it does less than patching PANL, the
        harness is broken -- that is a bug in us, not a finding about the model."""
        gates = evaluate_gates(self._signal(2.0, 1.0), self._summary(0.9, 0.01, 0.01, 0.2))
        assert not gates["cc_sanity"]
        assert not gates["overall"]

    def test_a_saturated_readout_is_flagged(self) -> None:
        """The failure mode that actually bit us on the 7B model: the distractors were so
        obviously wrong that the clean gap reached ~30 logits, and a patch worth 3 logits
        scored 0.10. The gate still fails, but it must not be read as "no PANL effect"."""
        gates = evaluate_gates(
            self._signal(60.0, 58.0), self._summary(0.1, 0.003, 0.001, 1.0, gap=30.0)
        )
        assert gates["saturated"]
        assert not gates["panl_effect_floor"]
        assert gates["panl_absolute_logits_moved"] == pytest.approx(3.0)

    def test_an_unsaturated_readout_is_not_flagged(self) -> None:
        gates = evaluate_gates(self._signal(2.6, 2.0), self._summary(0.9, 0.01, 0.01, 1.0, gap=2.6))
        assert not gates["saturated"]
        assert gates["overall"]


def test_confidence_signal_reports_calibration() -> None:
    frame = pd.concat(
        [
            _behavior({"q1a1": 3.0, "q1a2": -1.0, "q2a1": -2.0, "q2a2": 4.0}, block="b1"),
            _behavior({"q1a1": 2.0, "q1a2": -2.0, "q2a1": -1.0, "q2a2": 3.0}, block="b2"),
        ]
    )
    config = ExperimentConfig(model=ModelSpec(role="smoke"), n_blocks=2, n_boot=200)
    signal = confidence_signal(frame, config)

    assert signal["interaction"].mean > 0
    # Matched cells carry the higher margins here, so the margin ranks correct above incorrect.
    assert signal["calibration_auc"] == 1.0
    assert signal["mean_margin_matched"] > signal["mean_margin_crossed"]


class TestSelectRows:
    def _table(self, n_blocks: int) -> pd.DataFrame:
        rows = []
        for b in range(n_blocks):
            for cell in ("q1a1", "q1a2", "q2a1", "q2a2"):
                rows.append(
                    {
                        "block_id": f"fam/b{b:03d}",
                        "cell": cell,
                        "split": "train" if b % 2 == 0 else "test",
                        "relation_family": "capital_of",
                    }
                )
        return pd.DataFrame(rows)

    def test_blocks_are_kept_whole(self) -> None:
        config = ExperimentConfig(n_blocks=3, splits=("train",))
        selected = select_rows(self._table(20), config)
        assert selected["block_id"].nunique() == 3
        assert len(selected) == 12
        assert (selected.groupby("block_id").size() == 4).all()

    def test_only_the_requested_split_is_used(self) -> None:
        config = ExperimentConfig(n_blocks=3, splits=("train",))
        selected = select_rows(self._table(20), config)
        assert set(selected["split"]) == {"train"}

    def test_asking_for_more_blocks_than_exist_fails_loudly(self) -> None:
        config = ExperimentConfig(n_blocks=50, splits=("train",))
        with pytest.raises(ValueError, match="only 10 match"):
            select_rows(self._table(20), config)

    def test_row_order_is_deterministic(self) -> None:
        config = ExperimentConfig(n_blocks=5, splits=("train",))
        table = self._table(20)
        first = select_rows(table, config)
        second = select_rows(table.sample(frac=1, random_state=0), config)
        assert first["block_id"].tolist() == second["block_id"].tolist()
        assert first["cell"].tolist() == second["cell"].tolist()

    def test_null_n_blocks_takes_everything(self) -> None:
        config = ExperimentConfig(n_blocks=None, splits=("train",))
        selected = select_rows(self._table(20), config)
        assert selected["block_id"].nunique() == 10


def test_row_index_is_the_join_key_into_the_activation_store() -> None:
    """`row` must be a dense 0..n-1 index over the selected rows, because the Zarr array is
    indexed by it and a mismatch would silently pair activations with the wrong prompt."""
    config = ExperimentConfig(n_blocks=None, splits=("train",))
    table = pd.DataFrame(
        [
            {
                "block_id": f"fam/b{b}",
                "cell": c,
                "split": "train",
                "relation_family": "capital_of",
            }
            for b in range(3)
            for c in ("q1a1", "q1a2", "q2a1", "q2a2")
        ]
    )
    selected = select_rows(table, config)
    assert np.array_equal(selected.index.to_numpy(), np.arange(len(selected)))
