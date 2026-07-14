"""Route-ablation logic that runs without a GPU."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panl.experiments.routes import (
    ISOLATE_PANL,
    ROUTE_CONDITIONS,
    evaluate_route_gates,
    read_cliff,
)


class TestRouteConditions:
    def test_isolating_panl_cuts_exactly_the_two_bypass_routes(self) -> None:
        """The answer can reach CC directly, via PANL, or via PANL+1. Isolating PANL means
        cutting the other two, and only those -- cutting more would starve PANL as well."""
        assert set(ISOLATE_PANL) == {("CC", "answer"), ("PANL1", "answer")}
        assert ("PANL", "answer") not in ISOLATE_PANL

    def test_only_via_panl_leaves_panl_fed(self) -> None:
        edges = ROUTE_CONDITIONS["only via PANL"]
        assert ("PANL", "answer") not in edges  # PANL still sees the answer
        assert ("CC", "PANL") not in edges  # and CC still sees PANL

    def test_only_direct_starves_panl_completely(self) -> None:
        edges = ROUTE_CONDITIONS["only direct"]
        assert ("PANL", "answer") in edges  # PANL cannot see the answer
        assert ("CC", "PANL") in edges  # and CC cannot see PANL
        assert ("CC", "answer") not in edges  # only the direct edge survives

    def test_cut_everything_severs_every_route(self) -> None:
        """The floor condition. If the gap does not collapse here, the knockout is not
        working and no other row in the table means anything."""
        edges = set(ROUTE_CONDITIONS["cut everything"])
        assert {("CC", "answer"), ("PANL1", "answer"), ("PANL", "answer"), ("CC", "PANL")} <= edges

    def test_clean_cuts_nothing(self) -> None:
        assert ROUTE_CONDITIONS["clean"] == ()


def _conditions(floor: float, via_panl: float, direct: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"condition": "clean", "share_of_clean": 1.0},
            {"condition": "only via PANL", "share_of_clean": via_panl},
            {"condition": "only direct", "share_of_clean": direct},
            {"condition": "cut everything", "share_of_clean": floor},
        ]
    )


def _isolated(panl_flip: float, control_flip: float, effect: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"position": "PANL", "layer": 18, "effect": effect, "flip_rate": panl_flip},
            {"position": "PANL1", "layer": 8, "effect": 0.003, "flip_rate": control_flip},
            {"position": "AC", "layer": 14, "effect": 0.001, "flip_rate": control_flip},
        ]
    )


class TestRouteGates:
    def test_the_observed_qwen_result_passes(self) -> None:
        gates = evaluate_route_gates(_conditions(0.19, 0.97, 0.88), _isolated(0.96, 0.00))
        assert gates["overall"]

    def test_panl_is_reported_as_sufficient_but_not_necessary(self) -> None:
        """The headline. The direct route carries 88% of the gap on its own, so the intact
        model does not need PANL -- and the gate must say so rather than quietly pass."""
        gates = evaluate_route_gates(_conditions(0.19, 0.93, 0.88), _isolated(0.76, 0.02))
        assert gates["panl_alone_carries_the_signal"]
        assert not gates["panl_is_necessary"]

    def test_a_route_that_the_direct_edge_cannot_carry_makes_panl_necessary(self) -> None:
        gates = evaluate_route_gates(_conditions(0.19, 0.93, 0.10), _isolated(0.76, 0.02))
        assert gates["panl_is_necessary"]

    def test_a_knockout_that_does_not_collapse_the_gap_fails(self) -> None:
        """If severing every route leaves the gap intact, the attention mask is not doing
        what we think it is and every other number here is meaningless."""
        gates = evaluate_route_gates(_conditions(0.85, 0.93, 0.88), _isolated(0.76, 0.02))
        assert not gates["knockout_collapses_the_gap"]
        assert not gates["overall"]

    def test_panl_failing_to_carry_the_signal_fails(self) -> None:
        gates = evaluate_route_gates(_conditions(0.19, 0.22, 0.88), _isolated(0.76, 0.02))
        assert not gates["panl_alone_carries_the_signal"]
        assert not gates["overall"]

    def test_a_patch_that_moves_nothing_fails(self) -> None:
        gates = evaluate_route_gates(_conditions(0.19, 0.93, 0.88), _isolated(0.05, 0.02))
        assert not gates["isolated_panl_patch_flips_decisions"]
        assert not gates["overall"]

    def test_controls_that_move_too_fail(self) -> None:
        """If patching AC also flips decisions, the effect is not about PANL at all."""
        gates = evaluate_route_gates(_conditions(0.19, 0.93, 0.88), _isolated(0.76, 0.60))
        assert not gates["controls_stay_flat"]
        assert not gates["overall"]


class TestKeyMask:
    """The mask that selects which keys an attention edge severs."""

    def test_answer_mask_covers_exactly_the_answer_span(self, fake_tokenizer: object) -> None:
        import torch

        from panl.models.batching import make_batches
        from panl.models.positions import resolve_positions
        from panl.models.prompts import PromptRenderer, PromptStyle, PromptTemplate

        renderer = PromptRenderer(
            fake_tokenizer,  # type: ignore[arg-type]
            template=PromptTemplate(),
            style=PromptStyle.CHAT,
        )
        resolved = [
            resolve_positions(
                fake_tokenizer,  # type: ignore[arg-type]
                renderer.render("What is the capital of France?", "Paris"),
            )
        ]
        batch = next(iter(make_batches(resolved, max_batch_size=1)))

        start, end = (int(x) for x in batch.answer_spans[0])
        span = set(range(start, end))

        # LAT lies inside the answer; PANL and CC lie outside it.
        assert int(batch.positions["LAT"][0]) in span
        assert int(batch.positions["PANL"][0]) not in span
        assert int(batch.positions["CC"][0]) not in span
        assert torch.equal(
            batch.input_ids[0, start:end],
            torch.tensor(list(resolved[0].input_ids[start:end])),
        )


class TestReadCliff:
    """With a cumulative patch, the argmax layer is meaningless and the cliff is the answer."""

    def _profile(self, effects: dict[int, float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"position": "PANL", "layer": layer, "effect": effect, "cumulative": True}
                for layer, effect in effects.items()
            ]
        )

    def test_the_cliff_is_the_last_start_layer_that_still_works(self) -> None:
        """Every span starting below the read point contains the one starting at it, so they
        all tie at the peak. Taking the argmax would report L4 -- an arbitrary member of the
        tie that says nothing about where CC reads."""
        observed = self._profile(
            {0: 0.90, 4: 0.90, 8: 0.90, 12: 0.90, 16: 0.90, 18: 0.42, 20: -0.01, 24: 0.0}
        )
        assert read_cliff(observed) == 16
        # The naive answer, and the reason this function exists.
        assert int(observed.loc[observed["effect"].idxmax(), "layer"]) == 0

    def test_a_flat_null_profile_has_no_cliff(self) -> None:
        assert read_cliff(self._profile({0: 0.0, 8: 0.0, 16: -0.01})) == -1

    def test_an_absent_position_has_no_cliff(self) -> None:
        assert read_cliff(self._profile({0: 0.9}), position="AC") == -1

    def test_tolerance_controls_how_much_decay_still_counts(self) -> None:
        observed = self._profile({0: 1.0, 8: 0.95, 16: 0.80, 20: 0.05})
        assert read_cliff(observed, tolerance=0.9) == 8
        assert read_cliff(observed, tolerance=0.5) == 16


def test_block_gaps_use_the_block_as_the_unit() -> None:
    from panl.experiments.routes import _block_gaps

    behavior = pd.DataFrame(
        {
            "block_id": ["b1", "b1", "b1", "b1", "b2", "b2", "b2", "b2"],
            "matched": [True, False, False, True] * 2,
        }
    )
    margins = np.array([10.0, 0.0, 0.0, 10.0, 4.0, 2.0, 2.0, 4.0])
    gaps = _block_gaps(behavior, margins)
    assert sorted(gaps) == pytest.approx([2.0, 10.0])
