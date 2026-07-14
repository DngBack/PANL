"""Block-level resampling statistics."""

from __future__ import annotations

import numpy as np
import pytest

from panl.analysis.stats import auc, block_bootstrap, block_bootstrap_ratio, paired_effect_size


class TestBlockBootstrap:
    def test_recovers_a_known_mean(self) -> None:
        values = np.full(200, 2.0)
        est = block_bootstrap(values, n_boot=500, seed=0)
        assert est.mean == pytest.approx(2.0)
        assert est.ci_low == pytest.approx(2.0)
        assert est.ci_high == pytest.approx(2.0)
        assert est.sign_consistency == 1.0

    def test_a_centred_null_does_not_exclude_zero(self) -> None:
        rng = np.random.default_rng(0)
        est = block_bootstrap(rng.normal(0, 1, 200), n_boot=2000, seed=1)
        assert not est.excludes_zero
        assert est.p_permutation is not None and est.p_permutation > 0.05

    def test_a_real_effect_excludes_zero(self) -> None:
        rng = np.random.default_rng(0)
        est = block_bootstrap(rng.normal(1.5, 1.0, 200), n_boot=2000, seed=1)
        assert est.excludes_zero
        assert est.p_permutation is not None and est.p_permutation < 0.01

    def test_permutation_p_is_never_zero(self) -> None:
        """A permutation p-value is bounded below by 1/(n_boot+1); reporting 0 would be a lie."""
        est = block_bootstrap(np.full(50, 100.0), n_boot=100, seed=0)
        assert est.p_permutation is not None
        assert est.p_permutation >= 1 / 101

    def test_is_deterministic_given_the_seed(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(0.5, 1, 60)
        assert block_bootstrap(values, seed=7) == block_bootstrap(values, seed=7)

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no finite block values"):
            block_bootstrap(np.array([]))

    def test_more_blocks_narrow_the_interval(self) -> None:
        """The block count is what buys precision -- that is why cells must not be resampled."""
        rng = np.random.default_rng(0)
        small = block_bootstrap(rng.normal(1, 1, 30), n_boot=2000, seed=1)
        large = block_bootstrap(rng.normal(1, 1, 300), n_boot=2000, seed=1)
        assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


class TestRatioBootstrap:
    def test_recovers_a_known_ratio(self) -> None:
        moved = np.full(100, 0.5)
        gap = np.full(100, 1.0)
        est = block_bootstrap_ratio(moved, gap, n_boot=500, seed=0)
        assert est.mean == pytest.approx(0.5)
        assert est.sign_consistency == 1.0

    def test_a_near_zero_gap_does_not_blow_up_the_estimate(self) -> None:
        """The reason this is a ratio of means and not a mean of ratios: one block whose
        clean gap is ~0 would otherwise dominate the whole summary."""
        moved = np.concatenate([np.full(99, 0.5), [0.001]])
        gap = np.concatenate([np.full(99, 1.0), [1e-6]])
        est = block_bootstrap_ratio(moved, gap, n_boot=500, seed=0)
        assert 0.4 < est.mean < 0.6

    def test_no_effect_gives_a_ratio_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        est = block_bootstrap_ratio(
            rng.normal(0, 0.01, 200), np.full(200, 2.0), n_boot=1000, seed=0
        )
        assert abs(est.mean) < 0.05


class TestAuc:
    def test_perfect_separation(self) -> None:
        assert auc(np.array([3.0, 4.0, 1.0, 2.0]), np.array([1, 1, 0, 0])) == 1.0

    def test_perfect_inversion(self) -> None:
        assert auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 1, 0, 0])) == 0.0

    def test_all_ties_give_one_half(self) -> None:
        """Low-precision logits tie often enough that this must be handled, not assumed away."""
        assert auc(np.ones(6), np.array([1, 1, 1, 0, 0, 0])) == pytest.approx(0.5)

    def test_one_class_only_is_undefined(self) -> None:
        assert np.isnan(auc(np.array([1.0, 2.0]), np.array([1, 1])))


def test_paired_effect_size() -> None:
    assert paired_effect_size(np.array([2.0, 2.0, 2.0, 2.0])) != paired_effect_size(
        np.array([2.0, 0.0, 4.0, 2.0])
    )
    values = np.array([1.0, 2.0, 3.0])
    assert paired_effect_size(values) == pytest.approx(values.mean() / values.std(ddof=1))
