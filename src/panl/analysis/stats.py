"""Block-level resampling statistics.

Plan section 4.2: "The block -- not the cell -- is the independent resampling unit." Cells
inside a block share a question, an answer, and a template, so resampling cells would treat
four correlated measurements as four independent ones and shrink every interval by roughly a
factor of two. Every function here therefore takes per-block values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Estimate:
    mean: float
    ci_low: float
    ci_high: float
    #: Fraction of blocks whose value has the same sign as the mean.
    sign_consistency: float
    n_blocks: int
    #: Two-sided p-value from a sign-flip permutation test, or None if not requested.
    p_permutation: float | None = None

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def __str__(self) -> str:
        p = "" if self.p_permutation is None else f" p={self.p_permutation:.4f}"
        return (
            f"{self.mean:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
            f"sign={self.sign_consistency:.0%} n={self.n_blocks}{p}"
        )


def block_bootstrap(
    values: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    permutation: bool = True,
) -> Estimate:
    """Percentile bootstrap over blocks, plus a sign-flip permutation test.

    Args:
        values: [n_blocks] one value per block, e.g. the block's interaction contrast.

    The permutation test flips the sign of each block's value independently, which is the
    exact null for a within-block contrast: under "no interaction", relabelling matched and
    crossed inside a block negates that block's contrast and nothing else.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        msg = "no finite block values to bootstrap"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    mean = float(values.mean())

    draws = rng.integers(0, n, size=(n_boot, n))
    boot_means = values[draws].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])

    sign = float((np.sign(values) == np.sign(mean)).mean()) if mean != 0 else 0.0

    p_value: float | None = None
    if permutation:
        flips = rng.choice([-1.0, 1.0], size=(n_boot, n))
        null_means = (values * flips).mean(axis=1)
        # +1 in numerator and denominator: a permutation p-value is never exactly zero.
        p_value = float((np.abs(null_means) >= abs(mean)).sum() + 1) / (n_boot + 1)

    return Estimate(
        mean=mean,
        ci_low=float(lo),
        ci_high=float(hi),
        sign_consistency=sign,
        n_blocks=n,
        p_permutation=p_value,
    )


def block_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Bootstrap a ratio of means over blocks: mean(numerator) / mean(denominator).

    Used for the normalized patching effect, where the numerator is how far the patch moved
    the confidence margin and the denominator is how far it *could* have moved it (the clean
    source-target gap). A per-block ratio would be dominated by blocks whose gap is near zero;
    resampling the two means together and dividing inside each resample avoids that.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    numerator, denominator = numerator[finite], denominator[finite]
    n = len(numerator)
    if n == 0:
        msg = "no finite block values to bootstrap"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    mean = float(numerator.mean() / denominator.mean())

    draws = rng.integers(0, n, size=(n_boot, n))
    boot = numerator[draws].mean(axis=1) / denominator[draws].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    with np.errstate(divide="ignore", invalid="ignore"):
        per_block = np.where(denominator != 0, numerator / denominator, np.nan)
    valid = np.isfinite(per_block)
    sign = float((per_block[valid] > 0).mean()) if valid.any() else float("nan")

    return Estimate(
        mean=mean,
        ci_low=float(lo),
        ci_high=float(hi),
        sign_consistency=sign,
        n_blocks=n,
    )


def paired_effect_size(values: np.ndarray) -> float:
    """Standardized paired effect (Cohen's d_z) over blocks."""
    values = np.asarray(values, dtype=np.float64)
    std = values.std(ddof=1)
    return float(values.mean() / std) if std > 0 else float("nan")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve: how well the confidence margin ranks correct over incorrect.

    Computed from ranks rather than sklearn so that the tie handling is explicit -- logits
    tie often enough in low-precision dtypes to matter.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tied groups.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
