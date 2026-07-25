"""Honest statistics: bootstrap confidence intervals.

- ci_mean:      percentile bootstrap 95% CI on the mean of per-item scores
                (GLEU, accuracy-as-0/1, judge rubric means).
- paired_delta: paired bootstrap on the mean difference between two models
                over the SAME items; reports the CI and whether it excludes 0.
- ci_statistic: bootstrap a whole-sample statistic (QWK, macro-F1) by
                resampling (pred, gold) pairs, since those don't decompose
                into per-item means.

All seeded. No headline number leaves report.py without one of these.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_ITERS = 2000
ALPHA = 0.05


@dataclass
class CI:
    point: float
    lo: float
    hi: float
    n: int

    def fmt(self, digits: int = 3) -> str:
        return f"{self.point:.{digits}f} [{self.lo:.{digits}f}, {self.hi:.{digits}f}]"


@dataclass
class PairedDelta:
    delta: float
    lo: float
    hi: float
    excludes_zero: bool
    n: int


def ci_mean(values: Sequence[float], seed: int = 0, iters: int = DEFAULT_ITERS) -> CI:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return CI(point=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(iters, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [ALPHA / 2, 1 - ALPHA / 2])
    return CI(point=float(arr.mean()), lo=float(lo), hi=float(hi), n=int(arr.size))


def paired_delta(
    a: Sequence[float], b: Sequence[float], seed: int = 0, iters: int = DEFAULT_ITERS
) -> PairedDelta:
    """Mean(a) - Mean(b) over identical items. Lengths must match: callers
    align on sample_id before calling (runner guarantees shared manifests)."""
    xa, xb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if xa.shape != xb.shape:
        raise ValueError(f"paired_delta needs aligned items, got {xa.shape} vs {xb.shape}")
    if xa.size == 0:
        return PairedDelta(float("nan"), float("nan"), float("nan"), False, 0)
    diff = xa - xb
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(iters, diff.size))
    deltas = diff[idx].mean(axis=1)
    lo, hi = np.quantile(deltas, [ALPHA / 2, 1 - ALPHA / 2])
    return PairedDelta(
        delta=float(diff.mean()),
        lo=float(lo),
        hi=float(hi),
        excludes_zero=bool(lo > 0 or hi < 0),
        n=int(diff.size),
    )


def ci_statistic(
    pairs: Sequence[tuple[str, str]],
    statistic: Callable[[list[str], list[str]], float],
    seed: int = 0,
    iters: int = DEFAULT_ITERS,
) -> CI:
    """Bootstrap a non-decomposable statistic over (pred, gold) pairs."""
    items = list(pairs)
    if not items:
        return CI(point=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    point = statistic([p for p, _ in items], [g for _, g in items])
    rng = np.random.default_rng(seed)
    stats = []
    n = len(items)
    for _ in range(iters):
        sample = [items[i] for i in rng.integers(0, n, size=n)]
        stats.append(statistic([p for p, _ in sample], [g for _, g in sample]))
    lo, hi = np.quantile(np.asarray(stats), [ALPHA / 2, 1 - ALPHA / 2])
    return CI(point=float(point), lo=float(lo), hi=float(hi), n=n)
