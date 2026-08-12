"""Inference helpers: Wilson intervals, clustered bootstrap, clustered OLS.

Clustering by event is not a nicety. Both outcomes of a game are perfectly
negatively correlated, and consecutive snapshots of the same game are nearly
identical draws. Treating those as independent observations shrinks every
standard error by roughly sqrt(rows per event) and manufactures significance.
Everything here resamples or sandwiches at the *event* level.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

Statistic = Callable[[np.ndarray], float]


def wilson_interval(successes: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because calibration bins at the
    extremes routinely have observed frequencies of 0 or 1, where the Wald
    interval collapses to zero width.

    >>> lo, hi = wilson_interval(8, 10)
    >>> round(lo, 4), round(hi, 4)
    (0.4902, 0.9433)
    >>> wilson_interval(0, 0)
    (0.0, 1.0)
    """
    if n <= 0:
        return (0.0, 1.0)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z / denom * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def cluster_indices(clusters: Sequence) -> list[np.ndarray]:
    """Row positions belonging to each cluster, computed once and reused."""
    arr = np.asarray(clusters)
    if arr.size == 0:
        return []
    order = np.argsort(arr, kind="stable")
    sorted_arr = arr[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_arr[1:] != sorted_arr[:-1], True])
    return [order[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]


def cluster_bootstrap(
    statistic: Statistic,
    clusters: Sequence,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 20260810,
) -> dict[str, float]:
    """Percentile bootstrap over whole clusters.

    `statistic` receives an index array of resampled row positions and returns a
    scalar. Resampling *clusters* (not rows) preserves within-event correlation.

    Returns point estimate, CI bounds, the bootstrap SE, and the number of
    independent clusters the interval actually rests on.
    """
    groups = cluster_indices(clusters)
    n_groups = len(groups)
    if n_groups == 0:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan, "se": np.nan, "n_clusters": 0}

    all_idx = np.concatenate(groups)
    point = float(statistic(all_idx))
    if n_groups < 2:
        return {"point": point, "lo": np.nan, "hi": np.nan, "se": np.nan, "n_clusters": n_groups}

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([groups[k] for k in pick])
        draws[b] = statistic(idx)

    finite = draws[np.isfinite(draws)]
    if finite.size < 2:
        return {"point": point, "lo": np.nan, "hi": np.nan, "se": np.nan, "n_clusters": n_groups}
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "se": float(finite.std(ddof=1)),
        "n_clusters": n_groups,
    }


def ols_cluster(
    x: np.ndarray, y: np.ndarray, clusters: Sequence
) -> dict[str, float]:
    """Simple regression y = a + b*x with cluster-robust (CR1) standard errors.

    The favourite-longshot test: regress the realised 0/1 outcome on the
    forecast probability. A perfectly calibrated forecaster gives b = 1, a = 0.
    b < 1 is the classic signature -- longshots win less often than priced.

    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(0.05, 0.95, 4000)
    >>> y = (rng.uniform(size=4000) < p).astype(float)
    >>> fit = ols_cluster(p, y, np.arange(4000) // 2)
    >>> bool(abs(fit["slope"] - 1.0) < 3 * fit["se_slope"])
    True
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    groups = cluster_indices(np.asarray(clusters)[keep])

    design = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta

    meat = np.zeros((2, 2))
    for idx in groups:
        xg = design[idx]
        ug = xg.T @ resid[idx]
        meat += np.outer(ug, ug)

    n, k, g = len(y), 2, max(len(groups), 1)
    scale = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = scale * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.diag(cov))
    crit = float(stats.t.ppf(0.975, max(g - 1, 1)))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_intercept": float(se[0]),
        "se_slope": float(se[1]),
        "slope_lo": float(beta[1] - crit * se[1]),
        "slope_hi": float(beta[1] + crit * se[1]),
        "intercept_lo": float(beta[0] - crit * se[0]),
        "intercept_hi": float(beta[0] + crit * se[0]),
        "n": int(n),
        "n_clusters": int(g),
    }


def paired_difference(
    values_a: np.ndarray, values_b: np.ndarray, clusters: Sequence, **kwargs
) -> dict[str, float]:
    """Clustered bootstrap CI for mean(a) - mean(b) on paired rows.

    Used to compare two venues' Brier scores on exactly the same events, which
    is far more powerful than comparing two independently estimated scores.
    """
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)

    def stat(idx: np.ndarray) -> float:
        sample = diff[idx]
        sample = sample[np.isfinite(sample)]
        return float(sample.mean()) if sample.size else np.nan

    return cluster_bootstrap(stat, clusters, **kwargs)
