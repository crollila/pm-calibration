from __future__ import annotations

import numpy as np
import pytest

from pmcal.stats import cluster_bootstrap, ols_cluster, paired_difference, wilson_interval


def test_wilson_interval_hand_computed():
    lo, hi = wilson_interval(8, 10)
    assert (lo, hi) == pytest.approx((0.4901, 0.9433), abs=5e-4)


def test_wilson_stays_inside_zero_one_at_the_extremes():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.25
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and 0.75 < lo < 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_narrows_with_sample_size():
    small = wilson_interval(50, 100)
    large = wilson_interval(5000, 10_000)
    assert (large[1] - large[0]) < (small[1] - small[0]) / 5


def test_clustering_by_event_widens_intervals():
    """The point of the clustering requirement, demonstrated.

    Twenty identical copies of each observation carry no extra information.
    A naive row-level bootstrap thinks they do; a clustered one does not.
    """
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 60)
    values = np.repeat(base, 20)
    events = np.repeat(np.arange(60), 20)
    rows = np.arange(values.size)

    def mean(idx: np.ndarray) -> float:
        return float(values[idx].mean())

    clustered = cluster_bootstrap(mean, events, n_boot=2000)
    naive = cluster_bootstrap(mean, rows, n_boot=2000)
    assert (clustered["hi"] - clustered["lo"]) > 3 * (naive["hi"] - naive["lo"])
    assert clustered["n_clusters"] == 60
    assert naive["n_clusters"] == 1200


def test_bootstrap_point_estimate_uses_the_full_sample():
    values = np.arange(100, dtype=float)
    out = cluster_bootstrap(lambda i: float(values[i].mean()), np.arange(100) // 5, n_boot=500)
    assert out["point"] == pytest.approx(values.mean())
    assert out["lo"] < out["point"] < out["hi"]


def test_bootstrap_handles_degenerate_input():
    out = cluster_bootstrap(lambda i: 1.0, [], n_boot=10)
    assert out["n_clusters"] == 0 and np.isnan(out["point"])
    single = cluster_bootstrap(lambda i: 1.0, ["a", "a"], n_boot=10)
    assert single["n_clusters"] == 1 and np.isnan(single["lo"])


def test_ols_recovers_a_known_slope():
    rng = np.random.default_rng(1)
    x = rng.uniform(0.05, 0.95, 3000)
    y = 0.5 * x + 0.2 + rng.normal(0, 0.05, 3000)
    fit = ols_cluster(x, y, np.arange(3000) // 3)
    assert fit["slope"] == pytest.approx(0.5, abs=0.02)
    assert fit["intercept"] == pytest.approx(0.2, abs=0.02)
    assert fit["slope_lo"] < 0.5 < fit["slope_hi"]


def test_a_calibrated_forecaster_has_slope_one():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 6000)
    y = (rng.uniform(size=6000) < p).astype(float)
    fit = ols_cluster(p, y, np.arange(6000) // 2)
    assert abs(fit["slope"] - 1.0) < 3 * fit["se_slope"]
    assert abs(fit["intercept"]) < 3 * fit["se_intercept"]


def test_cluster_robust_se_exceeds_the_iid_se_under_correlation():
    rng = np.random.default_rng(3)
    n_groups, per = 80, 15
    shock = rng.normal(0, 0.3, n_groups).repeat(per)
    x = rng.uniform(0, 1, n_groups * per)
    y = x + shock + rng.normal(0, 0.05, n_groups * per)
    clustered = ols_cluster(x, y, np.arange(n_groups * per) // per)
    iid = ols_cluster(x, y, np.arange(n_groups * per))
    assert clustered["se_intercept"] > 2 * iid["se_intercept"]


def test_paired_difference_detects_a_known_gap():
    rng = np.random.default_rng(4)
    a = rng.normal(0.20, 0.02, 500)
    b = a + 0.01
    out = paired_difference(a, b, np.arange(500) // 2, n_boot=1000)
    assert out["point"] == pytest.approx(-0.01, abs=1e-3)
    assert out["hi"] < 0
