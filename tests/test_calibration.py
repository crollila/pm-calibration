from __future__ import annotations

import numpy as np
import pytest

from analysis import calibration


def test_brier_and_log_loss_hand_computed():
    p = np.array([0.7, 0.3])
    y = np.array([1.0, 0.0])
    assert calibration.brier_score(p, y) == pytest.approx(0.09)
    assert calibration.log_loss(p, y) == pytest.approx(-np.log(0.7))


def test_log_loss_punishes_confident_errors_far_harder_than_brier():
    """Why both metrics are reported instead of one."""
    confident_wrong = (np.array([0.01]), np.array([1.0]))
    unsure_wrong = (np.array([0.45]), np.array([1.0]))
    brier_ratio = calibration.brier_score(*confident_wrong) / calibration.brier_score(*unsure_wrong)
    ll_ratio = calibration.log_loss(*confident_wrong) / calibration.log_loss(*unsure_wrong)
    assert brier_ratio < 4
    assert ll_ratio > 5


def test_murphy_decomposition_reconstructs_the_brier_score():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 5000)
    y = (rng.uniform(size=5000) < p).astype(float)
    parts = calibration.murphy_decomposition(p, y)
    rebuilt = parts["reliability"] - parts["resolution"] + parts["uncertainty"]
    assert rebuilt == pytest.approx(parts["brier"], abs=2e-3)
    assert abs(parts["residual"]) < 2e-3


def test_perfect_calibration_has_near_zero_reliability():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.uniform(size=20_000) < p).astype(float)
    parts = calibration.murphy_decomposition(p, y)
    assert parts["reliability"] < 0.002
    assert parts["resolution"] > 0.05


def test_a_biased_forecaster_shows_up_as_reliability():
    rng = np.random.default_rng(2)
    truth = rng.uniform(0.05, 0.95, 20_000)
    y = (rng.uniform(size=20_000) < truth).astype(float)
    biased = np.clip(truth + 0.10, 0.01, 0.99)
    assert (
        calibration.murphy_decomposition(biased, y)["reliability"]
        > 20 * calibration.murphy_decomposition(truth, y)["reliability"]
    )


def test_uncertainty_is_the_base_rate_variance_and_venue_independent():
    y = np.array([1.0] * 30 + [0.0] * 70)
    a = calibration.murphy_decomposition(np.full(100, 0.3), y)
    b = calibration.murphy_decomposition(np.random.default_rng(3).uniform(size=100), y)
    assert a["uncertainty"] == pytest.approx(0.21)
    assert a["uncertainty"] == pytest.approx(b["uncertainty"])


def test_calibration_table_bins_and_covers():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.98, 4000)
    y = (rng.uniform(size=4000) < p).astype(float)
    table = calibration.calibration_table(p, y)
    assert table["n"].sum() == 4000
    assert (table["wilson_lo"] <= table["observed"]).all()
    assert (table["observed"] <= table["wilson_hi"]).all()
    # A calibrated forecaster sits inside its own interval nearly everywhere.
    inside = (table["wilson_lo"] <= table["mean_forecast"]) & (
        table["mean_forecast"] <= table["wilson_hi"]
    )
    assert inside.mean() >= 0.8


def test_empty_input_does_not_raise():
    empty = calibration.murphy_decomposition(np.array([]), np.array([]))
    assert np.isnan(empty["brier"])
    assert calibration.calibration_table(np.array([]), np.array([])).empty


def test_venue_summary_reports_every_venue_with_intervals(panel):
    summary = calibration.venue_summary(panel, n_boot=300)
    assert not summary.empty
    assert {"polymarket", "kalshi", "book_consensus", "pinnacle"} <= set(summary["venue"])
    assert (summary["brier_lo"] <= summary["brier"]).all()
    assert (summary["brier"] <= summary["brier_hi"]).all()
    assert (summary["n_events"] < summary["n_rows"]).all()


def test_favourite_longshot_slope_is_near_one_on_calibrated_synthetic_data(panel):
    flb = calibration.favourite_longshot(panel)
    row = flb.set_index("venue").loc["polymarket"]
    assert abs(row["slope"] - 1.0) < 4 * row["se_slope"]
    assert row["n_clusters"] < row["n"]


def test_head_to_head_is_paired_and_reports_significance(panel):
    out = calibration.head_to_head(panel, "polymarket", "book_consensus")
    assert out["n_rows"] > 0
    assert out["diff_lo"] <= out["brier_diff"] <= out["diff_hi"]
    assert isinstance(out["significant"], bool)


def test_sharpening_profile_buckets_by_time(panel):
    profile = calibration.sharpening_profile(panel, n_boot=200)
    assert not profile.empty
    assert profile["bucket_index"].nunique() > 1
    assert (profile["brier_lo"] <= profile["brier"]).all()


def test_devig_comparison_covers_all_three_methods(panel):
    out = calibration.devig_method_comparison(panel, n_boot=200)
    assert set(out["venue"]) == {"book_multiplicative", "book_additive", "book_shin"}
