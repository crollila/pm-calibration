from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import backtest, kelly
from pmcal.fees import kalshi_taker_fee


def _panel_with_edge(edge: float = 0.06, depth: float = 50.0, n: int = 60) -> pd.DataFrame:
    """A synthetic panel where the book is right and Polymarket is `edge` cheap."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p = float(rng.uniform(0.3, 0.7))
        rows.append(
            {
                "event_key": f"e{i}", "sport": "nfl", "outcome": "home",
                "ts": pd.Timestamp("2026-09-12 12:00") + pd.Timedelta(hours=i),
                "event_start_time": pd.Timestamp("2026-09-13 12:00") + pd.Timedelta(hours=i),
                "minutes_to_start": 1440.0,
                "pm_bid": p - edge - 0.01, "pm_ask": p - edge, "pm_mid": p - edge - 0.005,
                "pm_bid_size": depth, "pm_ask_size": depth,
                "kalshi_bid": np.nan, "kalshi_ask": np.nan, "kalshi_mid": np.nan,
                "kalshi_bid_size": np.nan, "kalshi_ask_size": np.nan,
                "book_multiplicative": p, "book_additive": p, "book_shin": p,
                "pinnacle_multiplicative": p, "pinnacle_additive": p, "pinnacle_shin": p,
                "n_books": 3.0,
                "y": float(rng.uniform() < p), "y_agreement": 1.0,
            }
        )
    return pd.DataFrame(rows)


def _mirror_to_kalshi(panel: pd.DataFrame) -> pd.DataFrame:
    """Copy the Polymarket quotes onto the Kalshi columns to exercise its fee model."""
    out = panel.copy()
    for src, dst in (("pm_bid", "kalshi_bid"), ("pm_ask", "kalshi_ask"),
                     ("pm_mid", "kalshi_mid"), ("pm_bid_size", "kalshi_bid_size"),
                     ("pm_ask_size", "kalshi_ask_size")):
        out[dst] = out[src]
    return out


def test_signals_require_the_edge_to_clear_the_threshold():
    panel = _panel_with_edge(edge=0.06)
    assert len(backtest.find_signals(panel, backtest.BacktestConfig(threshold=0.03))) == 60
    assert backtest.find_signals(panel, backtest.BacktestConfig(threshold=0.08)).empty


def test_execution_is_at_the_ask_never_the_mid():
    panel = _panel_with_edge()
    cfg = backtest.BacktestConfig(threshold=0.03)
    signals = backtest.find_signals(panel, cfg)
    assert np.allclose(signals["entry_price"], panel["pm_ask"])
    assert (signals["entry_price"] > signals["mid_at_entry"]).all()


def test_fill_size_is_capped_by_displayed_depth():
    panel = _panel_with_edge(depth=7.0)
    cfg = backtest.BacktestConfig(threshold=0.03, max_contracts=1000, bankroll=1_000_000)
    trades = backtest.size_trades(backtest.find_signals(panel, cfg), cfg)
    assert (trades["contracts"] <= 7.0).all()
    assert trades["depth_capped"].all()


def test_hard_contract_cap_also_binds():
    panel = _panel_with_edge(depth=10_000.0)
    cfg = backtest.BacktestConfig(threshold=0.03, max_contracts=25, bankroll=1_000_000)
    trades = backtest.size_trades(backtest.find_signals(panel, cfg), cfg)
    assert (trades["contracts"] <= 25).all()


def test_fees_are_charged_on_every_fill():
    panel = _panel_with_edge()
    cfg = backtest.BacktestConfig(threshold=0.03, trade_venue="polymarket", poly_fee_rate=0.018)
    trades = backtest.settle(backtest.size_trades(backtest.find_signals(panel, cfg), cfg), cfg)
    assert (trades["fee"] > 0).all()
    assert np.allclose(trades["capital"], trades["contracts"] * trades["entry_price"] + trades["fee"])


def test_kalshi_fees_match_the_published_formula():
    panel = _mirror_to_kalshi(_panel_with_edge())
    cfg = backtest.BacktestConfig(threshold=0.03, trade_venue="kalshi")
    trades = backtest.settle(backtest.size_trades(backtest.find_signals(panel, cfg), cfg), cfg)
    expected = [
        kalshi_taker_fee(c, p)
        for c, p in zip(trades["contracts"], trades["entry_price"], strict=True)
    ]
    assert trades["fee"].tolist() == pytest.approx(expected)


def test_fees_can_turn_a_thin_edge_negative():
    """The reason the cost model is not optional."""
    thin = _mirror_to_kalshi(_panel_with_edge(edge=0.012))
    thin["y"] = thin["book_shin"]  # settle exactly at the fair value: pure edge, no variance
    cfg = backtest.BacktestConfig(threshold=0.01, trade_venue="kalshi", max_contracts=1)
    trades = backtest.settle(backtest.size_trades(backtest.find_signals(thin, cfg), cfg), cfg)
    assert (trades["pnl"] < 0).mean() > 0.5


def test_capital_days_account_for_the_lock_up():
    panel = _panel_with_edge()
    cfg = backtest.BacktestConfig(threshold=0.03, settle_lag_hours=4.0)
    trades = backtest.settle(backtest.size_trades(backtest.find_signals(panel, cfg), cfg), cfg)
    assert np.allclose(trades["hold_days"], 1.0 + 4.0 / 24.0)
    assert np.allclose(trades["capital_days"], trades["capital"] * trades["hold_days"])


def test_one_trade_per_outcome_by_default():
    panel = _panel_with_edge(n=5)
    repeated = pd.concat(
        [panel.assign(ts=panel["ts"] + pd.Timedelta(minutes=15 * k)) for k in range(4)],
        ignore_index=True,
    )
    cfg = backtest.BacktestConfig(threshold=0.03)
    assert len(backtest.find_signals(repeated, cfg)) == 5
    assert len(backtest.find_signals(repeated, backtest.BacktestConfig(
        threshold=0.03, one_trade_per_outcome=False))) == 20


def test_summary_reports_intervals_and_event_count():
    panel = _panel_with_edge()
    cfg = backtest.BacktestConfig(threshold=0.03)
    trades, stats = backtest.run(panel, cfg, n_boot=400)
    assert stats["n_events"] == 60
    assert stats["rocd_lo"] <= stats["return_on_capital_days"] <= stats["rocd_hi"]
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert stats["max_drawdown"] >= 0.0
    assert len(trades) == 60


def test_empty_result_is_handled_not_crashed():
    panel = _panel_with_edge(edge=0.001)
    trades, stats = backtest.run(panel, backtest.BacktestConfig(threshold=0.05), n_boot=100)
    assert trades.empty and stats == {"n_trades": 0, "n_events": 0}


def test_sweep_reports_the_whole_curve_including_losing_thresholds():
    panel = _panel_with_edge()
    sweep = backtest.sweep(
        panel, backtest.BacktestConfig(), thresholds=np.array([0.0, 0.03, 0.09]), n_boot=200
    )
    assert list(sweep["threshold"]) == [0.0, 0.03, 0.09]
    assert sweep["n_trades"].iloc[-1] == 0  # published anyway, not dropped


def test_clv_is_measured_against_the_last_pre_kickoff_mark():
    panel = _panel_with_edge(n=3)
    late = panel.copy()
    late["ts"] = late["ts"] + pd.Timedelta(hours=23)
    late["minutes_to_start"] = 30.0
    late["pm_mid"] = late["book_shin"]  # the market converges to fair by the close
    both = pd.concat([panel, late], ignore_index=True)

    cfg = backtest.BacktestConfig(threshold=0.03)
    trades, _ = backtest.run(both, cfg, n_boot=100)
    assert trades["clv_valid"].all()
    assert (trades["clv"] > 0).all()  # we bought below where the market closed


def test_max_drawdown_on_a_known_sequence():
    pnl = pd.Series([10.0, -4.0, -3.0, 6.0, -1.0])
    assert backtest.max_drawdown(pnl) == pytest.approx(7.0)
    assert backtest.max_drawdown(pd.Series(dtype=float)) == 0.0


def test_kelly_sizing_hand_computed_and_scaled():
    assert kelly.kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly.kelly_fraction(0.6, 0.5, 0.25) == pytest.approx(0.05)
    assert kelly.kelly_fraction(0.4, 0.5) == 0.0
    with pytest.raises(ValueError):
        kelly.kelly_fraction(0.6, 1.0)


def test_expected_log_growth_is_unchanged_by_symmetric_uncertainty():
    """The negative result the module is built around.

    Log growth is affine in p, so averaging over a noisy p returns the growth at
    its mean. 'p is uncertain' is therefore NOT on its own an argument for
    betting less -- the real arguments are amplification and asymmetry, below.
    """
    out = kelly.expected_growth_is_linear_in_p(0.60, 0.50, sigma=0.08)
    assert out["mean_growth"] == pytest.approx(out["growth_at_mean_p"], abs=5e-4)
    # The spread of realised growth is large relative to the growth itself --
    # the mean is unmoved, but the risk it hides is not.
    assert out["sd_growth"] > out["growth_at_mean_p"]


def test_a_bigger_stake_amplifies_an_error_in_the_estimate():
    assert kelly.growth_slope(0.05, 0.5) < kelly.growth_slope(0.2, 0.5)
    assert kelly.growth_slope(0.2, 0.5) < kelly.growth_slope(0.6, 0.5)


def test_risk_adjusted_optimum_is_below_full_kelly_and_equals_a_shrunk_estimate():
    certain = kelly.optimal_fraction_under_uncertainty(0.60, 0.50, sigma=0.0)
    noisy = kelly.optimal_fraction_under_uncertainty(0.60, 0.50, sigma=0.08)
    assert certain["ratio"] == pytest.approx(1.0)
    assert noisy["optimal"] < noisy["full_kelly"]
    assert noisy["objective_at_optimal"] > noisy["objective_at_full"]
    # Fractional Kelly IS estimate shrinkage: the grid optimum matches the
    # closed form, full Kelly evaluated at p_hat - k*sigma.
    assert noisy["optimal"] == pytest.approx(noisy["closed_form"], abs=1e-3)
    assert noisy["equivalent_shrunk_p"] == pytest.approx(0.52)


def test_the_optimal_kelly_fraction_shrinks_as_noise_grows():
    curve = kelly.uncertainty_curve(0.60, 0.50, np.array([0.0, 0.03, 0.06, 0.09]))
    ratios = [row["ratio"] for row in curve]
    assert ratios == sorted(ratios, reverse=True)
    assert ratios[0] == pytest.approx(1.0)
    assert ratios[-1] < 0.6
    assert all(np.isfinite(r) for r in ratios)


def test_overbetting_costs_far_more_than_underbetting():
    """Half Kelly keeps ~75% of the growth; double Kelly keeps none of it."""
    out = kelly.overbetting_asymmetry(0.60, 0.50)
    assert out["retained_at_0.5x"] == pytest.approx(0.75, abs=0.03)
    assert out["retained_at_1.0x"] == pytest.approx(1.0)
    assert out["retained_at_2.0x"] < 0.01
    assert out["retained_at_0.25x"] > out["retained_at_2.0x"]
