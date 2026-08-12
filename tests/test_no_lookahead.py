"""The no-lookahead property, asserted rather than asserted-to-be-true.

The test: rebuild the panel from data truncated at time T, and require every row
at or before T to be byte-identical to the corresponding row built from the full
history. If any feature secretly consulted the future -- a later book quote
correcting a kickoff time, a forward-filled price, a normalisation computed over
the whole sample -- truncation changes it and this test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.backtest import BacktestConfig, find_signals, settle, size_trades
from pmcal.panel import build_panel
from tests.factories import make_snapshots

FEATURE_COLUMNS = [
    "pm_bid", "pm_ask", "pm_mid", "pm_bid_size", "pm_ask_size",
    "kalshi_bid", "kalshi_ask", "kalshi_mid",
    "book_multiplicative", "book_additive", "book_shin",
    "pinnacle_multiplicative", "pinnacle_additive", "pinnacle_shin",
    "n_books", "minutes_to_start", "event_start_time",
]
KEYS = ["event_key", "outcome", "ts"]


def _cutoffs(snapshots: pd.DataFrame) -> list[pd.Timestamp]:
    stamps = sorted(pd.to_datetime(snapshots["ts"]).unique())
    return [stamps[len(stamps) // 4], stamps[len(stamps) // 2], stamps[-2]]


@pytest.mark.parametrize("cut_index", [0, 1, 2])
def test_panel_features_do_not_change_when_the_future_is_removed(cut_index):
    snapshots, labels = make_snapshots(n_events=12, n_ts=6, seed=3)
    cutoff = _cutoffs(snapshots)[cut_index]

    full = build_panel(snapshots, labels)
    truncated = build_panel(snapshots[pd.to_datetime(snapshots["ts"]) <= cutoff], labels)

    full_head = full[pd.to_datetime(full["ts"]) <= cutoff].sort_values(KEYS).reset_index(drop=True)
    trunc = truncated.sort_values(KEYS).reset_index(drop=True)

    assert len(trunc) == len(full_head) > 0
    pd.testing.assert_frame_equal(
        trunc[KEYS + FEATURE_COLUMNS], full_head[KEYS + FEATURE_COLUMNS], check_dtype=False
    )


def test_kickoff_time_is_as_of_not_best_known_later():
    """A book that only shows up late must not retro-fix earlier rows."""
    snapshots, labels = make_snapshots(n_events=4, n_ts=4, seed=5)
    ts_values = sorted(pd.to_datetime(snapshots["ts"]).unique())

    # The sportsbook is authoritative for kickoff; delete its early rows and give
    # the prediction markets a deliberately wrong proxy start time.
    late_books = (snapshots["venue"] != "sportsbook") | (
        pd.to_datetime(snapshots["ts"]) >= ts_values[-1]
    )
    doctored = snapshots[late_books].copy()
    proxy = doctored["venue"] != "sportsbook"
    doctored.loc[proxy, "event_start_time"] = doctored.loc[
        proxy, "event_start_time"
    ] + pd.Timedelta(hours=3)

    panel = build_panel(doctored, labels)
    early = panel[pd.to_datetime(panel["ts"]) < ts_values[-1]]
    late = panel[pd.to_datetime(panel["ts"]) >= ts_values[-1]]

    # Early rows can only know the (wrong) proxy; late rows switch to the book.
    assert not early.empty and not late.empty
    assert early["event_start_time"].nunique() >= 1
    merged = early.merge(
        late[["event_key", "outcome", "event_start_time"]],
        on=["event_key", "outcome"], suffixes=("_early", "_late"),
    )
    assert (merged["event_start_time_early"] != merged["event_start_time_late"]).all()


def test_panel_excludes_post_kickoff_quotes():
    snapshots, labels = make_snapshots(n_events=5, n_ts=3, seed=8)
    after = snapshots.copy()
    after["ts"] = pd.to_datetime(after["event_start_time"]) + pd.Timedelta(minutes=30)
    panel = build_panel(pd.concat([snapshots, after], ignore_index=True), labels)
    assert (panel["minutes_to_start"] > 0).all()


def test_trade_entry_uses_only_the_signal_row():
    """A signal at ts must be executable with quotes visible at ts."""
    snapshots, labels = make_snapshots(n_events=25, n_ts=4, seed=13)
    panel = build_panel(snapshots, labels)
    # Force signals to exist regardless of the sample's natural divergence.
    panel = panel.copy()
    panel["book_shin"] = np.clip(panel["pm_ask"] + 0.05, 0.01, 0.99)

    cfg = BacktestConfig(threshold=0.02, trade_venue="polymarket")
    trades = settle(size_trades(find_signals(panel, cfg), cfg), cfg)
    assert len(trades) > 0

    joined = trades.merge(
        panel[KEYS + ["pm_ask", "pm_ask_size"]], on=KEYS, how="left", suffixes=("", "_panel")
    )
    # Entry price is the ask at the signal timestamp, never a later or mid price.
    assert np.allclose(joined["entry_price"], joined["pm_ask_panel"])
    assert (joined["contracts"] <= joined["pm_ask_size_panel"]).all()
    # And the trade is always struck strictly before the event starts.
    assert (trades["minutes_to_start"] > 0).all()


def test_backtest_never_fills_at_the_midpoint():
    snapshots, labels = make_snapshots(n_events=20, n_ts=3, seed=21)
    panel = build_panel(snapshots, labels).copy()
    panel["book_shin"] = np.clip(panel["pm_ask"] + 0.05, 0.01, 0.99)
    cfg = BacktestConfig(threshold=0.02, trade_venue="polymarket")
    trades = settle(size_trades(find_signals(panel, cfg), cfg), cfg)
    assert (trades["entry_price"] > trades["mid_at_entry"]).all()
