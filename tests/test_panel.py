from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmcal import matching
from pmcal import panel as panel_mod

NOW = pd.Timestamp("2026-09-13 12:00")


def _res(venue: str, outcome: str, value, market_id: str = "m", event_key: str = "e1") -> dict:
    return {"venue": venue, "market_id": market_id, "outcome": outcome,
            "event_key": event_key, "resolved": value}


def test_unanimous_venues_produce_a_label():
    frame = pd.DataFrame([_res("polymarket", "chiefs", 1.0), _res("kalshi", "chiefs", 1.0)])
    out = panel_mod.ground_truth(frame)
    assert out.loc[0, "y"] == 1.0
    assert out.loc[0, "y_agreement"] == pytest.approx(1.0)


def test_a_two_venue_disagreement_produces_no_label():
    frame = pd.DataFrame([_res("polymarket", "chiefs", 1.0), _res("kalshi", "chiefs", 0.0)])
    out = panel_mod.ground_truth(frame)
    assert np.isnan(out.loc[0, "y"])


def test_a_third_opinion_breaks_the_tie_and_records_agreement():
    frame = pd.DataFrame([
        _res("polymarket", "chiefs", 1.0),
        _res("kalshi", "chiefs", 0.0),
        _res("oddsapi_scores", "chiefs", 1.0),
    ])
    out = panel_mod.ground_truth(frame)
    assert out.loc[0, "y"] == 1.0
    assert out.loc[0, "y_agreement"] == pytest.approx(2 / 3)


def test_conflicts_are_recorded_rather_than_resolved():
    frame = pd.DataFrame([_res("polymarket", "chiefs", 1.0), _res("kalshi", "chiefs", 0.0)])
    conflicts = panel_mod.find_conflicts(frame, NOW)
    assert len(conflicts) == 1
    row = conflicts.iloc[0]
    assert {row["venue_a"], row["venue_b"]} == {"polymarket", "kalshi"}
    assert row["value_a"] != row["value_b"]


def test_three_way_disagreement_yields_all_pairs():
    frame = pd.DataFrame([
        _res("polymarket", "chiefs", 1.0),
        _res("kalshi", "chiefs", 0.0),
        _res("oddsapi_scores", "chiefs", 1.0),
    ])
    assert len(panel_mod.find_conflicts(frame, NOW)) == 2  # pm-kalshi, kalshi-scores


def test_agreement_produces_no_conflicts():
    frame = pd.DataFrame([_res("polymarket", "chiefs", 1.0), _res("kalshi", "chiefs", 1.0)])
    assert panel_mod.find_conflicts(frame, NOW).empty


def test_unresolved_rows_are_ignored_everywhere():
    frame = pd.DataFrame([_res("polymarket", "chiefs", None), _res("kalshi", "chiefs", 1.0)])
    assert panel_mod.ground_truth(frame).loc[0, "y"] == 1.0
    assert panel_mod.find_conflicts(frame, NOW).empty


def test_panel_has_one_row_per_event_outcome_timestamp(panel):
    assert not panel.duplicated(subset=["event_key", "outcome", "ts"]).any()
    assert set(panel_mod.PANEL_COLUMNS) == set(panel.columns)


def test_panel_carries_every_venue_and_devig_method(panel):
    for column in ("pm_mid", "kalshi_mid", "book_multiplicative", "book_additive",
                   "book_shin", "pinnacle_shin"):
        assert panel[column].notna().any(), column


def test_book_probabilities_sum_to_one_within_each_event(panel):
    totals = panel.groupby(["event_key", "ts"])["book_shin"].sum()
    assert totals.dropna().sub(1.0).abs().max() < 1e-9


def test_minutes_to_start_is_positive_and_decreasing(panel):
    assert (panel["minutes_to_start"] > 0).all()
    first_event = panel[panel["event_key"] == panel["event_key"].iloc[0]]
    per_outcome = first_event[first_event["outcome"] == first_event["outcome"].iloc[0]]
    assert per_outcome["minutes_to_start"].is_monotonic_decreasing


def test_empty_input_returns_an_empty_panel():
    out = panel_mod.build_panel(pd.DataFrame(), pd.DataFrame())
    assert out.empty and list(out.columns) == panel_mod.PANEL_COLUMNS


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Chiefs vs. Ravens", ("chiefs", "ravens")),
        ("Will the Kansas City Chiefs beat the Baltimore Ravens?", ("chiefs", "ravens")),
        ("Red Sox @ Yankees", ("red sox", "yankees")),
        ("Fed rate decision", ()),
    ],
)
def test_team_extraction(title, expected):
    assert matching.extract_teams(title, None if not expected else _sport(expected)) == expected


def _sport(teams):
    for sport, names in matching.NICKNAMES.items():
        if all(t in names for t in teams):
            return sport
    return None


def test_event_key_is_order_independent_and_timezone_sane():
    kickoff = pd.Timestamp("2026-09-13 00:20").to_pydatetime()  # Sunday night game, UTC Monday
    a = matching.build_event_key("nfl", kickoff, ("ravens", "chiefs"))
    b = matching.build_event_key("nfl", kickoff, ("chiefs", "ravens"))
    assert a == b == "nfl|2026-09-12|chiefs~ravens"


def test_event_key_requires_exactly_two_teams():
    kickoff = pd.Timestamp("2026-09-13 00:20").to_pydatetime()
    assert matching.build_event_key("nfl", kickoff, ("chiefs",)) is None
    assert matching.build_event_key("nfl", None, ("chiefs", "ravens")) is None
