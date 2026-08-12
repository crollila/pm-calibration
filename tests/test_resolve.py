"""Settlement parsing and the bounded pending-markets query."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

import resolve
from pmcal import db, resolution
from pmcal.util import utcnow

NOW = datetime(2026, 9, 13, 12, 0)


def _snapshot(venue: str, market_id: str, ts: datetime, start: datetime) -> list:
    values = {
        "ts": ts, "venue": venue, "book": "", "market_id": market_id, "outcome": "chiefs",
        "event_key": "nfl|2026-09-12|chiefs~ravens", "sport": "nfl", "title": "t",
        "bid": 0.5, "ask": 0.52, "mid": 0.51, "bid_size": 10.0, "ask_size": 10.0,
        "volume": 1.0, "event_start_time": start, "collected_at": ts, "run_id": "r", "source": "live",
        "raw_json": "{}",
    }
    return [values[c] for c in db.SNAPSHOT_COLUMNS]


@pytest.fixture()
def con(tmp_path):
    connection = db.connect(tmp_path / "r.duckdb")
    yield connection
    connection.close()


def test_pending_only_includes_events_past_the_grace_period(con):
    past = utcnow() - timedelta(days=1)
    future = utcnow() + timedelta(days=1)
    db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS,
                   [_snapshot("kalshi", "K1:yes", NOW, past),
                    _snapshot("kalshi", "K2:yes", NOW, future)], db.SNAPSHOT_KEY)
    pending = resolve.pending_markets(con, grace_minutes=240, limit=500)
    assert pending["market_id"].tolist() == ["K1:yes"]


def test_pending_excludes_already_settled_markets(con):
    past = utcnow() - timedelta(days=1)
    db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS,
                   [_snapshot("kalshi", "K1:yes", NOW, past)], db.SNAPSHOT_KEY)
    db.upsert_rows(
        con, "resolutions",
        ("venue", "market_id", "outcome", "event_key", "resolved", "status",
         "resolved_at", "observed_at", "raw_json"),
        [["kalshi", "K1:yes", "chiefs", "e1", 1.0, "settled", NOW, NOW, "{}"]],
        ("venue", "market_id", "outcome"),
    )
    assert resolve.pending_markets(con, 240, 500).empty


def test_pending_is_bounded_and_most_recent_first(con):
    base = utcnow() - timedelta(days=10)
    rows = [
        _snapshot("kalshi", f"K{i}:yes", NOW, base + timedelta(hours=i)) for i in range(20)
    ]
    db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, rows, db.SNAPSHOT_KEY)
    pending = resolve.pending_markets(con, 240, limit=5)
    assert len(pending) == 5
    assert pending["market_id"].iloc[0] == "K19:yes"


def test_kalshi_settlement_maps_both_sides():
    market = {"status": "settled", "result": "yes", "yes_sub_title": "Chiefs",
              "no_sub_title": "Ravens", "expiration_time": "2026-09-13T04:00:00Z"}
    records = resolution._kalshi_records("K1", market, NOW)
    by_id = {r["market_id"]: r for r in records}
    assert by_id["K1:yes"]["resolved"] == 1.0
    assert by_id["K1:no"]["resolved"] == 0.0
    assert by_id["K1:yes"]["resolved_at"] == datetime(2026, 9, 13, 4, 0)


def test_polymarket_resolutions_are_fetched_by_path_not_by_query_filter(monkeypatch):
    """Gamma's `?id=` filter answers 200 with an empty list instead of erroring.

    A batched `?id=a&id=b` implementation therefore resolves nothing while
    looking perfectly healthy -- which is what happened on the first real run.
    """
    calls: list[str] = []

    def fake_get(session, url, params=None, timeout=None):
        calls.append(url)
        if "?" in url or params:
            return []          # the trap
        return {"id": "512", "closed": True, "outcomes": '["Chiefs","Ravens"]',
                "outcomePrices": '["1","0"]', "endDate": "2026-09-13T04:00:00Z"}

    monkeypatch.setattr(resolution, "get_json", fake_get)
    cfg = type("Cfg", (), {"request_timeout": 5})()
    records = resolution.polymarket_resolutions(None, ["512"], cfg, NOW)

    assert calls == ["https://gamma-api.polymarket.com/markets/512"]
    assert [r["resolved"] for r in records] == [1.0, 0.0]


def test_missing_polymarket_markets_are_counted_not_silently_dropped(monkeypatch, caplog):
    monkeypatch.setattr(resolution, "get_json", lambda *a, **k: [])
    cfg = type("Cfg", (), {"request_timeout": 5})()
    with caplog.at_level("WARNING"):
        assert resolution.polymarket_resolutions(None, ["1", "2"], cfg, NOW) == []
    assert "2/2 market lookups returned nothing" in caplog.text


def test_kalshi_no_side_settles_to_the_opponent_not_a_repeated_subtitle():
    """Kalshi's `no_sub_title` echoes the YES team; the ticker names the real opponent.

    Taking the subtitle at face value would settle both outcomes of a game to
    the same team, which reads as a 100%-accurate forecaster on one side and a
    0% one on the other -- a silent, catastrophic labelling bug.
    """
    market = {"status": "settled", "result": "yes", "ticker": "KXMLBGAME-26AUG101945PHISTL-STL",
              "yes_sub_title": "St. Louis", "no_sub_title": "St. Louis",
              "expiration_time": "2026-08-11T02:54:44Z"}
    records = resolution._kalshi_records("KXMLBGAME-26AUG101945PHISTL-STL", market, NOW)
    by_id = {r["market_id"].rsplit(":", 1)[1]: r for r in records}
    assert by_id["yes"]["outcome_raw"] == "stl"
    assert by_id["no"]["outcome_raw"] == "phi"
    assert by_id["yes"]["resolved"] == 1.0
    assert by_id["no"]["resolved"] == 0.0


def test_resolution_labels_share_the_snapshot_namespace():
    """Labels must be team codes, matching what the collectors write."""
    records = [{"venue": "polymarket", "market_id": "m1", "outcome_raw": "St. Louis Cardinals",
                "resolved": 1.0, "status": "settled", "resolved_at": NOW,
                "observed_at": NOW, "raw_json": "{}"}]
    rows = resolve._to_rows(records, {("polymarket", "m1"): ("mlb|2026-08-10|phi~stl", "mlb")})
    assert rows[0][2] == "stl"
    assert rows[0][3] == "mlb|2026-08-10|phi~stl"


def test_kalshi_open_market_is_not_treated_as_settled():
    market = {"status": "active", "result": "", "yes_sub_title": "Chiefs"}
    assert all(r["resolved"] is None for r in resolution._kalshi_records("K1", market, NOW))


def test_polymarket_settlement_requires_prices_at_the_extremes():
    settled = {"id": "1", "closed": True, "outcomes": '["Chiefs","Ravens"]',
               "outcomePrices": '["1","0"]', "endDate": "2026-09-13T04:00:00Z"}
    records = resolution._poly_records(settled, NOW)
    assert [r["resolved"] for r in records] == [1.0, 0.0]
    assert all(r["status"] == "settled" for r in records)


def test_polymarket_closed_but_undecided_is_left_unresolved():
    """A 0.98 price is a market that stopped trading, not a settled outcome."""
    undecided = {"id": "1", "closed": True, "outcomes": '["Chiefs","Ravens"]',
                 "outcomePrices": '["0.98","0.02"]'}
    records = resolution._poly_records(undecided, NOW)
    assert all(r["resolved"] is None for r in records)
    assert all(r["status"] == "closed" for r in records)


def test_polymarket_open_market_is_unresolved():
    market = {"id": "1", "closed": False, "outcomes": '["Chiefs","Ravens"]',
              "outcomePrices": '["0.62","0.38"]'}
    assert all(r["status"] == "open" for r in resolution._poly_records(market, NOW))


def test_scores_resolution_picks_the_winner_and_skips_draws(monkeypatch):
    from pmcal.sources import oddsapi

    events = [
        {"id": "e1", "completed": True, "commence_time": "2026-09-13T00:20:00Z",
         "scores": [{"name": "Kansas City Chiefs", "score": "27"},
                    {"name": "Baltimore Ravens", "score": "20"}]},
        {"id": "e2", "completed": True, "commence_time": "2026-09-13T00:20:00Z",
         "scores": [{"name": "A", "score": "3"}, {"name": "B", "score": "3"}]},
        {"id": "e3", "completed": False, "scores": []},
    ]
    monkeypatch.setattr(oddsapi, "fetch_scores", lambda *a, **k: events)
    records = resolution.scores_resolutions(None, "americanfootball_nfl", object(), NOW)
    assert {r["market_id"] for r in records} == {"e1:h2h"}
    winner = next(r for r in records if r["outcome_raw"] == "Kansas City Chiefs")
    assert winner["resolved"] == 1.0
    assert json.loads(winner["raw_json"])["id"] == "e1"


