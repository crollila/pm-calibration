"""Collector behaviour, exercised with canned payloads instead of live APIs."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

import collect
from pmcal import db
from pmcal.config import Config
from pmcal.sources import kalshi, oddsapi, polymarket

NOW = datetime(2026, 9, 12, 17, 31, 12)
KICKOFF = "2026-09-13T00:20:00Z"

GAMMA_MARKET = {
    "id": "512",
    "question": "Chiefs vs. Ravens",
    "outcomes": '["Kansas City Chiefs", "Baltimore Ravens"]',
    "outcomePrices": '["0.62", "0.38"]',
    "clobTokenIds": '["tok_a", "tok_b"]',
    "volumeNum": 1_000_000,
    "gameStartTime": KICKOFF,
    "closed": False,
}
CLOB_BOOKS = [
    {"asset_id": "tok_a", "bids": [{"price": "0.60", "size": "300"},
                                   {"price": "0.61", "size": "150"}],
     "asks": [{"price": "0.64", "size": "200"}, {"price": "0.63", "size": "80"}]},
    {"asset_id": "tok_b", "bids": [{"price": "0.36", "size": "90"}],
     "asks": [{"price": "0.39", "size": "110"}]},
]
KALSHI_MARKET = {
    # Real Kalshi game tickers carry an ET timestamp and both team codes.
    "ticker": "KXNFLGAME-26SEP121300KCBAL-KC",
    "title": "Chiefs vs Ravens",
    "yes_sub_title": "Chiefs",
    "no_sub_title": "Ravens",
    "yes_bid": 61, "yes_ask": 63, "no_bid": 37, "no_ask": 39,
    "last_price": 62, "volume": 8000, "close_time": KICKOFF, "status": "open",
}
ODDS_EVENT = {
    "id": "abc123",
    "commence_time": KICKOFF,
    "home_team": "Baltimore Ravens",
    "away_team": "Kansas City Chiefs",
    "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Kansas City Chiefs", "price": -150},
            {"name": "Baltimore Ravens", "price": 130}]}]},
        {"key": "draftkings", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Kansas City Chiefs", "price": -160},
            {"name": "Baltimore Ravens", "price": 135}]}]},
    ],
}

CFG = Config(odds_api_key="test-key", sports=("americanfootball_nfl",))


def test_polymarket_uses_the_order_book_not_the_gamma_midpoint(monkeypatch):
    monkeypatch.setattr(polymarket, "fetch_markets", lambda *a, **k: [GAMMA_MARKET])
    monkeypatch.setattr(
        polymarket, "fetch_books", lambda *a, **k: {b["asset_id"]: b for b in CLOB_BOOKS}
    )
    rows = polymarket.fetch(None, CFG, NOW)

    assert len(rows) == 2
    chiefs = next(r for r in rows if r["outcome"] == "kc")
    assert chiefs["bid"] == pytest.approx(0.61)   # best (highest) bid, not the first level
    assert chiefs["ask"] == pytest.approx(0.63)   # best (lowest) ask
    assert chiefs["mid"] == pytest.approx(0.62)
    assert chiefs["bid_size"] == pytest.approx(150.0)
    assert chiefs["event_key"] == "nfl|2026-09-12|bal~kc"
    assert json.loads(chiefs["raw_json"])["market"]["id"] == "512"


def test_polymarket_falls_back_to_the_gamma_price_without_a_book(monkeypatch):
    monkeypatch.setattr(polymarket, "fetch_markets", lambda *a, **k: [GAMMA_MARKET])
    monkeypatch.setattr(polymarket, "fetch_books", lambda *a, **k: {})
    rows = polymarket.fetch(None, CFG, NOW)
    chiefs = next(r for r in rows if r["outcome"] == "kc")
    assert chiefs["bid"] is None and chiefs["ask"] is None
    assert chiefs["mid"] == pytest.approx(0.62)


def test_polymarket_ignores_market_open_date_as_the_event_time(monkeypatch):
    """`startDate` is when the market listed, not when the event happens.

    Using it would make every long-running market look ready to settle the
    moment it is created, and flood `resolve.py` with pointless lookups.
    """
    market = {**GAMMA_MARKET, "endDate": "2027-01-05T18:00:00Z",
              "startDate": "2024-01-01T00:00:00Z"}
    del market["gameStartTime"]
    monkeypatch.setattr(polymarket, "fetch_markets", lambda *a, **k: [market])
    monkeypatch.setattr(polymarket, "fetch_books", lambda *a, **k: {})
    rows = polymarket.fetch(None, CFG, NOW)
    assert all(r["event_start_time"] == datetime(2027, 1, 5, 18, 0) for r in rows)


def test_kalshi_parlay_subtitles_fall_back_to_plain_sides(monkeypatch):
    """Multi-leg markets concatenate every leg into the subtitle; that is not an outcome."""
    parlay = {
        **KALSHI_MARKET,
        "ticker": "KXMVECROSSCATEGORY-ABC",
        "yes_sub_title": "yes los angeles ayes milwaukeeyes tampa bayyes bobby witt jr 1",
        "no_sub_title": "yes los angeles ayes milwaukeeyes tampa bayyes bobby witt jr 1",
    }
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: [parlay])
    monkeypatch.setattr(kalshi, "fetch_orderbook", lambda *a, **k: None)
    rows = kalshi.fetch(None, CFG, NOW)
    assert {r["outcome"] for r in rows} == {"yes", "no"}


def test_kalshi_converts_cents_and_keeps_both_sides(monkeypatch):
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: [KALSHI_MARKET])
    monkeypatch.setattr(
        kalshi, "fetch_orderbook", lambda *a, **k: {"yes": [[61, 400]], "no": [[37, 250]]}
    )
    rows = kalshi.fetch(None, CFG, NOW)

    assert {r["outcome"] for r in rows} == {"kc", "bal"}
    yes = next(r for r in rows if r["outcome"] == "kc")
    assert (yes["bid"], yes["ask"]) == (pytest.approx(0.61), pytest.approx(0.63))
    assert yes["mid"] == pytest.approx(0.62)
    assert yes["bid_size"] == pytest.approx(400.0)
    assert yes["market_id"].endswith(":yes")
    assert yes["event_key"] == "nfl|2026-09-12|bal~kc"


def test_sportsbook_rows_carry_a_synthetic_two_sided_quote(monkeypatch):
    monkeypatch.setattr(oddsapi, "fetch_odds", lambda *a, **k: [ODDS_EVENT])
    rows = oddsapi.fetch(None, CFG, NOW)

    assert len(rows) == 4
    assert {r["book"] for r in rows} == {"pinnacle", "draftkings"}
    pinn = {r["outcome"]: r for r in rows if r["book"] == "pinnacle"}
    # -150 implies 0.60; +130 implies 0.434783; overround 3.48%.
    assert pinn["kc"]["ask"] == pytest.approx(0.6, abs=1e-6)
    assert pinn["kc"]["bid"] == pytest.approx(1 - 100 / 230, abs=1e-6)
    assert pinn["kc"]["ask"] > pinn["kc"]["bid"]
    # The synthetic midpoint reproduces the additive de-vig exactly.
    assert pinn["kc"]["mid"] + pinn["bal"]["mid"] == pytest.approx(1.0, abs=1e-9)
    assert all(r["event_key"] == "nfl|2026-09-12|bal~kc" for r in rows)


def test_sportsbook_collection_requires_a_key():
    from pmcal.http import FetchError

    with pytest.raises(FetchError):
        oddsapi.fetch(None, Config(odds_api_key=None), NOW)


def _stub_all(monkeypatch):
    monkeypatch.setattr(polymarket, "fetch_markets", lambda *a, **k: [GAMMA_MARKET])
    monkeypatch.setattr(
        polymarket, "fetch_books", lambda *a, **k: {b["asset_id"]: b for b in CLOB_BOOKS}
    )
    monkeypatch.setattr(kalshi, "fetch_markets", lambda *a, **k: [KALSHI_MARKET])
    monkeypatch.setattr(kalshi, "fetch_orderbook", lambda *a, **k: None)
    monkeypatch.setattr(oddsapi, "fetch_odds", lambda *a, **k: [ODDS_EVENT])
    monkeypatch.setattr(collect, "load_config", lambda: CFG)
    # Freeze the clock so the snapshot bucket (and therefore the run id) is
    # deterministic even if the test straddles a 15-minute boundary.
    monkeypatch.setattr(collect, "utcnow", lambda: NOW)


def test_dry_run_writes_nothing(monkeypatch, tmp_path):
    _stub_all(monkeypatch)
    path = tmp_path / "dry.duckdb"
    assert collect.main(["--dry-run", "--db", str(path)]) == 0
    assert not path.exists()


def test_end_to_end_run_is_idempotent(monkeypatch, tmp_path):
    _stub_all(monkeypatch)
    path = str(tmp_path / "c.duckdb")

    assert collect.main(["--db", path]) == 0
    con = db.connect(path)
    first = con.execute("SELECT count(*) FROM snapshots").fetchone()[0]
    assert first == 8  # 2 polymarket + 2 kalshi + 4 sportsbook
    con.close()

    assert collect.main(["--db", path]) == 0
    con = db.connect(path)
    try:
        assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == first
        runs = con.execute(
            "SELECT source, status, rows_inserted FROM collection_runs ORDER BY source"
        ).fetchall()
        # One bookkeeping row per source per bucket, reused by the rerun.
        assert [r[0] for r in runs] == ["kalshi", "oddsapi", "polymarket"]
        assert [r[1] for r in runs] == ["ok", "ok", "ok"]
        assert sum(r[2] for r in runs) == 0  # the rerun inserted nothing new
    finally:
        con.close()


def test_one_failing_source_does_not_lose_the_others(monkeypatch, tmp_path):
    _stub_all(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("kalshi is down")

    monkeypatch.setattr(kalshi, "fetch_markets", boom)
    path = str(tmp_path / "partial.duckdb")
    assert collect.main(["--db", path]) == 0

    con = db.connect(path)
    try:
        assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 6
        error = con.execute(
            "SELECT status, error FROM collection_runs WHERE source = 'kalshi'"
        ).fetchone()
        assert error[0] == "error" and "kalshi is down" in error[1]
    finally:
        con.close()


def test_snapshots_are_bucketed_so_venues_line_up(monkeypatch, tmp_path):
    _stub_all(monkeypatch)
    path = str(tmp_path / "bucket.duckdb")
    collect.main(["--db", path, "--bucket-minutes", "15"])
    con = db.connect(path)
    try:
        stamps = con.execute("SELECT DISTINCT ts FROM snapshots").fetchall()
        assert len(stamps) == 1
        assert stamps[0][0].minute % 15 == 0
        assert stamps[0][0].second == 0
    finally:
        con.close()


def test_unknown_source_is_rejected(monkeypatch, tmp_path):
    _stub_all(monkeypatch)
    assert collect.main(["--db", str(tmp_path / "x.duckdb"), "--sources", "betfair"]) == 2

