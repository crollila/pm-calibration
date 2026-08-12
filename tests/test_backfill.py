"""Backfill behaviour with canned historical payloads."""

from __future__ import annotations

from datetime import datetime

import pytest

import backfill
from pmcal import db
from pmcal.config import Config
from pmcal.sources import kalshi_history as kh
from pmcal.sources import polymarket_history as ph

NOW = datetime(2026, 8, 11, 12, 0)
KICKOFF = datetime(2026, 8, 10, 23, 45)
CFG = Config()

POLY_MARKET = {
    "id": "3155130",
    "conditionId": "0xabc",
    "question": "Philadelphia Phillies vs. St. Louis Cardinals",
    "slug": "mlb-phi-stl-2026-08-10",
    "sportsMarketType": "moneyline",
    "outcomes": '["Philadelphia Phillies", "St. Louis Cardinals"]',
    "clobTokenIds": '["tok_phi", "tok_stl"]',
    "gameStartTime": "2026-08-10 23:45:00+00",
    "endDate": "2026-08-17T23:45:00Z",
    "volumeNum": 1_000_000,
    "closed": True,
}

KALSHI_MARKET = {
    "ticker": "KXMLBGAME-26AUG101945PHISTL-STL",
    "title": "Philadelphia vs St. Louis Winner?",
    "yes_sub_title": "St. Louis",
    "no_sub_title": "St. Louis",
    "status": "settled",
    "result": "no",
    "volume": 5000,
}


def _candle(ts: int, bid: str, ask: str) -> dict:
    return {
        "end_period_ts": ts,
        "yes_bid": {"close_dollars": bid},
        "yes_ask": {"close_dollars": ask},
        "price": {"close_dollars": bid},
        "volume_fp": "10.0",
    }


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

def test_resample_takes_the_last_observation_at_or_before_each_grid_point():
    points = [(0, 0.40), (1800, 0.45), (3600, 0.50)]
    grid = [datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0)]
    assert ph.resample_asof(points, grid) == {grid[0]: 0.40, grid[1]: 0.50}


def test_resample_never_reaches_into_the_future():
    """A price set 1 second after T must not appear on the row stamped T."""
    grid = [datetime(1970, 1, 1, 0, 0), datetime(1970, 1, 1, 1, 0)]
    points = [(1, 0.9), (3601, 0.1)]
    out = ph.resample_asof(points, grid)
    assert grid[0] not in out            # nothing known yet at T=0
    assert out[grid[1]] == 0.9           # the 3601s point is still in the future


def test_resample_of_nothing_is_nothing():
    assert ph.resample_asof([], [datetime(1970, 1, 1)]) == {}


# --------------------------------------------------------------------------
# polymarket path selection
# --------------------------------------------------------------------------

def test_prices_history_is_the_primary_path(monkeypatch):
    monkeypatch.setattr(ph, "fetch_price_history", lambda *a, **k: [(0, 0.5)])
    monkeypatch.setattr(ph, "fetch_trade_series", lambda *a, **k: pytest.fail("should not fall back"))
    series, path = ph.series_for_token(None, "tok", "0xabc", 0, 10, 60, [datetime(1970, 1, 1)])
    assert path == "prices-history" and series


def test_empty_prices_history_falls_back_to_trades(monkeypatch):
    """The documented gotcha, guarded even though it did not reproduce in practice."""
    monkeypatch.setattr(ph, "fetch_price_history", lambda *a, **k: [])
    monkeypatch.setattr(ph, "fetch_trade_series", lambda *a, **k: [(0, 0.42)])
    series, path = ph.series_for_token(None, "tok", "0xabc", 0, 10, 1, [datetime(1970, 1, 1)])
    assert path == "trades"
    assert series == {datetime(1970, 1, 1): 0.42}


def test_both_paths_empty_is_reported_not_invented(monkeypatch):
    monkeypatch.setattr(ph, "fetch_price_history", lambda *a, **k: [])
    monkeypatch.setattr(ph, "fetch_trade_series", lambda *a, **k: [])
    series, path = ph.series_for_token(None, "tok", "0xabc", 0, 10, 1, [datetime(1970, 1, 1)])
    assert series == {} and path == "empty"


def test_trade_series_filters_to_the_requested_token(monkeypatch):
    trades = [
        {"asset": "tok_a", "price": "0.6", "timestamp": 100},
        {"asset": "tok_b", "price": "0.4", "timestamp": 100},
        {"asset": "tok_a", "price": "0.7", "timestamp": 5000},   # outside the window
    ]
    monkeypatch.setattr(ph, "get_json", lambda *a, **k: trades)
    out = ph.fetch_trade_series(None, "0xabc", "tok_a", 0, 1000, max_pages=1)
    assert out == [(100, 0.6)]


def test_polymarket_rows_use_the_slug_for_outcome_labels():
    series = {NOW: 0.55}
    rows = ph.market_rows(POLY_MARKET, {0: series, 1: {NOW: 0.45}},
                          "mlb|2026-08-10|phi~stl", "mlb", NOW)
    assert {r["outcome"] for r in rows} == {"phi", "stl"}
    assert all(r["bid"] is None and r["ask"] is None for r in rows)  # no book history
    assert all(r["mid"] is not None for r in rows)
    assert all(r["event_start_time"] == KICKOFF for r in rows)


# --------------------------------------------------------------------------
# kalshi candles
# --------------------------------------------------------------------------

def test_candles_produce_both_sides_with_a_real_spread():
    rows = kh.candle_rows(KALSHI_MARKET, [_candle(1786287600, "0.48", "0.52")],
                          "mlb|2026-08-10|phi~stl", "mlb", ("phi", "stl"), KICKOFF, NOW)
    by_outcome = {r["outcome"]: r for r in rows}
    assert set(by_outcome) == {"stl", "phi"}
    assert by_outcome["stl"]["bid"] == pytest.approx(0.48)
    assert by_outcome["stl"]["ask"] == pytest.approx(0.52)
    # The NO side is the complement, and bid/ask swap: buying NO lifts 1 - bid.
    assert by_outcome["phi"]["bid"] == pytest.approx(0.48)
    assert by_outcome["phi"]["ask"] == pytest.approx(0.52)
    assert by_outcome["stl"]["mid"] + by_outcome["phi"]["mid"] == pytest.approx(1.0)


def test_candle_timestamp_is_the_period_end():
    """A bar covering (E-1h, E] describes the state at E, so it is stamped E."""
    rows = kh.candle_rows(KALSHI_MARKET, [_candle(1786287600, "0.48", "0.52")],
                          "k", "mlb", ("phi", "stl"), KICKOFF, NOW)
    assert rows[0]["ts"] == datetime(2026, 8, 9, 15, 0)


def test_illiquid_candles_fall_back_to_previous_price():
    thin = {"end_period_ts": 1786287600, "price": {"previous_dollars": "0.49"}}
    rows = kh.candle_rows(KALSHI_MARKET, [thin], "k", "mlb", ("phi", "stl"), KICKOFF, NOW)
    assert rows and rows[0]["mid"] == pytest.approx(0.49)


def test_valueless_candles_are_dropped_not_zero_filled():
    empty = {"end_period_ts": 1786287600, "price": {}}
    assert kh.candle_rows(KALSHI_MARKET, [empty], "k", "mlb", ("phi", "stl"), KICKOFF, NOW) == []


def test_invalid_candle_interval_is_rejected():
    with pytest.raises(ValueError, match="period_interval"):
        kh.fetch_candlesticks(None, "KXMLBGAME", "T", 0, 100, interval=15)


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [("30d", datetime(2026, 7, 12)), ("2w", datetime(2026, 7, 28)),
     ("12mo", datetime(2025, 8, 16)), ("1y", datetime(2025, 8, 11)),
     ("2026-01-15", datetime(2026, 1, 15))],
)
def test_since_parsing(text, expected):
    assert backfill.parse_since(text, datetime(2026, 8, 11)) == expected


def test_bad_since_is_rejected():
    with pytest.raises(Exception, match="since"):
        backfill.parse_since("last tuesday", datetime(2026, 8, 11))


def test_grid_is_on_the_collector_bucket_and_stops_at_kickoff():
    grid = backfill.grid_for(datetime(2026, 8, 10, 23, 45), lookback_hours=3, bucket_minutes=60)
    assert grid[0] == datetime(2026, 8, 10, 20, 0)
    assert grid[-1] == datetime(2026, 8, 10, 23, 0)   # floored, so never past kickoff
    assert all(g.minute == 0 for g in grid)


def _stub(monkeypatch):
    monkeypatch.setattr(ph, "discover_markets", lambda *a, **k: [POLY_MARKET])
    monkeypatch.setattr(
        ph, "series_for_token",
        lambda s, tok, cond, a, b, f, grid, **k: ({grid[-1]: 0.55}, "prices-history"),
    )
    monkeypatch.setattr(kh, "fetch_settled_markets", lambda *a, **k: [KALSHI_MARKET])
    monkeypatch.setattr(
        kh, "fetch_candlesticks", lambda *a, **k: [_candle(1786280400, "0.48", "0.52")]
    )
    monkeypatch.setattr(backfill, "load_config", lambda: CFG)
    monkeypatch.setattr(backfill, "utcnow", lambda: NOW)


def test_dry_run_writes_nothing(monkeypatch, tmp_path):
    _stub(monkeypatch)
    path = tmp_path / "dry.duckdb"
    assert backfill.main(["--sport", "mlb", "--since", "30d", "--dry-run",
                          "--db", str(path)]) == 0
    assert not path.exists()


def test_backfilled_rows_are_tagged_and_idempotent(monkeypatch, tmp_path):
    _stub(monkeypatch)
    path = str(tmp_path / "b.duckdb")
    args = ["--sport", "mlb", "--since", "30d", "--db", path, "--rps", "1000"]

    assert backfill.main(args) == 0
    con = db.connect(path)
    first = con.execute("SELECT count(*) FROM snapshots").fetchone()[0]
    sources = con.execute("SELECT DISTINCT source FROM snapshots").fetchall()
    con.close()
    assert first > 0
    assert sources == [("backfill",)]

    # Rerun: progress table short-circuits, and the primary key would catch
    # anything that slipped through anyway.
    assert backfill.main(args) == 1  # nothing new written
    con = db.connect(path)
    try:
        assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == first
        done = con.execute("SELECT venue, status FROM backfill_progress ORDER BY venue").fetchall()
        assert done == [("kalshi", "ok"), ("polymarket", "ok")]
    finally:
        con.close()


def test_backfill_and_live_rows_coexist_in_one_table(monkeypatch, tmp_path):
    _stub(monkeypatch)
    path = str(tmp_path / "mixed.duckdb")
    backfill.main(["--sport", "mlb", "--since", "30d", "--db", path, "--rps", "1000"])

    con = db.connect(path)
    try:
        live = dict(zip(db.SNAPSHOT_COLUMNS,
                        [NOW, "polymarket", "", "live-1", "stl", "mlb|2026-08-10|phi~stl",
                         "mlb", "t", 0.5, 0.52, 0.51, 1.0, 1.0, 1.0, KICKOFF, NOW,
                         "r", "live", "{}"], strict=True))
        db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS,
                       [[live[c] for c in db.SNAPSHOT_COLUMNS]], db.SNAPSHOT_KEY)
        counts = dict(con.execute(
            "SELECT source, count(*) FROM snapshots GROUP BY source"
        ).fetchall())
        assert counts["live"] == 1 and counts["backfill"] > 0
        # And they share a key namespace, which is the whole point.
        keys = con.execute("SELECT DISTINCT event_key FROM snapshots").fetchall()
        assert keys == [("mlb|2026-08-10|phi~stl",)]
    finally:
        con.close()


def test_sportsbook_backfill_is_an_explicit_no_op(monkeypatch, tmp_path):
    _stub(monkeypatch)
    path = str(tmp_path / "sb.duckdb")
    con = db.connect(path)
    try:
        stats = backfill.backfill_sportsbooks(con, CFG, None, "run", backfill.Limiter(0))
        assert stats["rows"] == 0
        assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 0
    finally:
        con.close()


def test_limiter_spaces_requests(monkeypatch):
    sleeps: list[float] = []
    from pmcal import backfill_io

    monkeypatch.setattr(backfill_io.time, "sleep", sleeps.append)
    clock = iter([100.0, 100.0, 100.1, 100.1])
    monkeypatch.setattr(backfill_io.time, "monotonic", lambda: next(clock))
    limiter = backfill.Limiter(min_interval=0.25)
    limiter.wait()
    limiter.wait()
    assert sleeps == [pytest.approx(0.15, abs=1e-6)]


def test_resume_skips_markets_already_recorded(monkeypatch, tmp_path):
    _stub(monkeypatch)
    con = db.connect(tmp_path / "resume.duckdb")
    try:
        backfill.record_progress(con, ph.VENUE, "3155130", "ok", "prices-history", 5)
        assert backfill.done_markets(con, ph.VENUE) == {"3155130"}
        backfill.record_progress(con, ph.VENUE, "999", "error", None, 0, "boom")
        # Errors are retried on the next run; only ok/empty are considered done.
        assert backfill.done_markets(con, ph.VENUE) == {"3155130"}
    finally:
        con.close()


def test_missing_depth_blocks_the_backtest_rather_than_reporting_no_edge():
    """Backfilled rows have no displayed depth, so the cost model cannot be applied.

    'No data' and 'no edge' are different claims; the pipeline must not let the
    first masquerade as the second.
    """
    import pandas as pd

    from analysis.run_all import tradeability_blockers

    panel = pd.DataFrame({
        "pm_ask": [0.5], "pm_ask_size": [float("nan")], "book_shin": [float("nan")],
        "kalshi_ask": [0.5], "kalshi_ask_size": [100.0],
    })
    blockers = tradeability_blockers(panel, "polymarket", "book_shin")
    assert any("depth" in b for b in blockers)
    assert any("reference" in b for b in blockers)
    # A venue with depth still trips only on the missing reference.
    assert tradeability_blockers(panel, "kalshi", "book_shin") == [
        "no reference venue (book_shin is empty)"
    ]
    # Everything present -> nothing blocks.
    panel["book_shin"] = [0.55]
    assert tradeability_blockers(panel, "kalshi", "book_shin") == []


def test_window_is_validated():
    assert backfill.main(["--since", "2026-08-01", "--until", "2026-07-01"]) == 2


def test_discover_widens_the_query_window_for_resolution_lag(monkeypatch):
    """A market's endDate trails the game by days; the query must reach past it."""
    seen: list[str] = []

    def fake_get(session, url, params=None, timeout=None):
        seen.append(params["end_date_min"])
        return []

    monkeypatch.setattr(ph, "get_json", fake_get)
    ph.discover_markets(None, "mlb", datetime(2026, 8, 1), datetime(2026, 8, 2))
    assert seen[0].startswith("2026-08-01")
    assert seen[-1].startswith("2026-08-12")  # 2 days + RESOLUTION_LAG_DAYS
