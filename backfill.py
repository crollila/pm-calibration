#!/usr/bin/env python
"""Populate the database from historical endpoints instead of waiting for the collector.

    uv run backfill.py --sport mlb --since 30d --dry-run
    uv run backfill.py --sport mlb --since 30d
    uv run backfill.py --sport mlb --since 12mo --venues polymarket

Same DuckDB file, same `snapshots` schema and the same append-only writers as
`collect.py`; backfilled rows are tagged `source = 'backfill'` so analysis can
separate them from live ones. Reruns are no-ops: rows dedupe on the existing
primary key, and every market already processed is skipped via
`backfill_progress`, so an interrupted job resumes where it stopped.

Two honest limitations, both visible in the data rather than papered over:

* Polymarket's historical endpoints expose **no bid/ask**, only a price. Kalshi
  candlesticks do carry a spread but no resting depth. So backfilled rows cannot
  support the execution model in `analysis/backtest.py`, which needs both.
* Sportsbook history is a paid endpoint and is deliberately **not** collected
  (see `backfill_sportsbooks`), so the book consensus is absent from any
  backfilled sample and the venue-vs-book comparison cannot be run on it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from pmcal import db, gamekeys
from pmcal.backfill_io import (
    Limiter,
    done_markets,
    grid_for,
    parse_since,
    record_progress,
    write_rows,
)
from pmcal.config import load_config
from pmcal.http import FetchError, make_session
from pmcal.sources import kalshi_history as kh
from pmcal.sources import polymarket_history as ph
from pmcal.util import setup_logging, utcnow

log = logging.getLogger("backfill")


def backfill_polymarket(con: Any, cfg: Any, opts: argparse.Namespace, run_id: str,
                        limiter: Limiter) -> dict[str, int]:
    session = make_session(cfg.user_agent)
    limiter.wait()
    markets = ph.discover_markets(session, opts.sport, opts.since, opts.until)
    log.info("polymarket: %d settled %s moneyline market(s) in window",
             len(markets), opts.sport)

    stats = {"markets": len(markets), "rows": 0, "skipped": 0, "empty": 0,
             "prices_history": 0, "trades": 0}
    if opts.dry_run:
        stats["rows"] = len(markets) * 2 * len(
            grid_for(utcnow(), opts.lookback_hours, opts.bucket_minutes)
        )
        return stats

    already = done_markets(con, ph.VENUE)
    for index, market in enumerate(markets, 1):
        market_id = str(market.get("id"))
        if market_id in already:
            stats["skipped"] += 1
            continue
        slug = str(market.get("slug") or "")
        event_key = gamekeys.polymarket_event_key(slug)
        start = _start_of(market)
        if not event_key or start is None:
            record_progress(con, ph.VENUE, market_id, "empty", "unmatched", 0, slug)
            stats["empty"] += 1
            continue

        grid = grid_for(start, opts.lookback_hours, opts.bucket_minutes)
        start_ts = int((start - timedelta(hours=opts.lookback_hours)).replace(tzinfo=UTC).timestamp())
        end_ts = int(start.replace(tzinfo=UTC).timestamp())
        tokens = [str(t) for t in ph.as_list(market.get("clobTokenIds"))]
        condition = str(market.get("conditionId") or "")

        series_by_outcome: dict[int, dict[datetime, float]] = {}
        paths: list[str] = []
        try:
            for position, token in enumerate(tokens):
                limiter.wait()
                series, path = ph.series_for_token(
                    session, token, condition, start_ts, end_ts, opts.fidelity, grid
                )
                paths.append(path)
                if series:
                    series_by_outcome[position] = series
        except FetchError as exc:
            log.warning("polymarket %s failed: %s", slug, exc)
            record_progress(con, ph.VENUE, market_id, "error", None, 0, str(exc)[:400])
            continue

        rows = ph.market_rows(market, series_by_outcome, event_key,
                              opts.sport, utcnow()) if series_by_outcome else []
        written = write_rows(con, rows, run_id)
        stats["rows"] += written
        path = paths[0] if paths else "empty"
        stats["prices_history"] += paths.count("prices-history")
        stats["trades"] += paths.count("trades")
        if not rows:
            stats["empty"] += 1
        record_progress(con, ph.VENUE, market_id, "ok" if rows else "empty", path, written, slug)
        if index % 25 == 0:
            log.info("polymarket %d/%d markets, %d rows", index, len(markets), stats["rows"])
    return stats


def backfill_kalshi(con: Any, cfg: Any, opts: argparse.Namespace, run_id: str,
                    limiter: Limiter) -> dict[str, int]:
    series_ticker = gamekeys.SERIES_FOR_SPORT.get(opts.sport)
    if not series_ticker:
        log.warning("kalshi: no game series known for sport %r; skipping", opts.sport)
        return {"markets": 0, "rows": 0, "skipped": 0, "empty": 0}

    session = make_session(cfg.user_agent)
    limiter.wait()
    # Kalshi filters on close time, which is the end of the game.
    markets = kh.fetch_settled_markets(
        session, series_ticker,
        int(opts.since.replace(tzinfo=UTC).timestamp()),
        int((opts.until + timedelta(days=1)).replace(tzinfo=UTC).timestamp()),
    )
    log.info("kalshi: %d settled %s market(s) in window", len(markets), series_ticker)

    # The two tickers of one game name both teams unambiguously.
    by_game: dict[str, set[str]] = {}
    parsed_by_ticker: dict[str, dict[str, Any]] = {}
    for market in markets:
        parsed = gamekeys.parse_kalshi_ticker(str(market.get("ticker") or ""))
        if not parsed:
            continue
        parsed_by_ticker[str(market["ticker"])] = parsed
        by_game.setdefault(str(parsed["game_id"]), set()).add(str(parsed["side"]))

    stats = {"markets": len(parsed_by_ticker), "rows": 0, "skipped": 0, "empty": 0,
             "games": len(by_game)}
    if opts.dry_run:
        stats["rows"] = len(parsed_by_ticker) * 2 * (opts.lookback_hours * 60 // opts.interval)
        return stats

    already = done_markets(con, kh.VENUE)
    for index, market in enumerate(markets, 1):
        ticker = str(market.get("ticker") or "")
        parsed = parsed_by_ticker.get(ticker)
        if not parsed:
            continue
        if ticker in already:
            stats["skipped"] += 1
            continue
        start: datetime = parsed["start"]  # type: ignore[assignment]
        if not opts.since <= start <= opts.until:
            record_progress(con, kh.VENUE, ticker, "empty", "out-of-window", 0, None)
            continue
        codes = gamekeys.kalshi_game_codes(
            by_game.get(str(parsed["game_id"]), set()), str(parsed["blob"]), opts.sport
        )
        event_key = gamekeys.event_key_from_codes(opts.sport, start, codes)
        if not event_key:
            record_progress(con, kh.VENUE, ticker, "empty", "unmatched", 0, ticker)
            stats["empty"] += 1
            continue

        try:
            limiter.wait()
            candles = kh.fetch_candlesticks(
                session, series_ticker, ticker,
                int((start - timedelta(hours=opts.lookback_hours)).replace(tzinfo=UTC).timestamp()),
                int(start.replace(tzinfo=UTC).timestamp()),
                opts.interval,
            )
        except FetchError as exc:
            log.warning("kalshi %s failed: %s", ticker, exc)
            record_progress(con, kh.VENUE, ticker, "error", None, 0, str(exc)[:400])
            continue

        rows = kh.candle_rows(market, candles, event_key, opts.sport, codes, start)
        written = write_rows(con, rows, run_id)
        stats["rows"] += written
        if not rows:
            stats["empty"] += 1
        record_progress(con, kh.VENUE, ticker, "ok" if rows else "empty",
                        "candlesticks", written, None)
        if index % 50 == 0:
            log.info("kalshi %d/%d markets, %d rows", index, len(markets), stats["rows"])
    return stats


def backfill_sportsbooks(con: Any, cfg: Any, opts: argparse.Namespace, run_id: str,
                         limiter: Limiter) -> dict[str, int]:
    """Not implemented on purpose.

    TODO: The Odds API historical endpoint (`/v4/historical/sports/{sport}/odds`)
    is a paid tier and bills one credit per market per timestamp, so a month of
    15-minute snapshots across a dozen books is expensive enough to be a
    deliberate decision rather than a default. When enabled it should reuse
    `pmcal.sources.oddsapi._event_rows` verbatim -- the response shape matches
    the live endpoint, with `snapshot` timestamps added -- and write rows with
    `source = 'backfill'`.

    Until then the book consensus is absent from backfilled samples, and the
    headline "prediction markets vs sportsbooks" comparison cannot be run on
    them. That is stated in the README rather than hidden behind an empty table.
    """
    log.info("sportsbooks: skipped (paid historical endpoint; see backfill_sportsbooks)")
    return {"markets": 0, "rows": 0, "skipped": 0, "empty": 0}


VENUES = {
    "polymarket": backfill_polymarket,
    "kalshi": backfill_kalshi,
    "sportsbook": backfill_sportsbooks,
}


def _start_of(market: dict[str, Any]) -> datetime | None:
    from pmcal.util import parse_ts

    return parse_ts(market.get("gameStartTime") or market.get("endDate"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=None)
    parser.add_argument("--sport", default="mlb", choices=sorted(gamekeys.ALIASES))
    parser.add_argument("--since", default="30d", help="30d | 6w | 12mo | 2026-01-15")
    parser.add_argument("--until", default=None, help="ISO date; defaults to now")
    parser.add_argument("--venues", default="polymarket,kalshi")
    parser.add_argument("--bucket-minutes", type=int, default=60,
                        help="snapshot grid; 60 matches Kalshi's coarsest usable candle")
    parser.add_argument("--interval", type=int, default=60, choices=kh.VALID_INTERVALS,
                        help="Kalshi candle interval in minutes")
    parser.add_argument("--fidelity", type=int, default=60,
                        help="Polymarket prices-history fidelity in minutes")
    parser.add_argument("--lookback-hours", type=int, default=48,
                        help="how far before kickoff to reconstruct")
    parser.add_argument("--rps", type=float, default=4.0, help="max requests per second")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be fetched, write nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config()
    now = utcnow()
    args.since = parse_since(args.since, now)
    args.until = datetime.fromisoformat(args.until) if args.until else now
    if args.since >= args.until:
        log.error("--since must precede --until")
        return 2

    venues = [v.strip() for v in args.venues.split(",") if v.strip() in VENUES]
    if not venues:
        log.error("no valid venues selected")
        return 2

    limiter = Limiter(min_interval=1.0 / max(args.rps, 0.1))
    run_id = f"backfill-{now:%Y%m%dT%H%M}"
    con = None if args.dry_run else db.connect(args.db or cfg.db_path)
    log.info("backfill %s from %s to %s (venues: %s)%s",
             args.sport, args.since.date(), args.until.date(), ",".join(venues),
             " [dry-run]" if args.dry_run else "")
    try:
        totals: dict[str, dict[str, int]] = {}
        for venue in venues:
            try:
                totals[venue] = VENUES[venue](con, cfg, args, run_id, limiter)
            except Exception as exc:  # noqa: BLE001 - one venue must not sink the job
                log.error("venue %s failed: %s", venue, exc)
                totals[venue] = {"markets": 0, "rows": 0, "error": 1}
            log.info("%s: %s", venue, totals[venue])
        grand = sum(t.get("rows", 0) for t in totals.values())
        log.info("%s %d row(s) across %d venue(s)",
                 "would write" if args.dry_run else "wrote", grand, len(totals))
        return 0 if grand or args.dry_run else 1
    finally:
        if con is not None:
            con.close()


if __name__ == "__main__":
    sys.exit(main())
