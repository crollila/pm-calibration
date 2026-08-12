#!/usr/bin/env python
"""Snapshot live probabilities from Polymarket, Kalshi and the sportsbooks.

Design notes that matter more than the code:

* **Append only.** Rows are keyed on (ts, venue, book, market_id, outcome) and
  inserted with ON CONFLICT DO NOTHING. Re-running a bucket is a no-op.
* **Crash safe.** Each source writes inside its own transaction together with
  its `collection_runs` bookkeeping, so a kill -9 mid-run leaves the database
  consistent and the run visibly marked `running` (never silently missing).
* **One source failing never loses the others.** Failures are recorded with the
  exception text so uptime is provable from the database alone.
* **Raw payloads are stored verbatim** so any field can be re-derived later
  without re-collecting -- the data is the asset, the parsing is disposable.

    uv run collect.py --dry-run
    uv run collect.py --sources polymarket,kalshi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import Any

from pmcal import db
from pmcal.config import Config, load_config
from pmcal.http import make_session
from pmcal.sources import kalshi, oddsapi, polymarket
from pmcal.util import floor_to, run_id_for, setup_logging, utcnow

log = logging.getLogger("collect")

FETCHERS = {
    "polymarket": polymarket.fetch,
    "kalshi": kalshi.fetch,
    "oddsapi": oddsapi.fetch,
}


def to_row(record: dict[str, Any], ts: datetime, run_id: str) -> list[Any]:
    """Project a source dict onto the snapshot column order."""
    record = {**record, "ts": ts, "run_id": run_id, "source": "live"}
    record.setdefault("collected_at", ts)
    return [record.get(col) for col in db.SNAPSHOT_COLUMNS]


def collect_source(
    con: Any,
    name: str,
    cfg: Config,
    session: Any,
    ts: datetime,
    run_id: str,
    dry_run: bool,
) -> tuple[int, int, str | None]:
    """Fetch one source and persist it. Returns (seen, inserted, error)."""
    started = utcnow()
    if not dry_run:
        with db.transaction(con):
            db.start_run(con, run_id, name, started, dry_run)

    try:
        records = FETCHERS[name](session, cfg, started)
    except Exception as exc:  # noqa: BLE001 - recorded, never fatal to other sources
        log.error("source %s failed: %s", name, exc)
        if not dry_run:
            with db.transaction(con):
                db.finish_run(con, run_id, name, utcnow(), "error", 0, 0, str(exc)[:2000])
        return 0, 0, str(exc)

    rows = [to_row(r, ts, run_id) for r in records]
    if dry_run:
        log.info("[dry-run] %s: %d rows (nothing written)", name, len(rows))
        _preview(name, records)
        return len(rows), 0, None

    with db.transaction(con):
        inserted = db.insert_rows(
            con, "snapshots", db.SNAPSHOT_COLUMNS, rows, db.SNAPSHOT_KEY
        )
        db.finish_run(con, run_id, name, utcnow(), "ok", len(rows), inserted, None)
    log.info("%s: %d rows seen, %d new", name, len(rows), inserted)
    return len(rows), inserted, None


def _preview(name: str, records: list[dict[str, Any]], limit: int = 3) -> None:
    for record in records[:limit]:
        slim = {
            k: v
            for k, v in record.items()
            if k in ("market_id", "outcome", "event_key", "bid", "ask", "mid", "event_start_time")
        }
        log.info("[dry-run] %s sample: %s", name, json.dumps(slim, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=None, help="DuckDB path (overrides PMCAL_DB)")
    parser.add_argument(
        "--sources",
        default="polymarket,kalshi,oddsapi",
        help="comma-separated subset of polymarket,kalshi,oddsapi",
    )
    parser.add_argument(
        "--bucket-minutes",
        type=int,
        default=15,
        help="snapshot bucket; must match the cron/timer interval",
    )
    parser.add_argument("--dry-run", action="store_true", help="fetch and log, write nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config()
    db_path = args.db or cfg.db_path

    now = utcnow()
    ts = floor_to(now, args.bucket_minutes)
    run_id = run_id_for(ts)
    sources = [s.strip() for s in args.sources.split(",") if s.strip() in FETCHERS]
    if not sources:
        log.error("no valid sources selected")
        return 2

    session = make_session(cfg.user_agent)
    con = None if args.dry_run else db.connect(db_path)
    try:
        total_seen = total_new = 0
        errors: list[str] = []
        for name in sources:
            seen, new, err = collect_source(con, name, cfg, session, ts, run_id, args.dry_run)
            total_seen += seen
            total_new += new
            if err:
                errors.append(f"{name}: {err}")
        log.info(
            "run %s complete: %d rows seen, %d inserted, %d source error(s)",
            run_id, total_seen, total_new, len(errors),
        )
        # Partial failure is expected (rate limits, key exhaustion) and must not
        # stop the timer; only a total wipeout is worth a non-zero exit.
        return 1 if len(errors) == len(sources) else 0
    finally:
        if con is not None:
            con.close()


if __name__ == "__main__":
    sys.exit(main())
