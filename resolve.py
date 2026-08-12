#!/usr/bin/env python
"""Join settled outcomes back onto the snapshots and rebuild the analysis panel.

    uv run resolve.py                 # settle what has finished, rebuild panel
    uv run resolve.py --with-scores   # add The Odds API /scores as a third opinion
    uv run resolve.py --panel-only    # rebuild the panel without re-fetching

Polymarket and Kalshi are the ground truth. Where both cover the same event they
are cross-checked; disagreements land in `resolution_conflicts` and the affected
label is dropped rather than decided by fiat.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from pmcal import db, gamekeys, panel, resolution
from pmcal.config import load_config
from pmcal.http import make_session
from pmcal.util import setup_logging, utcnow

log = logging.getLogger("resolve")

RESOLUTION_COLUMNS = (
    "venue", "market_id", "outcome", "event_key", "resolved", "status",
    "resolved_at", "observed_at", "raw_json",
)
CONFLICT_COLUMNS = (
    "event_key", "outcome", "venue_a", "value_a", "venue_b", "value_b", "detected_at", "note",
)


def pending_markets(con: Any, grace_minutes: int, limit: int) -> pd.DataFrame:
    """Markets whose event has started long enough ago to plausibly be settled.

    Bounded and ordered most-recently-finished first. Kalshi has no bulk lookup,
    so an unbounded query would fire thousands of requests on the first run
    after a long collection; the remainder is simply picked up next run.
    """
    return con.execute(
        """
        SELECT s.venue, s.market_id, any_value(s.event_key) AS event_key,
               any_value(s.sport) AS sport, max(s.event_start_time) AS started
        FROM snapshots s
        LEFT JOIN resolutions r
          ON r.venue = s.venue AND r.market_id = s.market_id AND r.resolved IS NOT NULL
        WHERE s.event_start_time IS NOT NULL
          AND s.event_start_time < now() - INTERVAL (?) MINUTE
          AND r.market_id IS NULL
        GROUP BY s.venue, s.market_id
        ORDER BY started DESC
        LIMIT ?
        """,
        [grace_minutes, limit],
    ).df()


def _key_lookup(con: Any) -> dict[tuple[str, str], tuple[str | None, str | None]]:
    rows = con.execute(
        "SELECT DISTINCT venue, market_id, event_key, sport FROM snapshots"
    ).fetchall()
    return {(v, m): (k, sp) for v, m, k, sp in rows}


def _to_rows(records: list[dict[str, Any]], lookup: dict[Any, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for rec in records:
        event_key, sport = lookup.get((rec["venue"], rec["market_id"]), (None, None))
        # Must land in the same namespace the snapshots use, or the label joins
        # to nothing and the event silently drops out of the panel.
        outcome = gamekeys.outcome_code(sport, str(rec.get("outcome_raw") or ""))
        rows.append(
            [
                rec["venue"], rec["market_id"], outcome, event_key, rec["resolved"],
                rec["status"], rec["resolved_at"], rec["observed_at"], rec["raw_json"],
            ]
        )
    return rows


def fetch_all(
    con: Any, cfg: Any, with_scores: bool, grace_minutes: int, limit: int = 500
) -> int:
    pending = pending_markets(con, grace_minutes, limit)
    log.info("%d market(s) pending settlement", len(pending))
    session = make_session(cfg.user_agent)
    now = utcnow()

    poly_ids = pending.loc[pending["venue"] == "polymarket", "market_id"].astype(str).tolist()
    kalshi_tickers = sorted(
        {
            mid.split(":")[0]
            for mid in pending.loc[pending["venue"] == "kalshi", "market_id"].astype(str)
        }
    )

    records: list[dict[str, Any]] = []
    if poly_ids:
        records += resolution.polymarket_resolutions(session, poly_ids, cfg, now)
    if kalshi_tickers:
        records += resolution.kalshi_resolutions(session, kalshi_tickers, cfg, now)
    if with_scores and cfg.odds_api_key:
        for sport_key in cfg.sports:
            records += resolution.scores_resolutions(session, sport_key, cfg, now)
    elif with_scores:
        log.warning("--with-scores requested but ODDS_API_KEY is unset; skipping")

    lookup = _key_lookup(con)
    rows = _to_rows(records, lookup)
    with db.transaction(con):
        written = db.upsert_rows(
            con, "resolutions", RESOLUTION_COLUMNS, rows, ("venue", "market_id", "outcome")
        )
    log.info("wrote %d resolution record(s)", written)
    return written


def reconcile(con: Any) -> tuple[pd.DataFrame, int]:
    resolutions = con.execute(
        "SELECT venue, market_id, outcome, event_key, resolved FROM resolutions"
    ).df()
    conflicts = panel.find_conflicts(resolutions, pd.Timestamp(utcnow()))
    if not conflicts.empty:
        with db.transaction(con):
            db.upsert_rows(
                con,
                "resolution_conflicts",
                CONFLICT_COLUMNS,
                conflicts[list(CONFLICT_COLUMNS)].values.tolist(),
                ("event_key", "outcome", "venue_a", "venue_b"),
            )
        log.warning("%d cross-venue settlement conflict(s) recorded", len(conflicts))
    return resolutions, len(conflicts)


def rebuild_panel(con: Any, out_parquet: str | None) -> pd.DataFrame:
    snapshots = con.execute(
        """
        SELECT ts, venue, book, market_id, outcome, event_key, sport,
               bid, ask, mid, bid_size, ask_size, event_start_time
        FROM snapshots
        WHERE event_key IS NOT NULL
        """
    ).df()
    resolutions = con.execute(
        "SELECT venue, market_id, outcome, event_key, resolved FROM resolutions"
    ).df()

    labels = panel.ground_truth(resolutions)
    table = panel.build_panel(snapshots, labels)
    con.execute("DROP TABLE IF EXISTS analysis_panel")
    con.register("panel_df", table)
    con.execute("CREATE TABLE analysis_panel AS SELECT * FROM panel_df")
    con.unregister("panel_df")
    if out_parquet:
        table.to_parquet(out_parquet, index=False)
    labelled = int(table["y"].notna().sum())
    log.info(
        "panel: %d rows (%d labelled) across %d event(s)",
        len(table), labelled, table["event_key"].nunique(),
    )
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=None)
    parser.add_argument("--with-scores", action="store_true", help="use The Odds API /scores too")
    parser.add_argument("--panel-only", action="store_true", help="skip network, rebuild panel")
    parser.add_argument(
        "--grace-minutes", type=int, default=240,
        help="how long after kickoff to start checking for settlement",
    )
    parser.add_argument(
        "--max-markets", type=int, default=500,
        help="cap settlement lookups per run; the rest are picked up next run",
    )
    parser.add_argument("--parquet", default=None, help="also write the panel to this parquet path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config()
    con = db.connect(args.db or cfg.db_path)
    try:
        if not args.panel_only:
            fetch_all(con, cfg, args.with_scores, args.grace_minutes, args.max_markets)
        _, n_conflicts = reconcile(con)
        table = rebuild_panel(con, args.parquet)
        if n_conflicts:
            log.warning("see resolution_conflicts; %d label(s) may be dropped", n_conflicts)
        return 0 if len(table) else 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())

