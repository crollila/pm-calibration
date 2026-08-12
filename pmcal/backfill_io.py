"""Shared plumbing for `backfill.py`: pacing, windowing, the resume ledger.

Kept separate from the CLI so the venue walkers stay readable and so the resume
behaviour can be tested without touching argument parsing.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pmcal import db
from pmcal.util import floor_to, utcnow

PROGRESS_COLUMNS = ("venue", "market_id", "status", "path", "rows_written", "updated_at", "note")
DURATION = re.compile(r"^(\d+)\s*(d|w|mo|y)$")


@dataclass
class Limiter:
    """Minimum spacing between requests.

    Deliberately dumb: one long-running job against two public APIs does not
    need a token bucket, it needs to not hammer them. Kalshi starts returning
    HTTP 429 above roughly 4 requests/second.
    """

    min_interval: float = 0.25
    _last: float = field(default=0.0, repr=False)

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()


def parse_since(text: str, now: datetime) -> datetime:
    """Accept `30d`, `6w`, `12mo`, `1y` or an ISO date.

    >>> from datetime import datetime
    >>> parse_since("30d", datetime(2026, 8, 11))
    datetime.datetime(2026, 7, 12, 0, 0)
    >>> parse_since("2026-01-15", datetime(2026, 8, 11))
    datetime.datetime(2026, 1, 15, 0, 0)
    """
    match = DURATION.match(text.strip().lower())
    if match:
        count, unit = int(match.group(1)), match.group(2)
        days = {"d": 1, "w": 7, "mo": 30, "y": 365}[unit] * count
        return now - timedelta(days=days)
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"cannot parse --since {text!r}") from exc


def grid_for(start: datetime, lookback_hours: int, bucket_minutes: int) -> list[datetime]:
    """Snapshot times before kickoff, on the same bucket grid the collector uses.

    >>> from datetime import datetime
    >>> grid_for(datetime(2026, 8, 10, 23, 45), 2, 60)
    [datetime.datetime(2026, 8, 10, 21, 0), datetime.datetime(2026, 8, 10, 22, 0), datetime.datetime(2026, 8, 10, 23, 0)]
    """
    first = floor_to(start - timedelta(hours=lookback_hours), bucket_minutes)
    last = floor_to(start, bucket_minutes)
    grid, moment = [], first
    while moment <= last:
        grid.append(moment)
        moment += timedelta(minutes=bucket_minutes)
    return grid


def done_markets(con: Any, venue: str) -> set[str]:
    """Markets already handled. Errors are excluded so the next run retries them."""
    rows = con.execute(
        "SELECT market_id FROM backfill_progress WHERE venue = ? AND status IN ('ok','empty')",
        [venue],
    ).fetchall()
    return {r[0] for r in rows}


def record_progress(con: Any, venue: str, market_id: str, status: str, path: str | None,
                    rows: int, note: str | None = None) -> None:
    with db.transaction(con):
        db.upsert_rows(
            con, "backfill_progress", PROGRESS_COLUMNS,
            [[venue, market_id, status, path, rows, utcnow(), note]],
            ("venue", "market_id"),
        )


def write_rows(con: Any, records: list[dict[str, Any]], run_id: str) -> int:
    """Append through the shared collector writer, tagged as backfilled."""
    if not records:
        return 0
    rows = []
    for record in records:
        payload = {**record, "run_id": run_id, "source": "backfill"}
        payload.setdefault("collected_at", payload["ts"])
        rows.append([payload.get(col) for col in db.SNAPSHOT_COLUMNS])
    with db.transaction(con):
        return db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, rows, db.SNAPSHOT_KEY)
