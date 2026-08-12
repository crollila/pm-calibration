"""Small shared helpers: timestamps, run ids, logging setup."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any


def utcnow() -> datetime:
    """Timezone-naive UTC 'now'. DuckDB TIMESTAMP is naive; we keep everything UTC."""
    return datetime.now(UTC).replace(tzinfo=None)


def floor_to(dt: datetime, minutes: int) -> datetime:
    """Floor a timestamp to a snapshot bucket.

    Bucketing makes the primary key stable: two collectors that start 40 seconds
    apart still write the same `ts`, so the venues line up row-for-row.

    >>> from datetime import datetime
    >>> floor_to(datetime(2026, 9, 12, 17, 43, 21), 15)
    datetime.datetime(2026, 9, 12, 17, 30)
    """
    discard = timedelta(
        minutes=dt.minute % minutes, seconds=dt.second, microseconds=dt.microsecond
    )
    return dt - discard


def parse_ts(value: Any) -> datetime | None:
    """Parse the ISO-8601 shapes the three APIs use, returning naive UTC.

    >>> parse_ts("2026-09-12T17:00:00Z")
    datetime.datetime(2026, 9, 12, 17, 0)
    >>> parse_ts(None) is None
    True
    """
    if value in (None, "", 0):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo is None else value.astimezone(UTC).replace(tzinfo=None)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(float(value), UTC).replace(tzinfo=None)
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed


def run_id_for(ts: datetime) -> str:
    """Deterministic run id so a retried run reuses its `collection_runs` row."""
    return ts.strftime("%Y%m%dT%H%M")


NOISY_LOGGERS = ("matplotlib", "urllib3", "PIL", "requests")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # --verbose is for our own diagnostics; matplotlib's font search at DEBUG
    # emits thousands of lines and buries them.
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
