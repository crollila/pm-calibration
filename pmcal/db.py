"""DuckDB schema and append-only, idempotent writes.

Every write goes through a single transaction so a crash mid-run leaves the file
either fully updated or untouched. Snapshot inserts are `ON CONFLICT DO NOTHING`
against a primary key, so re-running a collection for the same minute is a no-op
rather than a duplicate.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import duckdb

SNAPSHOT_KEY = ("ts", "venue", "book", "market_id", "outcome")

SNAPSHOT_COLUMNS = (
    "ts",
    "venue",
    "book",
    "market_id",
    "outcome",
    "event_key",
    "sport",
    "title",
    "bid",
    "ask",
    "mid",
    "bid_size",
    "ask_size",
    "volume",
    "event_start_time",
    "collected_at",
    "run_id",
    "source",
    "raw_json",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    ts                TIMESTAMP NOT NULL,   -- snapshot bucket (UTC, floored to the run)
    venue             VARCHAR   NOT NULL,   -- polymarket | kalshi | sportsbook
    book              VARCHAR   NOT NULL,   -- bookmaker key; '' for prediction markets
    market_id         VARCHAR   NOT NULL,
    outcome           VARCHAR   NOT NULL,   -- normalised outcome label
    event_key         VARCHAR,              -- cross-venue join key (may be NULL)
    sport             VARCHAR,
    title             VARCHAR,
    bid               DOUBLE,               -- probability units, 0-1
    ask               DOUBLE,
    mid               DOUBLE,
    bid_size          DOUBLE,               -- top-of-book depth in contracts
    ask_size          DOUBLE,
    volume            DOUBLE,
    event_start_time  TIMESTAMP,
    collected_at      TIMESTAMP NOT NULL,   -- wall clock of the actual fetch
    run_id            VARCHAR   NOT NULL,
    -- Ingestion mode: 'live' (collect.py, sampled at ts) or 'backfill'
    -- (backfill.py, reconstructed from historical endpoints). NOT the same
    -- thing as collection_runs.source, which names the venue.
    source            VARCHAR   NOT NULL DEFAULT 'live',
    raw_json          VARCHAR,              -- full upstream payload, verbatim
    PRIMARY KEY (ts, venue, book, market_id, outcome)
);

CREATE TABLE IF NOT EXISTS backfill_progress (
    venue        VARCHAR NOT NULL,
    market_id    VARCHAR NOT NULL,
    status       VARCHAR NOT NULL,          -- ok | empty | error
    path         VARCHAR,                   -- which historical endpoint served it
    rows_written BIGINT DEFAULT 0,
    updated_at   TIMESTAMP NOT NULL,
    note         VARCHAR,
    PRIMARY KEY (venue, market_id)
);

CREATE TABLE IF NOT EXISTS collection_runs (
    run_id       VARCHAR NOT NULL,
    source       VARCHAR NOT NULL,
    started_at   TIMESTAMP NOT NULL,
    finished_at  TIMESTAMP,
    status       VARCHAR NOT NULL,          -- ok | error | skipped
    rows_seen    BIGINT DEFAULT 0,
    rows_inserted BIGINT DEFAULT 0,
    dry_run      BOOLEAN DEFAULT FALSE,
    error        VARCHAR,
    PRIMARY KEY (run_id, source)
);

CREATE TABLE IF NOT EXISTS resolutions (
    venue        VARCHAR NOT NULL,
    market_id    VARCHAR NOT NULL,
    outcome      VARCHAR NOT NULL,
    event_key    VARCHAR,
    resolved     DOUBLE,                    -- 1.0 yes, 0.0 no, NULL unresolved
    status       VARCHAR,
    resolved_at  TIMESTAMP,
    observed_at  TIMESTAMP NOT NULL,
    raw_json     VARCHAR,
    PRIMARY KEY (venue, market_id, outcome)
);

CREATE TABLE IF NOT EXISTS resolution_conflicts (
    event_key    VARCHAR NOT NULL,
    outcome      VARCHAR NOT NULL,
    venue_a      VARCHAR NOT NULL,
    value_a      DOUBLE,
    venue_b      VARCHAR NOT NULL,
    value_b      DOUBLE,
    detected_at  TIMESTAMP NOT NULL,
    note         VARCHAR,
    PRIMARY KEY (event_key, outcome, venue_a, venue_b)
);

CREATE TABLE IF NOT EXISTS event_map (
    event_key    VARCHAR NOT NULL,
    venue        VARCHAR NOT NULL,
    market_id    VARCHAR NOT NULL,
    match_score  DOUBLE,
    method       VARCHAR,
    PRIMARY KEY (event_key, venue, market_id)
);
"""


# Columns added after the first release. Applied on every open so a database
# collected before the column existed keeps working without a manual migration.
MIGRATIONS = (
    "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS source VARCHAR DEFAULT 'live'",
)


def connect(db_path: str | Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    if not read_only:
        con.execute(SCHEMA)
        for statement in MIGRATIONS:
            con.execute(statement)
    return con


@contextlib.contextmanager
def transaction(con: duckdb.DuckDBPyConnection) -> Iterator[duckdb.DuckDBPyConnection]:
    con.execute("BEGIN TRANSACTION")
    try:
        yield con
    except BaseException:
        con.execute("ROLLBACK")
        raise
    else:
        con.execute("COMMIT")


def _values_clause(n_cols: int, n_rows: int) -> str:
    row = "(" + ",".join(["?"] * n_cols) + ")"
    return ",".join([row] * n_rows)


def insert_rows(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    key: Sequence[str],
    chunk: int = 500,
) -> int:
    """Append rows, skipping any that collide with an existing primary key.

    Duplicates *within* `rows` are dropped first: DuckDB refuses to resolve two
    conflicting inserts to the same key inside one statement.
    """
    if not rows:
        return 0
    key_idx = [columns.index(k) for k in key]
    seen: set[tuple[Any, ...]] = set()
    deduped: list[Sequence[Any]] = []
    for row in rows:
        k = tuple(row[i] for i in key_idx)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(row)

    col_sql = ",".join(columns)
    inserted = 0
    for start in range(0, len(deduped), chunk):
        batch = deduped[start : start + chunk]
        flat: list[Any] = [v for row in batch for v in row]
        sql = (
            f"INSERT INTO {table} ({col_sql}) VALUES "
            f"{_values_clause(len(columns), len(batch))} ON CONFLICT DO NOTHING"
        )
        result = con.execute(sql, flat).fetchone()
        inserted += int(result[0]) if result else 0
    return inserted


def upsert_rows(
    con: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    key: Sequence[str],
) -> int:
    """Like `insert_rows` but overwrites non-key columns on conflict.

    Used for resolutions, which can legitimately change (unresolved -> resolved).
    """
    if not rows:
        return 0
    updates = ",".join(f"{c}=excluded.{c}" for c in columns if c not in key)
    key_idx = [columns.index(k) for k in key]
    seen: set[tuple[Any, ...]] = set()
    deduped: list[Sequence[Any]] = []
    for row in rows:
        k = tuple(row[i] for i in key_idx)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(row)
    col_sql = ",".join(columns)
    sql = (
        f"INSERT INTO {table} ({col_sql}) VALUES "
        f"{_values_clause(len(columns), len(deduped))} "
        f"ON CONFLICT ({','.join(key)}) DO UPDATE SET {updates}"
    )
    con.execute(sql, [v for row in deduped for v in row])
    return len(deduped)


def start_run(
    con: duckdb.DuckDBPyConnection, run_id: str, source: str, started_at: Any, dry_run: bool
) -> None:
    con.execute(
        "INSERT INTO collection_runs (run_id, source, started_at, status, dry_run) "
        "VALUES (?, ?, ?, 'running', ?) ON CONFLICT DO NOTHING",
        [run_id, source, started_at, dry_run],
    )


def finish_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    source: str,
    finished_at: Any,
    status: str,
    rows_seen: int,
    rows_inserted: int,
    error: str | None,
) -> None:
    con.execute(
        "UPDATE collection_runs SET finished_at=?, status=?, rows_seen=?, rows_inserted=?, "
        "error=? WHERE run_id=? AND source=?",
        [finished_at, status, rows_seen, rows_inserted, error, run_id, source],
    )
