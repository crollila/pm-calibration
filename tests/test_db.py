from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from pmcal import db

TS = datetime(2026, 9, 12, 17, 30)


def _row(outcome: str = "chiefs", bid: float = 0.51) -> list:
    values = {
        "ts": TS, "venue": "polymarket", "book": "", "market_id": "m1", "outcome": outcome,
        "event_key": "nfl|2026-09-12|chiefs~ravens", "sport": "nfl", "title": "t",
        "bid": bid, "ask": bid + 0.02, "mid": bid + 0.01, "bid_size": 100.0, "ask_size": 100.0,
        "volume": 1.0, "event_start_time": TS, "collected_at": TS, "run_id": "r1", "source": "live",
        "raw_json": "{}",
    }
    return [values[c] for c in db.SNAPSHOT_COLUMNS]


@pytest.fixture()
def con(tmp_path):
    connection = db.connect(tmp_path / "t.duckdb")
    yield connection
    connection.close()


def test_insert_is_idempotent_across_reruns(con):
    first = db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, [_row()], db.SNAPSHOT_KEY)
    second = db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, [_row()], db.SNAPSHOT_KEY)
    assert (first, second) == (1, 0)
    assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 1


def test_rerunning_never_overwrites_an_existing_price(con):
    db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, [_row(bid=0.51)], db.SNAPSHOT_KEY)
    db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, [_row(bid=0.99)], db.SNAPSHOT_KEY)
    assert con.execute("SELECT bid FROM snapshots").fetchone()[0] == pytest.approx(0.51)


def test_duplicate_keys_within_one_batch_are_collapsed(con):
    inserted = db.insert_rows(
        con, "snapshots", db.SNAPSHOT_COLUMNS, [_row(), _row(), _row()], db.SNAPSHOT_KEY
    )
    assert inserted == 1


def test_distinct_outcomes_are_separate_rows(con):
    inserted = db.insert_rows(
        con, "snapshots", db.SNAPSHOT_COLUMNS,
        [_row("chiefs"), _row("ravens")], db.SNAPSHOT_KEY,
    )
    assert inserted == 2


def test_batching_handles_more_rows_than_one_chunk(con):
    rows = [_row(f"o{i}") for i in range(1200)]
    assert db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, rows, db.SNAPSHOT_KEY) == 1200


def test_transaction_rolls_back_on_failure(con):
    with pytest.raises(RuntimeError), db.transaction(con):
        db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, [_row()], db.SNAPSHOT_KEY)
        raise RuntimeError("simulated crash mid-run")
    assert con.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 0


def test_run_bookkeeping_records_counts_and_errors(con):
    with db.transaction(con):
        db.start_run(con, "r1", "polymarket", TS, False)
    with db.transaction(con):
        db.finish_run(con, "r1", "polymarket", TS, "ok", 10, 7, None)
    with db.transaction(con):
        db.start_run(con, "r1", "kalshi", TS, False)
        db.finish_run(con, "r1", "kalshi", TS, "error", 0, 0, "HTTP 503")

    rows = con.execute(
        "SELECT source, status, rows_seen, rows_inserted, error FROM collection_runs ORDER BY source"
    ).fetchall()
    assert rows == [
        ("kalshi", "error", 0, 0, "HTTP 503"),
        ("polymarket", "ok", 10, 7, None),
    ]


def test_reopening_the_file_preserves_data(tmp_path):
    path = tmp_path / "persist.duckdb"
    con = db.connect(path)
    db.insert_rows(con, "snapshots", db.SNAPSHOT_COLUMNS, [_row()], db.SNAPSHOT_KEY)
    con.close()

    reopened = db.connect(path)
    try:
        assert reopened.execute("SELECT count(*) FROM snapshots").fetchone()[0] == 1
        # Schema creation is idempotent too.
        reopened.execute(db.SCHEMA)
    finally:
        reopened.close()


def test_upsert_updates_resolutions_in_place(con):
    columns = ("venue", "market_id", "outcome", "event_key", "resolved", "status",
               "resolved_at", "observed_at", "raw_json")
    key = ("venue", "market_id", "outcome")
    db.upsert_rows(con, "resolutions", columns,
                   [["kalshi", "K1:yes", "chiefs", "e1", None, "open", None, TS, "{}"]], key)
    db.upsert_rows(con, "resolutions", columns,
                   [["kalshi", "K1:yes", "chiefs", "e1", 1.0, "settled", TS, TS, "{}"]], key)
    rows = con.execute("SELECT resolved, status FROM resolutions").fetchall()
    assert rows == [(1.0, "settled")]


def test_primary_key_is_enforced_by_the_database(con):
    con.execute("BEGIN TRANSACTION")
    con.execute(
        f"INSERT INTO snapshots ({','.join(db.SNAPSHOT_COLUMNS)}) "
        f"VALUES ({','.join(['?'] * len(db.SNAPSHOT_COLUMNS))})",
        _row(),
    )
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            f"INSERT INTO snapshots ({','.join(db.SNAPSHOT_COLUMNS)}) "
            f"VALUES ({','.join(['?'] * len(db.SNAPSHOT_COLUMNS))})",
            _row(),
        )
    con.execute("ROLLBACK")

