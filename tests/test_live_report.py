"""Sanitization, cross-checks and reconciliation of the published case study.

The raw payload built here is SYNTHETIC: it mimics the shape of the Polymarket US
activities API with invented ids and prices. The last group of tests runs on the
committed sanitized records in data/live/ and proves the published document is
exactly what the pipeline produces from them.
"""

from __future__ import annotations

import csv
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from live import analytics as an
from live.ingest import sanitize

D = Decimal

SALT = b"test-salt"
RAW_SECRETS = ["RAWTRADE1", "RAWORDER1", "COUNTERPARTY1", "RAWTRADE2", "RAWORDER2", "acct-xyz"]


def _money(v):
    return {"value": str(v), "currency": "USD"}


def _trade(tid, oid, slug, intent, side, qty, px_yes, fee, cost, t, title="A vs. B", outcome="A"):
    meta = {"title": title, "outcome": outcome, "eventSlug": slug[4:], "team": {"league": "nba"}}
    return {
        "type": "ACTIVITY_TYPE_TRADE",
        "trade": {
            "id": tid,
            "marketSlug": slug,
            "createTime": t,
            "isAggressor": True,
            "qtyDecimal": str(qty),
            "price": _money(px_yes),
            "cost": _money(cost),
            "comboLegDetails": [],
            "aggressorExecution": {
                "lastShares": str(qty),
                "lastPx": _money(px_yes),
                "commissionNotionalCollected": _money(fee),
                "order": {
                    "id": oid,
                    "type": "ORDER_TYPE_LIMIT",
                    "tif": "TIME_IN_FORCE_DAY",
                    "intent": f"ORDER_INTENT_{intent}",
                    "outcomeSide": f"OUTCOME_SIDE_{side}",
                    "marketMetadata": meta,
                },  # fmt: skip
            },
            "passiveExecution": {"order": {"id": "COUNTERPARTY1", "marketMetadata": meta}},
        },
    }


def _resolution(slug, prices, net, t):
    pos = {"netPositionDecimal": str(net), "qtyBoughtDecimal": "0", "qtySoldDecimal": "0",
           "realized": _money(0), "cost": _money(0)}  # fmt: skip
    return {
        "type": "ACTIVITY_TYPE_POSITION_RESOLUTION",
        "positionResolution": {
            "marketSlug": slug,
            "updateTime": t,
            "beforePosition": pos,
            "afterPosition": {**pos, "netPositionDecimal": "0"},
            "market": {
                "outcomePrices": json.dumps(prices),
                "status": "MARKET_STATUS_RESOLVED",
                "marketSides": [{"long": True}, {"long": False}],
            },
        },
    }


@pytest.fixture
def raw():
    return {
        "pulled_at": 1789637844,
        "positions": {"positions": {}, "eof": True},
        "activities": [
            _resolution(
                "aec-nba-a-b-2026-01-01", ["1", "0"], -10, "2026-01-02T00:00:00.123456789Z"
            ),
            # Buy NO at YES-price 0.65: 10 NO shares cost 3.50 + 0.02 fee.
            _trade(
                "RAWTRADE1",
                "RAWORDER1",
                "aec-nba-a-b-2026-01-01",
                "BUY_SHORT",
                "NO",
                10,
                "0.65",
                "0.02",
                "3.52",
                "2026-01-01T20:00:00.5Z",
                outcome="B",
            ),  # fmt: skip
            {
                "type": "ACTIVITY_TYPE_ACCOUNT_WITHDRAWAL",
                "accountBalanceChange": {"amount": _money(999), "description": "acct-xyz"},
            },
            {
                "type": "ACTIVITY_TYPE_REFERRAL_BONUS",
                "accountBalanceChange": {"amount": _money(10), "description": "acct-xyz"},
            },
        ],
    }


def test_sanitize_prices_no_side_and_reconciles_to_trade_cost(raw):
    out = sanitize(raw, SALT)
    (f,) = out["fills"]
    assert f["side"] == "NO" and f["action"] == "BUY"
    assert f["px_side"] == D("0.35")
    assert f["cash_flow"] == D("-3.52") == -f["exch_trade_cost"]
    assert f["ts_utc"] == "2026-01-01T20:00:00.500000Z"
    (s,) = out["settlements"]
    assert (s["yes_price"], s["no_price"]) == (D(1), D(0))


def test_sanitized_output_contains_no_raw_ids_or_account_text(raw):
    text = json.dumps(sanitize(raw, SALT), default=str)
    for secret in RAW_SECRETS:
        assert secret not in text
    assert "999" not in text  # withdrawal amount


def test_ids_are_stable_and_salt_dependent(raw):
    a = sanitize(raw, SALT)["fills"][0]
    assert a["fill_id"] == sanitize(raw, SALT)["fills"][0]["fill_id"]
    assert a["fill_id"] != sanitize(raw, b"other-salt")["fills"][0]["fill_id"]
    assert re.fullmatch(r"F-[0-9a-f]{12}", a["fill_id"])


def test_passive_fill_is_refused_rather_than_misread(raw):
    raw["activities"][1]["trade"]["isAggressor"] = False
    with pytest.raises(ValueError, match="passive"):
        sanitize(raw, SALT)


# ------------------------------------------------------------- analytics
def test_outcome_mapping_needs_all_three_checks_to_agree():
    ref = {"outcome0": "Houston Rockets", "outcome1": "Los Angeles Lakers", "winner0": "1"}
    assert an.map_outcome("Rockets", "YES", ref, D(1)) == (0, "")
    assert an.map_outcome("Lakers", "NO", ref, D(0)) == (1, "")
    assert an.map_outcome("Rockets", "NO", ref, None)[0] is None  # order disagrees
    assert an.map_outcome("Rockets", "YES", ref, D(0))[0] is None  # winner disagrees
    assert an.map_outcome("s", "YES", ref, None)[0] is None  # ambiguous label


def test_drawdown_starts_from_zero_peak():
    d = an.max_drawdown([D(-3), D(1)], ["a", "b"])
    assert d["max_drawdown"] == D(3) and d["peak_at"] is None and d["final"] == D(-2)


# ------------------------------------------------- committed case study
DATA = Path("data/live")
needs_data = pytest.mark.skipif(not (DATA / "fills.csv").exists(), reason="no sanitized data")


@pytest.fixture(scope="module")
def built():
    from live.report import compute, load

    return compute(load(DATA))


def _rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@needs_data
def test_published_aggregates_reconcile_to_sanitized_fills(built):
    rows, s = built
    fills = _rows("fills.csv")
    settlements = {r["market_slug"]: r for r in _rows("settlements.csv")}
    closed = [r for r in rows if r["status"] != "open"]

    # Recompute P&L straight from the fills and settlement prices, without the ledger.
    held = {}
    cash_by_pos = {}
    for f in fills:
        key = (f["market_slug"], f["side"])
        sign = 1 if f["action"] == "BUY" else -1
        held[key] = held.get(key, D(0)) + sign * D(f["qty"])
        cash_by_pos[key] = cash_by_pos.get(key, D(0)) + D(f["cash_flow"])
    realized = D(0)
    for (slug, side), q in held.items():
        st = settlements.get(slug)
        if q and st is None:
            continue  # open
        pay = q * D(st["yes_price"] if side == "YES" else st["no_price"]) if q else D(0)
        realized += cash_by_pos[(slug, side)] + pay

    assert s["realized_pnl"] == realized
    assert s["fees_all"] == sum(D(f["fee"]) for f in fills)
    assert s["counts"]["fills"] == len(fills)
    assert s["counts"]["closed"] == len(closed)
    assert s["capital_at_risk_closed"] == sum(r["capital_at_risk"] for r in closed)


@needs_data
def test_committed_positions_and_summary_match_a_fresh_build(built):
    from live.report import _json_default

    rows, s = built
    committed = _rows("positions.csv")
    assert len(committed) == len(rows)
    assert sum(D(r["realized_pnl"]) for r in committed if r["realized_pnl"]) == s["realized_pnl"]
    fresh = json.loads(json.dumps(s, default=_json_default, sort_keys=True))
    assert json.loads((DATA / "summary.json").read_text()) == fresh


@needs_data
def test_committed_document_is_exactly_the_pipeline_output(built):
    from live.render import render

    rows, s = built
    doc = Path("LIVE_TRADING_CASE_STUDY.md").read_text(encoding="utf-8")
    assert doc == render(rows, s), "case study is stale: run `uv run -m live.report`"


@needs_data
def test_sanitized_files_carry_only_hmac_ids():
    for r in _rows("fills.csv"):
        assert re.fullmatch(r"F-[0-9a-f]{12}", r["fill_id"])
        assert re.fullmatch(r"O-[0-9a-f]{12}", r["order_id"])
        assert re.fullmatch(r"P-[0-9a-f]{10}", r["position_id"])
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert "ACCOUNT_WITHDRAWAL" not in manifest.get("non_trading_credits_usd", {})
