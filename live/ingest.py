"""Private step: raw Polymarket US account history -> sanitized, committable CSVs.

    uv run -m live.ingest --raw PATH/TO/my_us_history.json

Input is the JSON written by a read-only pull of the account's own
`/v1/portfolio/activities` and `/v1/portfolio/positions` endpoints (the format is
documented in LIVE_TRADING_CASE_STUDY.md). The raw file never enters the repo.

What is kept: public market identifiers (slugs, titles, outcome labels), fill
times, quantities, prices, fees, order type and time-in-force, settlement prices.

What is dropped: every exchange order/trade/transaction id (replaced by salted
HMACs so they stay stable but cannot be looked up), the counterparty's order on
every fill, withdrawal amounts, payment metadata, and anything describing the
account itself. The salt lives outside the repo (PMCAL_LIVE_SALT or
~/.pmcal/live_id_salt) so the ids cannot be reversed by brute force.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import os
import secrets
from collections import Counter
from decimal import Decimal
from pathlib import Path

from live.ledger import fill_cash_flow, side_price

OUT_DIR = Path("data/live")
SALT_FILE = Path.home() / ".pmcal" / "live_id_salt"

FILL_FIELDS = [
    "seq", "fill_id", "order_id", "position_id", "ts_utc", "market_slug", "side",
    "outcome_label", "action", "intent", "order_type", "tif", "liquidity",
    "qty", "px_yes", "px_side", "fee", "cash_flow", "exch_trade_cost",
]  # fmt: skip
MARKET_FIELDS = [
    "market_slug", "kind", "league", "title", "event_slug", "event_start_utc",
    "start_source", "n_legs", "legs_json", "exch_open_net",
]  # fmt: skip
SETTLEMENT_FIELDS = [
    "market_slug", "settled_ts", "yes_price", "no_price", "market_status",
    "exch_net_before", "exch_qty_bought", "exch_qty_sold", "exch_realized_before",
    "exch_realized_delta", "exch_cost_after",
]  # fmt: skip


def load_salt() -> bytes:
    env = os.environ.get("PMCAL_LIVE_SALT")
    if env:
        return env.encode()
    if not SALT_FILE.exists():
        SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SALT_FILE.write_text(secrets.token_hex(32))
        print(f"created new id salt at {SALT_FILE}; keep it or every sanitized id changes")
    return SALT_FILE.read_text().strip().encode()


def make_id(salt: bytes, prefix: str, raw: str, n: int = 12) -> str:
    """Stable, non-reversible id: HMAC-SHA256 of the raw exchange id."""
    return f"{prefix}-{hmac.new(salt, raw.encode(), hashlib.sha256).hexdigest()[:n]}"


def ts(value: str) -> str:
    """Normalise exchange nanosecond timestamps to fixed-width microsecond ISO UTC.

    Fixed width matters: the ledger orders events by comparing these strings.

    >>> ts("2026-09-17T09:19:24.966804068Z"), ts("2026-01-01T00:00:00Z")
    ('2026-09-17T09:19:24.966804Z', '2026-01-01T00:00:00.000000Z')
    """
    head, _, frac = value.rstrip("Z").partition(".")
    return f"{head}.{(frac + '000000')[:6]}Z"


def usd(obj) -> Decimal:
    return Decimal(obj["value"]) if obj else Decimal(0)


def _short(enum: str, prefix: str) -> str:
    return enum[len(prefix) :] if enum.startswith(prefix) else enum


def _strip_aec(slug: str) -> str:
    return slug[4:] if slug.startswith("aec-") else slug


def _legs(details: list[dict]) -> list[dict]:
    return [
        {
            "league": (d.get("team") or {}).get("league")
            or (d.get("eventGroupTitle") or "").lower(),
            "title": d.get("title"),
            "outcome": d.get("outcome"),
            "side": _short(d.get("outcomeSide", ""), "OUTCOME_SIDE_"),
            "event_slug": d.get("eventSlug"),
            "start_utc": ts(d["eventStartTime"]) if d.get("eventStartTime") else "",
            "state": _short(d.get("state", ""), "COMBO_LEG_STATE_"),
        }
        for d in details
    ]


def sanitize(raw: dict, salt: bytes) -> dict:
    """Pure transform of the raw pull into the four sanitized tables."""
    acts = raw["activities"]
    trades = [a["trade"] for a in acts if a["type"] == "ACTIVITY_TYPE_TRADE"]
    trades.sort(key=lambda t: (ts(t["createTime"]), t["id"]))

    fills, markets = [], {}
    for seq, t in enumerate(trades, start=1):
        if not t.get("isAggressor"):
            # Every fill in the current data is aggressive. A passive fill would
            # mean our execution is `passiveExecution`; fail loudly rather than
            # silently reading the counterparty's order.
            raise ValueError("passive fill encountered; extend ingest before trusting it")
        ex = t["aggressorExecution"]
        o = ex["order"]
        side = _short(o["outcomeSide"], "OUTCOME_SIDE_")
        intent = _short(o["intent"], "ORDER_INTENT_")
        action = intent.split("_")[0]
        qty = Decimal(ex["lastShares"])
        px_yes = usd(ex["lastPx"])
        px = side_price(px_yes, side)
        fee = usd(ex.get("commissionNotionalCollected"))
        slug = t["marketSlug"]
        meta = o.get("marketMetadata") or {}
        fills.append(
            {
                "seq": seq,
                "fill_id": make_id(salt, "F", t["id"]),
                "order_id": make_id(salt, "O", o["id"]),
                "position_id": make_id(salt, "P", f"{slug}|{side}", 10),
                "ts_utc": ts(t["createTime"]),
                "market_slug": slug,
                "side": side,
                "outcome_label": meta.get("outcome") or "",
                "action": action,
                "intent": intent,
                "order_type": _short(o["type"], "ORDER_TYPE_"),
                "tif": _short(o["tif"], "TIME_IN_FORCE_"),
                "liquidity": "taker",
                "qty": qty,
                "px_yes": px_yes,
                "px_side": px,
                "fee": fee,
                "cash_flow": fill_cash_flow(action, qty, px, fee),
                "exch_trade_cost": usd(t["cost"]),
            }
        )
        m = markets.setdefault(slug, {"market_slug": slug, "exch_open_net": ""})
        legs = t.get("comboLegDetails") or []
        if legs:
            m.update(kind="combo", n_legs=len(legs), _legs=legs)
        elif meta.get("title"):
            m.update(
                kind="single",
                n_legs=1,
                title=meta["title"],
                league=(meta.get("team") or {}).get("league") or slug.split("-")[1],
                event_slug=_strip_aec(meta.get("eventSlug") or slug),
            )

    settlements = {}
    for a in acts:
        if a["type"] != "ACTIVITY_TYPE_POSITION_RESOLUTION":
            continue
        r = a["positionResolution"]
        slug = r["marketSlug"]
        if slug in settlements:
            raise ValueError(f"{slug}: more than one resolution")
        sides = r["market"]["marketSides"]
        prices = json.loads(r["market"]["outcomePrices"])
        if len(sides) != 2 or not sides[0]["long"] or sides[1]["long"]:
            raise ValueError(f"{slug}: unexpected market side layout")
        before, after = r["beforePosition"], r["afterPosition"]
        settlements[slug] = {
            "market_slug": slug,
            "settled_ts": ts(r["updateTime"]),
            "yes_price": Decimal(prices[0]),
            "no_price": Decimal(prices[1]),
            "market_status": _short(r["market"]["status"], "MARKET_STATUS_"),
            # The exchange's own position record, kept only for cross-checks.
            # Quantities are in YES terms: buying NO counts as selling YES.
            "exch_net_before": Decimal(before["netPositionDecimal"]),
            "exch_qty_bought": Decimal(before["qtyBoughtDecimal"]),
            "exch_qty_sold": Decimal(before["qtySoldDecimal"]),
            "exch_realized_before": usd(before["realized"]),
            "exch_realized_delta": usd(after["realized"]) - usd(before["realized"]),
            "exch_cost_after": usd(after["cost"]),
        }
        if r.get("comboLegDetails") and slug in markets:
            markets[slug]["_legs"] = r["comboLegDetails"]  # final leg states

    for slug, p in (raw.get("positions") or {}).get("positions", {}).items():
        if slug in markets:
            markets[slug]["exch_open_net"] = Decimal(p["netPositionDecimal"])

    for m in markets.values():
        legs = _legs(m.pop("_legs", []))
        if m.get("kind") == "combo":
            leagues = sorted({lg["league"] for lg in legs if lg["league"]})
            starts = sorted(lg["start_utc"] for lg in legs if lg["start_utc"])
            m.update(
                league="+".join(leagues),
                title=" | ".join(f"{lg['outcome']} ({lg['title']})" for lg in legs),
                event_slug="",
                event_start_utc=starts[0] if starts else "",
                start_source="exchange: earliest combo leg start" if starts else "",
                legs_json=json.dumps(legs, sort_keys=True),
            )
        else:
            m.update(event_start_utc="", start_source="", legs_json="")
        if "kind" not in m:
            raise ValueError(f"{m['market_slug']}: no market metadata")

    types = Counter(a["type"] for a in acts)
    credits = Counter()
    for a in acts:
        if a["type"] in ("ACTIVITY_TYPE_REFERRAL_BONUS", "ACTIVITY_TYPE_TRANSFER"):
            credits[_short(a["type"], "ACTIVITY_TYPE_")] += usd(a["accountBalanceChange"]["amount"])
    manifest = {
        "source": "Polymarket US API: GET /v1/portfolio/activities (all pages) and "
        "GET /v1/portfolio/positions, pulled read-only with the account's own key",
        "pulled_at_utc_date": _utc_date(raw["pulled_at"]),
        "activity_counts": {_short(k, "ACTIVITY_TYPE_"): v for k, v in sorted(types.items())},
        "non_trading_credits_usd": {k: str(v) for k, v in sorted(credits.items())},
        "withdrawals": "count only; amounts are personal and not trading results",
        "positions_endpoint_eof": (raw.get("positions") or {}).get("eof"),
        "first_fill_utc": fills[0]["ts_utc"] if fills else None,
        "last_fill_utc": fills[-1]["ts_utc"] if fills else None,
    }
    return {
        "fills": fills,
        "markets": sorted(markets.values(), key=lambda m: m["market_slug"]),
        "settlements": sorted(settlements.values(), key=lambda s: s["market_slug"]),
        "manifest": manifest,
    }


def _utc_date(epoch: int) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%d")


def fmt(value) -> str:
    """Exact, plain-notation decimals.

    >>> fmt(Decimal("-9.51000000")), fmt(Decimal("100")), fmt(Decimal("1E+1")), fmt(None)
    ('-9.51', '100', '10', '')
    """
    if isinstance(value, Decimal):
        text = format(value, "f")
        return text.rstrip("0").rstrip(".") if "." in text else text
    return "" if value is None else str(value)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k)) for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True, type=Path, help="private activities JSON")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    raw_bytes = args.raw.read_bytes()
    out = sanitize(json.loads(raw_bytes), load_salt())
    out["manifest"]["raw_sha256"] = hashlib.sha256(raw_bytes).hexdigest()

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "fills.csv", FILL_FIELDS, out["fills"])
    write_csv(args.out / "markets.csv", MARKET_FIELDS, out["markets"])
    write_csv(args.out / "settlements.csv", SETTLEMENT_FIELDS, out["settlements"])
    (args.out / "manifest.json").write_text(json.dumps(out["manifest"], indent=1) + "\n")
    print(
        f"{len(out['fills'])} fills, {len(out['markets'])} markets, "
        f"{len(out['settlements'])} settlements -> {args.out}"
    )


if __name__ == "__main__":
    main()
