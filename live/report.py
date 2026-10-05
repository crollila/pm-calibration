"""Public step: sanitized records -> positions.csv, summary.json, LIVE_TRADING_CASE_STUDY.md.

    uv run -m live.report

Reads only `data/live/*.csv` and `manifest.json`, all of which are committed, so
anyone can rebuild the case study and check every number. Fails loudly if the
ledger does not reconcile to the fills to the last decimal.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np

from live import analytics as an
from live.ledger import OPEN, SETTLED, ZERO, build_positions

DATA = Path("data/live")
DOC = Path("LIVE_TRADING_CASE_STUDY.md")

POSITION_FIELDS = [
    "position_id", "entry_utc", "close_utc", "market_slug", "event_slug", "kind", "league",
    "market", "side", "outcome", "order_types", "fills", "orders", "qty_bought", "qty_sold",
    "entry_price", "capital_at_risk", "fees", "exit_price", "settle_price", "status",
    "realized_pnl", "return", "timing", "event_start_utc", "start_source",
    "clv_close_price", "clv", "clv_note",
]  # fmt: skip


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load(data: Path = DATA) -> dict:
    ref_path = data / "event_reference.csv"
    return {
        "fills": read_csv(data / "fills.csv"),
        "markets": {m["market_slug"]: m for m in read_csv(data / "markets.csv")},
        "settlements": {s["market_slug"]: s for s in read_csv(data / "settlements.csv")},
        "reference": {r["event_slug"]: r for r in read_csv(ref_path)} if ref_path.exists() else {},
        "manifest": json.loads((data / "manifest.json").read_text()),
    }


def position_rows(src: dict) -> tuple[list[dict], dict]:
    fills, markets, settlements, ref = (
        src["fills"], src["markets"], src["settlements"], src["reference"]
    )  # fmt: skip
    positions = build_positions(fills, settlements)
    by_pos = defaultdict(list)
    for f in fills:
        by_pos[(f["market_slug"], f["side"])].append(f)

    rows = []
    for key, p in sorted(positions.items(), key=lambda kv: (kv[1].first_ts or "", kv[0])):
        slug, side = key
        m = markets[slug]
        fs = by_pos[key]
        buys = [f["ts_utc"] for f in fs if f["action"] == "BUY"]
        r = ref.get(m["event_slug"]) if m["kind"] == "single" else None
        start = m["event_start_utc"] or (r["start_utc"] if r else "")
        start_source = m["start_source"] or ("public reference: " + r["source"] if r else "")
        row = {
            "position_id": p.position_id,
            "entry_utc": p.first_ts,
            "close_utc": p.close_ts or "",
            "market_slug": slug,
            "event_slug": m["event_slug"],
            "kind": m["kind"],
            "league": m["league"],
            "market": m["title"],
            "side": side,
            "outcome": fs[0]["outcome_label"] if m["kind"] == "single" else f"combo {side}",
            "order_types": "+".join(sorted({f"{f['order_type']}/{f['tif']}" for f in fs})),
            "fills": p.fills,
            "orders": len(p.orders),
            "qty_bought": p.qty_bought,
            "qty_sold": p.qty_sold,
            "entry_price": p.entry_price,
            "capital_at_risk": p.capital_at_risk,
            "fees": p.fees,
            "exit_price": p.exit_price,
            "settle_price": p.settle_price,
            "status": p.status,
            "realized_pnl": p.realized_pnl,
            "return": (p.realized_pnl / p.capital_at_risk) if p.realized_pnl is not None else None,
            "timing": an.timing(buys, start or None),
            "event_start_utc": start,
            "start_source": start_source,
            "clv_close_price": None,
            "clv": None,
            "clv_note": "",
        }
        _attach_clv(row, r, p)
        rows.append(row)
    return rows, positions


def _attach_clv(row: dict, ref: dict | None, p) -> None:
    """Cross-venue closing-line proxy for single-market pregame positions only."""
    if row["kind"] != "single":
        row["clv_note"] = "combo: no closing price for the combined contract"
        return
    if ref is None:
        row["clv_note"] = "event not in public reference"
        return
    if row["timing"] != "pregame":
        row["clv_note"] = f"{row['timing']} entry: no pre-start close to compare"
        return
    if not ref["close0_pre_start"]:
        row["clv_note"] = "no pre-start print in reference"
        return
    idx, why = an.map_outcome(row["outcome"], row["side"], ref, p.settle_price)
    if idx is None:
        row["clv_note"] = why
        return
    c0 = Decimal(ref["close0_pre_start"])
    close = c0 if idx == 0 else 1 - c0
    row["clv_close_price"] = close
    row["clv"] = close - p.entry_price
    row["clv_note"] = "cross-venue proxy (Polymarket intl last print before start)"


def compute(src: dict) -> tuple[list[dict], dict]:
    rows, positions = position_rows(src)
    fills, settlements = src["fills"], src["settlements"]

    # ---- exact reconciliation: the ledger is a partition of the fills ----
    cash = sum((Decimal(f["cash_flow"]) for f in fills), ZERO)
    payouts = sum((p.payout for p in positions.values()), ZERO)
    closed = [r for r in rows if r["status"] != OPEN]
    open_ = [r for r in rows if r["status"] == OPEN]
    realized = sum((r["realized_pnl"] for r in closed), ZERO)
    open_net_cash = sum(
        (Decimal(f["cash_flow"]) for f in fills
         if positions[(f["market_slug"], f["side"])].status == OPEN),
        ZERO,
    )  # fmt: skip
    assert cash + payouts == realized + open_net_cash, "ledger does not reconcile to fills"
    assert sum(r["fills"] for r in rows) == len(fills)
    fee_total = sum((Decimal(f["fee"]) for f in fills), ZERO)
    assert sum((r["fees"] for r in rows), ZERO) == fee_total

    # ---- headline ----
    car = sum((r["capital_at_risk"] for r in closed), ZERO)
    closed_fees = sum((r["fees"] for r in closed), ZERO)
    notional = sum((Decimal(f["qty"]) * Decimal(f["px_side"]) for f in fills), ZERO)
    pnl_f = np.array([float(r["realized_pnl"]) for r in closed])
    car_f = np.array([float(r["capital_at_risk"]) for r in closed])
    day = [r["close_utc"][:10] for r in closed]
    ids = list(range(len(closed)))

    held = [r for r in closed if r["status"] == SETTLED]
    won = np.array([float(r["settle_price"]) for r in held])
    implied = np.array([float(r["entry_price"]) for r in held])
    fee_ps = np.array([float(r["fees"] / r["qty_bought"]) for r in held])
    held_day = [r["close_utc"][:10] for r in held]

    order = sorted(closed, key=lambda r: (r["close_utc"], r["position_id"]))
    dd = an.max_drawdown([r["realized_pnl"] for r in order], [r["close_utc"] for r in order])

    by_pnl = sorted(closed, key=lambda r: r["realized_pnl"], reverse=True)
    top2 = by_pnl[:2]
    rest = by_pnl[2:]
    rest_pnl = sum((r["realized_pnl"] for r in rest), ZERO)
    rest_car = sum((r["capital_at_risk"] for r in rest), ZERO)

    clv_rows = [r for r in rows if r["clv"] is not None]
    clv = np.array([float(r["clv"]) for r in clv_rows])
    clv_fee = np.array([float(r["clv"] - r["fees"] / r["qty_bought"]) for r in clv_rows])
    clv_ev = [r["event_slug"] for r in clv_rows]
    clv_reasons = defaultdict(int)
    for r in rows:
        if r["clv"] is None:
            clv_reasons[r["clv_note"]] += 1

    summary = {
        "manifest": src["manifest"],
        "counts": {
            "fills": len(fills),
            "orders": len({f["order_id"] for f in fills}),
            "markets": len(src["markets"]),
            "positions": len(rows),
            **an.status_counts(rows),
            "closed": len(closed),
            "settled_markets": len(settlements),
        },
        "first_entry": rows[0]["entry_utc"] if rows else None,
        "last_fill": max(f["ts_utc"] for f in fills) if fills else None,
        "realized_pnl": realized,
        "capital_at_risk_closed": car,
        "roi": realized / car if car else None,
        "gross_pnl_before_fees": realized + closed_fees,
        "fees_all": fee_total,
        "fees_closed": closed_fees,
        "traded_notional": notional,
        "fee_pct_notional": fee_total / notional if notional else None,
        "open_capital_at_risk": sum((r["capital_at_risk"] for r in open_), ZERO),
        "open_net_cash": open_net_cash,
        "fill_cash_total": cash,
        "payouts_total": payouts,
        "roi_ci_iid": an.ratio_ci(pnl_f, car_f, ids),
        "roi_ci_day": an.ratio_ci(pnl_f, car_f, day),
        "pnl_ci_day": an.sum_ci(pnl_f, day),
        "win_rate": an.win_rate(closed),
        "held": {
            "n": len(held),
            "won": float(won.sum()),
            "implied": float(implied.sum()),
            "breakeven_with_fees": float((implied + fee_ps).sum()),
            "excess": an.mean_ci(won - implied, held_day),
            "excess_after_fees": an.mean_ci(won - implied - fee_ps, held_day),
            "brier_entry_price": float(np.mean((implied - won) ** 2)) if len(held) else None,
            "roi_ci_day": an.ratio_ci(
                np.array([float(r["realized_pnl"]) for r in held]),
                np.array([float(r["capital_at_risk"]) for r in held]),
                held_day,
            ),
        },
        "drawdown": dd,
        "concentration": {
            "top2_pnl": sum((r["realized_pnl"] for r in top2), ZERO),
            "top2_ids": [r["position_id"] for r in top2],
            "rest_pnl": rest_pnl,
            "rest_roi": rest_pnl / rest_car if rest_car else None,
            "best5": [r["position_id"] for r in by_pnl[:5]],
            "worst5": [r["position_id"] for r in by_pnl[-5:][::-1]],
        },
        "size_quantiles": an.quantiles([r["capital_at_risk"] for r in rows]),
        "clv": {
            "n": len(clv_rows),
            "events": len(set(clv_ev)),
            "mean": an.mean_ci(clv, clv_ev) if len(clv_rows) else None,
            "mean_after_fees": an.mean_ci(clv_fee, clv_ev) if len(clv_rows) else None,
            "share_positive": float((clv > 0).mean()) if len(clv_rows) else None,
            "excluded": dict(sorted(clv_reasons.items(), key=lambda kv: -kv[1])),
        },
        "groups": {
            "kind": an.group_table(closed, lambda r: r["kind"]),
            "league": an.group_table(
                closed, lambda r: r["league"] if r["kind"] == "single" else "combo"
            ),
            "timing": an.group_table(closed, lambda r: r["timing"]),
            "side": an.group_table(closed, lambda r: r["side"]),
            "exit": an.group_table(
                closed,
                lambda r: (
                    "held to settlement" if r["status"] == SETTLED else "sold before settlement"
                ),
            ),
            "price": an.group_table(closed, lambda r: an.price_bucket(r["entry_price"])),
            "size": an.group_table(closed, lambda r: an.size_bucket(r["capital_at_risk"])),
            "month": an.group_table(closed, lambda r: r["close_utc"][:7]),
        },
        "checks": an.exchange_checks(positions, fills, settlements, src["markets"]),
    }
    return rows, summary


def _plain(v):
    if isinstance(v, Decimal):
        t = format(v, "f")
        return t.rstrip("0").rstrip(".") if "." in t else t
    return "" if v is None else v


def write_positions(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=POSITION_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _plain(r[k]) for k in POSITION_FIELDS})


def _json_default(o):
    if isinstance(o, Decimal):
        return _plain(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def main() -> None:
    from live.render import render

    src = load()
    rows, summary = compute(src)
    write_positions(rows, DATA / "positions.csv")
    (DATA / "summary.json").write_text(
        json.dumps(summary, indent=1, default=_json_default, sort_keys=True) + "\n"
    )
    DOC.write_text(render(rows, summary), encoding="utf-8", newline="\n")
    c = summary["counts"]
    print(
        f"{c['positions']} positions ({c['closed']} closed, {c['open']} open); "
        f"realized {summary['realized_pnl']:.2f} on {summary['capital_at_risk_closed']:.2f} "
        f"-> {DOC}"
    )


if __name__ == "__main__":
    main()
