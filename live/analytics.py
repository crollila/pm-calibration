"""Aggregates over sanitized positions. Money stays Decimal; statistics use floats.

Every aggregate is a function of the position rows alone, so a reader holding
`data/live/positions.csv` can recompute any number in the case study.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

import numpy as np

from live.ledger import EXITED, OPEN, SETTLED, ZERO, Position
from pmcal.stats import cluster_bootstrap, wilson_interval

N_BOOT = 10_000


# --------------------------------------------------------------------- enrich
def timing(buy_times: list[str], start: str | None) -> str:
    """pregame / in-play / mixed relative to the scheduled start, else unknown.

    >>> timing(["2026-01-01T10:00:00.000000Z"], "2026-01-01T12:00:00.000000Z")
    'pregame'
    >>> timing(["2026-01-01T10:00:00.000000Z", "2026-01-01T13:00:00.000000Z"],
    ...        "2026-01-01T12:00:00.000000Z")
    'mixed'
    >>> timing(["2026-01-01T10:00:00.000000Z"], None)
    'unknown'
    """
    if not start or not buy_times:
        return "unknown"
    before = [t < start for t in buy_times]
    return "pregame" if all(before) else "in-play" if not any(before) else "mixed"


def map_outcome(label: str, side: str, ref: dict, settle_price) -> tuple:
    """Index of the backed outcome in the reference event, or (None, reason).

    Three independent checks must agree, because a wrong mapping silently flips
    the sign of CLV: the outcome label matches exactly one reference outcome
    name; the side agrees with title order (YES is the first-named team); and,
    if the position settled, the reference winner agrees with the settlement.
    """
    names = [ref["outcome0"], ref["outcome1"]]
    hits = [i for i, n in enumerate(names) if label and label.lower() in n.lower()]
    if len(hits) != 1:
        return None, "label does not match exactly one reference outcome"
    idx = hits[0]
    if (side == "YES") != (idx == 0):
        return None, "side disagrees with reference outcome order"
    if settle_price is not None and settle_price in (0, 1):
        won = settle_price == 1
        ref_won = (int(ref["winner0"]) == 1) == (idx == 0)
        if won != ref_won:
            return None, "reference winner disagrees with settlement"
    return idx, ""


# ---------------------------------------------------------------- statistics
def ratio_ci(num: np.ndarray, den: np.ndarray, clusters, n_boot: int = N_BOOT) -> dict:
    return cluster_bootstrap(lambda i: num[i].sum() / den[i].sum(), clusters, n_boot=n_boot)


def sum_ci(x: np.ndarray, clusters, n_boot: int = N_BOOT) -> dict:
    return cluster_bootstrap(lambda i: x[i].sum(), clusters, n_boot=n_boot)


def mean_ci(x: np.ndarray, clusters, n_boot: int = N_BOOT) -> dict:
    return cluster_bootstrap(lambda i: x[i].mean(), clusters, n_boot=n_boot)


def max_drawdown(pnls: list[Decimal], stamps: list[str]) -> dict:
    """Largest peak-to-trough fall of cumulative realized P&L (peak starts at 0).

    >>> d = max_drawdown([Decimal(5), Decimal(-8), Decimal(2), Decimal(-1)], list("abcd"))
    >>> d["max_drawdown"], d["peak_at"], d["trough_at"]
    (Decimal('8'), 'a', 'b')
    """
    cum = peak = ZERO
    peak_at = None
    best = {"max_drawdown": ZERO, "peak_at": None, "trough_at": None, "peak": ZERO}
    for pnl, stamp in zip(pnls, stamps, strict=True):
        cum += pnl
        if cum > peak:
            peak, peak_at = cum, stamp
        if peak - cum > best["max_drawdown"]:
            best = {
                "max_drawdown": peak - cum,
                "peak_at": peak_at,
                "trough_at": stamp,
                "peak": peak,
            }
    best["final"] = cum
    return best


def quantiles(values: list[Decimal]) -> dict:
    a = np.array([float(v) for v in values])
    qs = {"min": 0, "p25": 25, "median": 50, "p75": 75, "p90": 90, "max": 100}
    return {k: float(np.percentile(a, q)) for k, q in qs.items()} if a.size else {}


# ------------------------------------------------------------------- groups
def group_table(rows: list[dict], key) -> list[dict]:
    """Per-group count, wins, capital, P&L, ROI and price-implied win expectation."""
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    out = []
    for k, rs in g.items():
        car = sum((r["capital_at_risk"] for r in rs), ZERO)
        pnl = sum((r["realized_pnl"] for r in rs), ZERO)
        held = [r for r in rs if r["status"] == SETTLED]
        out.append(
            {
                "group": k,
                "n": len(rs),
                "wins": sum(r["realized_pnl"] > 0 for r in rs),
                "capital": car,
                "pnl": pnl,
                "fees": sum((r["fees"] for r in rs), ZERO),
                "roi": pnl / car if car else None,
                "held": len(held),
                # Decimal sums: built-in float sum() rounds differently on 3.11 and 3.12.
                "held_won": float(sum((r["settle_price"] for r in held), ZERO)),
                "held_implied": float(sum((r["entry_price"] for r in held), ZERO)),
            }
        )
    return sorted(out, key=lambda x: (-x["n"], str(x["group"])))


def price_bucket(p: Decimal) -> str:
    """
    >>> price_bucket(Decimal("0.05")), price_bucket(Decimal("0.5")), price_bucket(Decimal("1"))
    ('0.00-0.20', '0.40-0.60', '0.80-1.00')
    """
    edges = [Decimal("0.2"), Decimal("0.4"), Decimal("0.6"), Decimal("0.8")]
    labels = ["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"]
    for e, lab in zip(edges, labels, strict=False):
        if p < e:
            return lab
    return labels[-1]


def size_bucket(car: Decimal) -> str:
    for hi, lab in [
        (10, "< $10"),
        (25, "$10-25"),
        (50, "$25-50"),
        (100, "$50-100"),
        (250, "$100-250"),
    ]:
        if car < hi:
            return lab
    return ">= $250"


SIZE_ORDER = ["< $10", "$10-25", "$25-50", "$50-100", "$100-250", ">= $250"]


# ------------------------------------------------------------ cross-checks
def exchange_checks(
    positions: dict[tuple, Position], fills: list[dict], settlements: dict, markets: dict
) -> dict:
    """Compare the fill-derived ledger with the exchange's own position records."""
    from live.ledger import market_net_yes

    cost_ok = sum(abs(Decimal(f["cash_flow"])) == Decimal(f["exch_trade_cost"]) for f in fills)

    # Net YES-equivalent position at settlement vs the exchange's pre-settlement record.
    held = defaultdict(lambda: ZERO)
    for (slug, side), p in positions.items():
        if p.settle_price is not None:
            held[slug] += p.qty_at_settlement if side == "YES" else -p.qty_at_settlement
    qty_match = [s for s in settlements if held[s] == Decimal(settlements[s]["exch_net_before"])]

    # Gross volume in YES terms (buying NO = selling YES). A mismatch means the
    # exchange's position saw executions that the activity feed did not return.
    vol = defaultdict(lambda: [ZERO, ZERO])
    for f in fills:
        yes_buy = (f["side"] == "YES") == (f["action"] == "BUY")
        vol[f["market_slug"]][0 if yes_buy else 1] += Decimal(f["qty"])
    volume_gaps = [
        {
            "market_slug": slug,
            "exch_bought": Decimal(s["exch_qty_bought"]),
            "exch_sold": Decimal(s["exch_qty_sold"]),
            "feed_bought": vol[slug][0],
            "feed_sold": vol[slug][1],
            "exch_realized_before": Decimal(s["exch_realized_before"]),
        }
        for slug, s in settlements.items()
        if (Decimal(s["exch_qty_bought"]), Decimal(s["exch_qty_sold"])) != tuple(vol[slug])
    ]

    # Settlement P&L ex fees, where it is unambiguous: no sells in the market.
    # The exchange reports money to 4 dp and sometimes books the settlement gain
    # in its `cost` field instead of `realized`; both are classified, not hidden.
    sold = {slug for (slug, _), p in positions.items() if p.qty_sold > 0}
    pnl = {"comparable": 0, "exact": 0, "rounding_4dp": 0, "booked_in_cost_field": 0, "other": []}
    for slug, s in settlements.items():
        if slug in sold:
            continue
        pnl["comparable"] += 1
        mine = sum(
            (p.payout - p.buy_notional for (sl, _), p in positions.items() if sl == slug), ZERO
        )
        exch = Decimal(s["exch_realized_delta"])
        if mine == exch:
            pnl["exact"] += 1
        elif abs(mine - exch) < Decimal("0.0001"):
            pnl["rounding_4dp"] += 1
        elif exch == 0 and mine == Decimal(s["exch_cost_after"]):
            pnl["booked_in_cost_field"] += 1
        else:
            pnl["other"].append({"market_slug": slug, "ledger": mine, "exchange": exch})

    net = market_net_yes(positions)
    open_slugs = {slug for (slug, _), p in positions.items() if p.status == OPEN}
    snap = {s: Decimal(m["exch_open_net"]) for s, m in markets.items() if m["exch_open_net"]}
    open_match = open_slugs == set(snap) and all(net[s] == snap[s] for s in snap)
    return {
        "fills": len(fills),
        "fill_cost_match": cost_ok,
        "settled_markets": len(settlements),
        "settled_qty_match": len(qty_match),
        "volume_gaps": volume_gaps,
        "settlement_pnl": pnl,
        "open_positions": len(open_slugs),
        "exchange_open_positions": len(snap),
        "open_match": open_match,
    }


def status_counts(rows: list[dict]) -> dict:
    return {s: sum(r["status"] == s for r in rows) for s in (SETTLED, EXITED, OPEN)}


def win_rate(rows: list[dict]) -> dict:
    n = len(rows)
    w = sum(r["realized_pnl"] > 0 for r in rows)
    lo, hi = wilson_interval(w, n)
    return {"n": n, "wins": w, "rate": w / n if n else float("nan"), "lo": lo, "hi": hi}
