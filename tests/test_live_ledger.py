"""P&L, fee and settlement arithmetic of the live-trading ledger.

Every fill in this file is SYNTHETIC: hand-made numbers chosen so the right
answer can be worked out on paper. None of them is a real trade.
"""

from __future__ import annotations

import random
from decimal import Decimal

import pytest

from live.ledger import (
    EXITED,
    OPEN,
    SETTLED,
    build_positions,
    fill_cash_flow,
    market_net_yes,
    side_price,
)

D = Decimal

_seq = iter(range(1, 10_000))


def fill(slug, side, action, qty, px_side, fee, ts, order="O-1"):
    """One synthetic fill row in the sanitized-CSV shape (all strings)."""
    return {
        "position_id": f"P-{slug}-{side}",
        "order_id": order,
        "market_slug": slug,
        "side": side,
        "action": action,
        "qty": str(qty),
        "px_side": str(px_side),
        "fee": str(fee),
        "ts_utc": f"2026-01-01T{ts}:00.000000Z",
        "seq": next(_seq),
        "order_type": "LIMIT",
    }


def settle(yes, ts="23:00"):
    return {
        "yes_price": str(yes),
        "no_price": str(1 - D(str(yes))),
        "settled_ts": f"2026-01-01T{ts}:00.000000Z",
    }


# ------------------------------------------------------------- primitives
def test_no_price_is_complement_of_yes_price():
    assert side_price(D("0.70"), "NO") == D("0.30")
    assert side_price(D("0.70"), "YES") == D("0.70")
    with pytest.raises(ValueError):
        side_price(D("0.5"), "MAYBE")


def test_buy_cash_includes_fee_and_sell_cash_nets_it():
    assert fill_cash_flow("BUY", D(10), D("0.40"), D("0.05")) == D("-4.05")
    assert fill_cash_flow("SELL", D(10), D("0.40"), D("0.05")) == D("3.95")


@pytest.mark.parametrize(
    "args",
    [("BUY", D(0), D("0.5"), D(0)), ("BUY", D(1), D("1.2"), D(0)), ("BUY", D(1), D("0.5"), D(-1)),
     ("HOLD", D(1), D("0.5"), D(0))],
)  # fmt: skip
def test_invalid_fills_are_rejected(args):
    with pytest.raises(ValueError):
        fill_cash_flow(*args)


# ------------------------------------------------------------- positions
def test_yes_held_to_a_win():
    pos = build_positions([fill("m", "YES", "BUY", 10, "0.40", "0.05", "10:00")], {"m": settle(1)})
    p = pos[("m", "YES")]
    assert p.status == SETTLED
    assert p.capital_at_risk == D("4.05")
    assert p.payout == D(10)
    assert p.realized_pnl == D("5.95")  # 10 - 4.00 - 0.05
    assert p.entry_price == D("0.40")


def test_no_held_to_a_loss_loses_exactly_the_capital():
    # NO bought at YES-price 0.70 -> 0.30 per NO share; YES wins, NO pays 0.
    pos = build_positions([fill("m", "NO", "BUY", 10, "0.30", "0.02", "10:00")], {"m": settle(1)})
    p = pos[("m", "NO")]
    assert p.settle_price == D(0)
    assert p.realized_pnl == D("-3.02") == -p.capital_at_risk


def test_partial_exit_then_settlement():
    fills = [
        fill("m", "YES", "BUY", 10, "0.50", "0.10", "10:00"),
        fill("m", "YES", "SELL", 4, "0.60", "0.05", "11:00"),
    ]
    p = build_positions(fills, {"m": settle(0)})[("m", "YES")]
    assert p.qty_at_settlement == D(6)
    assert p.status == SETTLED
    assert p.fees == D("0.15")
    assert p.capital_at_risk == D("5.10")  # buys + buy fees only
    assert p.realized_pnl == D("-2.75")  # 2.40 - 5.00 - 0.15 + 0
    assert p.exit_price == D("0.60")


def test_full_exit_before_settlement_is_realized_without_a_payout():
    fills = [
        fill("m", "YES", "BUY", 10, "0.50", "0.10", "10:00"),
        fill("m", "YES", "SELL", 10, "0.45", "0.10", "11:00"),
    ]
    p = build_positions(fills, {})[("m", "YES")]
    assert p.status == EXITED
    assert p.payout == 0
    assert p.realized_pnl == D("-0.70")  # 4.50 - 5.00 - 0.20
    assert p.close_ts == "2026-01-01T11:00:00.000000Z"


def test_exited_position_ignores_later_settlement():
    fills = [
        fill("m", "YES", "BUY", 5, "0.50", "0", "10:00"),
        fill("m", "YES", "SELL", 5, "0.70", "0", "11:00"),
    ]
    p = build_positions(fills, {"m": settle(1)})[("m", "YES")]
    assert p.status == EXITED
    assert p.realized_pnl == D("1.00")


def test_open_position_has_no_realized_pnl():
    p = build_positions([fill("m", "YES", "BUY", 3, "0.20", "0.01", "10:00")], {})[("m", "YES")]
    assert p.status == OPEN
    assert p.realized_pnl is None
    assert p.close_ts is None


def test_void_settles_at_half():
    p = build_positions([fill("m", "YES", "BUY", 10, "0.40", "0", "10:00")], {"m": settle("0.5")})
    assert p[("m", "YES")].realized_pnl == D("1.00")


def test_vwap_entry_across_fills():
    fills = [
        fill("m", "YES", "BUY", 10, "0.40", "0", "10:00", "O-1"),
        fill("m", "YES", "BUY", 30, "0.60", "0", "10:01", "O-2"),
    ]
    p = build_positions(fills, {"m": settle(1)})[("m", "YES")]
    assert p.entry_price == D("0.55")
    assert p.fills == 2 and len(p.orders) == 2
    assert p.first_ts == "2026-01-01T10:00:00.000000Z"


def test_both_sides_of_one_market_are_separate_positions():
    fills = [
        fill("m", "YES", "BUY", 10, "0.40", "0", "10:00"),
        fill("m", "YES", "SELL", 10, "0.30", "0", "11:00"),
        fill("m", "NO", "BUY", 5, "0.70", "0", "12:00"),
    ]
    pos = build_positions(fills, {"m": settle(0)})
    assert pos[("m", "YES")].realized_pnl == D("-1.00")
    assert pos[("m", "NO")].realized_pnl == D("1.50")  # 5 * (1 - 0.70)
    assert market_net_yes(pos)["m"] == D(-5)


def test_selling_more_than_held_is_corrupt_input():
    with pytest.raises(ValueError, match="sold more"):
        build_positions([fill("m", "YES", "SELL", 1, "0.5", "0", "10:00")], {})


def test_fill_after_settlement_is_corrupt_input():
    with pytest.raises(ValueError, match="after settlement"):
        build_positions(
            [fill("m", "YES", "BUY", 1, "0.5", "0", "23:30")], {"m": settle(1, "23:00")}
        )


def test_input_order_does_not_matter():
    fills = [
        fill("m", "YES", "BUY", 10, "0.50", "0.10", "10:00"),
        fill("m", "YES", "SELL", 4, "0.60", "0.05", "11:00"),
    ]
    a = build_positions(fills, {"m": settle(1)})[("m", "YES")].realized_pnl
    b = build_positions(fills[::-1], {"m": settle(1)})[("m", "YES")].realized_pnl
    assert a == b


def test_ledger_partitions_cash_exactly_on_random_books():
    """Sum of fill cash + payouts == realized P&L + net cash still in open positions."""
    rng = random.Random(7)
    for _ in range(50):
        fills, held = [], {}
        settlements = {}
        for m in range(8):
            slug = f"m{m}"
            for k in range(rng.randint(1, 6)):
                side = rng.choice(["YES", "NO"])
                key = (slug, side)
                qty = D(rng.randint(1, 500)) / 100
                if held.get(key, 0) >= qty and rng.random() < 0.4:
                    action, held[key] = "SELL", held[key] - qty
                else:
                    action, held[key] = "BUY", held.get(key, 0) + qty
                px = D(rng.randint(1, 99)) / 100
                fee = D(rng.randint(0, 50)) / 1000
                fills.append(fill(slug, side, action, qty, px, fee, f"1{k}:{m:02d}"))
            if rng.random() < 0.75:
                settlements[slug] = settle(rng.choice(["0", "1", "0.5"]))
        pos = build_positions(fills, settlements)
        cash = sum(
            fill_cash_flow(f["action"], D(f["qty"]), D(f["px_side"]), D(f["fee"])) for f in fills
        )
        payouts = sum(p.payout for p in pos.values())
        realized = sum(p.realized_pnl for p in pos.values() if p.status != OPEN)
        open_cash = sum(
            fill_cash_flow(f["action"], D(f["qty"]), D(f["px_side"]), D(f["fee"]))
            for f in fills
            if pos[(f["market_slug"], f["side"])].status == OPEN
        )
        assert cash + payouts == realized + open_cash
