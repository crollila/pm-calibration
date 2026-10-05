"""Exact position accounting from sanitized fills and settlements.

Money is `Decimal` end to end; floats appear only later, in the statistics.
Conventions, each verified against the raw Polymarket US records (see
LIVE_TRADING_CASE_STUDY.md, "Ledger conventions"):

- Every market is binary. `px_yes` is the YES price of the fill even when the
  order was on the NO side, so a NO share costs `1 - px_yes`.
- A buy costs `qty * px_side + fee`; a sell returns `qty * px_side - fee`.
  The exchange's own `trade.cost` field equals the buy identity on every fill.
- At settlement each held share pays its side's settlement price (1, 0, or a
  split such as 0.5 on a void).
- A position is one (market, side). Buying NO in a market where YES was held
  earlier is a different bet and a different position.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

ZERO = Decimal(0)
ONE = Decimal(1)

# Statuses. A position is realized only once nothing is left to settle.
SETTLED = "settled"  # shares still held when the market resolved
EXITED = "exited"  # fully sold before resolution
OPEN = "open"  # shares held, no resolution in the data


def side_price(px_yes: Decimal, side: str) -> Decimal:
    """Price per share of the side actually traded.

    >>> side_price(Decimal("0.35"), "NO")
    Decimal('0.65')
    >>> side_price(Decimal("0.35"), "YES")
    Decimal('0.35')
    """
    if side == "YES":
        return px_yes
    if side == "NO":
        return ONE - px_yes
    raise ValueError(f"unknown side {side!r}")


def fill_cash_flow(action: str, qty: Decimal, px_side: Decimal, fee: Decimal) -> Decimal:
    """Signed cash effect of one fill on the account (negative = cash out).

    >>> fill_cash_flow("BUY", Decimal("26.03"), Decimal("0.368"), Decimal("0.42"))
    Decimal('-9.99904')
    >>> fill_cash_flow("SELL", Decimal("10"), Decimal("0.5"), Decimal("0.1"))
    Decimal('4.9')
    """
    if qty <= 0 or fee < 0 or not ZERO <= px_side <= ONE:
        raise ValueError(f"bad fill qty={qty} px={px_side} fee={fee}")
    notional = qty * px_side
    if action == "BUY":
        return -(notional + fee)
    if action == "SELL":
        return notional - fee
    raise ValueError(f"unknown action {action!r}")


@dataclass
class Position:
    position_id: str
    market_slug: str
    side: str
    fills: int = 0
    orders: set = field(default_factory=set)
    qty_bought: Decimal = ZERO
    qty_sold: Decimal = ZERO
    buy_notional: Decimal = ZERO
    sell_notional: Decimal = ZERO
    fees: Decimal = ZERO
    buy_fees: Decimal = ZERO
    first_ts: str | None = None
    last_fill_ts: str | None = None
    order_types: set = field(default_factory=set)
    settle_price: Decimal | None = None  # price per share of *this side*
    settled_ts: str | None = None
    qty_at_settlement: Decimal = ZERO

    @property
    def inventory(self) -> Decimal:
        return self.qty_bought - self.qty_sold

    @property
    def capital_at_risk(self) -> Decimal:
        """Cash committed to the position: every buy plus its fee."""
        return self.buy_notional + self.buy_fees

    @property
    def payout(self) -> Decimal:
        if self.settle_price is None:
            return ZERO
        return self.qty_at_settlement * self.settle_price

    @property
    def status(self) -> str:
        if self.qty_at_settlement > 0 and self.settle_price is not None:
            return SETTLED
        if self.inventory == 0:
            return EXITED
        return OPEN

    @property
    def realized_pnl(self) -> Decimal | None:
        """Cash in minus cash out, only once the position is fully closed."""
        if self.status == OPEN:
            return None
        return self.sell_notional - self.buy_notional - self.fees + self.payout

    @property
    def entry_price(self) -> Decimal | None:
        """Volume-weighted buy price of this side, before fees."""
        return self.buy_notional / self.qty_bought if self.qty_bought else None

    @property
    def exit_price(self) -> Decimal | None:
        """Volume-weighted sell price, before fees (None if never sold)."""
        return self.sell_notional / self.qty_sold if self.qty_sold else None

    @property
    def close_ts(self) -> str | None:
        if self.status == SETTLED:
            return self.settled_ts
        if self.status == EXITED:
            return self.last_fill_ts
        return None


def build_positions(fills: list[dict], settlements: dict[str, dict]) -> dict[tuple, Position]:
    """Fold fills (any order) and settlements into positions keyed by (slug, side).

    `fills` rows need: position_id, order_id, market_slug, side, action, qty,
    px_side, fee, ts_utc, seq, order_type. `settlements` maps slug ->
    {yes_price, no_price, settled_ts}. Settlement only pays shares still held at
    the settlement time, so a sell after resolution is rejected as corrupt input.
    """
    positions: dict[tuple, Position] = {}
    for f in sorted(fills, key=lambda r: (r["ts_utc"], int(r["seq"]))):
        key = (f["market_slug"], f["side"])
        p = positions.get(key)
        if p is None:
            p = positions[key] = Position(f["position_id"], f["market_slug"], f["side"])
        elif p.position_id != f["position_id"]:
            raise ValueError(f"position id drift on {key}")
        qty, px, fee = Decimal(f["qty"]), Decimal(f["px_side"]), Decimal(f["fee"])
        fill_cash_flow(f["action"], qty, px, fee)  # validates the row
        p.fills += 1
        p.orders.add(f["order_id"])
        p.order_types.add(f["order_type"])
        p.fees += fee
        if f["action"] == "BUY":
            p.qty_bought += qty
            p.buy_notional += qty * px
            p.buy_fees += fee
            if p.first_ts is None:
                p.first_ts = f["ts_utc"]
        else:
            p.qty_sold += qty
            p.sell_notional += qty * px
        if p.inventory < 0:
            raise ValueError(f"{key}: sold more than bought at {f['ts_utc']}")
        p.last_fill_ts = f["ts_utc"]

    for (slug, side), p in positions.items():
        s = settlements.get(slug)
        if s is None:
            continue
        if p.last_fill_ts and p.last_fill_ts > s["settled_ts"]:
            raise ValueError(f"{slug}: fill after settlement")
        p.settle_price = Decimal(s["yes_price"] if side == "YES" else s["no_price"])
        p.settled_ts = s["settled_ts"]
        p.qty_at_settlement = p.inventory
    return positions


def market_net_yes(positions: dict[tuple, Position]) -> dict[str, Decimal]:
    """Net YES-equivalent shares per market (YES held minus NO held).

    This is how the exchange reports a position, so it is the quantity the
    reconciliation compares against the exchange's own pre-settlement record.
    """
    net: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for (slug, side), p in positions.items():
        net[slug] += p.inventory if side == "YES" else -p.inventory
    return dict(net)
