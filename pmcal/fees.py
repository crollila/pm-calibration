"""Venue fee models. Explicit, testable, and applied to every simulated fill.

Fees are the single most common way a backtested prediction-market edge
disappears. The Kalshi schedule in particular is *not* proportional to notional:
it peaks at p = 0.5 and vanishes at the extremes, so a 2-cent edge on a coin-flip
market is roughly a break-even trade before any spread cost at all.
"""

from __future__ import annotations

import math

KALSHI_FEE_RATE = 0.07


def _ceil_cents(dollars: float) -> float:
    """Round up to the next whole cent, the way an exchange bills it.

    >>> _ceil_cents(0.0163)
    0.02
    >>> _ceil_cents(1.75)
    1.75
    """
    return math.ceil(round(dollars * 100.0, 9)) / 100.0


def kalshi_taker_fee(contracts: float, price: float, rate: float = KALSHI_FEE_RATE) -> float:
    """`ceil(0.07 * C * P * (1 - P))`, rounded up to the cent.

    Maximum bite is at p = 0.5 (1.75 cents per contract) and it falls away
    toward the extremes.

    >>> kalshi_taker_fee(100, 0.50)
    1.75
    >>> kalshi_taker_fee(100, 0.60)
    1.68
    >>> kalshi_taker_fee(1, 0.63)
    0.02
    >>> kalshi_taker_fee(100, 0.95)
    0.34
    >>> kalshi_taker_fee(0, 0.5)
    0.0
    """
    if contracts <= 0:
        return 0.0
    if not 0.0 <= price <= 1.0:
        raise ValueError(f"price must be in [0, 1], got {price!r}")
    return _ceil_cents(rate * contracts * price * (1.0 - price))


def polymarket_taker_fee(contracts: float, price: float, rate: float) -> float:
    """Proportional taker fee on notional traded.

    Polymarket's taker fee is category-tiered (roughly 0.75%-1.80%), so `rate`
    is a configuration parameter rather than a constant, and the backtest
    sweeps it. Charged on the cash actually paid, `contracts * price`.

    >>> round(polymarket_taker_fee(100, 0.50, 0.01), 6)
    0.5
    >>> round(polymarket_taker_fee(100, 0.50, 0.018), 6)
    0.9
    >>> polymarket_taker_fee(0, 0.5, 0.01)
    0.0
    """
    if contracts <= 0:
        return 0.0
    if not 0.0 <= price <= 1.0:
        raise ValueError(f"price must be in [0, 1], got {price!r}")
    if not 0.0 <= rate < 0.5:
        raise ValueError(f"implausible fee rate {rate!r}")
    return _ceil_cents(rate * contracts * price)


def taker_fee(venue: str, contracts: float, price: float, poly_rate: float = 0.01) -> float:
    """Dispatch by venue so the backtest never hard-codes a schedule.

    >>> taker_fee("kalshi", 100, 0.5)
    1.75
    >>> taker_fee("polymarket", 100, 0.5, poly_rate=0.01)
    0.5
    """
    if venue == "kalshi":
        return kalshi_taker_fee(contracts, price)
    if venue == "polymarket":
        return polymarket_taker_fee(contracts, price, poly_rate)
    raise ValueError(f"no fee model for venue {venue!r}")


def breakeven_edge(venue: str, price: float, poly_rate: float = 0.01) -> float:
    """Edge in probability points a single contract must clear to pay its fee.

    A coin-flip Kalshi market costs 1.75 probability points in fees alone --
    wider than most of the cross-venue divergences this study finds.

    >>> round(breakeven_edge("kalshi", 0.5), 4)
    0.0175
    >>> round(breakeven_edge("kalshi", 0.9), 4)
    0.0063
    >>> round(breakeven_edge("polymarket", 0.5, 0.018), 4)
    0.009
    """
    return taker_fee(venue, 1000.0, price, poly_rate) / 1000.0
