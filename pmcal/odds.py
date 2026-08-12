"""Odds conversions.

All probabilities in this repo are floats in [0, 1]. Sportsbook quotes arrive as
American odds; prediction-market quotes arrive as prices that already *are*
probabilities. These helpers are the only place the two representations meet.
"""

from __future__ import annotations

import math


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal (European) odds.

    >>> round(american_to_decimal(-110), 6)
    1.909091
    >>> round(american_to_decimal(150), 6)
    2.5
    """
    if american == 0 or math.isnan(american):
        raise ValueError(f"invalid American odds: {american!r}")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds back to American odds.

    >>> round(decimal_to_american(2.5), 6)
    150.0
    >>> round(decimal_to_american(american_to_decimal(-110)), 6)
    -110.0
    """
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {decimal!r}")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def american_to_implied(american: float) -> float:
    """Implied probability of American odds, *including* the vig.

    A -110/-110 two-way market implies 0.52381 on each side; the two sum to
    1.04762, and that 4.76% overround is what `devig` strips out.

    >>> round(american_to_implied(-110), 6)
    0.52381
    >>> round(american_to_implied(150), 6)
    0.4
    """
    return 1.0 / american_to_decimal(american)


def implied_to_american(prob: float) -> float:
    """Inverse of `american_to_implied`.

    >>> round(implied_to_american(0.4), 6)
    150.0
    >>> round(implied_to_american(american_to_implied(-110)), 6)
    -110.0
    """
    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability must be strictly inside (0, 1), got {prob!r}")
    return decimal_to_american(1.0 / prob)


def overround(implied: list[float]) -> float:
    """Book overround (vig) as a fraction: sum of implied probabilities minus 1.

    >>> round(overround([american_to_implied(-110), american_to_implied(-110)]), 6)
    0.047619
    """
    return sum(implied) - 1.0


def cents_to_prob(cents: float) -> float:
    """Kalshi quotes in whole cents (1-99) map linearly to probability.

    >>> cents_to_prob(63)
    0.63
    """
    return cents / 100.0


def clip_prob(p: float, eps: float = 1e-6) -> float:
    """Keep probabilities strictly inside (0, 1) so log loss stays finite."""
    return min(max(p, eps), 1.0 - eps)
