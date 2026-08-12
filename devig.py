"""Vig removal: three methods, one signature, no favourite.

Each method maps a vector of *quoted* implied probabilities (which sum to the
overround, > 1) to fair probabilities summing to 1:

    method(implied: Sequence[float]) -> np.ndarray

They disagree most on longshots, which is exactly where the favourite-longshot
bias lives, so `analysis/calibration.py` reports all three rather than picking.

A result worth stating up front, because it changes how to read the comparison
--------------------------------------------------------------------------
**On a two-outcome market, Shin and additive de-vig are the same number.**

Shin's relation between fair `p_i` and quoted `pi_i` is
`pi_i = sqrt(Pi * p_i * ((1-z) p_i + z))`, where `Pi = sum(pi)`. Write
`A = sqrt(p1 q1)`, `B = sqrt(p2 q2)` with `q_i = (1-z) p_i + z`. Additive de-vig
shifts both outcomes by the same constant, so it is characterised by
`pi_1 - pi_2 = p_1 - p_2`. Combined with `A + B = sqrt(Pi)` that requires
`A^2 - B^2 = p_1 - p_2`, and

    p1*q1 - p2*q2 = (1-z)(p1^2 - p2^2) + z(p1 - p2)
                  = (p1 - p2)[(1-z)(p1 + p2) + z]
                  = (p1 - p2)          because p1 + p2 = 1

holds *identically in z*. So the additive solution always satisfies Shin's
system; `z` merely adjusts to fit the overround.

Consequence: for h2h (two-way) sports markets the only real comparison is
multiplicative vs. the other two. The three methods genuinely separate only on
three-or-more-way markets. `tests/test_devig.py` asserts both halves of this.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

Method = str
METHODS: tuple[Method, ...] = ("multiplicative", "additive", "shin")


def _validate(implied: Sequence[float]) -> np.ndarray:
    arr = np.asarray(implied, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("need at least two outcomes")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"implied probabilities must be positive and finite: {implied!r}")
    if arr.sum() <= 1.0:
        # An underround is arbitrage, not vig. Refuse rather than invent a fair price.
        raise ValueError(f"expected an overround, got sum={arr.sum():.6f}")
    return arr


def multiplicative(implied: Sequence[float]) -> np.ndarray:
    """Normalise so the probabilities sum to 1 (proportional vig split).

    Assumes the book applies vig *proportionally*, which understates the true
    margin on longshots -- the reason it reads as the most longshot-friendly
    of the three.

    >>> np.round(multiplicative([0.55, 0.50]), 6)
    array([0.52381, 0.47619])
    >>> np.round(multiplicative([0.60, 0.30, 0.15]), 6)
    array([0.571429, 0.285714, 0.142857])
    """
    arr = _validate(implied)
    return arr / arr.sum()


def additive(implied: Sequence[float]) -> np.ndarray:
    """Subtract the overround equally across outcomes (balanced-book split).

    Assumes the book takes the same absolute margin from every outcome, which
    strips proportionally *more* from longshots than from favourites.

    >>> np.round(additive([0.55, 0.50]), 6)
    array([0.525, 0.475])
    >>> np.round(additive([0.60, 0.30, 0.15]), 6)
    array([0.583333, 0.283333, 0.133333])
    """
    arr = _validate(implied)
    fair = arr - (arr.sum() - 1.0) / arr.size
    if np.any(fair <= 0.0):
        # Happens on heavy longshots in wide books; fall back rather than emit
        # a negative probability, and let the caller see it in the residual.
        return multiplicative(arr)
    return fair


def _shin_probs(arr: np.ndarray, z: float) -> np.ndarray:
    """Shin fair probabilities for a given insider fraction `z`.

    Uses the algebraically equivalent but numerically stable form

        p_i = 2 pi_i^2 / (Pi * (z + sqrt(z^2 + 4 (1-z) pi_i^2 / Pi)))

    which avoids the 1/(1-z) blow-up of the textbook expression as z -> 1.
    """
    total = arr.sum()
    inner = z * z + 4.0 * (1.0 - z) * arr * arr / total
    return 2.0 * arr * arr / (total * (z + np.sqrt(inner)))


def shin_z(implied: Sequence[float], tol: float = 1e-12) -> float:
    """Solve for Shin's insider-trading fraction `z` in [0, 1).

    `sum(p(z))` falls monotonically from `sqrt(Pi) > 1` at z=0 to
    `sum(pi^2)/Pi < 1` at z=1, so a root always exists and Brent's method is
    safe without a hand-rolled bracket search.

    >>> round(shin_z([0.55, 0.50]), 6)
    0.050006
    >>> round(shin_z([0.60, 0.30, 0.15]), 6)
    0.025265
    """
    arr = _validate(implied)

    def gap(z: float) -> float:
        return float(_shin_probs(arr, z).sum() - 1.0)

    return float(brentq(gap, 0.0, 1.0, xtol=tol))


def shin(implied: Sequence[float]) -> np.ndarray:
    """Shin (1993): back out the fair prices a book would set facing insiders.

    Two-way markets reproduce `additive` exactly (see the module docstring);
    three-way markets sit between `additive` and `multiplicative`.

    >>> np.round(shin([0.55, 0.50]), 6)
    array([0.525, 0.475])
    >>> np.round(shin([0.60, 0.30, 0.15]), 6)
    array([0.580262, 0.283863, 0.135875])
    """
    arr = _validate(implied)
    return _shin_probs(arr, shin_z(arr))


DISPATCH = {"multiplicative": multiplicative, "additive": additive, "shin": shin}


def devig(implied: Sequence[float], method: Method = "shin") -> np.ndarray:
    """Dispatch by name so callers can sweep all three from a config string.

    >>> np.round(devig([0.55, 0.50], "multiplicative"), 6)
    array([0.52381, 0.47619])
    """
    if method not in DISPATCH:
        raise ValueError(f"unknown de-vig method {method!r}; expected one of {METHODS}")
    return DISPATCH[method](implied)
