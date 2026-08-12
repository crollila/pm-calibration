"""Kelly sizing for binary contracts, and why full Kelly is the wrong default.

A binary contract bought at `price` pays 1 on YES and 0 on NO. If the true
probability is `p`, the log-optimal bankroll fraction is

    f* = (p - price) / (1 - price)

Why full Kelly is wrong here -- stated carefully, because the usual argument is wrong
--------------------------------------------------------------------------------
The common claim is "p is uncertain, so expected log growth is maximised below
f*". That is false, and `expected_growth_is_linear_in_p` demonstrates it: log
growth is *affine* in p,

    g(f, p) = log(1 - f) + p * [log(1 + f*b) - log(1 - f)],    b = (1-price)/price

so averaging over any distribution of p just substitutes its mean. If all you
had was mean-zero noise on a single bet, full Kelly on the posterior mean would
be correct.

The two things that are actually true, and that this module computes:

1. **The error does not average out.** The same estimate is reused on every
   bet, so a biased `p_hat` is a permanent tilt, not noise. Because g is affine
   in the true p with slope `s(f) = log(1+f*b) - log(1-f)`, which grows with f,
   a larger stake linearly amplifies that permanent error. Penalising it -- any
   mean-variance or certainty-equivalent objective -- gives an interior optimum
   strictly below f*, and it is exactly full Kelly evaluated at a shrunk
   probability `p_hat - k*sigma`. Fractional Kelly *is* estimate shrinkage.

2. **The penalty is violently asymmetric.** Betting half of f* keeps ~75% of the
   growth rate. Betting twice f* keeps *none* of it -- growth goes to zero and
   then negative. When you are unsure which side of the peak you are on, the
   cheap side is the low one. `overbetting_asymmetry` prints the numbers.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(p: float, price: float, fraction: float = 1.0) -> float:
    """Fraction of bankroll to stake, scaled by `fraction`.

    Negative edges return 0: this study never shorts, because the mirror
    contract is a different instrument with its own spread and fee.

    >>> round(kelly_fraction(0.60, 0.50), 6)
    0.2
    >>> round(kelly_fraction(0.60, 0.50, fraction=0.25), 6)
    0.05
    >>> kelly_fraction(0.40, 0.50)
    0.0
    """
    if not 0.0 < price < 1.0:
        raise ValueError(f"price must be inside (0, 1), got {price!r}")
    edge = (p - price) / (1.0 - price)
    return max(0.0, edge * fraction)


def growth_rate(f: float, p: float, price: float) -> float:
    """Expected log growth per bet at stake fraction `f`.

    >>> round(growth_rate(0.2, 0.6, 0.5), 6)
    0.020136
    """
    if not 0.0 <= f < 1.0:
        return -np.inf
    odds = (1.0 - price) / price
    return float(p * np.log1p(f * odds) + (1.0 - p) * np.log1p(-f))


def growth_slope(f: float, price: float) -> float:
    """d(growth)/dp -- how hard a stake amplifies an error in the estimate.

    Rises monotonically with `f`, which is the whole reason a noisy estimate
    argues for a smaller stake.

    >>> round(growth_slope(0.1, 0.5), 6) < round(growth_slope(0.4, 0.5), 6)
    True
    """
    odds = (1.0 - price) / price
    return float(np.log1p(f * odds) - np.log1p(-f))


def expected_growth_is_linear_in_p(
    p_hat: float, price: float, sigma: float, f: float = 0.2, n_draws: int = 50_000, seed: int = 7
) -> dict[str, float]:
    """The negative result, computed rather than asserted.

    Averaging log growth over a distribution of true `p` returns the growth at
    the mean of that distribution, to Monte-Carlo error. Uncertainty alone does
    not move the expected-growth optimum.

    >>> out = expected_growth_is_linear_in_p(0.6, 0.5, sigma=0.08)
    >>> abs(out["mean_growth"] - out["growth_at_mean_p"]) < 5e-4
    True
    """
    draws = sample_true_p(p_hat, sigma, n_draws, seed)
    odds = (1.0 - price) / price
    realised = draws * np.log1p(f * odds) + (1.0 - draws) * np.log1p(-f)
    return {
        "f": f,
        "mean_growth": float(realised.mean()),
        "growth_at_mean_p": growth_rate(f, p_hat, price),
        "sd_growth": float(realised.std(ddof=1)),
    }


def sample_true_p(p_hat: float, sigma: float, n_draws: int = 20_000, seed: int = 7) -> np.ndarray:
    """Beta draws for the true probability with mean `p_hat` and sd `sigma`."""
    if sigma <= 0:
        return np.full(n_draws, p_hat)
    var = min(sigma**2, p_hat * (1.0 - p_hat) * 0.98)
    concentration = p_hat * (1.0 - p_hat) / var - 1.0
    rng = np.random.default_rng(seed)
    return rng.beta(p_hat * concentration, (1.0 - p_hat) * concentration, size=n_draws)


def optimal_fraction_under_uncertainty(
    p_hat: float, price: float, sigma: float, risk_aversion: float = 1.0, n_grid: int = 2001
) -> dict[str, float]:
    """Stake that maximises `mean growth - risk_aversion * sd(growth)`.

    Since realised growth is affine in the true p, its standard deviation is
    `sigma * growth_slope(f)`, so the objective is
    `g(f, p_hat) - k * sigma * s(f)`. Maximising it is algebraically identical
    to full Kelly on the shrunk probability `p_hat - k*sigma`, which is the
    cleanest way to say what a fractional Kelly actually buys you.

    >>> out = optimal_fraction_under_uncertainty(0.60, 0.50, sigma=0.08)
    >>> out["optimal"] < out["full_kelly"]
    True
    >>> round(out["equivalent_shrunk_p"], 4)
    0.52
    """
    full = kelly_fraction(p_hat, price)
    shrunk = p_hat - risk_aversion * sigma
    if sigma <= 0 or full <= 0:
        return {
            "sigma": sigma,
            "full_kelly": full,
            "optimal": full,
            "ratio": 1.0 if full > 0 else np.nan,
            "equivalent_shrunk_p": p_hat,
            "objective_at_full": growth_rate(full, p_hat, price),
            "objective_at_optimal": growth_rate(full, p_hat, price),
        }

    grid = np.linspace(0.0, 0.995, n_grid)
    objective = np.array(
        [growth_rate(f, p_hat, price) - risk_aversion * sigma * growth_slope(f, price)
         for f in grid]
    )
    best_idx = int(np.nanargmax(objective))
    best = float(grid[best_idx])
    return {
        "sigma": sigma,
        "full_kelly": full,
        "optimal": best,
        "ratio": best / full,
        # The identity: the risk-adjusted optimum is full Kelly on a shrunk p.
        "equivalent_shrunk_p": shrunk,
        "closed_form": kelly_fraction(shrunk, price),
        "objective_at_full": float(
            growth_rate(full, p_hat, price) - risk_aversion * sigma * growth_slope(full, price)
        ),
        "objective_at_optimal": float(objective[best_idx]),
    }


def overbetting_asymmetry(p: float, price: float) -> dict[str, float]:
    """Growth retained at multiples of full Kelly. Half keeps most; double keeps none.

    In the small-edge limit the retained fraction is `lambda * (2 - lambda)`:
    0.75 at half Kelly, 1.00 at full, 0.00 at double.

    >>> out = overbetting_asymmetry(0.60, 0.50)
    >>> 0.70 < out["retained_at_0.5x"] < 0.80
    True
    >>> out["retained_at_2.0x"] < 0.01
    True
    """
    full = kelly_fraction(p, price)
    best = growth_rate(full, p, price)
    out = {"full_kelly": full, "growth_at_full": best}
    for multiple in (0.25, 0.5, 1.0, 1.5, 2.0):
        g = growth_rate(min(full * multiple, 0.999), p, price)
        out[f"retained_at_{multiple}x"] = g / best if best > 0 else np.nan
    return out


def uncertainty_curve(
    p_hat: float, price: float, sigmas: np.ndarray | None = None, risk_aversion: float = 1.0
) -> list[dict[str, float]]:
    """Optimal-to-full Kelly ratio as estimate noise grows. Feeds the writeup plot."""
    sigmas = np.linspace(0.0, 0.09, 10) if sigmas is None else sigmas
    return [
        optimal_fraction_under_uncertainty(p_hat, price, float(s), risk_aversion) for s in sigmas
    ]
