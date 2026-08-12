from __future__ import annotations

import numpy as np
import pytest

import devig
from pmcal.odds import american_to_implied

TWO_WAY = [0.55, 0.50]           # 5.0% overround, mild favourite
THREE_WAY = [0.60, 0.30, 0.15]   # 5.0% overround, genuine longshot
STANDARD = [american_to_implied(-110)] * 2


@pytest.mark.parametrize("method", devig.METHODS)
@pytest.mark.parametrize("quotes", [TWO_WAY, THREE_WAY, STANDARD, [0.90, 0.15]])
def test_every_method_returns_a_probability_vector(method, quotes):
    fair = devig.devig(quotes, method)
    assert fair.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(fair > 0.0) and np.all(fair < 1.0)
    assert len(fair) == len(quotes)


def test_multiplicative_hand_computed():
    # 0.55 / 1.05 and 0.50 / 1.05
    assert devig.multiplicative(TWO_WAY) == pytest.approx([0.55 / 1.05, 0.50 / 1.05])


def test_additive_hand_computed():
    # overround 0.05 split two ways = 0.025 off each side
    assert devig.additive(TWO_WAY) == pytest.approx([0.525, 0.475])
    # three ways = 0.016666... off each
    assert devig.additive(THREE_WAY) == pytest.approx(
        [0.60 - 0.05 / 3, 0.30 - 0.05 / 3, 0.15 - 0.05 / 3]
    )


def test_minus_110_both_ways_is_a_coin_flip_under_every_method():
    for method in devig.METHODS:
        assert devig.devig(STANDARD, method) == pytest.approx([0.5, 0.5], abs=1e-9)


@pytest.mark.parametrize("quotes", [TWO_WAY, [0.90, 0.15], [0.70, 0.36], STANDARD])
def test_shin_equals_additive_on_two_way_markets(quotes):
    """The identity proved in devig.py's module docstring, checked numerically."""
    assert devig.shin(quotes) == pytest.approx(devig.additive(quotes), abs=1e-9)


def test_shin_is_strictly_between_the_others_on_three_way_markets():
    mult = devig.multiplicative(THREE_WAY)
    add = devig.additive(THREE_WAY)
    shin = devig.shin(THREE_WAY)
    assert not np.allclose(shin, add)
    assert not np.allclose(shin, mult)
    longshot = -1  # the 0.15 quote
    assert add[longshot] < shin[longshot] < mult[longshot]
    favourite = 0
    assert mult[favourite] < shin[favourite] < add[favourite]


def test_methods_disagree_most_on_longshots():
    """The whole reason all three are reported rather than one being chosen."""
    quotes = [0.70, 0.20, 0.15]
    spread = [
        abs(devig.multiplicative(quotes)[i] - devig.additive(quotes)[i]) / quotes[i]
        for i in range(3)
    ]
    assert spread[2] > spread[0]  # relative disagreement is larger on the longshot


def test_shin_z_is_a_sensible_insider_fraction():
    z = devig.shin_z(THREE_WAY)
    assert 0.0 < z < 1.0
    # A wider book implies more insider trading.
    assert devig.shin_z([0.62, 0.31, 0.16]) > z


def test_shin_z_satisfies_its_own_defining_equation():
    """z is defined by sum(p(z)) = 1; check the root, not a remembered constant."""
    z = devig.shin_z(TWO_WAY)
    assert z == pytest.approx(0.050006, abs=1e-5)
    assert devig._shin_probs(np.asarray(TWO_WAY), z).sum() == pytest.approx(1.0, abs=1e-12)
    # And the root is unique: the sum is strictly decreasing in z.
    assert devig._shin_probs(np.asarray(TWO_WAY), z - 0.01).sum() > 1.0
    assert devig._shin_probs(np.asarray(TWO_WAY), z + 0.01).sum() < 1.0


def test_shin_is_numerically_stable_for_tiny_overrounds():
    fair = devig.shin([0.5000001, 0.5000001])
    assert fair == pytest.approx([0.5, 0.5], abs=1e-6)


@pytest.mark.parametrize("bad", [[0.5, 0.4], [0.5], [0.6, -0.1], [0.6, float("nan")]])
def test_invalid_input_rejected(bad):
    with pytest.raises(ValueError):
        devig.shin(bad)


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="unknown de-vig method"):
        devig.devig(TWO_WAY, "power")


def test_additive_falls_back_rather_than_emitting_negative_probabilities():
    # A wide three-way book where equal subtraction would push the longshot below zero.
    quotes = [0.80, 0.30, 0.02]
    fair = devig.additive(quotes)
    assert np.all(fair > 0.0)
    assert fair == pytest.approx(devig.multiplicative(quotes))
