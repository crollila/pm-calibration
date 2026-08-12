from __future__ import annotations

import math

import pytest

from pmcal import odds


def test_hand_computed_american_conversions():
    # -110 is the standard US spread price: risk 110 to win 100.
    assert odds.american_to_decimal(-110) == pytest.approx(1 + 100 / 110)
    assert odds.american_to_implied(-110) == pytest.approx(110 / 210)
    # +150 pays 150 on a 100 stake.
    assert odds.american_to_decimal(150) == pytest.approx(2.5)
    assert odds.american_to_implied(150) == pytest.approx(0.4)
    # An even-money price is the hinge between the two branches.
    assert odds.american_to_decimal(100) == pytest.approx(2.0)
    assert odds.american_to_implied(-100) == pytest.approx(0.5)


@pytest.mark.parametrize("american", [-1000, -250, -110, -101, 100, 105, 150, 900])
def test_roundtrip(american):
    assert odds.decimal_to_american(odds.american_to_decimal(american)) == pytest.approx(american)
    assert odds.implied_to_american(odds.american_to_implied(american)) == pytest.approx(american)


def test_standard_two_way_overround_is_476_bps():
    both = [odds.american_to_implied(-110)] * 2
    assert odds.overround(both) == pytest.approx(0.047619, abs=1e-6)


@pytest.mark.parametrize("bad", [0, float("nan")])
def test_invalid_american_rejected(bad):
    with pytest.raises(ValueError):
        odds.american_to_decimal(bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.4])
def test_implied_to_american_rejects_impossible_probabilities(bad):
    with pytest.raises(ValueError):
        odds.implied_to_american(bad)


def test_decimal_below_one_rejected():
    with pytest.raises(ValueError):
        odds.decimal_to_american(0.9)


def test_cents_and_clipping():
    assert odds.cents_to_prob(63) == pytest.approx(0.63)
    assert 0.0 < odds.clip_prob(0.0) < 1e-5
    assert math.isfinite(math.log(odds.clip_prob(0.0)))
    assert odds.clip_prob(1.0) < 1.0
