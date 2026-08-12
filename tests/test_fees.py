from __future__ import annotations

import math

import pytest

from pmcal import fees


@pytest.mark.parametrize(
    ("contracts", "price", "expected"),
    [
        (100, 0.50, 1.75),   # 0.07 * 100 * 0.25 = 1.75 exactly
        (100, 0.60, 1.68),   # 0.07 * 100 * 0.24
        (100, 0.40, 1.68),   # symmetric about 0.5
        (100, 0.95, 0.34),   # 0.3325 rounds UP to the cent
        (1, 0.63, 0.02),     # 0.016317 rounds UP, not to nearest
        (1, 0.50, 0.02),     # 0.0175 -> 0.02
        (10, 0.99, 0.01),    # 0.00693 -> 0.01, never free
    ],
)
def test_kalshi_fee_hand_computed(contracts, price, expected):
    assert fees.kalshi_taker_fee(contracts, price) == pytest.approx(expected)


def test_kalshi_fee_rounds_up_not_to_nearest():
    raw = 0.07 * 1 * 0.63 * 0.37
    assert raw < 0.0165  # would round DOWN to 0.02's neighbour under round-half
    assert fees.kalshi_taker_fee(1, 0.63) == 0.02


def test_kalshi_fee_peaks_at_a_coin_flip():
    at_half = fees.kalshi_taker_fee(1000, 0.50)
    assert at_half > fees.kalshi_taker_fee(1000, 0.30)
    assert at_half > fees.kalshi_taker_fee(1000, 0.70)
    assert at_half == max(fees.kalshi_taker_fee(1000, p / 100) for p in range(1, 100))


def test_kalshi_fee_is_zero_at_the_boundaries_and_for_no_trade():
    assert fees.kalshi_taker_fee(100, 0.0) == 0.0
    assert fees.kalshi_taker_fee(100, 1.0) == 0.0
    assert fees.kalshi_taker_fee(0, 0.5) == 0.0


@pytest.mark.parametrize(
    ("contracts", "price", "rate", "expected"),
    [
        (100, 0.50, 0.0075, 0.38),   # 0.375 rounds up
        (100, 0.50, 0.0100, 0.50),
        (100, 0.50, 0.0180, 0.90),
        (200, 0.25, 0.0100, 0.50),   # charged on notional paid, not contracts
    ],
)
def test_polymarket_fee_is_proportional_to_notional(contracts, price, rate, expected):
    assert fees.polymarket_taker_fee(contracts, price, rate) == pytest.approx(expected)


@pytest.mark.parametrize("bad_price", [-0.01, 1.01])
def test_invalid_price_rejected(bad_price):
    with pytest.raises(ValueError):
        fees.kalshi_taker_fee(10, bad_price)
    with pytest.raises(ValueError):
        fees.polymarket_taker_fee(10, bad_price, 0.01)


def test_implausible_polymarket_rate_rejected():
    with pytest.raises(ValueError):
        fees.polymarket_taker_fee(10, 0.5, 1.5)


def test_dispatch_matches_the_direct_calls():
    assert fees.taker_fee("kalshi", 100, 0.5) == fees.kalshi_taker_fee(100, 0.5)
    assert fees.taker_fee("polymarket", 100, 0.5, 0.012) == fees.polymarket_taker_fee(
        100, 0.5, 0.012
    )
    with pytest.raises(ValueError):
        fees.taker_fee("betfair", 100, 0.5)


def test_breakeven_edge_is_material_at_the_middle_of_the_book():
    # 1.75 probability points on a coin flip: wider than most cross-venue gaps.
    assert fees.breakeven_edge("kalshi", 0.5) == pytest.approx(0.0175)
    assert fees.breakeven_edge("kalshi", 0.9) == pytest.approx(0.0063)
    assert fees.breakeven_edge("polymarket", 0.5, 0.018) == pytest.approx(0.009)


def test_fees_are_always_a_whole_number_of_cents():
    for contracts in (1, 7, 33, 100):
        for price in (0.03, 0.17, 0.5, 0.81, 0.97):
            fee = fees.kalshi_taker_fee(contracts, price)
            assert math.isclose(fee * 100, round(fee * 100), abs_tol=1e-9)
