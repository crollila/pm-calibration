"""Synthetic snapshots with a known data-generating process.

Tests need data whose right answer is known in advance. Each event has a true
probability; the book quotes it with a configurable bias plus vig, the
prediction markets quote it with noise and a spread, and outcomes are drawn from
the true probability. That lets a test assert calibration properties rather than
merely that the code runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BOOKS = ("pinnacle", "draftkings", "fanduel")


def make_snapshots(
    n_events: int = 40,
    n_ts: int = 4,
    seed: int = 11,
    vig: float = 0.045,
    pm_noise: float = 0.01,
    pm_spread: float = 0.02,
    pm_bias: float = 0.0,
    start: datetime = datetime(2026, 9, 12, 12, 0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (snapshots, labels) in the shapes `build_panel` expects.

    `pm_bias` misprices Polymarket against the truth in a way that still sums to
    one across outcomes: the home side is shifted by `+pm_bias` and the away
    side by `-pm_bias`. Use it to make the backtest fire deterministically.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    labels: list[dict] = []

    for e in range(n_events):
        p_true = float(rng.uniform(0.2, 0.8))
        kickoff = start + timedelta(hours=24 + e)
        event_key = f"nfl|2026-09-{12 + e % 10:02d}|chiefs~ravens{e}"
        outcomes = {"home": p_true, "away": 1.0 - p_true}
        winner = "home" if rng.uniform() < p_true else "away"
        for outcome in outcomes:
            labels.append(
                {
                    "event_key": event_key,
                    "outcome": outcome,
                    "y": 1.0 if outcome == winner else 0.0,
                    "y_agreement": 1.0,
                }
            )

        for t in range(n_ts):
            ts = kickoff - timedelta(hours=12 * (n_ts - t))
            for outcome, p in outcomes.items():
                quoted = p * (1.0 + vig)
                for book in BOOKS:
                    jitter = float(rng.normal(0.0, 0.004))
                    rows.append(
                        _row("sportsbook", book, f"{e}:h2h", outcome, event_key, ts, kickoff,
                             bid=None, ask=min(max(quoted + jitter, 0.01), 0.99),
                             mid=None, bid_size=None, ask_size=None)
                    )
                bias = pm_bias if outcome == "home" else -pm_bias
                pm_mid = float(np.clip(p + bias + rng.normal(0.0, pm_noise), 0.02, 0.98))
                rows.append(
                    _row("polymarket", "", f"pm{e}", outcome, event_key, ts, kickoff,
                         bid=pm_mid - pm_spread / 2, ask=pm_mid + pm_spread / 2, mid=pm_mid,
                         bid_size=500.0, ask_size=500.0)
                )
                k_mid = float(np.clip(p + rng.normal(0.0, pm_noise), 0.02, 0.98))
                rows.append(
                    _row("kalshi", "", f"K{e}:yes", outcome, event_key, ts, kickoff,
                         bid=k_mid - 0.01, ask=k_mid + 0.01, mid=k_mid,
                         bid_size=200.0, ask_size=200.0)
                )

    return pd.DataFrame(rows), pd.DataFrame(labels)


def _row(venue, book, market_id, outcome, event_key, ts, kickoff, **quote) -> dict:
    return {
        "ts": ts,
        "venue": venue,
        "book": book,
        "market_id": market_id,
        "outcome": outcome,
        "event_key": event_key,
        "sport": "nfl",
        "title": f"{event_key} {outcome}",
        "event_start_time": kickoff,
        "volume": 1000.0,
        **quote,
    }
