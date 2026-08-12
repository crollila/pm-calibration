"""Fetch settlement status from each venue and reconcile the disagreements.

Ground truth comes from Polymarket and Kalshi, which both publish resolution
status. Where the same event exists on both, the two are cross-checked and any
disagreement is written to `resolution_conflicts` rather than silently resolved
in favour of one venue -- a venue that settles a game differently from its
competitor is itself a finding, and quietly picking a winner would hide it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import requests

from pmcal import gamekeys
from pmcal.http import FetchError, get_json
from pmcal.util import parse_ts

log = logging.getLogger(__name__)

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"

RESOLVED_KALSHI_STATUS = {"settled", "finalized", "determined"}


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def polymarket_resolutions(
    session: requests.Session, market_ids: list[str], cfg: Any, now: datetime
) -> list[dict[str, Any]]:
    """Look up Gamma markets by id and emit one record per outcome.

    Fetched one at a time via `/markets/{id}`. The tempting bulk form,
    `/markets?id=a&id=b`, is a trap: it answers HTTP 200 with an empty list
    rather than an error, so a batched implementation silently resolves nothing
    and looks like a settlement drought. Misses are counted and logged here for
    exactly that reason.
    """
    records: list[dict[str, Any]] = []
    misses = 0
    for market_id in market_ids:
        try:
            payload = get_json(
                session, f"{GAMMA_URL}/{market_id}", timeout=cfg.request_timeout
            )
        except FetchError as exc:
            log.warning("polymarket resolution %s failed: %s", market_id, exc)
            misses += 1
            continue
        if not isinstance(payload, dict) or not payload.get("id"):
            misses += 1
            continue
        records.extend(_poly_records(payload, now))
    if misses:
        log.warning("polymarket: %d/%d market lookups returned nothing",
                    misses, len(market_ids))
    return records


def _poly_records(market: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    outcomes = [str(o) for o in _as_list(market.get("outcomes"))]
    prices = [_to_float(p) for p in _as_list(market.get("outcomePrices"))]
    closed = bool(market.get("closed"))
    market_id = str(market.get("id") or market.get("conditionId") or "")
    if not outcomes or not market_id:
        return []

    # Gamma reports a settled market as outcomePrices of exactly 1 and 0. A
    # closed-but-not-yet-settled market keeps interior prices; we leave those
    # unresolved rather than rounding a 0.98 into a win.
    settled = closed and len(prices) == len(outcomes) and all(
        p is not None and (p <= 1e-6 or p >= 1.0 - 1e-6) for p in prices
    )
    status = "settled" if settled else ("closed" if closed else "open")
    resolved_at = parse_ts(market.get("closedTime") or market.get("endDate")) if settled else None

    return [
        {
            "venue": "polymarket",
            "market_id": market_id,
            "outcome_raw": outcome,
            "resolved": (1.0 if prices[i] >= 0.5 else 0.0) if settled else None,
            "status": status,
            "resolved_at": resolved_at,
            "observed_at": now,
            "raw_json": json.dumps(market, default=str),
        }
        for i, outcome in enumerate(outcomes)
    ]


def kalshi_resolutions(
    session: requests.Session, tickers: list[str], cfg: Any, now: datetime
) -> list[dict[str, Any]]:
    """One GET per ticker; Kalshi has no bulk lookup for arbitrary tickers."""
    records: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            payload = get_json(
                session, f"{KALSHI_BASE}/markets/{ticker}", timeout=cfg.request_timeout
            )
        except FetchError as exc:
            log.warning("kalshi resolution %s failed: %s", ticker, exc)
            continue
        market = payload.get("market") if isinstance(payload, dict) else None
        if market:
            records.extend(_kalshi_records(ticker, market, now))
    return records


def _kalshi_records(ticker: str, market: dict[str, Any], now: datetime) -> list[dict[str, Any]]:
    status = str(market.get("status") or "").lower()
    result = str(market.get("result") or "").lower()
    settled = status in RESOLVED_KALSHI_STATUS and result in {"yes", "no"}
    resolved_at = parse_ts(market.get("expiration_time") or market.get("close_time"))
    yes_value = 1.0 if result == "yes" else 0.0
    labels = _kalshi_side_labels(ticker, market)

    return [
        {
            "venue": "kalshi",
            "market_id": f"{ticker}:{side}",
            "outcome_raw": labels[side],
            "resolved": (yes_value if side == "yes" else 1.0 - yes_value) if settled else None,
            "status": status or "unknown",
            "resolved_at": resolved_at if settled else None,
            "observed_at": now,
            "raw_json": json.dumps(market, default=str),
        }
        for side in ("yes", "no")
    ]


def _kalshi_side_labels(ticker: str, market: dict[str, Any]) -> dict[str, str]:
    """Which team each side of a Kalshi market refers to.

    `no_sub_title` repeats the YES team rather than naming the opponent, so a
    game ticker has to be parsed to learn who NO actually is. Getting this
    wrong would settle both outcomes of a game to the same team and quietly
    destroy every label in the panel.
    """
    parsed = gamekeys.parse_kalshi_ticker(ticker)
    if parsed:
        side_code = str(parsed["side"])
        pair = gamekeys.kalshi_game_codes({side_code}, str(parsed["blob"]), str(parsed["sport"]))
        other = next((c for c in pair if c != side_code), "no")
        return {"yes": side_code, "no": other}
    return {side: str(market.get(f"{side}_sub_title") or side) for side in ("yes", "no")}


def scores_resolutions(
    session: requests.Session, sport_key: str, cfg: Any, now: datetime, days_from: int = 3
) -> list[dict[str, Any]]:
    """Optional third opinion from The Odds API `/scores` (costs API credits)."""
    from pmcal.sources.oddsapi import fetch_scores

    records: list[dict[str, Any]] = []
    for event in fetch_scores(session, sport_key, cfg, days_from):
        if not event.get("completed"):
            continue
        scores = {str(s.get("name")): _to_float(s.get("score")) for s in event.get("scores") or []}
        if len(scores) != 2 or any(v is None for v in scores.values()):
            continue
        (name_a, score_a), (name_b, score_b) = scores.items()
        if score_a == score_b:
            continue  # a draw is not a two-way h2h resolution
        winner = name_a if score_a > score_b else name_b
        for name in (name_a, name_b):
            records.append(
                {
                    "venue": "oddsapi_scores",
                    "market_id": f"{event.get('id')}:h2h",
                    "outcome_raw": name,
                    "resolved": 1.0 if name == winner else 0.0,
                    "status": "completed",
                    "resolved_at": parse_ts(event.get("commence_time")),
                    "observed_at": now,
                    "raw_json": json.dumps(event, default=str),
                }
            )
    return records


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
