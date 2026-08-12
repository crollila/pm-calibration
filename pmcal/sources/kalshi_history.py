"""Kalshi historical candlesticks for settled game markets.

Kalshi's candlesticks are richer than Polymarket's price history: each bar
carries `yes_bid` and `yes_ask` as well as a traded price, so backfilled Kalshi
rows have a genuine spread rather than a midpoint alone.

Two constraints discovered by probing, both enforced here:

* `period_interval` accepts only 1, 60 and 1440 minutes. 5 and 15 are rejected.
* A request is capped at 5000 candles, so minute bars need a short window.

A candle whose period ends at E summarises `(E - interval, E]`, so its close is
the market state *at* E. Using `end_period_ts` as the snapshot timestamp is
therefore as-of correct and needs no further alignment.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import requests

from pmcal import gamekeys
from pmcal.http import get_json
from pmcal.util import utcnow

BASE = "https://api.elections.kalshi.com/trade-api/v2"
VENUE = "kalshi"
VALID_INTERVALS = (1, 60, 1440)
MAX_CANDLES = 5000

log = logging.getLogger(__name__)


def fetch_settled_markets(
    session: requests.Session, series_ticker: str, min_close_ts: int, max_close_ts: int,
    timeout: float = 30.0, max_pages: int = 60,
) -> list[dict[str, Any]]:
    """Every settled market in one series over a close-time window."""
    markets: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "series_ticker": series_ticker, "status": "settled", "limit": 200,
            "min_close_ts": min_close_ts, "max_close_ts": max_close_ts,
        }
        if cursor:
            params["cursor"] = cursor
        payload = get_json(session, f"{BASE}/markets", params=params, timeout=timeout)
        batch = (payload or {}).get("markets") or []
        markets.extend(batch)
        cursor = (payload or {}).get("cursor") or None
        if not cursor or not batch:
            break
    return markets


def fetch_candlesticks(
    session: requests.Session, series_ticker: str, ticker: str,
    start_ts: int, end_ts: int, interval: int, timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Candles for one market. `interval` must be one Kalshi accepts."""
    if interval not in VALID_INTERVALS:
        raise ValueError(f"period_interval must be one of {VALID_INTERVALS}, got {interval}")
    span = max(end_ts - start_ts, 0) // 60
    if span / interval > MAX_CANDLES:
        # Trim the *older* end: the run-up to kickoff is what the study needs.
        start_ts = end_ts - MAX_CANDLES * interval * 60
    payload = get_json(
        session, f"{BASE}/series/{series_ticker}/markets/{ticker}/candlesticks",
        params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": interval},
        timeout=timeout,
    )
    return (payload or {}).get("candlesticks") or []


def _dollars(node: Any, *keys: str) -> float | None:
    """Pull the first present `*_dollars` value from a candle sub-object.

    Candles for illiquid periods omit `close_dollars` and carry only
    `previous_dollars`; treating that as missing would throw away most of the
    pre-game history on thin markets.
    """
    if not isinstance(node, dict):
        return None
    for key in keys:
        value = node.get(key)
        if value in (None, ""):
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if 0.0 < price < 1.0:
            return price
    return None


def candle_rows(
    market: dict[str, Any],
    candles: list[dict[str, Any]],
    event_key: str,
    sport: str,
    codes: tuple[str, ...],
    start: datetime,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Snapshot rows for both sides of one Kalshi market from its candles.

    A game ticker is one team's market: YES is that team and NO is the
    opponent, priced as the complement. Both sides are emitted so the panel
    sees a full two-outcome market from a single ticker.
    """
    now = now or utcnow()
    ticker = str(market.get("ticker") or "")
    parsed = gamekeys.parse_kalshi_ticker(ticker)
    yes_code = str(parsed["side"]) if parsed else "yes"
    no_code = next((c for c in codes if c != yes_code), "no")
    volume = _to_float(market.get("volume"))

    rows: list[dict[str, Any]] = []
    for candle in candles:
        ts = candle.get("end_period_ts")
        if not ts:
            continue
        moment = datetime.fromtimestamp(int(ts), UTC).replace(tzinfo=None)
        bid = _dollars(candle.get("yes_bid"), "close_dollars", "open_dollars")
        ask = _dollars(candle.get("yes_ask"), "close_dollars", "open_dollars")
        last = _dollars(candle.get("price"), "close_dollars", "mean_dollars", "previous_dollars")
        mid = (bid + ask) / 2.0 if bid is not None and ask is not None else last
        if mid is None:
            continue
        raw = json.dumps({"ticker": ticker, "candle": candle}, default=str)
        rows.append(_row(ticker, "yes", yes_code, bid, ask, mid, event_key, sport,
                         market, moment, start, volume, now, raw))
        rows.append(_row(ticker, "no", no_code, _flip(ask), _flip(bid), _flip(mid),
                         event_key, sport, market, moment, start, volume, now, raw))
    return rows


def _flip(price: float | None) -> float | None:
    """The NO side of a YES quote: a bid to buy YES is an offer to sell NO."""
    return None if price is None else 1.0 - price


def _row(ticker, side, label, bid, ask, mid, event_key, sport, market, moment,
         start, volume, now, raw) -> dict[str, Any]:
    return {
        "ts": moment,
        "venue": VENUE,
        "book": "",
        "market_id": f"{ticker}:{side}",
        "outcome": label,
        "event_key": event_key,
        "sport": sport,
        "title": str(market.get("title") or ticker),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "bid_size": None,      # candlesticks carry no resting depth
        "ask_size": None,
        "volume": volume,
        "event_start_time": start,
        "collected_at": now,
        "raw_json": raw,
    }


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
