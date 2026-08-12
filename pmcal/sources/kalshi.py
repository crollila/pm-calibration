"""Kalshi collector: public v2 market data, no auth required.

Kalshi quotes in whole cents and exposes both sides directly (`yes_bid`,
`yes_ask`, `no_bid`, `no_ask`), so top-of-book comes for free. Depth does not,
so we pull `/orderbook` for the most active markets.

`event_start_time` here is the market's `close_time`, which for game markets is
the scheduled start but is not guaranteed to be. `resolve.py` prefers the Odds
API `commence_time` whenever an event is matched across venues.
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

BASE = "https://api.elections.kalshi.com/trade-api/v2"
VENUE = "kalshi"
# Multi-leg parlay markets concatenate every leg into the side subtitle. That is
# not an outcome name, so anything this long falls back to plain yes/no.
MAX_LABEL_CHARS = 40

log = logging.getLogger(__name__)


def fetch_markets(session: requests.Session, cfg: Any) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    cursor: str | None = None
    while len(markets) < cfg.kalshi_market_limit:
        params: dict[str, Any] = {"limit": 200, "status": "open"}
        if cursor:
            params["cursor"] = cursor
        page = get_json(session, f"{BASE}/markets", params=params, timeout=cfg.request_timeout)
        batch = page.get("markets") or []
        markets.extend(batch)
        cursor = page.get("cursor") or None
        if not cursor or not batch:
            break
    return markets[: cfg.kalshi_market_limit]


def fetch_orderbook(
    session: requests.Session, ticker: str, cfg: Any
) -> dict[str, Any] | None:
    try:
        payload = get_json(
            session,
            f"{BASE}/markets/{ticker}/orderbook",
            params={"depth": 1},
            timeout=cfg.request_timeout,
        )
    except FetchError as exc:
        log.warning("kalshi orderbook %s failed: %s", ticker, exc)
        return None
    return payload.get("orderbook") if isinstance(payload, dict) else None


def _depth(orderbook: dict[str, Any] | None, side: str) -> float:
    """Size resting at the best bid on `side` ('yes' or 'no'), in contracts."""
    if not orderbook:
        return 0.0
    levels = orderbook.get(side) or []
    if not levels:
        return 0.0
    best = max(levels, key=lambda lvl: lvl[0])
    return float(best[1])


def _price(cents: Any) -> float | None:
    try:
        value = float(cents)
    except (TypeError, ValueError):
        return None
    return value / 100.0 if 0.0 < value < 100.0 else None


def _mid(bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return last


def _side_rows(
    market: dict[str, Any],
    orderbook: dict[str, Any] | None,
    now: datetime,
) -> list[dict[str, Any]]:
    title = " ".join(
        str(market.get(k) or "") for k in ("title", "subtitle", "yes_sub_title")
    ).strip()
    ticker = str(market.get("ticker") or "")
    start = parse_ts(market.get("close_time"))

    # The ticker is structured and authoritative for game markets: it carries
    # the sport, the kickoff and the teams, none of which the prose title
    # reliably gives us (Kalshi names sides by city, not by nickname).
    parsed = gamekeys.parse_kalshi_ticker(ticker)
    if parsed:
        sport = str(parsed["sport"])
        start = parsed["start"]  # type: ignore[assignment]
        codes = gamekeys.kalshi_game_codes({str(parsed["side"])}, str(parsed["blob"]), sport)
        event_key = gamekeys.event_key_from_codes(sport, start, codes)
    else:
        sport = gamekeys.infer_sport(title)
        event_key = gamekeys.event_key_from_codes(
            sport, start, gamekeys.codes_from_title(sport, title)
        ) if sport else None
    volume = _to_float(market.get("volume"))
    last = _price(market.get("last_price"))
    labels = _side_labels(market, parsed, sport)

    sides = (
        ("yes", _price(market.get("yes_bid")),
         _price(market.get("yes_ask")), _depth(orderbook, "yes")),
        ("no", _price(market.get("no_bid")),
         _price(market.get("no_ask")), _depth(orderbook, "no")),
    )

    rows: list[dict[str, Any]] = []
    for side, bid, ask, depth in sides:
        label = labels[side]
        rows.append(
            {
                "venue": VENUE,
                "book": "",
                "market_id": f"{ticker}:{side}",
                "outcome": label,
                "event_key": event_key,
                "sport": sport,
                "title": title,
                "bid": bid,
                "ask": ask,
                "mid": _mid(bid, ask, last if side == "yes" else _complement(last)),
                "bid_size": depth,
                "ask_size": depth,
                "volume": volume,
                "event_start_time": start,
                "collected_at": now,
                "raw_json": json.dumps(
                    {"market": market, "side": side, "orderbook": orderbook}, default=str
                ),
            }
        )
    return rows


def _side_labels(
    market: dict[str, Any], parsed: dict[str, Any] | None, sport: str | None
) -> dict[str, str]:
    """Outcome label for each side of a Kalshi market.

    A game ticker is a single team's market: YES is that team, NO is the
    opponent. Kalshi's own `no_sub_title` repeats the YES team rather than
    naming the opponent, so it cannot be used for this.
    """
    if parsed:
        side_code = str(parsed["side"])
        pair = gamekeys.kalshi_game_codes({side_code}, str(parsed["blob"]), str(parsed["sport"]))
        other = next((c for c in pair if c != side_code), "no")
        return {"yes": side_code, "no": other}

    out = {}
    for side in ("yes", "no"):
        sub_title = market.get(f"{side}_sub_title")
        label = gamekeys.outcome_code(sport, str(sub_title)) if sub_title else side
        out[side] = label if 0 < len(label) <= MAX_LABEL_CHARS else side
    return out


def _complement(p: float | None) -> float | None:
    return None if p is None else 1.0 - p


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch(session: requests.Session, cfg: Any, now: datetime) -> list[dict[str, Any]]:
    markets = fetch_markets(session, cfg)
    ranked = sorted(markets, key=lambda m: _to_float(m.get("volume")) or 0.0, reverse=True)
    depth_tickers = {str(m.get("ticker")) for m in ranked[:120]}

    rows: list[dict[str, Any]] = []
    for market in markets:
        ticker = str(market.get("ticker") or "")
        orderbook = fetch_orderbook(session, ticker, cfg) if ticker in depth_tickers else None
        rows.extend(_side_rows(market, orderbook, now))
    return rows
