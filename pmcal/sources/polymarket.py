"""Polymarket collector: Gamma metadata + CLOB top-of-book.

Gamma gives us the market universe and a midpoint; it does not give a reliable
tradeable spread. We therefore hit the CLOB `/books` endpoint for the most
liquid markets so the backtest can execute at the far side of the book instead
of pretending the midpoint is fillable.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import requests

from pmcal import gamekeys
from pmcal.http import FetchError, get_json, post_json
from pmcal.util import parse_ts

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_BOOKS_URL = "https://clob.polymarket.com/books"
VENUE = "polymarket"

log = logging.getLogger(__name__)


def _as_list(value: Any) -> list[Any]:
    """Gamma returns several list fields as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def fetch_markets(session: requests.Session, cfg: Any) -> list[dict[str, Any]]:
    """Page through open Gamma markets, most-traded first."""
    markets: list[dict[str, Any]] = []
    page_size = 100
    while len(markets) < cfg.poly_market_limit:
        params = {
            "closed": "false",
            "active": "true",
            "limit": page_size,
            "offset": len(markets),
            "order": "volumeNum",
            "ascending": "false",
        }
        page = get_json(session, GAMMA_URL, params=params, timeout=cfg.request_timeout)
        if not isinstance(page, list) or not page:
            break
        markets.extend(page)
        if len(page) < page_size:
            break
    return markets[: cfg.poly_market_limit]


def fetch_books(
    session: requests.Session, token_ids: list[str], cfg: Any
) -> dict[str, dict[str, Any]]:
    """Batch top-of-book lookup. Returns {token_id: book}; missing ids are absent."""
    books: dict[str, dict[str, Any]] = {}
    for start in range(0, len(token_ids), 50):
        batch = token_ids[start : start + 50]
        payload = [{"token_id": t} for t in batch]
        try:
            result = post_json(session, CLOB_BOOKS_URL, payload, timeout=cfg.request_timeout)
        except FetchError as exc:
            log.warning("CLOB book batch failed, continuing without depth: %s", exc)
            continue
        for book in result or []:
            token = str(book.get("asset_id") or book.get("token_id") or "")
            if token:
                books[token] = book
    return books


def _top_of_book(book: dict[str, Any] | None) -> tuple[float | None, float | None, float, float]:
    """Best bid/ask and their sizes. CLOB levels are strings and unordered."""
    if not book:
        return None, None, 0.0, 0.0
    bids = [(float(b["price"]), float(b["size"])) for b in book.get("bids") or []]
    asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks") or []]
    best_bid = max(bids, default=None)
    best_ask = min(asks, default=None)
    return (
        best_bid[0] if best_bid else None,
        best_ask[0] if best_ask else None,
        best_bid[1] if best_bid else 0.0,
        best_ask[1] if best_ask else 0.0,
    )


def _mid(bid: float | None, ask: float | None, fallback: float | None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return fallback


def _market_rows(
    market: dict[str, Any], books: dict[str, dict[str, Any]], now: datetime
) -> list[dict[str, Any]]:
    outcomes = [str(o) for o in _as_list(market.get("outcomes"))]
    prices = [float(p) for p in _as_list(market.get("outcomePrices")) or []]
    tokens = [str(t) for t in _as_list(market.get("clobTokenIds"))]
    if not outcomes:
        return []

    title = str(market.get("question") or market.get("slug") or "")
    # `startDate` is when the *market* opened, not when the event happens; using
    # it would make every long-running market look ready to settle the moment it
    # is listed. Fall back to `endDate`, the resolution deadline, instead.
    start = parse_ts(market.get("gameStartTime") or market.get("endDate"))
    # The slug is structured and authoritative; the title is the fallback.
    event_key = gamekeys.polymarket_event_key(str(market.get("slug") or ""))
    sport = event_key.split("|")[0] if event_key else gamekeys.infer_sport(title)
    if event_key is None and sport:
        event_key = gamekeys.event_key_from_codes(
            sport, start, gamekeys.codes_from_title(sport, title)
        )
    volume = _to_float(market.get("volumeNum") or market.get("volume"))

    rows: list[dict[str, Any]] = []
    for i, outcome in enumerate(outcomes):
        token = tokens[i] if i < len(tokens) else ""
        book = books.get(token)
        bid, ask, bid_size, ask_size = _top_of_book(book)
        fallback = prices[i] if i < len(prices) else None
        rows.append(
            {
                "venue": VENUE,
                "book": "",
                "market_id": str(market.get("id") or market.get("conditionId") or token),
                "outcome": gamekeys.outcome_code(sport, outcome),
                "event_key": event_key,
                "sport": sport,
                "title": title,
                "bid": bid,
                "ask": ask,
                "mid": _mid(bid, ask, fallback),
                "bid_size": bid_size,
                "ask_size": ask_size,
                "volume": volume,
                "event_start_time": start,
                "collected_at": now,
                "raw_json": json.dumps(
                    {"market": market, "token_id": token, "book": book}, default=str
                ),
            }
        )
    return rows


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fetch(session: requests.Session, cfg: Any, now: datetime) -> list[dict[str, Any]]:
    markets = fetch_markets(session, cfg)
    # Only book the top-N by volume: each book call is expensive and the
    # long tail has no depth worth simulating against anyway.
    ranked = sorted(markets, key=lambda m: _to_float(m.get("volumeNum")) or 0.0, reverse=True)
    tokens: list[str] = []
    for market in ranked[: cfg.poly_book_limit]:
        tokens.extend(str(t) for t in _as_list(market.get("clobTokenIds")))
    books = fetch_books(session, tokens, cfg) if tokens else {}

    rows: list[dict[str, Any]] = []
    for market in markets:
        rows.extend(_market_rows(market, books, now))
    return rows
