"""The Odds API v4 collector: `h2h` markets, `us` region, American odds, all books.

A sportsbook publishes one take-it price per outcome, so there is no native
bid/ask. We synthesise one: backing every *other* outcome is economically the
same as laying this one, so

    ask_i = implied(price_i)                  (what you pay to back outcome i)
    bid_i = 1 - sum_{j != i} ask_j            (what backing the complement pays)

`mid` is the midpoint of that synthetic quote. For a two-way market that
midpoint is algebraically identical to the additive de-vig, which is why
`devig.py` always works from the raw prices rather than from `mid`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import requests

from pmcal import gamekeys, matching
from pmcal.http import FetchError, get_json
from pmcal.odds import american_to_implied
from pmcal.util import parse_ts

BASE = "https://api.the-odds-api.com/v4"
VENUE = "sportsbook"

log = logging.getLogger(__name__)


def fetch_odds(session: requests.Session, sport: str, cfg: Any) -> list[dict[str, Any]]:
    params = {
        "apiKey": cfg.odds_api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    payload = get_json(session, f"{BASE}/sports/{sport}/odds", params=params, timeout=cfg.request_timeout)
    return payload if isinstance(payload, list) else []


def fetch_scores(
    session: requests.Session, sport: str, cfg: Any, days_from: int = 3
) -> list[dict[str, Any]]:
    """Completed-game scores, used by `resolve.py` as an independent third opinion."""
    params = {"apiKey": cfg.odds_api_key, "daysFrom": days_from, "dateFormat": "iso"}
    payload = get_json(session, f"{BASE}/sports/{sport}/scores", params=params, timeout=cfg.request_timeout)
    return payload if isinstance(payload, list) else []


def _event_rows(event: dict[str, Any], sport: str, now: datetime) -> list[dict[str, Any]]:
    start = parse_ts(event.get("commence_time"))
    home = str(event.get("home_team") or "")
    away = str(event.get("away_team") or "")
    codes = tuple(sorted({c for c in (gamekeys.canonical_code(sport, home),
                                      gamekeys.canonical_code(sport, away)) if c}))
    event_key = gamekeys.event_key_from_codes(sport, start, codes)
    title = f"{away} @ {home}"
    event_id = str(event.get("id") or "")

    rows: list[dict[str, Any]] = []
    for bookmaker in event.get("bookmakers") or []:
        h2h = next((m for m in bookmaker.get("markets") or [] if m.get("key") == "h2h"), None)
        if not h2h:
            continue
        prices: list[tuple[str, float]] = []
        for outcome in h2h.get("outcomes") or []:
            try:
                prices.append((str(outcome["name"]), american_to_implied(float(outcome["price"]))))
            except (KeyError, TypeError, ValueError):
                continue
        if len(prices) < 2:
            continue
        total = sum(p for _, p in prices)
        for name, ask in prices:
            bid = 1.0 - (total - ask)
            rows.append(
                {
                    "venue": VENUE,
                    "book": str(bookmaker.get("key") or "unknown"),
                    "market_id": f"{event_id}:h2h",
                    "outcome": gamekeys.outcome_code(sport, name),
                    "event_key": event_key,
                    "sport": sport,
                    "title": title,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                    "bid_size": None,
                    "ask_size": None,
                    "volume": None,
                    "event_start_time": start,
                    "collected_at": now,
                    "raw_json": json.dumps(
                        {
                            "event": {k: v for k, v in event.items() if k != "bookmakers"},
                            "bookmaker": bookmaker,
                            "outcome_name": name,
                        },
                        default=str,
                    ),
                }
            )
    return rows


def fetch(session: requests.Session, cfg: Any, now: datetime) -> list[dict[str, Any]]:
    if not cfg.odds_api_key:
        raise FetchError("ODDS_API_KEY is not set; skipping sportsbook collection")
    rows: list[dict[str, Any]] = []
    for sport_key in cfg.sports:
        sport = matching.SPORT_FROM_ODDS_KEY.get(sport_key, sport_key)
        events = fetch_odds(session, sport_key, cfg)
        log.info("oddsapi %s: %d events", sport_key, len(events))
        for event in events:
            rows.extend(_event_rows(event, sport, now))
    return rows

