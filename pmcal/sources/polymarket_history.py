"""Polymarket historical prices for settled game markets.

Two paths, chosen automatically per market and recorded so the mix is auditable:

* **prices-history** (`clob.polymarket.com/prices-history`) is the primary. It
  returns a `{t, p}` series at a requested `fidelity` in minutes.
* **trades** (`data-api.polymarket.com/trades`) is the fallback for when
  prices-history comes back empty. Executed trades are bucketed into a series by
  taking the last trade at or before each grid point.

The documented gotcha -- prices-history returning empty for resolved markets
below 12-hour fidelity -- did not reproduce on any market sampled in August
2026; fidelity=1 returned dense minute bars for settled markets. The fallback is
kept because the failure mode is plausible and cheap to guard against, and
`backfill_progress.path` records which route each market actually took, so the
claim is checkable rather than assumed.

Neither endpoint exposes bid/ask. Backfilled Polymarket rows therefore carry a
midpoint only, which is why the backtest cannot run on them.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from pmcal import gamekeys
from pmcal.http import FetchError, get_json
from pmcal.util import parse_ts

GAMMA = "https://gamma-api.polymarket.com/markets"
PRICES_HISTORY = "https://clob.polymarket.com/prices-history"
TRADES = "https://data-api.polymarket.com/trades"
VENUE = "polymarket"

# Gamma 500s on deep offsets, so discovery walks one day at a time instead.
GAMMA_PAGE = 100
GAMMA_MAX_OFFSET = 1400
# A market's endDate is its resolution deadline, up to a week after the game.
RESOLUTION_LAG_DAYS = 10

log = logging.getLogger(__name__)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def discover_markets(
    session: requests.Session, sport: str, since: datetime, until: datetime,
    timeout: float = 30.0, on_day: Any = None,
) -> list[dict[str, Any]]:
    """Settled moneyline markets for one sport, found by walking day windows.

    Gamma returns HTTP 500 past an offset of roughly 1500, so a month cannot be
    paged directly. Filtering `end_date_*` to a single day keeps every page
    shallow. The window is widened by `RESOLUTION_LAG_DAYS` because a market's
    `endDate` is its resolution deadline, which for MLB sits a week after the
    game; markets are then narrowed back to the requested range by their actual
    `gameStartTime`.
    """
    prefix = f"{sport}-"
    seen: dict[str, dict[str, Any]] = {}
    day = since.date()
    last = (until + timedelta(days=RESOLUTION_LAG_DAYS)).date()
    while day <= last:
        lo = day.isoformat() + "T00:00:00Z"
        hi = (day + timedelta(days=1)).isoformat() + "T00:00:00Z"
        offset = 0
        while offset <= GAMMA_MAX_OFFSET:
            page = get_json(
                session, GAMMA,
                params={"closed": "true", "limit": GAMMA_PAGE, "offset": offset,
                        "end_date_min": lo, "end_date_max": hi},
                timeout=timeout,
            )
            if not isinstance(page, list) or not page:
                break
            for market in page:
                if market.get("sportsMarketType") != "moneyline":
                    continue
                if not str(market.get("slug") or "").startswith(prefix):
                    continue
                start = parse_ts(market.get("gameStartTime"))
                if start is None or not since <= start <= until:
                    continue
                seen[str(market.get("id"))] = market
            if len(page) < GAMMA_PAGE:
                break
            offset += GAMMA_PAGE
        if on_day:
            on_day(day, len(seen))
        day += timedelta(days=1)
    return list(seen.values())


def fetch_price_history(
    session: requests.Session, token_id: str, start_ts: int, end_ts: int,
    fidelity: int, timeout: float = 30.0,
) -> list[tuple[int, float]]:
    """`(unix_ts, price)` points, oldest first. Empty is a valid answer."""
    payload = get_json(
        session, PRICES_HISTORY,
        params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity},
        timeout=timeout,
    )
    history = (payload or {}).get("history") or []
    points = [(int(p["t"]), float(p["p"])) for p in history if "t" in p and "p" in p]
    return sorted(points)


def fetch_trade_series(
    session: requests.Session, condition_id: str, token_id: str, start_ts: int, end_ts: int,
    timeout: float = 30.0, max_pages: int = 20, page_size: int = 500,
) -> list[tuple[int, float]]:
    """Fallback series built from executed trades on one outcome token.

    The Data API filters by `conditionId` (a token id returns nothing), so the
    market's trades are pulled and then narrowed to the token we want.
    """
    points: list[tuple[int, float]] = []
    for page in range(max_pages):
        batch = get_json(
            session, TRADES,
            params={"market": condition_id, "limit": page_size, "offset": page * page_size},
            timeout=timeout,
        )
        if not isinstance(batch, list) or not batch:
            break
        oldest = end_ts
        for trade in batch:
            ts = int(trade.get("timestamp") or 0)
            oldest = min(oldest, ts)
            if str(trade.get("asset")) != str(token_id) or not start_ts <= ts <= end_ts:
                continue
            try:
                points.append((ts, float(trade["price"])))
            except (KeyError, TypeError, ValueError):
                continue
        # Trades come newest-first; stop once the page predates the window.
        if len(batch) < page_size or oldest < start_ts:
            break
    return sorted(points)


def resample_asof(points: list[tuple[int, float]], grid: list[datetime]) -> dict[datetime, float]:
    """Last observation at or before each grid time.

    As-of, never as-of-later: a bar covering [T, T+1h) must not be reported at T,
    because at T that hour has not happened yet. Grid points with no prior
    observation are simply absent.

    >>> from datetime import datetime
    >>> pts = [(0, 0.4), (3600, 0.5), (7200, 0.6)]
    >>> grid = [datetime(1970, 1, 1, h) for h in (0, 1, 2, 3)]
    >>> sorted(resample_asof(pts, grid).values())
    [0.4, 0.5, 0.6, 0.6]
    """
    out: dict[datetime, float] = {}
    if not points:
        return out
    i, last = 0, None
    for moment in sorted(grid):
        # Grid times are naive UTC. `.timestamp()` on a naive datetime assumes
        # *local* time, which would silently shift every backfilled row by the
        # collecting machine's UTC offset.
        target = moment.replace(tzinfo=UTC).timestamp()
        while i < len(points) and points[i][0] <= target:
            last = points[i][1]
            i += 1
        if last is not None:
            out[moment] = last
    return out


def market_rows(
    market: dict[str, Any],
    series_by_outcome: dict[int, dict[datetime, float]],
    event_key: str,
    sport: str,
    now: datetime,
) -> list[dict[str, Any]]:
    """Snapshot rows for one market from its per-outcome resampled series."""
    outcomes = [str(o) for o in as_list(market.get("outcomes"))]
    tokens = [str(t) for t in as_list(market.get("clobTokenIds"))]
    slug = str(market.get("slug") or "")
    parsed = gamekeys.parse_polymarket_slug(slug)
    codes = list(parsed["codes"]) if parsed else []  # type: ignore[index]
    start = parse_ts(market.get("gameStartTime") or market.get("endDate"))
    market_id = str(market.get("id") or market.get("conditionId") or slug)
    volume = _to_float(market.get("volumeNum") or market.get("volume"))

    rows: list[dict[str, Any]] = []
    for index, series in series_by_outcome.items():
        # The slug lists teams away-then-home and `outcomes` follows the same
        # order, so position is the reliable link between them -- the outcome
        # text is a full team name and the slug holds a code.
        label = (
            codes[index] if index < len(codes)
            else gamekeys.outcome_code(sport, outcomes[index] if index < len(outcomes) else "")
        )
        for ts, price in series.items():
            rows.append(
                {
                    "ts": ts,
                    "venue": VENUE,
                    "book": "",
                    "market_id": market_id,
                    "outcome": label,
                    "event_key": event_key,
                    "sport": sport,
                    "title": str(market.get("question") or slug),
                    "bid": None,       # prices-history exposes no order book
                    "ask": None,
                    "mid": price,
                    "bid_size": None,
                    "ask_size": None,
                    "volume": volume,
                    "event_start_time": start,
                    "collected_at": now,
                    "raw_json": json.dumps(
                        {"slug": slug, "outcome_index": index,
                         "token_id": tokens[index] if index < len(tokens) else None,
                         "price": price, "source": "prices-history"},
                        default=str,
                    ),
                }
            )
    return rows


def series_for_token(
    session: requests.Session,
    token_id: str,
    condition_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int,
    grid: list[datetime],
    timeout: float = 30.0,
) -> tuple[dict[datetime, float], str]:
    """Resampled series for one token plus the endpoint that produced it."""
    try:
        points = fetch_price_history(session, token_id, start_ts, end_ts, fidelity, timeout)
    except FetchError as exc:
        log.warning("prices-history failed for %s: %s", token_id[:16], exc)
        points = []
    path = "prices-history"
    if not points and condition_id:
        log.info("prices-history empty for %s; falling back to trades", token_id[:16])
        try:
            points = fetch_trade_series(session, condition_id, token_id, start_ts, end_ts, timeout)
            path = "trades"
        except FetchError as exc:
            log.warning("trades fallback failed for %s: %s", token_id[:16], exc)
            path = "failed"
    return resample_asof(points, grid), (path if points else "empty")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
