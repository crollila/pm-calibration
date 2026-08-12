"""Thin HTTP helper: one session, bounded retries, explicit timeouts."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """Raised when a source cannot be fetched after retries."""


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})
    return session


def get_json(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in RETRY_STATUS:
                raise FetchError(f"{url} -> HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - retried, then re-raised as FetchError
            last = exc
            if attempt == retries - 1:
                break
            sleep_for = backoff**attempt
            log.warning("fetch failed (%s), retrying in %.1fs: %s", attempt + 1, sleep_for, exc)
            time.sleep(sleep_for)
    raise FetchError(f"GET {url} failed after {retries} attempts: {last}") from last


def post_json(
    session: requests.Session,
    url: str,
    payload: Any,
    timeout: float = 20.0,
    retries: int = 3,
) -> Any:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.post(url, json=payload, timeout=timeout)
            if resp.status_code in RETRY_STATUS:
                raise FetchError(f"{url} -> HTTP {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt == retries - 1:
                break
            time.sleep(1.5**attempt)
    raise FetchError(f"POST {url} failed after {retries} attempts: {last}") from last
