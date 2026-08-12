"""Configuration: environment + a tiny .env reader (no extra dependency)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "pm_calibration.duckdb"
FIGURES_DIR = REPO_ROOT / "figures"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env into os.environ without overwriting."""
    path = path or REPO_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class Config:
    db_path: Path = DEFAULT_DB
    odds_api_key: str | None = None
    sports: tuple[str, ...] = ("americanfootball_nfl",)
    poly_fee_rate: float = 0.01
    request_timeout: float = 20.0
    # Polymarket Gamma returns a lot of long-tail markets; cap the pull per run.
    poly_market_limit: int = 500
    # Order-book fetches are the slow part, so only book the most liquid markets.
    poly_book_limit: int = 120
    kalshi_market_limit: int = 1000
    user_agent: str = "pm-calibration/0.1 (research; contact via repo)"
    sources: tuple[str, ...] = field(default=("polymarket", "kalshi", "oddsapi"))


def load_config() -> Config:
    load_dotenv()
    sports = os.environ.get("PMCAL_SPORTS", "americanfootball_nfl")
    db = os.environ.get("PMCAL_DB")
    return Config(
        db_path=Path(db) if db else DEFAULT_DB,
        odds_api_key=os.environ.get("ODDS_API_KEY") or None,
        sports=tuple(s.strip() for s in sports.split(",") if s.strip()),
        poly_fee_rate=float(os.environ.get("PMCAL_POLY_FEE", "0.01")),
    )
