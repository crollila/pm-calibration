"""Cross-venue event matching.

Polymarket, Kalshi and the sportsbooks all name the same game differently. We
reduce every title to a canonical `event_key` of the form

    nfl|2026-09-12|chiefs~ravens

Team names come from a nickname table; the date is bucketed in US time so a
Sunday-night game that starts at 00:20 UTC Monday still lands on Sunday.

Matching is best-effort and deliberately conservative: a market that yields
fewer than two known nicknames gets `event_key = None` and is simply excluded
from the cross-venue analysis rather than guessed at.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime, timedelta

# Multi-word nicknames must be listed before their single-word suffixes so the
# longest match wins.
NICKNAMES: dict[str, tuple[str, ...]] = {
    "nfl": (
        "49ers", "bears", "bengals", "bills", "broncos", "browns", "buccaneers",
        "cardinals", "chargers", "chiefs", "colts", "commanders", "cowboys",
        "dolphins", "eagles", "falcons", "giants", "jaguars", "jets", "lions",
        "packers", "panthers", "patriots", "raiders", "rams", "ravens",
        "saints", "seahawks", "steelers", "texans", "titans", "vikings",
    ),
    "nba": (
        "trail blazers", "76ers", "bucks", "bulls", "cavaliers", "celtics",
        "clippers", "grizzlies", "hawks", "heat", "hornets", "jazz", "kings",
        "knicks", "lakers", "magic", "mavericks", "nets", "nuggets", "pacers",
        "pelicans", "pistons", "raptors", "rockets", "spurs", "suns",
        "thunder", "timberwolves", "warriors", "wizards",
    ),
    "mlb": (
        "red sox", "white sox", "blue jays", "angels", "astros", "athletics",
        "braves", "brewers", "cardinals", "cubs", "diamondbacks", "dodgers",
        "giants", "guardians", "mariners", "marlins", "mets", "nationals",
        "orioles", "padres", "phillies", "pirates", "rangers", "rays", "reds",
        "rockies", "royals", "tigers", "twins", "yankees",
    ),
}

SPORT_FROM_ODDS_KEY = {
    "americanfootball_nfl": "nfl",
    "basketball_nba": "nba",
    "baseball_mlb": "mlb",
}

# Ambiguous nicknames shared across leagues; only trusted when the sport is known.
AMBIGUOUS = {"cardinals", "giants", "rangers", "kings", "panthers", "jets"}

_WS = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    >>> normalize("Will the K.C. Chiefs  beat the Ravens?")
    'will the kc chiefs beat the ravens'
    """
    return re.sub(r"\s+", " ", _WS.sub("", text.lower())).strip()


def us_date(start: datetime) -> str:
    """Bucket a UTC kickoff into the US calendar date fans would call it.

    Shifting back 8 hours puts every US start time (earliest ~12:00 ET) on the
    right side of midnight without needing a timezone database.
    """
    return (start - timedelta(hours=8)).date().isoformat()


def extract_teams(title: str, sport: str | None = None) -> tuple[str, ...]:
    """Pull canonical nicknames out of a free-text market title.

    >>> extract_teams("Chiefs vs. Ravens", "nfl")
    ('chiefs', 'ravens')
    >>> extract_teams("Will the Boston Red Sox win the World Series?", "mlb")
    ('red sox',)
    """
    text = " " + normalize(title) + " "
    pools = [sport] if sport in NICKNAMES else list(NICKNAMES)
    found: list[tuple[int, str]] = []
    for pool in pools:
        for name in sorted(NICKNAMES[pool], key=len, reverse=True):
            if sport is None and name in AMBIGUOUS:
                continue
            idx = text.find(" " + name + " ")
            if idx >= 0 and not any(name in seen for _, seen in found):
                found.append((idx, name))
    ordered = [name for _, name in sorted(found)]
    # Drop nicknames fully contained in a longer match ("sox" inside "red sox").
    return tuple(n for n in ordered if not any(n != o and n in o for o in ordered))


def build_event_key(sport: str, start: datetime | None, teams: tuple[str, ...]) -> str | None:
    """Canonical join key, or None when the market is not a two-team game.

    >>> from datetime import datetime
    >>> build_event_key("nfl", datetime(2026, 9, 13, 0, 20), ("ravens", "chiefs"))
    'nfl|2026-09-12|chiefs~ravens'
    """
    if start is None or len(teams) != 2:
        return None
    return f"{sport}|{us_date(start)}|{'~'.join(sorted(teams))}"


def title_similarity(a: str, b: str) -> float:
    """0-1 similarity used only as a diagnostic `match_score`, never as the key.

    >>> title_similarity("Chiefs vs Ravens", "chiefs vs. ravens")
    1.0
    """
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def outcome_label(text: str, sport: str | None = None) -> str:
    """Normalise an outcome name to a nickname when possible, else lowercase text.

    >>> outcome_label("Kansas City Chiefs", "nfl")
    'chiefs'
    >>> outcome_label("Draw")
    'draw'
    """
    teams = extract_teams(text, sport)
    if len(teams) == 1:
        return teams[0]
    return normalize(text)
