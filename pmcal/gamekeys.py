"""Derive `event_key` and outcome labels from venues' structured identifiers.

Polymarket and Kalshi both encode the teams and the date in machine-readable
form -- a slug and a ticker respectively -- and that is a far more reliable join
than their free-text titles, which use different naming conventions entirely.
Everything here funnels into one canonical namespace of short team codes so the
two venues (and the live free-text collectors) produce identical keys.

    mlb-stl-nyy-2026-08-03            -> mlb|2026-08-03|nyy~stl
    KXMLBGAME-26AUG101945PHISTL-STL   -> mlb|2026-08-10|phi~stl  (outcome 'stl')
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from pmcal import teams
from pmcal.matching import normalize, us_date

# alias -> code, per sport. Codes map to themselves.
ALIASES: dict[str, dict[str, str]] = {}
for _sport, _table in teams.BY_SPORT.items():
    _lookup: dict[str, str] = {}
    for _code, _names in _table.items():
        _lookup[_code] = _code
        for _name in _names:
            _lookup.setdefault(_name, _code)
    ALIASES[_sport] = _lookup

# Kalshi series ticker -> our sport label.
KALSHI_SERIES = {
    "KXMLBGAME": "mlb",
    "KXNFLGAME": "nfl",
    "KXNBAGAME": "nba",
    "KXWNBAGAME": "wnba",
}
SERIES_FOR_SPORT = {v: k for k, v in KALSHI_SERIES.items()}

# Polymarket slug: <sport>-<away>-<home>-<yyyy>-<mm>-<dd>[-suffix]
SLUG_RE = re.compile(r"^(mlb|nfl|nba|wnba)-([a-z]{2,4})-([a-z]{2,4})-(\d{4})-(\d{2})-(\d{2})$")
# Kalshi ticker: <SERIES>-<YYMMMDD><HHMM><TEAMS>-<SIDE>
TICKER_RE = re.compile(r"^([A-Z0-9]+)-(\d{2}[A-Z]{3}\d{2})(\d{4})([A-Z]+)-([A-Z0-9]+)$")

# Kalshi stamps its game tickers in US Eastern time; we normalise to UTC before
# bucketing so both venues land on the same US calendar date.
ET_OFFSET_HOURS = 4


def canonical_code(sport: str, text: str, allow_short: bool = True) -> str | None:
    """Resolve any spelling of a team to its canonical code.

    >>> canonical_code("mlb", "St. Louis Cardinals")
    'stl'
    >>> canonical_code("mlb", "ATH")
    'oak'
    >>> canonical_code("mlb", "AZ")
    'ari'
    >>> canonical_code("nfl", "chiefs")
    'kc'
    >>> canonical_code("mlb", "Nonexistent FC") is None
    True

    `allow_short=False` refuses to resolve two-letter inputs, which is required
    when the text is a *display label* rather than a code: Kalshi's side
    subtitle "No" would otherwise become the New Orleans Saints, and "NE" New
    England. Slug and ticker parsing keeps the default, because there a short
    token really is a team code.

    >>> canonical_code("nfl", "No", allow_short=False) is None
    True
    >>> canonical_code("nfl", "no")
    'no'
    """
    lookup = ALIASES.get(sport)
    if not lookup:
        return None
    text = normalize(text)
    if text in lookup and (allow_short or len(text) > 2):
        return lookup[text]
    padded = f" {text} "
    # The fuzzy pass only ever considers aliases long enough to be unambiguous
    # inside a sentence, so "no" never matches the word "no" in a title.
    hits = [a for a in lookup if len(a) >= 3 and (a in text.split() or f" {a} " in padded)]
    if not hits:
        return None
    return lookup[max(hits, key=len)]


def event_key_from_codes(sport: str, start: datetime, codes: tuple[str, ...]) -> str | None:
    """Canonical key from a UTC start time and exactly two team codes.

    >>> from datetime import datetime
    >>> event_key_from_codes("mlb", datetime(2026, 8, 4, 2, 10), ("nyy", "stl"))
    'mlb|2026-08-03|nyy~stl'
    """
    codes = tuple(sorted({c for c in codes if c}))
    if start is None or len(codes) != 2:
        return None
    return f"{sport}|{us_date(start)}|{'~'.join(codes)}"


def parse_polymarket_slug(slug: str) -> dict[str, object] | None:
    """Pull sport, team codes and date out of a Polymarket game slug.

    The slug date is already the US calendar date, so it is used directly
    rather than re-derived from `gameStartTime`.

    >>> out = parse_polymarket_slug("mlb-stl-nyy-2026-08-03")
    >>> out["sport"], out["codes"], out["date"]
    ('mlb', ('stl', 'nyy'), '2026-08-03')
    >>> parse_polymarket_slug("mlb-wsh-phi-2026-08-03-nrfi") is None
    True
    """
    match = SLUG_RE.match(str(slug).strip().lower())
    if not match:
        return None
    sport, away, home, year, month, day = match.groups()
    codes = tuple(canonical_code(sport, c) or c for c in (away, home))
    return {"sport": sport, "codes": codes, "date": f"{year}-{month}-{day}"}


def polymarket_event_key(slug: str) -> str | None:
    """Event key straight from the slug.

    >>> polymarket_event_key("mlb-stl-nyy-2026-08-03")
    'mlb|2026-08-03|nyy~stl'
    """
    parsed = parse_polymarket_slug(slug)
    if not parsed:
        return None
    codes = tuple(sorted(parsed["codes"]))  # type: ignore[arg-type]
    if len(set(codes)) != 2:
        return None
    return f"{parsed['sport']}|{parsed['date']}|{'~'.join(codes)}"


def parse_kalshi_ticker(ticker: str) -> dict[str, object] | None:
    """Pull sport, kickoff and the side's own team out of a Kalshi game ticker.

    The concatenated team blob (`CWSTB`) cannot be split reliably on its own --
    both `CWS`+`TB` and `CW`+`STB` are lexically valid -- so only the side code
    is taken from here. `game_id` groups the two tickers of one game, and
    `kalshi_game_codes` recovers the pair from their two suffixes.

    >>> out = parse_kalshi_ticker("KXMLBGAME-26AUG101945PHISTL-STL")
    >>> out["sport"], out["side"], out["game_id"]
    ('mlb', 'stl', 'KXMLBGAME-26AUG101945PHISTL')
    >>> out["start"].isoformat()
    '2026-08-10T23:45:00'
    >>> parse_kalshi_ticker("KXSOLE-26AUG10-X") is None
    True
    """
    match = TICKER_RE.match(str(ticker).strip().upper())
    if not match:
        return None
    series, stamp, clock, blob, side = match.groups()
    sport = KALSHI_SERIES.get(series)
    if not sport:
        return None
    try:
        day = datetime.strptime(stamp, "%y%b%d")
        start_et = day.replace(hour=int(clock[:2]), minute=int(clock[2:]))
    except ValueError:
        return None
    return {
        "sport": sport,
        "side": canonical_code(sport, side.lower()) or side.lower(),
        "blob": blob.lower(),
        "start": start_et + timedelta(hours=ET_OFFSET_HOURS),
        "game_id": f"{series}-{stamp}{clock}{blob}",
    }


def infer_sport(title: str) -> str | None:
    """Best-guess sport for a free-text title, or None if it is not a known game.

    >>> infer_sport("Chiefs vs. Ravens")
    'nfl'
    >>> infer_sport("Fed rate decision") is None
    True
    """
    from pmcal.matching import NICKNAMES, extract_teams

    for sport in NICKNAMES:
        found = extract_teams(title, sport)
        if len(found) == 2:
            return sport
    return None


def codes_from_title(sport: str | None, title: str) -> tuple[str, ...]:
    """Team codes for a free-text title, via the nickname matcher.

    This is the bridge that keeps the live collectors -- which only ever see
    prose -- in the same namespace as the structurally-parsed backfill.

    >>> codes_from_title("nfl", "Kansas City Chiefs at Baltimore Ravens")
    ('bal', 'kc')
    """
    from pmcal.matching import extract_teams

    if not sport:
        return ()
    found = extract_teams(title, sport)
    return tuple(sorted({canonical_code(sport, name) or name for name in found}))


def outcome_code(sport: str | None, text: str) -> str:
    """Canonical outcome label: a team code where possible, else normalised text.

    >>> outcome_code("mlb", "St. Louis Cardinals")
    'stl'
    >>> outcome_code(None, "Yes")
    'yes'
    >>> outcome_code("nfl", "No")
    'no'
    """
    if sport:
        # Display text, so a bare "No" must stay "no" rather than becoming a team.
        code = canonical_code(sport, text, allow_short=False)
        if code:
            return code
    return normalize(text)


def kalshi_game_codes(side_codes: set[str], blob: str, sport: str) -> tuple[str, ...]:
    """The two team codes of a game, preferring the unambiguous ticker suffixes.

    With both tickers present the two suffixes *are* the pair. With only one,
    the partner is whatever remains of the concatenated blob.

    >>> kalshi_game_codes({"stl", "phi"}, "phistl", "mlb")
    ('phi', 'stl')
    >>> kalshi_game_codes({"stl"}, "phistl", "mlb")
    ('phi', 'stl')
    >>> kalshi_game_codes({"tb"}, "cwstb", "mlb")
    ('cws', 'tb')
    """
    codes = {c for c in side_codes if c}
    if len(codes) >= 2:
        return tuple(sorted(codes))
    if len(codes) == 1:
        known = next(iter(codes))
        for raw in (blob.replace(known, "", 1), blob):
            partner = canonical_code(sport, raw)
            if partner and partner != known:
                return tuple(sorted({known, partner}))
    return tuple(sorted(codes))
