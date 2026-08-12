"""Team-code canonicalisation and structural key parsing.

These are the join between venues. If they drift, rows do not merge in the
panel -- they silently vanish from the cross-venue comparison instead of
failing loudly, which is why the coverage here is heavier than the code size
suggests it needs.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from pmcal import gamekeys, teams


@pytest.mark.parametrize(
    ("sport", "text", "expected"),
    [
        ("mlb", "St. Louis Cardinals", "stl"),      # Polymarket full name
        ("mlb", "St. Louis", "stl"),                # Kalshi city name
        ("mlb", "stl", "stl"),                      # slug / ticker code
        ("mlb", "STL", "stl"),
        ("mlb", "Red Sox", "bos"),                  # multi-word nickname
        ("mlb", "Chicago White Sox", "cws"),
        ("mlb", "ATH", "oak"),                      # Kalshi's code for Oakland
        ("mlb", "AZ", "ari"),                       # Kalshi's code for Arizona
        ("nfl", "Kansas City Chiefs", "kc"),
        ("nfl", "chiefs", "kc"),
        ("nba", "Golden State Warriors", "gsw"),
        ("wnba", "Toronto Tempo", "tor"),
    ],
)
def test_every_spelling_resolves_to_one_code(sport, text, expected):
    assert gamekeys.canonical_code(sport, text) == expected


def test_the_two_venues_agree_on_every_mlb_team():
    """Kalshi and Polymarket differ on exactly two codes; both must map across."""
    kalshi_codes = ["ath", "az", "cws", "wsh", "sd", "sf", "tb", "kc"]
    poly_codes = ["oak", "ari", "cws", "wsh", "sd", "sf", "tb", "kc"]
    for k, p in zip(kalshi_codes, poly_codes, strict=True):
        assert gamekeys.canonical_code("mlb", k) == gamekeys.canonical_code("mlb", p)


def test_unknown_team_returns_none():
    assert gamekeys.canonical_code("mlb", "Nonexistent FC") is None
    assert gamekeys.canonical_code("curling", "anything") is None


def test_short_labels_do_not_masquerade_as_teams():
    """A Kalshi side subtitle of 'No' must not resolve to the New Orleans Saints."""
    assert gamekeys.canonical_code("nfl", "No", allow_short=False) is None
    assert gamekeys.canonical_code("nfl", "Yes", allow_short=False) is None
    assert gamekeys.canonical_code("nba", "NO", allow_short=False) is None
    # ... but a slug or ticker token of the same shape still resolves.
    assert gamekeys.canonical_code("nfl", "no") == "no"


def test_polymarket_slug_parsing():
    parsed = gamekeys.parse_polymarket_slug("mlb-stl-nyy-2026-08-03")
    assert parsed == {"sport": "mlb", "codes": ("stl", "nyy"), "date": "2026-08-03"}
    assert gamekeys.polymarket_event_key("mlb-stl-nyy-2026-08-03") == "mlb|2026-08-03|nyy~stl"


@pytest.mark.parametrize(
    "slug",
    [
        "mlb-wsh-phi-2026-08-03-nrfi",   # a derivative market, not the moneyline
        "will-argentina-win-the-2026-fifa-world-cup-245",
        "lec-summer-2026-final",
        "",
    ],
)
def test_non_game_slugs_are_rejected(slug):
    assert gamekeys.parse_polymarket_slug(slug) is None
    assert gamekeys.polymarket_event_key(slug) is None


def test_kalshi_ticker_parsing():
    parsed = gamekeys.parse_kalshi_ticker("KXMLBGAME-26AUG101945PHISTL-STL")
    assert parsed["sport"] == "mlb"
    assert parsed["side"] == "stl"
    assert parsed["blob"] == "phistl"
    assert parsed["game_id"] == "KXMLBGAME-26AUG101945PHISTL"
    # The ticker stamp is US Eastern; we store UTC.
    assert parsed["start"] == datetime(2026, 8, 10, 23, 45)


@pytest.mark.parametrize(
    "ticker",
    ["KXSOLE-26AUG10-X", "KXMVECROSSCATEGORY-S2026-ABC", "", "not-a-ticker"],
)
def test_non_game_tickers_are_rejected(ticker):
    assert gamekeys.parse_kalshi_ticker(ticker) is None


def test_game_codes_prefer_the_unambiguous_ticker_suffixes():
    """`CWSTB` cannot be split lexically; the two tickers' suffixes can."""
    assert gamekeys.kalshi_game_codes({"cws", "tb"}, "cwstb", "mlb") == ("cws", "tb")
    # With only one side seen, the partner comes out of the remaining blob.
    assert gamekeys.kalshi_game_codes({"tb"}, "cwstb", "mlb") == ("cws", "tb")
    assert gamekeys.kalshi_game_codes({"cws"}, "cwstb", "mlb") == ("cws", "tb")
    assert gamekeys.kalshi_game_codes({"stl"}, "phistl", "mlb") == ("phi", "stl")


def test_game_codes_give_up_rather_than_guess():
    assert gamekeys.kalshi_game_codes(set(), "phistl", "mlb") == ()


def test_event_key_is_order_independent():
    start = datetime(2026, 8, 4, 2, 10)
    assert (gamekeys.event_key_from_codes("mlb", start, ("nyy", "stl"))
            == gamekeys.event_key_from_codes("mlb", start, ("stl", "nyy")))


def test_event_key_needs_two_distinct_teams_and_a_time():
    start = datetime(2026, 8, 4, 2, 10)
    assert gamekeys.event_key_from_codes("mlb", start, ("nyy",)) is None
    assert gamekeys.event_key_from_codes("mlb", start, ("nyy", "nyy")) is None
    assert gamekeys.event_key_from_codes("mlb", None, ("nyy", "stl")) is None


def test_late_games_land_on_the_us_calendar_date():
    """A 10:10pm ET first pitch is 02:10 UTC the next day, but still that day's game."""
    assert gamekeys.event_key_from_codes(
        "mlb", datetime(2026, 8, 4, 2, 10), ("nyy", "stl")
    ) == "mlb|2026-08-03|nyy~stl"


def test_both_venues_produce_the_same_key_for_one_game():
    """The whole point: a Polymarket slug and a Kalshi ticker must agree."""
    poly = gamekeys.polymarket_event_key("mlb-phi-stl-2026-08-10")
    parsed = gamekeys.parse_kalshi_ticker("KXMLBGAME-26AUG101945PHISTL-STL")
    codes = gamekeys.kalshi_game_codes({str(parsed["side"])}, str(parsed["blob"]), "mlb")
    kalshi = gamekeys.event_key_from_codes("mlb", parsed["start"], codes)
    assert poly == kalshi == "mlb|2026-08-10|phi~stl"


def test_free_text_titles_reach_the_same_namespace():
    """Live collectors only see prose; they must still land on codes."""
    assert gamekeys.codes_from_title("nfl", "Kansas City Chiefs at Baltimore Ravens") == ("bal", "kc")
    assert gamekeys.infer_sport("Chiefs vs. Ravens") == "nfl"
    assert gamekeys.infer_sport("Fed rate decision") is None
    assert gamekeys.outcome_code("mlb", "St. Louis Cardinals") == "stl"
    assert gamekeys.outcome_code(None, "Yes") == "yes"


def test_team_tables_have_no_duplicate_codes_or_empty_aliases():
    for sport, table in teams.BY_SPORT.items():
        assert len(table) == len(set(table)), sport
        for code, aliases in table.items():
            assert code and aliases, f"{sport}:{code}"
            assert all(a.strip() for a in aliases), f"{sport}:{code}"


def test_alias_collisions_within_a_sport_are_deliberate():
    """Two teams must never share an alias, or the join becomes ambiguous."""
    for sport, table in teams.BY_SPORT.items():
        seen: dict[str, str] = {}
        for code, aliases in table.items():
            for alias in aliases:
                assert alias not in seen, f"{sport}: {alias!r} claimed by {seen.get(alias)} and {code}"
                seen[alias] = code
