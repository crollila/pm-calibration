"""Team reference data: canonical code -> (nickname, other aliases).

The canonical identifier for a team in this repo is its short code, because
that is what both venues put in their *structured* identifiers -- Polymarket's
slug (`mlb-stl-nyy-2026-08-03`) and Kalshi's ticker
(`KXMLBGAME-26AUG101945PHISTL-STL`). Free text cannot be joined: Kalshi names
markets by city ("St. Louis"), Polymarket by full team name ("St. Louis
Cardinals"), sportsbooks by full name again.

Aliases cover every spelling seen in the wild: the nickname, the city, and the
other venue's code where they disagree (Polymarket says `ari`/`oak`, Kalshi says
`az`/`ath`). The first alias of each entry is the nickname, which is what the
free-text matcher produces.
"""

from __future__ import annotations

# code -> aliases (nickname first, then cities / alternate codes)
MLB: dict[str, tuple[str, ...]] = {
    "ari": ("diamondbacks", "arizona", "az", "dbacks"),
    "atl": ("braves", "atlanta"),
    "bal": ("orioles", "baltimore"),
    "bos": ("red sox", "boston"),
    "chc": ("cubs", "chicago cubs"),
    "cws": ("white sox", "chicago white sox", "chw"),
    "cin": ("reds", "cincinnati"),
    "cle": ("guardians", "cleveland"),
    "col": ("rockies", "colorado"),
    "det": ("tigers", "detroit"),
    "hou": ("astros", "houston"),
    "kc": ("royals", "kansas city"),
    "laa": ("angels", "los angeles a", "anaheim"),
    "lad": ("dodgers", "los angeles d"),
    "mia": ("marlins", "miami"),
    "mil": ("brewers", "milwaukee"),
    "min": ("twins", "minnesota"),
    "nym": ("mets", "new york m"),
    "nyy": ("yankees", "new york y"),
    "oak": ("athletics", "oakland", "ath"),
    "phi": ("phillies", "philadelphia"),
    "pit": ("pirates", "pittsburgh"),
    "sd": ("padres", "san diego"),
    "sf": ("giants", "san francisco"),
    "sea": ("mariners", "seattle"),
    "stl": ("cardinals", "st louis", "st. louis"),
    "tb": ("rays", "tampa bay"),
    "tex": ("rangers", "texas"),
    "tor": ("blue jays", "toronto"),
    "wsh": ("nationals", "washington", "was"),
    # All-Star game "teams", which both venues list.
    "al": ("american league",),
    "nl": ("national league",),
}

NFL: dict[str, tuple[str, ...]] = {
    "ari": ("cardinals", "arizona"),
    "atl": ("falcons", "atlanta"),
    "bal": ("ravens", "baltimore"),
    "buf": ("bills", "buffalo"),
    "car": ("panthers", "carolina"),
    "chi": ("bears", "chicago"),
    "cin": ("bengals", "cincinnati"),
    "cle": ("browns", "cleveland"),
    "dal": ("cowboys", "dallas"),
    "den": ("broncos", "denver"),
    "det": ("lions", "detroit"),
    "gb": ("packers", "green bay"),
    "hou": ("texans", "houston"),
    "ind": ("colts", "indianapolis"),
    "jax": ("jaguars", "jacksonville", "jac"),
    "kc": ("chiefs", "kansas city"),
    "lv": ("raiders", "las vegas", "oak"),
    "lac": ("chargers", "los angeles c"),
    "lar": ("rams", "los angeles r"),
    "mia": ("dolphins", "miami"),
    "min": ("vikings", "minnesota"),
    "ne": ("patriots", "new england"),
    "no": ("saints", "new orleans"),
    "nyg": ("giants", "new york g"),
    "nyj": ("jets", "new york j"),
    "phi": ("eagles", "philadelphia"),
    "pit": ("steelers", "pittsburgh"),
    "sf": ("49ers", "san francisco"),
    "sea": ("seahawks", "seattle"),
    "tb": ("buccaneers", "tampa bay"),
    "ten": ("titans", "tennessee"),
    "wsh": ("commanders", "washington", "was"),
}

NBA: dict[str, tuple[str, ...]] = {
    "atl": ("hawks", "atlanta"),
    "bos": ("celtics", "boston"),
    "bkn": ("nets", "brooklyn", "bro"),
    "cha": ("hornets", "charlotte"),
    "chi": ("bulls", "chicago"),
    "cle": ("cavaliers", "cleveland"),
    "dal": ("mavericks", "dallas"),
    "den": ("nuggets", "denver"),
    "det": ("pistons", "detroit"),
    "gsw": ("warriors", "golden state", "gs"),
    "hou": ("rockets", "houston"),
    "ind": ("pacers", "indiana"),
    "lac": ("clippers", "la clippers"),
    "lal": ("lakers", "la lakers"),
    "mem": ("grizzlies", "memphis"),
    "mia": ("heat", "miami"),
    "mil": ("bucks", "milwaukee"),
    "min": ("timberwolves", "minnesota"),
    "nop": ("pelicans", "new orleans", "no"),
    "nyk": ("knicks", "new york"),
    "okc": ("thunder", "oklahoma city"),
    "orl": ("magic", "orlando"),
    "phi": ("76ers", "philadelphia"),
    "phx": ("suns", "phoenix", "pho"),
    "por": ("trail blazers", "portland"),
    "sac": ("kings", "sacramento"),
    "sas": ("spurs", "san antonio"),
    "tor": ("raptors", "toronto"),
    "uta": ("jazz", "utah", "uth"),
    "wsh": ("wizards", "washington", "was"),
}

WNBA: dict[str, tuple[str, ...]] = {
    "atl": ("dream", "atlanta"),
    "chi": ("sky", "chicago"),
    "con": ("sun", "connecticut"),
    "dal": ("wings", "dallas"),
    "gs": ("valkyries", "golden state"),
    "ind": ("fever", "indiana"),
    "la": ("sparks", "los angeles"),
    "lv": ("aces", "las vegas"),
    "min": ("lynx", "minnesota"),
    "ny": ("liberty", "new york"),
    "phx": ("mercury", "phoenix"),
    "sea": ("storm", "seattle"),
    "tor": ("tempo", "toronto"),
    "wsh": ("mystics", "washington", "was"),
}

BY_SPORT: dict[str, dict[str, tuple[str, ...]]] = {
    "mlb": MLB,
    "nfl": NFL,
    "nba": NBA,
    "wnba": WNBA,
}
