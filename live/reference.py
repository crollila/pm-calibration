"""Public step: scheduled starts and last pre-start prices for the traded events.

    uv run -m live.reference PATH/games_year.jsonl [more.jsonl ...]

Input is public data only: minute price paths of settled sports moneylines from
Polymarket's international venue (Gamma events + CLOB `prices-history`), as
written by the Polymarket-Algo-V2 `backfill.py`, one JSON object per line:
`{slug, start, outcomes, winner0, hist: [[minutes_from_start, price0], ...]}`.

That venue is a *different order book* from Polymarket US, where the trades were
placed. Its pre-start price is used only as a labelled closing-line proxy; the
report validates every mapping against the US settlement before using it.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

OUT = Path("data/live/event_reference.csv")
MARKETS = Path("data/live/markets.csv")
FIELDS = [
    "event_slug", "start_utc", "outcome0", "outcome1", "winner0",
    "close0_pre_start", "close_minute", "source",
]  # fmt: skip
SOURCE = "Polymarket international Gamma startTime + CLOB prices-history (fidelity 1m)"


def last_pre_start(hist: list[list]) -> tuple[float, int] | None:
    """Last price print strictly before the scheduled start (minute < 0).

    >>> last_pre_start([[-3, 0.4], [-1, 0.42], [0, 0.5]])
    (0.42, -1)
    >>> last_pre_start([[0, 0.5]]) is None
    True
    """
    pre = [h for h in hist if h[0] < 0]
    return (pre[-1][1], pre[-1][0]) if pre else None


def build(paths: list[Path], wanted: set[str]) -> list[dict]:
    rows = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                g = json.loads(line)
                if g["slug"] not in wanted or g["slug"] in rows:
                    continue
                close = last_pre_start(sorted(g.get("hist") or []))
                rows[g["slug"]] = {
                    "event_slug": g["slug"],
                    "start_utc": datetime.fromtimestamp(g["start"], UTC).strftime(
                        "%Y-%m-%dT%H:%M:%S.000000Z"
                    ),
                    "outcome0": g["outcomes"][0],
                    "outcome1": g["outcomes"][1],
                    "winner0": int(bool(g["winner0"])),
                    "close0_pre_start": "" if close is None else close[0],
                    "close_minute": "" if close is None else close[1],
                    "source": SOURCE,
                }
    return sorted(rows.values(), key=lambda r: r["event_slug"])


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        sys.exit(__doc__)
    with open(MARKETS, encoding="utf-8") as f:
        wanted = {r["event_slug"] for r in csv.DictReader(f) if r["kind"] == "single"}
    rows = build(paths, wanted)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} of {len(wanted)} single-market events found -> {OUT}")


if __name__ == "__main__":
    main()
