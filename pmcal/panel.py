"""Build the tidy analysis panel: one row per (event_key, outcome, ts).

Every column in a row is derived from information available *at* `ts`. The one
exception is the label `y`, which by construction comes from after the event and
is never used as a feature. `tests/test_no_lookahead.py` asserts this by
truncating the input at T and checking that no row at or before T changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmcal.consensus import book_reference

PANEL_COLUMNS = [
    "event_key", "sport", "outcome", "ts", "event_start_time", "minutes_to_start",
    "pm_bid", "pm_ask", "pm_mid", "pm_bid_size", "pm_ask_size",
    "kalshi_bid", "kalshi_ask", "kalshi_mid", "kalshi_bid_size", "kalshi_ask_size",
    "book_multiplicative", "book_additive", "book_shin",
    "pinnacle_multiplicative", "pinnacle_additive", "pinnacle_shin",
    "n_books", "y", "y_agreement",
]

_VENUE_PREFIX = {"polymarket": "pm", "kalshi": "kalshi"}


def _market_side(snapshots: pd.DataFrame, venue: str, prefix: str) -> pd.DataFrame:
    """Top-of-book for one prediction market, one row per (event_key, outcome, ts)."""
    cols = ["event_key", "outcome", "ts", "bid", "ask", "mid", "bid_size", "ask_size"]
    side = snapshots.loc[snapshots["venue"] == venue, cols].copy()
    if side.empty:
        return pd.DataFrame(columns=["event_key", "outcome", "ts"])
    # A venue can list the same game twice (duplicate markets); keep the most
    # liquid quote, which is the one with the tightest two-sided spread.
    side["spread"] = (side["ask"] - side["bid"]).abs().fillna(np.inf)
    side = (
        side.sort_values("spread")
        .drop_duplicates(subset=["event_key", "outcome", "ts"], keep="first")
        .drop(columns="spread")
    )
    return side.rename(
        columns={c: f"{prefix}_{c}" for c in ("bid", "ask", "mid", "bid_size", "ask_size")}
    )


def _book_side(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Consensus and Pinnacle fair probabilities, wide by method."""
    quotes = snapshots.loc[snapshots["venue"] == "sportsbook"]
    reference = book_reference(quotes)
    if reference.empty:
        return pd.DataFrame(columns=["event_key", "outcome", "ts"])
    wide = reference.pivot_table(
        index=["event_key", "outcome", "ts"],
        columns=["venue", "method"],
        values="prob",
        aggfunc="first",
    )
    wide.columns = [
        f"{'book' if venue == 'book_consensus' else venue}_{method}" for venue, method in wide.columns
    ]
    counts = (
        reference.groupby(["event_key", "outcome", "ts"], sort=False)["n_books"].max().rename("n_books")
    )
    return wide.join(counts).reset_index()


def _event_start_asof(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Best kickoff time *known at each ts*, per event.

    The Odds API `commence_time` is the scheduled start; Kalshi's `close_time`
    and Polymarket's `gameStartTime` are weaker proxies. We prefer the
    sportsbook value -- but only from the moment we have actually observed it.
    Taking the best value over the whole history would let a book quote that
    arrives at 18:00 change `minutes_to_start` on a 12:00 row, which is exactly
    the kind of quiet lookahead this panel is supposed to rule out.
    """
    cols = ["event_key", "ts", "event_start_time", "sport"]
    frame = snapshots.dropna(subset=["event_key", "event_start_time"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=cols)

    frame["_rank"] = np.where(frame["venue"] == "sportsbook", 0, 1)
    per_ts = (
        frame.sort_values(["event_key", "ts", "_rank", "event_start_time"])
        .drop_duplicates(subset=["event_key", "ts"], keep="first")
        .loc[:, ["event_key", "ts", "_rank", "event_start_time", "sport"]]
        .sort_values(["event_key", "ts"])
        .reset_index(drop=True)
    )
    best_so_far = per_ts.groupby("event_key", sort=False)["_rank"].cummin()
    fresh = per_ts["_rank"] <= best_so_far
    per_ts["event_start_time"] = per_ts["event_start_time"].where(fresh)
    per_ts["sport"] = per_ts["sport"].where(fresh)
    per_ts[["event_start_time", "sport"]] = per_ts.groupby("event_key", sort=False)[
        ["event_start_time", "sport"]
    ].ffill()
    return per_ts[cols]


def build_panel(snapshots: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join market snapshots to realised outcomes.

    `snapshots` -- rows from the `snapshots` table (all venues).
    `labels`    -- event_key, outcome, y, y_agreement from `ground_truth`.
    """
    if snapshots.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    snapshots = snapshots.dropna(subset=["event_key"]).copy()
    frames = [
        _market_side(snapshots, venue, prefix) for venue, prefix in _VENUE_PREFIX.items()
    ]
    frames.append(_book_side(snapshots))

    panel: pd.DataFrame | None = None
    for frame in frames:
        if frame.empty:
            continue
        panel = frame if panel is None else panel.merge(
            frame, on=["event_key", "outcome", "ts"], how="outer"
        )
    if panel is None:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    panel = panel.merge(_event_start_asof(snapshots), on=["event_key", "ts"], how="left")
    panel = panel.merge(labels, on=["event_key", "outcome"], how="left")

    start = pd.to_datetime(panel["event_start_time"])
    panel["minutes_to_start"] = (start - pd.to_datetime(panel["ts"])).dt.total_seconds() / 60.0

    for col in PANEL_COLUMNS:
        if col not in panel.columns:
            panel[col] = np.nan
    panel = panel[PANEL_COLUMNS].sort_values(["event_key", "outcome", "ts"]).reset_index(drop=True)

    # Pre-event only: a quote taken after kickoff is a different forecasting
    # problem (live betting) and would flatter whichever venue is slower to halt.
    return panel[panel["minutes_to_start"] > 0].reset_index(drop=True)


def ground_truth(resolutions: pd.DataFrame) -> pd.DataFrame:
    """Reconcile per-venue settlements into a single label.

    Unanimous -> that value. Strict majority -> the majority value, with
    `y_agreement` recording the fraction that agreed. A tie -> `y` is NULL and
    the row drops out of the analysis; we do not break ties by venue preference.
    """
    cols = ["event_key", "outcome", "y", "y_agreement"]
    resolved = resolutions.dropna(subset=["resolved", "event_key"])
    if resolved.empty:
        return pd.DataFrame(columns=cols)

    def _reduce(group: pd.DataFrame) -> pd.Series:
        votes = group.drop_duplicates(subset=["venue"])["resolved"]
        counts = votes.value_counts()
        top = counts.max()
        winners = counts[counts == top].index.tolist()
        value = float(winners[0]) if len(winners) == 1 else np.nan
        return pd.Series({"y": value, "y_agreement": float(top) / float(len(votes))})

    out = (
        resolved.groupby(["event_key", "outcome"], sort=False)[["venue", "resolved"]]
        .apply(_reduce)
        .reset_index()
    )
    return out[cols]


def find_conflicts(resolutions: pd.DataFrame, detected_at: pd.Timestamp) -> pd.DataFrame:
    """Every pairwise cross-venue settlement disagreement, as its own row."""
    cols = ["event_key", "outcome", "venue_a", "value_a", "venue_b", "value_b",
            "detected_at", "note"]
    resolved = resolutions.dropna(subset=["resolved", "event_key"])
    if resolved.empty:
        return pd.DataFrame(columns=cols)

    records: list[dict[str, object]] = []
    for (event_key, outcome), group in resolved.groupby(["event_key", "outcome"], sort=False):
        votes = group.drop_duplicates(subset=["venue"])[["venue", "resolved"]].values.tolist()
        for i in range(len(votes)):
            for j in range(i + 1, len(votes)):
                (venue_a, value_a), (venue_b, value_b) = votes[i], votes[j]
                if float(value_a) == float(value_b):
                    continue
                records.append(
                    {
                        "event_key": event_key,
                        "outcome": outcome,
                        "venue_a": venue_a,
                        "value_a": float(value_a),
                        "venue_b": venue_b,
                        "value_b": float(value_b),
                        "detected_at": detected_at,
                        "note": "cross-venue settlement disagreement",
                    }
                )
    return pd.DataFrame.from_records(records, columns=cols)
