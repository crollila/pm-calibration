"""Turn per-book quotes into a book forecast.

Two references are produced at every timestamp:

* **consensus** -- the median fair probability across all books quoting the
  event. The median (not the mean) because a single stale or erroneous book
  should not move the reference.
* **pinnacle** -- Pinnacle alone when present, the standard sharp benchmark.
  Reported separately, never folded into the consensus.

De-vig runs *per book* before aggregation. Averaging quoted (vig-inclusive)
prices and de-vigging afterwards is a different, worse estimator: books have
different margins, and the overround of an average is not the average overround.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from devig import DISPATCH, METHODS

PINNACLE_KEYS = ("pinnacle",)


def devig_per_book(quotes: pd.DataFrame, methods: tuple[str, ...] = METHODS) -> pd.DataFrame:
    """De-vig each (ts, event_key, book) quote set with every method.

    `quotes` needs columns: ts, event_key, book, outcome, ask.
    Returns long format: ts, event_key, book, outcome, method, prob.
    """
    required = {"ts", "event_key", "book", "outcome", "ask"}
    missing = required - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes missing columns: {sorted(missing)}")

    clean = quotes.dropna(subset=["ask", "event_key"]).copy()
    records: list[dict[str, object]] = []
    for (ts, event_key, book), group in clean.groupby(["ts", "event_key", "book"], sort=False):
        group = group.drop_duplicates(subset=["outcome"])
        if len(group) < 2 or group["ask"].sum() <= 1.0:
            # Fewer than two outcomes, or an underround: not a de-viggable book.
            continue
        asks = group["ask"].to_numpy(dtype=float)
        outcomes = group["outcome"].tolist()
        for method in methods:
            fair = DISPATCH[method](asks)
            for outcome, prob in zip(outcomes, fair, strict=True):
                records.append(
                    {
                        "ts": ts,
                        "event_key": event_key,
                        "book": book,
                        "outcome": outcome,
                        "method": method,
                        "prob": float(prob),
                    }
                )
    return pd.DataFrame.from_records(
        records, columns=["ts", "event_key", "book", "outcome", "method", "prob"]
    )


def aggregate(per_book: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-book fair probabilities into consensus and Pinnacle series.

    Returns: ts, event_key, outcome, method, venue, prob, n_books.
    `venue` is 'book_consensus' or 'pinnacle' so it slots straight into the
    same venue column the prediction markets use.
    """
    if per_book.empty:
        return pd.DataFrame(
            columns=["ts", "event_key", "outcome", "method", "venue", "prob", "n_books"]
        )

    keys = ["ts", "event_key", "outcome", "method"]
    consensus = (
        per_book.groupby(keys, sort=False)["prob"]
        .agg(prob="median", n_books="size")
        .reset_index()
    )
    consensus["venue"] = "book_consensus"

    pinn = per_book[per_book["book"].str.lower().isin(PINNACLE_KEYS)]
    if pinn.empty:
        return consensus[["ts", "event_key", "outcome", "method", "venue", "prob", "n_books"]]

    pinn_agg = (
        pinn.groupby(keys, sort=False)["prob"].agg(prob="median", n_books="size").reset_index()
    )
    pinn_agg["venue"] = "pinnacle"
    cols = ["ts", "event_key", "outcome", "method", "venue", "prob", "n_books"]
    return pd.concat([consensus[cols], pinn_agg[cols]], ignore_index=True)


def renormalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Rescale each (ts, event_key, method, venue) group to sum to 1.

    Taking a median outcome-by-outcome breaks the adding-up constraint by a
    fraction of a percent. Renormalising restores it; the size of the
    adjustment is kept in `median_residual` because a large residual means the
    books disagree about the *shape* of the market, not just its level.
    """
    if frame.empty:
        return frame.assign(median_residual=[])
    keys = ["ts", "event_key", "method", "venue"]
    totals = frame.groupby(keys, sort=False)["prob"].transform("sum")
    out = frame.copy()
    out["median_residual"] = totals - 1.0
    out["prob"] = np.where(totals > 0, out["prob"] / totals, np.nan)
    return out


def book_reference(quotes: pd.DataFrame) -> pd.DataFrame:
    """End-to-end: raw book quotes -> normalised consensus and Pinnacle series."""
    return renormalise(aggregate(devig_per_book(quotes)))
