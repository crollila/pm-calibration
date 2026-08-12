"""Calibration: Brier with Murphy decomposition, log loss, reliability curves,
favourite-longshot regression, and sharpening over time-to-event.

Every number that gets reported carries a clustered interval. Rows are clustered
by `event_key`: the two sides of a game are the same observation seen twice, and
snapshots 15 minutes apart are not new information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmcal.odds import clip_prob
from pmcal.stats import cluster_bootstrap, ols_cluster, paired_difference, wilson_interval

VENUE_COLUMNS = {
    "polymarket": "pm_mid",
    "kalshi": "kalshi_mid",
    "book_consensus": "book_shin",
    "pinnacle": "pinnacle_shin",
}

DEFAULT_BINS = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
TIME_BUCKETS = np.array([0, 60, 180, 360, 720, 1440, 4320, np.inf])


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of a probabilistic forecast. Lower is better.

    >>> round(brier_score(np.array([0.7, 0.3]), np.array([1.0, 0.0])), 4)
    0.09
    """
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log likelihood.

    Unlike Brier, log loss is unbounded: a forecast of 0.001 that resolves YES
    costs 6.9 nats on its own, where Brier caps the damage at 1.0. Venues that
    quote extreme prices near expiry are punished far harder here, so the two
    metrics answering differently is informative, not a contradiction.

    >>> round(log_loss(np.array([0.7, 0.3]), np.array([1.0, 0.0])), 4)
    0.3567
    """
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def murphy_decomposition(
    p: np.ndarray, y: np.ndarray, bins: np.ndarray = DEFAULT_BINS
) -> dict[str, float]:
    """Brier = reliability - resolution + uncertainty.

    * **reliability** -- how far observed frequency sits from the forecast in
      each bin. Zero for a perfectly calibrated forecaster. Lower is better.
    * **resolution** -- how far the bin frequencies sit from the base rate, i.e.
      how much the forecaster discriminates. Higher is better.
    * **uncertainty** -- base-rate variance. A property of the events, identical
      across venues on the same sample, and not a score anyone can improve.

    The identity is exact only when every forecast inside a bin is identical, so
    `residual` reports the binning error rather than hiding it.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return dict.fromkeys(
            ("brier", "reliability", "resolution", "uncertainty", "residual", "n"), np.nan
        )
    base = float(y.mean())
    idx = np.clip(np.digitize(p, bins[1:-1], right=True), 0, len(bins) - 2)

    reliability = resolution = 0.0
    for k in np.unique(idx):
        mask = idx == k
        n_k = int(mask.sum())
        f_k, o_k = float(p[mask].mean()), float(y[mask].mean())
        reliability += n_k * (f_k - o_k) ** 2
        resolution += n_k * (o_k - base) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)
    total = brier_score(p, y)
    return {
        "brier": total,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "residual": total - (reliability - resolution + uncertainty),
        "n": float(n),
    }


def calibration_table(
    p: np.ndarray, y: np.ndarray, bins: np.ndarray = DEFAULT_BINS, alpha: float = 0.05,
    clusters: np.ndarray | None = None, n_boot: int = 2_000,
) -> pd.DataFrame:
    """Binned observed frequency vs forecast, with Wilson and clustered intervals.

    The Wilson interval treats every row as an independent Bernoulli draw. It is
    not: forty hourly snapshots of one game are one observation seen forty
    times. Passing `clusters` adds an event-clustered bootstrap interval, which
    on real data comes out roughly `sqrt(rows per event)` times wider -- about
    6x here. Reporting only the Wilson band would show statistically
    significant miscalibration that does not survive clustering, so both are
    computed and the clustered one is what the plots draw.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    idx = np.clip(np.digitize(p, bins[1:-1], right=True), 0, len(bins) - 2)
    rows = []
    for k in range(len(bins) - 1):
        mask = idx == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        successes = float(y[mask].sum())
        lo, hi = wilson_interval(successes, n_k, alpha)
        record = {
            "bin_lo": float(bins[k]),
            "bin_hi": float(bins[k + 1]),
            "n": n_k,
            "mean_forecast": float(p[mask].mean()),
            "observed": successes / n_k,
            "wilson_lo": lo,
            "wilson_hi": hi,
        }
        if clusters is not None:
            values = y[mask]
            ci = cluster_bootstrap(
                lambda i, v=values: float(v[i].mean()),
                np.asarray(clusters)[mask], n_boot=n_boot, alpha=alpha,
            )
            record |= {
                "n_events": int(ci["n_clusters"]),
                "cluster_lo": ci["lo"],
                "cluster_hi": ci["hi"],
            }
        rows.append(record)
    return pd.DataFrame(rows)


def _clean(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return frame.dropna(subset=[column, "y", "event_key"])


def venue_summary(
    panel: pd.DataFrame,
    venue_columns: dict[str, str] = VENUE_COLUMNS,
    n_boot: int = 10_000,
) -> pd.DataFrame:
    """Brier, its decomposition, and log loss per venue with clustered CIs."""
    rows = []
    for venue, column in venue_columns.items():
        if column not in panel.columns:
            continue
        frame = _clean(panel, column)
        if frame.empty:
            continue
        p = frame[column].to_numpy(float)
        y = frame["y"].to_numpy(float)
        clusters = frame["event_key"].to_numpy()

        # Default arguments bind the current venue's arrays into each closure.
        brier_ci = cluster_bootstrap(
            lambda i, p=p, y=y: brier_score(p[i], y[i]), clusters, n_boot=n_boot
        )
        ll_ci = cluster_bootstrap(
            lambda i, p=p, y=y: log_loss(p[i], y[i]), clusters, n_boot=n_boot
        )
        parts = murphy_decomposition(p, y)
        rows.append(
            {
                "venue": venue,
                "column": column,
                "n_rows": len(frame),
                "n_events": int(frame["event_key"].nunique()),
                "brier": brier_ci["point"],
                "brier_lo": brier_ci["lo"],
                "brier_hi": brier_ci["hi"],
                "log_loss": ll_ci["point"],
                "log_loss_lo": ll_ci["lo"],
                "log_loss_hi": ll_ci["hi"],
                **{k: parts[k] for k in ("reliability", "resolution", "uncertainty", "residual")},
            }
        )
    return pd.DataFrame(rows)


def favourite_longshot(
    panel: pd.DataFrame, venue_columns: dict[str, str] = VENUE_COLUMNS
) -> pd.DataFrame:
    """Regress realised outcome on forecast probability, per venue."""
    rows = []
    for venue, column in venue_columns.items():
        if column not in panel.columns:
            continue
        frame = _clean(panel, column)
        if len(frame) < 30:
            continue
        fit = ols_cluster(
            frame[column].to_numpy(float),
            frame["y"].to_numpy(float),
            frame["event_key"].to_numpy(),
        )
        rows.append({"venue": venue, **fit})
    return pd.DataFrame(rows)


def sharpening_profile(
    panel: pd.DataFrame,
    venue_columns: dict[str, str] = VENUE_COLUMNS,
    buckets: np.ndarray = TIME_BUCKETS,
    n_boot: int = 2_000,
) -> pd.DataFrame:
    """Brier score by minutes-to-event bucket: who sharpens, and how fast.

    A lower bootstrap count here than in the headline table: this is a
    descriptive breakdown across many thin buckets, not a headline claim.
    """
    rows = []
    labels = [f"{int(buckets[i])}-{buckets[i + 1]:.0f}m" for i in range(len(buckets) - 1)]
    for venue, column in venue_columns.items():
        if column not in panel.columns:
            continue
        frame = _clean(panel, column)
        if frame.empty:
            continue
        bucket = np.digitize(frame["minutes_to_start"].to_numpy(float), buckets[1:-1], right=True)
        for k, label in enumerate(labels):
            sub = frame[bucket == k]
            if len(sub) < 20:
                continue
            p = sub[column].to_numpy(float)
            y = sub["y"].to_numpy(float)
            ci = cluster_bootstrap(
                lambda i, p=p, y=y: brier_score(p[i], y[i]),
                sub["event_key"].to_numpy(),
                n_boot=n_boot,
            )
            rows.append(
                {
                    "venue": venue,
                    "bucket": label,
                    "bucket_index": k,
                    "n_rows": len(sub),
                    "n_events": int(sub["event_key"].nunique()),
                    "brier": ci["point"],
                    "brier_lo": ci["lo"],
                    "brier_hi": ci["hi"],
                }
            )
    return pd.DataFrame(rows)


def head_to_head(
    panel: pd.DataFrame, venue_a: str, venue_b: str,
    venue_columns: dict[str, str] = VENUE_COLUMNS, n_boot: int = 10_000,
) -> dict[str, float]:
    """Paired Brier difference (a - b) on rows where both venues quote.

    Negative means venue A is better calibrated. Because it is paired, this is
    the comparison to quote -- two separately estimated Brier scores with
    overlapping CIs can still differ significantly on the same events.
    """
    col_a, col_b = venue_columns[venue_a], venue_columns[venue_b]
    frame = panel.dropna(subset=[col_a, col_b, "y", "event_key"])
    if frame.empty:
        return {"venue_a": venue_a, "venue_b": venue_b, "n_rows": 0}
    y = frame["y"].to_numpy(float)
    sq_a = (frame[col_a].to_numpy(float) - y) ** 2
    sq_b = (frame[col_b].to_numpy(float) - y) ** 2
    ci = paired_difference(sq_a, sq_b, frame["event_key"].to_numpy(), n_boot=n_boot)
    return {
        "venue_a": venue_a,
        "venue_b": venue_b,
        "n_rows": len(frame),
        "n_events": int(frame["event_key"].nunique()),
        "brier_diff": ci["point"],
        "diff_lo": ci["lo"],
        "diff_hi": ci["hi"],
        "significant": bool(np.isfinite(ci["lo"]) and (ci["lo"] > 0 or ci["hi"] < 0)),
    }


def devig_method_comparison(panel: pd.DataFrame, n_boot: int = 2_000) -> pd.DataFrame:
    """Does the choice of de-vig method change the book's calibration verdict?

    On two-way markets Shin and additive are algebraically identical (see
    `devig.py`), so any spread here is multiplicative against the other two --
    and it will be concentrated in the longshot bins.
    """
    columns = {
        f"book_{m}": f"book_{m}" for m in ("multiplicative", "additive", "shin")
    }
    summary = venue_summary(panel, columns, n_boot=n_boot)
    if summary.empty:
        return summary
    longshots = panel[panel["book_shin"] < 0.25] if "book_shin" in panel else panel.iloc[0:0]
    tail = venue_summary(longshots, columns, n_boot=n_boot) if len(longshots) > 30 else None
    if tail is not None and not tail.empty:
        summary = summary.merge(
            tail[["venue", "brier", "n_rows"]].rename(
                columns={"brier": "brier_longshots", "n_rows": "n_rows_longshots"}
            ),
            on="venue",
            how="left",
        )
    return summary


def clip_panel_probabilities(panel: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep probabilities strictly inside (0, 1) so log loss stays finite."""
    out = panel.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].map(lambda v: np.nan if pd.isna(v) else clip_prob(float(v)))
    return out
