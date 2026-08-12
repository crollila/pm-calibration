"""Figures for the writeup. Every plot shows its uncertainty."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.calibration import DEFAULT_BINS, calibration_table  # noqa: E402

PALETTE = {
    "polymarket": "#1f77b4",
    "kalshi": "#d62728",
    "book_consensus": "#2ca02c",
    "pinnacle": "#9467bd",
}


def _finish(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def calibration_curve(
    panel: pd.DataFrame, venue_columns: dict[str, str], path: Path,
    bins: np.ndarray = DEFAULT_BINS,
) -> Path:
    """Observed frequency vs forecast, all venues overlaid on the 45 degree line."""
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.plot([0, 1], [0, 1], color="#444444", lw=1, ls="--", label="perfect calibration", zorder=1)

    clustered = False
    for venue, column in venue_columns.items():
        if column not in panel.columns:
            continue
        frame = panel.dropna(subset=[column, "y"])
        if frame.empty:
            continue
        has_events = "event_key" in frame.columns
        table = calibration_table(
            frame[column].to_numpy(float), frame["y"].to_numpy(float), bins,
            clusters=frame["event_key"].to_numpy() if has_events else None,
        )
        if table.empty:
            continue
        colour = PALETTE.get(venue, "#666666")
        # The thin inner bar is the naive per-row interval; the outer bar is the
        # event-clustered one. Showing both makes the gap between them the point.
        if "cluster_lo" in table.columns:
            clustered = True
            outer = np.vstack(
                [table["observed"] - table["cluster_lo"], table["cluster_hi"] - table["observed"]]
            ).clip(min=0)
            ax.errorbar(table["mean_forecast"], table["observed"], yerr=outer,
                        fmt="none", ecolor=colour, alpha=0.45, elinewidth=1.2, capsize=5, zorder=2)
        inner = np.vstack(
            [table["observed"] - table["wilson_lo"], table["wilson_hi"] - table["observed"]]
        ).clip(min=0)
        ax.errorbar(
            table["mean_forecast"], table["observed"], yerr=inner,
            marker="o", ms=4, lw=1.4, capsize=2, elinewidth=2.2,
            color=colour, label=f"{venue} (n={int(table['n'].sum())})", zorder=3,
        )

    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(
        "Calibration: thick bar = per-row Wilson 95%,\n"
        "faint bar = event-clustered 95% (the honest one)"
        if clustered else "Calibration, with Wilson 95% intervals per bin"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    return _finish(fig, path)


def reliability_residuals(
    panel: pd.DataFrame, venue_columns: dict[str, str], path: Path,
    bins: np.ndarray = DEFAULT_BINS,
) -> Path:
    """observed - forecast by bin: the favourite-longshot signature, magnified."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(0.0, color="#444444", lw=1, ls="--")
    for venue, column in venue_columns.items():
        if column not in panel.columns:
            continue
        frame = panel.dropna(subset=[column, "y"])
        if frame.empty:
            continue
        table = calibration_table(frame[column].to_numpy(float), frame["y"].to_numpy(float), bins)
        if table.empty:
            continue
        ax.plot(
            table["mean_forecast"], table["observed"] - table["mean_forecast"],
            marker="o", ms=4, color=PALETTE.get(venue, "#666666"), label=venue,
        )
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed - forecast")
    ax.set_title("Calibration error by bin (negative on longshots = favourite-longshot bias)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return _finish(fig, path)


def sharpening(profile: pd.DataFrame, path: Path) -> Path:
    """Brier score by time-to-event bucket, with bootstrap intervals."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for venue, group in profile.groupby("venue"):
        group = group.sort_values("bucket_index")
        ax.errorbar(
            group["bucket_index"], group["brier"],
            yerr=np.vstack(
                [group["brier"] - group["brier_lo"], group["brier_hi"] - group["brier"]]
            ).clip(min=0),
            marker="o", ms=4, capsize=3, color=PALETTE.get(str(venue), "#666666"), label=str(venue),
        )
    if not profile.empty:
        labels = profile.sort_values("bucket_index")["bucket"].unique()
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("minutes to event start (closer to kickoff on the left)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Do forecasts sharpen as the event approaches?")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return _finish(fig, path)


def threshold_sweep(sweep_df: pd.DataFrame, path: Path) -> Path:
    """The full k-sweep: return on capital-days, sample size, and CLV together."""
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.4), sharex=True)
    ax0, ax1, ax2 = axes

    ax0.axhline(0.0, color="#444444", lw=1, ls="--")
    ax0.plot(sweep_df["threshold"], sweep_df["return_on_capital_days"], marker="o", ms=4,
             color="#1f77b4")
    if {"rocd_lo", "rocd_hi"} <= set(sweep_df.columns):
        ax0.fill_between(sweep_df["threshold"], sweep_df["rocd_lo"], sweep_df["rocd_hi"],
                         alpha=0.18, color="#1f77b4")
    ax0.set_ylabel("return per capital-day")
    ax0.set_title("Threshold sweep, reported in full (no cherry-picked k)")
    ax0.grid(alpha=0.25)

    ax1.plot(sweep_df["threshold"], sweep_df["n_events"], marker="o", ms=4, color="#7f7f7f")
    ax1.set_ylabel("independent events")
    ax1.grid(alpha=0.25)

    if "clv_mean" in sweep_df.columns:
        ax2.axhline(0.0, color="#444444", lw=1, ls="--")
        ax2.plot(sweep_df["threshold"], sweep_df["clv_mean"], marker="o", ms=4, color="#2ca02c")
        if {"clv_lo", "clv_hi"} <= set(sweep_df.columns):
            ax2.fill_between(sweep_df["threshold"], sweep_df["clv_lo"], sweep_df["clv_hi"],
                             alpha=0.18, color="#2ca02c")
    ax2.set_ylabel("mean CLV (prob. points)")
    ax2.set_xlabel("divergence threshold k")
    ax2.grid(alpha=0.25)
    return _finish(fig, path)


def kelly_uncertainty(curve: list[dict[str, float]], path: Path) -> Path:
    """Optimal stake as a fraction of full Kelly, as the estimate gets noisier."""
    frame = pd.DataFrame(curve)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(frame["sigma"], frame["ratio"], marker="o", ms=4, color="#d62728")
    ax.axhline(1.0, color="#444444", lw=1, ls="--", label="full Kelly")
    ax.axhline(0.25, color="#2ca02c", lw=1, ls=":", label="quarter Kelly")
    ax.set_xlabel("standard deviation of the probability estimate")
    ax.set_ylabel("optimal stake / full-Kelly stake")
    ax.set_title("Full Kelly is only optimal when p is known exactly")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return _finish(fig, path)


def devig_disagreement(panel: pd.DataFrame, path: Path) -> Path:
    """Where the three de-vig methods disagree, as a function of price level."""
    cols = ["book_multiplicative", "book_additive", "book_shin"]
    frame = panel.dropna(subset=cols)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    if not frame.empty:
        base = frame["book_shin"].to_numpy(float)
        for col, colour in zip(cols, ("#1f77b4", "#ff7f0e", "#2ca02c"), strict=True):
            ax.scatter(base, frame[col].to_numpy(float) - base, s=6, alpha=0.35,
                       color=colour, label=col.replace("book_", ""))
    ax.axhline(0.0, color="#444444", lw=1, ls="--")
    ax.set_xlabel("Shin fair probability")
    ax.set_ylabel("method - Shin")
    ax.set_title("De-vig methods diverge on longshots (and coincide on two-way markets)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    return _finish(fig, path)
