#!/usr/bin/env python
"""Run the whole analysis and write figures plus a results summary.

    uv run -m analysis.run_all
    uv run -m analysis.run_all --panel data/panel.parquet --n-boot 10000

Calibration runs first and unconditionally. The backtest runs second and its
output is written next to the calibration result, never instead of it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import backtest, calibration, kelly, plots
from pmcal import db
from pmcal.config import FIGURES_DIR, load_config
from pmcal.util import setup_logging

log = logging.getLogger("run_all")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PAIRS = [("polymarket", "book_consensus"), ("kalshi", "book_consensus"),
         ("polymarket", "kalshi"), ("polymarket", "pinnacle")]


def load_panel(db_path: str | Path, parquet: str | None) -> pd.DataFrame:
    if parquet:
        return pd.read_parquet(parquet)
    con = db.connect(db_path, read_only=True)
    try:
        return con.execute("SELECT * FROM analysis_panel").df()
    finally:
        con.close()


def _write(frame: pd.DataFrame, name: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / f"{name}.csv", index=False)


def available_venues(panel: pd.DataFrame) -> dict[str, str]:
    """Only report venues that actually have data in this sample."""
    return {
        venue: column
        for venue, column in calibration.VENUE_COLUMNS.items()
        if column in panel.columns and panel[column].notna().any()
    }


def run_calibration(panel: pd.DataFrame, n_boot: int) -> dict[str, object]:
    venues = available_venues(panel)
    log.info("venues with data: %s", ", ".join(venues) or "(none)")

    summary = calibration.venue_summary(panel, venues, n_boot=n_boot)
    _write(summary, "calibration_summary")

    flb = calibration.favourite_longshot(panel, venues)
    _write(flb, "favourite_longshot")

    # Per-bin table with both interval flavours, so the gap between them is on
    # record rather than only visible in the plot.
    bins = []
    for venue, column in venues.items():
        frame = panel.dropna(subset=[column, "y", "event_key"])
        if frame.empty:
            continue
        table = calibration.calibration_table(
            frame[column].to_numpy(float), frame["y"].to_numpy(float),
            clusters=frame["event_key"].to_numpy(),
        )
        bins.append(table.assign(venue=venue))
    bin_table = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()
    _write(bin_table, "calibration_bins")

    profile = calibration.sharpening_profile(panel, venues, n_boot=max(n_boot // 5, 500))
    _write(profile, "sharpening_profile")

    devig_cmp = calibration.devig_method_comparison(panel, n_boot=max(n_boot // 5, 500))
    _write(devig_cmp, "devig_comparison")

    pairs = [
        calibration.head_to_head(panel, a, b, venues, n_boot=n_boot)
        for a, b in PAIRS
        if a in venues and b in venues
    ]
    head = pd.DataFrame(pairs)
    _write(head, "head_to_head")

    plots.calibration_curve(panel, venues, FIGURES_DIR / "calibration_curve.png")
    plots.reliability_residuals(panel, venues, FIGURES_DIR / "reliability_residuals.png")
    if not profile.empty:
        plots.sharpening(profile, FIGURES_DIR / "sharpening.png")
    plots.devig_disagreement(panel, FIGURES_DIR / "devig_disagreement.png")

    return {
        "venues": venues,
        "summary": summary,
        "bins": bin_table,
        "flb": flb,
        "profile": profile,
        "head_to_head": head,
        "devig": devig_cmp,
    }


def tradeability_blockers(panel: pd.DataFrame, venue: str, reference: str) -> list[str]:
    """Why a backtest cannot honestly run, stated rather than reported as zero trades.

    A sample with no executable ask, no displayed depth, or no reference venue
    does not produce "no edge" -- it produces no test at all, and the two must
    not look the same in the writeup.
    """
    ask_col, size_col, _ = backtest.TRADE_COLUMNS[venue]
    blockers = []
    if ask_col not in panel.columns or not panel[ask_col].notna().any():
        blockers.append(f"no executable ask price for {venue} ({ask_col} is empty)")
    if size_col not in panel.columns or not panel[size_col].notna().any():
        blockers.append(f"no top-of-book depth for {venue} ({size_col} is empty)")
    if reference not in panel.columns or not panel[reference].notna().any():
        blockers.append(f"no reference venue ({reference} is empty)")
    return blockers


def run_tradeability(panel: pd.DataFrame, cfg_poly_fee: float, n_boot: int) -> dict[str, object]:
    results: dict[str, object] = {}
    for venue in ("polymarket", "kalshi"):
        ask_col = backtest.TRADE_COLUMNS[venue][0]
        blockers = tradeability_blockers(panel, venue, "book_shin")
        if blockers:
            log.warning("backtest skipped for %s: %s", venue, "; ".join(blockers))
            results[venue] = {"skipped": blockers}
            continue
        if ask_col not in panel.columns or not panel[ask_col].notna().any():
            continue
        cfg = backtest.BacktestConfig(trade_venue=venue, poly_fee_rate=cfg_poly_fee)
        trades, stats = backtest.run(panel, cfg, n_boot=n_boot)
        sweep = backtest.sweep(panel, cfg, n_boot=max(n_boot // 5, 500))
        _write(sweep, f"threshold_sweep_{venue}")
        if not trades.empty:
            _write(trades, f"trades_{venue}")
            plots.threshold_sweep(sweep, FIGURES_DIR / f"threshold_sweep_{venue}.png")
        results[venue] = {"stats": stats, "sweep": sweep, "n_trades": len(trades)}
        log.info("%s: %d trades, %s", venue, len(trades), json.dumps(stats, default=str)[:400])

    curve = kelly.uncertainty_curve(p_hat=0.55, price=0.50)
    _write(pd.DataFrame(curve), "kelly_uncertainty")
    plots.kelly_uncertainty(curve, FIGURES_DIR / "kelly_uncertainty.png")
    results["kelly_curve"] = curve
    return results


def write_summary(panel: pd.DataFrame, calib: dict, trade: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "summary.md"
    labelled = panel[panel["y"].notna()]
    lines = [
        "# Results",
        "",
        "_Generated by `analysis/run_all.py`. Numbers here are what the README quotes._",
        "",
        "## Sample",
        "",
        f"- panel rows: **{len(panel):,}** ({len(labelled):,} labelled)",
        f"- independent events: **{labelled['event_key'].nunique():,}**",
        f"- collection window: {panel['ts'].min()} to {panel['ts'].max()} (UTC)",
        f"- sports: {', '.join(sorted(str(s) for s in panel['sport'].dropna().unique())) or 'n/a'}",
        "",
        "## Calibration",
        "",
        _table(calib["summary"]),
        "",
        "### Reliability bins: per-row Wilson vs event-clustered intervals",
        "",
        _table(calib["bins"]),
        "",
        "### Favourite-longshot regression (y ~ a + b p, cluster-robust SE)",
        "",
        _table(calib["flb"]),
        "",
        "### Paired Brier differences (negative favours venue A)",
        "",
        _table(calib["head_to_head"]),
        "",
        "### De-vig method comparison",
        "",
        _table(calib["devig"]),
        "",
        "## Tradeability",
        "",
    ]
    for venue, payload in trade.items():
        if venue == "kelly_curve":
            continue
        if "skipped" in payload:
            lines += [
                f"### Trading {venue}: not run",
                "",
                "The cost model cannot be applied honestly on this sample:",
                "",
                *[f"- {reason}" for reason in payload["skipped"]],
                "",
                "This is a gap in the data, not a finding of no edge. The two are",
                "different claims and are not reported as if they were the same.",
                "",
            ]
            continue
        stats = payload["stats"]
        lines += [
            f"### Trading {venue} against the book consensus",
            "",
            "```json",
            json.dumps(stats, indent=2, default=str),
            "```",
            "",
            "Full threshold sweep:",
            "",
            _table(_sweep_view(payload["sweep"])),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote %s", path)
    return path


SWEEP_VIEW = ["threshold", "n_trades", "n_events", "return_on_capital_days",
              "rocd_lo", "rocd_hi", "sharpe", "clv_mean"]


def _sweep_view(sweep: pd.DataFrame) -> pd.DataFrame:
    """Trim the sweep to the columns worth reading, tolerating missing ones.

    A threshold with no trades produces no CLV columns at all, and that row
    still belongs in the table -- the sweep is reported in full.
    """
    if sweep.empty:
        return sweep
    return sweep[[c for c in SWEEP_VIEW if c in sweep.columns]]


def _table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_no rows_"
    rounded = frame.copy()
    for col in rounded.select_dtypes(include=[np.floating]).columns:
        rounded[col] = rounded[col].round(5)
    return rounded.to_markdown(index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", default=None)
    parser.add_argument("--panel", default=None, help="read the panel from parquet instead")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup_logging(args.verbose)
    cfg = load_config()
    panel = load_panel(args.db or cfg.db_path, args.panel)
    if panel.empty:
        log.error("analysis_panel is empty; run collect.py and resolve.py first")
        return 1
    panel = panel[panel["y"].notna()].copy()
    if panel.empty:
        log.error("no labelled rows yet; nothing has settled")
        return 1

    calib = run_calibration(panel, args.n_boot)
    trade = {} if args.skip_backtest else run_tradeability(panel, cfg.poly_fee_rate, args.n_boot)
    write_summary(panel, calib, trade)
    return 0


if __name__ == "__main__":
    sys.exit(main())
