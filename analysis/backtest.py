"""Tradeability, simulated honestly. Run this only after calibration is settled.

Rule: when the reference venue's fair probability exceeds the trade venue's
**ask** by more than `threshold`, buy at that ask. Never at the midpoint --
crossing the spread is not optional, and a midpoint backtest is the single
easiest way to manufacture an edge that does not exist.

Guard rails baked in:
* fills capped at displayed top-of-book depth (`pmcal.fees` handles the rest);
* one trade per (event, outcome) by default, because 40 snapshots of the same
  mispricing is one bet, not forty;
* profitability reported as return on **capital-days**, since capital is locked
  until settlement and that, not per-trade ROI, is the binding constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from analysis.kelly import kelly_fraction
from pmcal.fees import taker_fee
from pmcal.stats import cluster_bootstrap

TRADE_COLUMNS = {"polymarket": ("pm_ask", "pm_ask_size", "pm_mid"),
                 "kalshi": ("kalshi_ask", "kalshi_ask_size", "kalshi_mid")}


@dataclass(frozen=True)
class BacktestConfig:
    threshold: float = 0.03
    trade_venue: str = "polymarket"
    reference: str = "book_shin"
    max_contracts: float = 100.0
    poly_fee_rate: float = 0.01
    kelly_fraction: float = 0.25
    bankroll: float = 10_000.0
    one_trade_per_outcome: bool = True
    settle_lag_hours: float = 4.0
    min_price: float = 0.02
    max_price: float = 0.98


def find_signals(panel: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Rows where the reference exceeds the executable ask by more than `k`."""
    ask_col, size_col, mid_col = TRADE_COLUMNS[cfg.trade_venue]
    needed = [ask_col, size_col, cfg.reference, "y", "event_key"]
    if any(c not in panel.columns for c in needed):
        return pd.DataFrame()

    frame = panel.dropna(subset=needed).copy()
    frame = frame[(frame[ask_col] >= cfg.min_price) & (frame[ask_col] <= cfg.max_price)]
    frame = frame[frame[size_col] > 0]
    frame["entry_price"] = frame[ask_col]
    frame["depth"] = frame[size_col]
    frame["reference_prob"] = frame[cfg.reference]
    frame["edge"] = frame["reference_prob"] - frame["entry_price"]
    frame = frame[frame["edge"] > cfg.threshold]
    if frame.empty:
        return frame

    frame = frame.sort_values("ts")
    if cfg.one_trade_per_outcome:
        frame = frame.drop_duplicates(subset=["event_key", "outcome"], keep="first")
    frame["mid_at_entry"] = frame[mid_col]
    return frame.reset_index(drop=True)


def size_trades(signals: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Fractional-Kelly stake, then capped by displayed depth and a hard limit."""
    if signals.empty:
        return signals
    out = signals.copy()
    fractions = [
        kelly_fraction(float(p), float(price), cfg.kelly_fraction)
        for p, price in zip(out["reference_prob"], out["entry_price"], strict=True)
    ]
    desired = np.array(fractions) * cfg.bankroll / out["entry_price"].to_numpy(float)
    out["desired_contracts"] = desired
    out["contracts"] = np.floor(
        np.minimum(np.minimum(desired, out["depth"].to_numpy(float)), cfg.max_contracts)
    )
    out["depth_capped"] = out["contracts"] < np.floor(np.minimum(desired, cfg.max_contracts))
    return out[out["contracts"] > 0].reset_index(drop=True)


def settle(trades: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Apply fees, realise the outcome, and account for locked capital."""
    if trades.empty:
        return trades
    out = trades.copy()
    out["fee"] = [
        taker_fee(cfg.trade_venue, float(c), float(p), cfg.poly_fee_rate)
        for c, p in zip(out["contracts"], out["entry_price"], strict=True)
    ]
    out["capital"] = out["contracts"] * out["entry_price"] + out["fee"]
    out["payoff"] = out["contracts"] * out["y"].astype(float)
    out["pnl"] = out["payoff"] - out["capital"]
    out["roi"] = out["pnl"] / out["capital"]
    out["hold_days"] = (out["minutes_to_start"] / 1440.0) + cfg.settle_lag_hours / 24.0
    out["capital_days"] = out["capital"] * out["hold_days"]
    out["realised_edge"] = out["y"].astype(float) - out["entry_price"]
    out["resolution_time"] = pd.to_datetime(out["event_start_time"]) + pd.Timedelta(
        hours=cfg.settle_lag_hours
    )
    return out


def add_clv(trades: pd.DataFrame, panel: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Closing line value: did the price move our way before the event started?

    The closing mark is the last pre-kickoff midpoint on the venue we traded.
    CLV is the cleanest evidence of real edge because it does not depend on the
    outcome at all -- a genuine informational advantage shows up as the market
    coming to you, well before any variance from actual results.
    """
    if trades.empty:
        return trades
    _, _, mid_col = TRADE_COLUMNS[cfg.trade_venue]
    closing = (
        panel.dropna(subset=[mid_col])
        .sort_values("minutes_to_start")
        .drop_duplicates(subset=["event_key", "outcome"], keep="first")
        .loc[:, ["event_key", "outcome", mid_col, "minutes_to_start"]]
        .rename(columns={mid_col: "closing_mid", "minutes_to_start": "closing_minutes"})
    )
    out = trades.merge(closing, on=["event_key", "outcome"], how="left")
    out["clv"] = out["closing_mid"] - out["entry_price"]
    # A close captured hours early is not a close; flag rather than silently use it.
    out["clv_valid"] = out["closing_minutes"] <= 60.0
    return out


def max_drawdown(pnl_by_time: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative P&L, in currency units."""
    if pnl_by_time.empty:
        return 0.0
    equity = pnl_by_time.cumsum()
    return float((equity.cummax() - equity).max())


def _sharpe(roi: np.ndarray, hold_days: np.ndarray) -> float:
    """Annualised Sharpe assuming capital recycles once per holding period.

    Per-trade ROI is not a time series, so an annualisation factor has to be
    assumed: `sqrt(365 / mean_hold_days)`. Stated explicitly because it is the
    number a desk will push back on first.
    """
    roi = roi[np.isfinite(roi)]
    if roi.size < 2 or roi.std(ddof=1) == 0:
        return np.nan
    mean_hold = float(np.nanmean(hold_days)) or 1.0
    return float(roi.mean() / roi.std(ddof=1) * np.sqrt(365.0 / max(mean_hold, 1e-6)))


def summarise(trades: pd.DataFrame, n_boot: int = 10_000) -> dict[str, float]:
    """Headline metrics, each with a clustered bootstrap interval where it matters."""
    if trades.empty:
        return {"n_trades": 0, "n_events": 0}

    roi = trades["roi"].to_numpy(float)
    pnl = trades["pnl"].to_numpy(float)
    capital_days = trades["capital_days"].to_numpy(float)
    hold = trades["hold_days"].to_numpy(float)
    clusters = trades["event_key"].to_numpy()

    rocd = cluster_bootstrap(
        lambda i: float(pnl[i].sum() / capital_days[i].sum()) if capital_days[i].sum() else np.nan,
        clusters, n_boot=n_boot,
    )
    sharpe = cluster_bootstrap(lambda i: _sharpe(roi[i], hold[i]), clusters, n_boot=n_boot)
    hit = cluster_bootstrap(lambda i: float((pnl[i] > 0).mean()), clusters, n_boot=n_boot)

    ordered = trades.sort_values("resolution_time")["pnl"]
    clv_rows = trades[trades.get("clv_valid", pd.Series(False, index=trades.index))]
    clv_stats: dict[str, float] = {"n_clv": len(clv_rows)}
    if not clv_rows.empty:
        clv = clv_rows["clv"].to_numpy(float)
        clv_ci = cluster_bootstrap(
            lambda i: float(np.nanmean(clv[i])), clv_rows["event_key"].to_numpy(), n_boot=n_boot
        )
        clv_stats |= {
            "clv_mean": clv_ci["point"],
            "clv_lo": clv_ci["lo"],
            "clv_hi": clv_ci["hi"],
            "clv_positive_rate": float(np.nanmean(clv > 0)),
        }

    return {
        "n_trades": int(len(trades)),
        "n_events": int(trades["event_key"].nunique()),
        "hit_rate": hit["point"],
        "hit_rate_lo": hit["lo"],
        "hit_rate_hi": hit["hi"],
        "predicted_edge": float(trades["edge"].mean()),
        "realised_edge": float(trades["realised_edge"].mean()),
        "total_pnl": float(pnl.sum()),
        "total_capital_days": float(capital_days.sum()),
        "return_on_capital_days": rocd["point"],
        "rocd_lo": rocd["lo"],
        "rocd_hi": rocd["hi"],
        "sharpe": sharpe["point"],
        "sharpe_lo": sharpe["lo"],
        "sharpe_hi": sharpe["hi"],
        "max_drawdown": max_drawdown(ordered),
        "mean_hold_days": float(np.nanmean(hold)),
        "depth_capped_share": float(trades["depth_capped"].mean()),
        **clv_stats,
    }


def run(panel: pd.DataFrame, cfg: BacktestConfig, n_boot: int = 10_000) -> tuple[pd.DataFrame, dict]:
    trades = settle(size_trades(find_signals(panel, cfg), cfg), cfg)
    trades = add_clv(trades, panel, cfg)
    return trades, summarise(trades, n_boot=n_boot)


def sweep(
    panel: pd.DataFrame,
    cfg: BacktestConfig,
    thresholds: np.ndarray | None = None,
    n_boot: int = 2_000,
) -> pd.DataFrame:
    """The whole threshold curve, reported in full.

    Publishing only the best `k` is p-hacking with extra steps. The shape of
    this curve is the evidence: a real edge decays smoothly and rests on many
    events, whereas a spurious one spikes at one threshold on a handful of them.
    """
    grid = np.arange(0.00, 0.12, 0.005) if thresholds is None else thresholds
    rows = []
    for k in grid:
        variant = replace(cfg, threshold=float(k))
        _, stats = run(panel, variant, n_boot=n_boot)
        rows.append({"threshold": float(k), **stats})
    return pd.DataFrame(rows)
