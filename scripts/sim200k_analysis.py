#!/usr/bin/env python3
"""Post-processing analysis for sim200k wheel-strategy backtest artifacts.

Reads per-window artifacts produced by the sim200k backtest campaign and
computes:
  1. Returns & drawdown from the equity curve
  2. Candidate funnel metrics
  3. Calibration: Spearman rho, hit rate, EV-decile table, prob_profit bins
  4. Friction-level sensitivity (none / bid_ask / full)
  5. Concentration audit (R9/R10 running peaks — NOT end-state)
  6. Benchmark: equal-weight buy-and-hold over the same universe + SPY
  7. Combined campaign_table.md with --all

Usage
-----
  py -3.12 scripts/sim200k_analysis.py --root <dir> --window <id>
  py -3.12 scripts/sim200k_analysis.py --root <dir> --all
  py -3.12 scripts/sim200k_analysis.py --selftest

§2: read-only — never imports ev_engine / wheel_runner / candidate_dossier.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# Force UTF-8 output on Windows consoles (cp1252 chokes on non-ASCII)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repo bootstrap: allow ``python scripts/sim200k_analysis.py`` direct
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Only import universes — no engine decision-layer imports.
try:
    from backtests.regression.universes import UNIVERSE_100
except Exception:  # pragma: no cover
    UNIVERSE_100 = ()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRICTION_LEVELS = ("none", "bid_ask", "full")
R10_CAP = 0.10  # single-name 10%
R9_CAP = 0.25  # sector 25%
OHLCV_CSV = _REPO_ROOT / "data" / "bloomberg" / "sp500_ohlcv.csv"
FUNDAMENTALS_CSV = _REPO_ROOT / "data" / "bloomberg" / "sp500_fundamentals.csv"
DEFAULT_ROOT = Path(os.environ.get("TEMP", "/tmp")) / "sim200k_backtest"
SELFTEST_ROOT = Path(os.environ.get("TEMP", "/tmp")) / "sim200k_smoke"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARN  cannot read {path}: {exc}")
        return None


def _load_rank_log(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, parse_dates=["date", "expiration_date"])
        return df
    except Exception as exc:
        print(f"  WARN  cannot read rank_log {path}: {exc}")
        return None


def _safe_fmt(x: float | None, fmt: str = ".2f") -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "N/A"
    return format(x, fmt)


# ---------------------------------------------------------------------------
# 1. Returns & drawdown
# ---------------------------------------------------------------------------


def compute_returns(equity_curve: list[dict], initial_capital: float) -> dict:
    if not equity_curve:
        return {}
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    final_nav = df["portfolio_value"].iloc[-1]
    final_cash = df["cash"].iloc[-1] if "cash" in df.columns else float("nan")
    total_return_pct = (final_nav / initial_capital - 1) * 100

    # Max drawdown
    rolling_peak = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - rolling_peak) / rolling_peak
    max_dd_pct = drawdown.min() * 100
    trough_idx = drawdown.idxmin()
    peak_idx = (df["portfolio_value"].iloc[: trough_idx + 1]).idxmax()
    peak_date = df["date"].iloc[peak_idx].strftime("%Y-%m-%d")
    trough_date = df["date"].iloc[trough_idx].strftime("%Y-%m-%d")

    # Monthly returns
    df_m = df.set_index("date")["portfolio_value"].resample("ME").last().dropna()
    if len(df_m) >= 2:
        monthly_returns = df_m.pct_change().dropna() * 100
    else:
        # Short window — single month; compute start-to-end
        monthly_returns = pd.Series(dtype=float)

    start_date = df["date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["date"].iloc[-1].strftime("%Y-%m-%d")

    return {
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "final_nav": final_nav,
        "final_cash": final_cash,
        "total_return_pct": total_return_pct,
        "max_dd_pct": max_dd_pct,
        "dd_peak_date": peak_date,
        "dd_trough_date": trough_date,
        "monthly_returns": monthly_returns.tolist(),
        "equity_curve_points": len(df),
    }


# ---------------------------------------------------------------------------
# 2. Funnel
# ---------------------------------------------------------------------------


def compute_funnel(rank_log: pd.DataFrame, metrics_agg: dict) -> dict:
    total_rows = len(rank_log)
    unique_candidates = rank_log[["date", "ticker"]].drop_duplicates().shape[0]
    pos_ev_rows = (rank_log["ev_dollars"] > 0).sum()
    pos_ev_pct = pos_ev_rows / total_rows * 100 if total_rows else 0.0

    executed = metrics_agg.get("executed_trades", 0)
    open_at_end = metrics_agg.get("open_at_end", 0)
    put_assignments = metrics_agg.get("put_assignments", 0)

    executed_over_ranked_pos = executed / pos_ev_rows if pos_ev_rows else float("nan")

    return {
        "total_rank_rows": total_rows,
        "unique_date_ticker_candidates": unique_candidates,
        "pos_ev_rows": int(pos_ev_rows),
        "pos_ev_pct": pos_ev_pct,
        "executed_trades": executed,
        "open_at_end": open_at_end,
        "put_assignments": put_assignments,
        "executed_over_ranked_positive": executed_over_ranked_pos,
    }


# ---------------------------------------------------------------------------
# 3. Calibration
# ---------------------------------------------------------------------------


def compute_calibration(rank_log: pd.DataFrame, metrics_agg: dict) -> dict:
    settled = rank_log.dropna(subset=["realized_pnl"])
    unsettled_count = len(rank_log) - len(settled)

    spearman_rho = metrics_agg.get("spearman_rho")
    spearman_p = metrics_agg.get("spearman_p")
    hit_rate = metrics_agg.get("hit_rate")
    mean_realized = metrics_agg.get("mean_realized")

    # EV-decile table (10 deciles by ev_dollars over settled rows)
    decile_table: list[dict] = []
    if len(settled) >= 10:
        settled = settled.copy()
        settled["ev_decile"] = pd.qcut(settled["ev_dollars"], q=10, labels=False, duplicates="drop")
        for d, grp in settled.groupby("ev_decile", observed=True):
            decile_table.append(
                {
                    "decile": int(d),
                    "n": len(grp),
                    "mean_predicted_ev": grp["ev_dollars"].mean(),
                    "mean_realized_pnl": grp["realized_pnl"].mean(),
                }
            )

    # prob_profit reliability bins [0.5-0.6, ..., 0.9-1.0]
    pp_table: list[dict] = []
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    if "prob_profit" in settled.columns:
        settled_pp = settled.dropna(subset=["prob_profit"])
        settled_pp = settled_pp[settled_pp["prob_profit"].between(0.5, 1.0)]
        for i, label in enumerate(bin_labels):
            lo, hi = bins[i], bins[i + 1]
            mask = (settled_pp["prob_profit"] >= lo) & (settled_pp["prob_profit"] < hi)
            grp = settled_pp[mask]
            if len(grp) == 0:
                continue
            midpoint = (lo + hi) / 2
            realized_win_rate = (grp["realized_pnl"] > 0).mean()
            pp_table.append(
                {
                    "bin": label,
                    "predicted_midpoint": midpoint,
                    "realized_win_rate": realized_win_rate,
                    "n": len(grp),
                }
            )

    return {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "hit_rate": hit_rate,
        "mean_realized": mean_realized,
        "settled_rows": len(settled),
        "unsettled_rows": unsettled_count,
        "ev_decile_table": decile_table,
        "pp_reliability_table": pp_table,
    }


# ---------------------------------------------------------------------------
# 4. Friction sensitivity
# ---------------------------------------------------------------------------


def compute_friction_sensitivity(
    friction_metrics: dict[str, dict],
) -> list[dict]:
    rows = []
    for level in FRICTION_LEVELS:
        agg = friction_metrics.get(level, {}).get("aggregate", {})
        rows.append(
            {
                "friction": level,
                "final_nav": agg.get("final_nav"),
                "executed_trades": agg.get("executed_trades"),
                "hit_rate": agg.get("hit_rate"),
                "spearman_rho": agg.get("spearman_rho"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 5. Concentration audit (R9/R10 running peaks)
# ---------------------------------------------------------------------------


def _build_sector_map() -> dict[str, str]:
    """Load ticker->gics_sector_name from fundamentals CSV."""
    if not FUNDAMENTALS_CSV.exists():
        print(f"  WARN  fundamentals CSV not found: {FUNDAMENTALS_CSV}")
        return {}
    try:
        df = pd.read_csv(FUNDAMENTALS_CSV, usecols=["ticker", "gics_sector_name"])
        # Normalize tickers (strip exchange suffix like " UW Equity")
        df["ticker_norm"] = df["ticker"].str.split().str[0]
        return dict(zip(df["ticker_norm"], df["gics_sector_name"], strict=False))
    except Exception as exc:
        print(f"  WARN  cannot load sector map: {exc}")
        return {}


def compute_concentration(
    tracker_state: dict,
    equity_curve_df: pd.DataFrame,
    sector_map: dict[str, str],
) -> dict:
    """Reconstruct daily open-position timeline and compute running peaks."""
    positions_raw = tracker_state.get("positions", {})
    closed_positions = tracker_state.get("closed_positions", [])

    # Build list of (ticker, entry_date, exit_date, strike, contracts)
    # For open positions: exit_date = put_expiration_date (or last ec date)
    ec_last_date = equity_curve_df["date"].max() if len(equity_curve_df) else pd.Timestamp.now()

    all_positions: list[dict] = []

    # Open positions
    for ticker, pos in positions_raw.items():
        entry = pos.get("entry_date") or pos.get("put_entry_date")
        strike = pos.get("put_strike")
        exp = pos.get("put_expiration_date")
        exit_date = pd.Timestamp(exp) if exp else ec_last_date
        if entry and strike is not None:
            all_positions.append(
                {
                    "ticker": ticker,
                    "entry_date": pd.Timestamp(entry),
                    "exit_date": exit_date,
                    "strike": float(strike),
                    "contracts": 1,
                }
            )

    # Closed positions
    for cp in closed_positions:
        ticker = cp.get("ticker")
        entry = cp.get("entry_date") or cp.get("put_entry_date")
        strike = cp.get("put_strike")
        # If no put_strike, try parsing from notes
        if strike is None:
            notes = cp.get("notes", [])
            if isinstance(notes, str):
                notes = [notes]
            for note in notes:
                import re

                m = re.search(r"(\d+(?:\.\d+)?)\s*[Pp]", str(note))
                if m:
                    strike = float(m.group(1))
                    break
        exp = cp.get("put_expiration_date") or cp.get("exit_date") or cp.get("close_date")
        exit_date = pd.Timestamp(exp) if exp else ec_last_date
        if ticker and entry and strike is not None:
            all_positions.append(
                {
                    "ticker": ticker,
                    "entry_date": pd.Timestamp(entry),
                    "exit_date": exit_date,
                    "strike": float(strike),
                    "contracts": 1,
                }
            )

    if equity_curve_df.empty or not all_positions:
        return {
            "max_single_name_pct": None,
            "max_sector_pct": None,
            "single_name_r10_fire_days": 0,
            "sector_r9_fire_days": 0,
            "unknown_sector_tickers": [],
        }

    ec = equity_curve_df.set_index("date")["portfolio_value"]
    all_dates = ec.index

    # Per day: compute single-name and sector notionals
    single_name_peak = 0.0
    sector_peak = 0.0
    r10_fire_days = 0
    r9_fire_days = 0
    unknown_tickers: set[str] = set()

    for date in all_dates:
        nav = ec.get(date, None)
        if nav is None or nav <= 0:
            continue
        # Find positions open on this date
        day_positions = [p for p in all_positions if p["entry_date"] <= date <= p["exit_date"]]

        # Single-name notionals
        single_name_notionals: dict[str, float] = {}
        sector_notionals: dict[str, float] = {}
        for p in day_positions:
            notional = p["strike"] * 100 * p["contracts"]
            t = p["ticker"]
            single_name_notionals[t] = single_name_notionals.get(t, 0) + notional
            sector = sector_map.get(t, "Unknown")
            if sector == "Unknown":
                unknown_tickers.add(t)
            sector_notionals[sector] = sector_notionals.get(sector, 0) + notional

        if single_name_notionals:
            max_sn = max(v / nav for v in single_name_notionals.values())
            if max_sn > single_name_peak:
                single_name_peak = max_sn
            if max_sn > R10_CAP:
                r10_fire_days += 1

        if sector_notionals:
            max_sec = max(v / nav for v in sector_notionals.values())
            if max_sec > sector_peak:
                sector_peak = max_sec
            if max_sec > R9_CAP:
                r9_fire_days += 1

    return {
        "max_single_name_pct": single_name_peak * 100,
        "max_sector_pct": sector_peak * 100,
        "single_name_r10_fire_days": r10_fire_days,
        "sector_r9_fire_days": r9_fire_days,
        "unknown_sector_tickers": sorted(unknown_tickers),
    }


# ---------------------------------------------------------------------------
# 6. Benchmark
# ---------------------------------------------------------------------------


def compute_benchmark(
    universe_tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    """Equal-weight buy-and-hold over the universe + SPY check."""
    result: dict = {
        "ew_return_pct": None,
        "spy_return_pct": None,
        "spy_available": False,
        "n_tickers_in_universe": 0,
    }

    if not OHLCV_CSV.exists():
        print(f"  WARN  OHLCV CSV not found: {OHLCV_CSV}")
        return result

    try:
        ohlcv = pd.read_csv(OHLCV_CSV, parse_dates=["date"])
    except Exception as exc:
        print(f"  WARN  cannot load OHLCV: {exc}")
        return result

    # Normalize tickers (strip exchange suffix)
    ohlcv["ticker_norm"] = ohlcv["ticker"].str.split().str[0]

    t_start = pd.Timestamp(start_date)
    t_end = pd.Timestamp(end_date)

    # Check SPY
    spy_df = ohlcv[ohlcv["ticker_norm"] == "SPY"].sort_values("date")
    if spy_df.empty:
        result["spy_available"] = False
    else:
        result["spy_available"] = True
        spy_before = spy_df[spy_df["date"] <= t_end]
        spy_after = spy_df[spy_df["date"] >= t_start]
        if not spy_before.empty and not spy_after.empty:
            spy_start_price = spy_after.iloc[0]["close"]
            spy_end_price = spy_before.iloc[-1]["close"]
            result["spy_return_pct"] = (spy_end_price / spy_start_price - 1) * 100

    # Equal-weight universe
    univ_set = set(universe_tickers)
    ew_returns = []
    n_found = 0
    for ticker in univ_set:
        t_df = ohlcv[ohlcv["ticker_norm"] == ticker].sort_values("date")
        if t_df.empty:
            continue
        t_before = t_df[t_df["date"] <= t_end]
        t_after = t_df[t_df["date"] >= t_start]
        if t_before.empty or t_after.empty:
            continue
        p_start = t_after.iloc[0]["close"]
        p_end = t_before.iloc[-1]["close"]
        if p_start > 0:
            ew_returns.append((p_end / p_start - 1) * 100)
            n_found += 1

    result["n_tickers_in_universe"] = n_found
    result["ew_return_pct"] = float(np.mean(ew_returns)) if ew_returns else None

    return result


# ---------------------------------------------------------------------------
# Per-window analysis
# ---------------------------------------------------------------------------


def analyze_window(
    window_dir: Path,
    window_id: str,
    sector_map: dict[str, str],
) -> dict | None:
    """Analyze one window directory. Returns analysis dict or None on skip."""
    full_dir = window_dir / "full"
    if not full_dir.is_dir():
        print(f"  SKIP  {window_id}: no full/ subfolder")
        return None

    # Load full-friction artifacts (headline)
    metrics_full_path = full_dir / "metrics.json"
    rank_log_full_path = full_dir / "rank_log.csv"
    tracker_state_full_path = full_dir / "tracker_state.json"

    metrics_full = _load_json(metrics_full_path)
    if metrics_full is None:
        print(f"  SKIP  {window_id}: cannot load full/metrics.json")
        return None

    rank_log_full = _load_rank_log(rank_log_full_path)
    if rank_log_full is None:
        print(f"  SKIP  {window_id}: cannot load full/rank_log.csv")
        return None

    tracker_full = _load_json(tracker_state_full_path)
    if tracker_full is None:
        print(f"  SKIP  {window_id}: cannot load full/tracker_state.json")
        return None

    # Load all friction levels for sensitivity
    friction_metrics: dict[str, dict] = {}
    for level in FRICTION_LEVELS:
        level_dir = window_dir / level
        m_path = level_dir / "metrics.json"
        if m_path.exists():
            m = _load_json(m_path)
            if m:
                friction_metrics[level] = m

    agg = metrics_full.get("aggregate", {})
    fingerprint = metrics_full.get("fingerprint", {})
    initial_capital = tracker_full.get("initial_capital", 200000.0)

    # Read summary.json for dates if present
    summary_path = window_dir / "summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else None

    # Equity curve
    equity_curve_raw = tracker_full.get("equity_curve", [])
    ec_df = pd.DataFrame(equity_curve_raw)
    if not ec_df.empty:
        ec_df["date"] = pd.to_datetime(ec_df["date"])
        ec_df = ec_df.sort_values("date").reset_index(drop=True)

    # --- 1. Returns ---
    returns_data = compute_returns(equity_curve_raw, initial_capital)

    # Dates: prefer summary.json, then fingerprint, then rank_log min/max
    start_date = (
        (summary or {}).get("start_date")
        or fingerprint.get("start")
        or (rank_log_full["date"].min().strftime("%Y-%m-%d") if len(rank_log_full) else None)
    )
    end_date = (
        (summary or {}).get("end_date")
        or fingerprint.get("end")
        or (rank_log_full["date"].max().strftime("%Y-%m-%d") if len(rank_log_full) else None)
    )

    # --- 2. Funnel ---
    funnel_data = compute_funnel(rank_log_full, agg)

    # --- 3. Calibration ---
    calib_data = compute_calibration(rank_log_full, agg)

    # --- 4. Friction sensitivity ---
    friction_data = compute_friction_sensitivity(friction_metrics)

    # --- 5. Concentration ---
    conc_data = compute_concentration(tracker_full, ec_df, sector_map)

    # --- 6. Benchmark ---
    universe_tickers = fingerprint.get("tickers") or list(UNIVERSE_100)
    bmark_data = compute_benchmark(
        universe_tickers,
        start_date or "2021-01-01",
        end_date or "2021-12-31",
    )

    # Engine minus benchmark
    engine_return = returns_data.get("total_return_pct")
    ew_return = bmark_data.get("ew_return_pct")
    engine_minus_bmark = (
        engine_return - ew_return if (engine_return is not None and ew_return is not None) else None
    )

    analysis = {
        "window_id": window_id,
        "start_date": start_date,
        "end_date": end_date,
        "returns": returns_data,
        "funnel": funnel_data,
        "calibration": calib_data,
        "friction_sensitivity": friction_data,
        "concentration": conc_data,
        "benchmark": bmark_data,
        "engine_minus_benchmark_pct": engine_minus_bmark,
    }

    return analysis


# ---------------------------------------------------------------------------
# Markdown output
# ---------------------------------------------------------------------------

_MD_SEP = "---"


def _pct(v: float | None, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "N/A"
    return f"{v:.{decimals}f}%"


def _fmt(v: float | None, decimals: int = 2) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "N/A"
    return f"{v:,.{decimals}f}"


def _fmt_int(v: int | None) -> str:
    if v is None:
        return "N/A"
    return f"{v:,}"


def render_summary_table(analyses: list[dict]) -> str:
    headers = [
        "Window",
        "Start",
        "End",
        "Final NAV",
        "Return %",
        "Max DD %",
        "Trades",
        "Assignments",
        "Hit Rate",
        "Spearman ρ",
        "EW BH Return %",
        "Engine − BM %",
    ]
    rows = []
    for a in analyses:
        r = a.get("returns", {})
        f = a.get("funnel", {})
        c = a.get("calibration", {})
        b = a.get("benchmark", {})
        rows.append(
            [
                a.get("window_id", "?"),
                a.get("start_date", "?"),
                a.get("end_date", "?"),
                _fmt(r.get("final_nav"), 0),
                _pct(r.get("total_return_pct")),
                _pct(r.get("max_dd_pct")),
                _fmt_int(f.get("executed_trades")),
                _fmt_int(f.get("put_assignments")),
                _pct(c.get("hit_rate", 0) * 100 if c.get("hit_rate") is not None else None),
                _safe_fmt(c.get("spearman_rho"), ".4f"),
                _pct(b.get("ew_return_pct")),
                _pct(a.get("engine_minus_benchmark_pct")),
            ]
        )
    return _md_table(headers, rows)


def render_calibration_tables(analyses: list[dict]) -> str:
    lines = []
    for a in analyses:
        c = a.get("calibration", {})
        wid = a.get("window_id", "?")
        lines.append(f"\n### Calibration — {wid}\n")
        lines.append(
            f"settled rows: {c.get('settled_rows', 'N/A')}  "
            f"unsettled (excluded): {c.get('unsettled_rows', 'N/A')}"
        )
        lines.append("")

        # EV-decile table
        dec = c.get("ev_decile_table", [])
        if dec:
            lines.append("**EV Decile Table** (settled rows only)\n")
            hdrs = ["Decile", "N", "Mean Pred EV ($)", "Mean Realized PnL ($)"]
            rows = [
                [
                    str(d["decile"]),
                    str(d["n"]),
                    _fmt(d["mean_predicted_ev"]),
                    _fmt(d["mean_realized_pnl"]),
                ]
                for d in dec
            ]
            lines.append(_md_table(hdrs, rows))

        # prob_profit bins
        pp = c.get("pp_reliability_table", [])
        if pp:
            lines.append("**Prob-Profit Reliability Bins** (settled rows)\n")
            hdrs = ["Bin", "Predicted Midpoint", "Realized Win Rate", "N"]
            rows = [
                [
                    d["bin"],
                    _pct(d["predicted_midpoint"] * 100),
                    _pct(d["realized_win_rate"] * 100),
                    str(d["n"]),
                ]
                for d in pp
            ]
            lines.append(_md_table(hdrs, rows))
    return "\n".join(lines)


def render_concentration_table(analyses: list[dict]) -> str:
    headers = [
        "Window",
        "Max Single-Name %",
        "R10 (10%) Fired Days",
        "Max Sector %",
        "R9 (25%) Fired Days",
        "Unknown-Sector Tickers",
    ]
    rows = []
    for a in analyses:
        c = a.get("concentration", {})
        unknown = c.get("unknown_sector_tickers", [])
        rows.append(
            [
                a.get("window_id", "?"),
                _pct(c.get("max_single_name_pct")),
                _fmt_int(c.get("single_name_r10_fire_days")),
                _pct(c.get("max_sector_pct")),
                _fmt_int(c.get("sector_r9_fire_days")),
                ", ".join(unknown[:5]) + ("…" if len(unknown) > 5 else ""),
            ]
        )
    return _md_table(headers, rows)


def render_friction_table(analyses: list[dict]) -> str:
    headers = ["Window", "Friction", "Final NAV", "Executed Trades", "Hit Rate", "Spearman ρ"]
    rows = []
    for a in analyses:
        wid = a.get("window_id", "?")
        for fs in a.get("friction_sensitivity", []):
            rows.append(
                [
                    wid,
                    fs.get("friction", "?"),
                    _fmt(fs.get("final_nav"), 0),
                    _fmt_int(fs.get("executed_trades")),
                    _pct(fs.get("hit_rate", 0) * 100 if fs.get("hit_rate") is not None else None),
                    _safe_fmt(fs.get("spearman_rho"), ".4f"),
                ]
            )
    return _md_table(headers, rows)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def print_analysis(a: dict) -> None:
    r = a.get("returns", {})
    f = a.get("funnel", {})
    c = a.get("calibration", {})
    b = a.get("benchmark", {})
    conc = a.get("concentration", {})
    wid = a.get("window_id", "?")

    print(f"\n{'=' * 70}")
    print(f"  WINDOW: {wid}  [{a.get('start_date')} -> {a.get('end_date')}]")
    print(f"{'=' * 70}")

    print("\n--- 1. RETURNS (full friction) ---")
    print(f"  Initial capital : ${r.get('initial_capital', 0):,.0f}")
    print(f"  Final NAV       : ${r.get('final_nav', 0):,.2f}")
    print(f"  Final cash      : ${r.get('final_cash', 0):,.2f}")
    print(f"  Total return    : {_pct(r.get('total_return_pct'))}")
    print(f"  Max drawdown    : {_pct(r.get('max_dd_pct'))}")
    print(f"  DD peak date    : {r.get('dd_peak_date', 'N/A')}")
    print(f"  DD trough date  : {r.get('dd_trough_date', 'N/A')}")
    print(f"  Equity pts      : {r.get('equity_curve_points', 0)}")
    monthly = r.get("monthly_returns", [])
    if monthly:
        print(f"  Monthly returns : {[round(x, 2) for x in monthly]}")

    print("\n--- 2. FUNNEL (full friction) ---")
    print(f"  Total rank rows        : {f.get('total_rank_rows', 0):,}")
    print(f"  Unique (date,ticker)   : {f.get('unique_date_ticker_candidates', 0):,}")
    print(f"  Positive-EV rows       : {f.get('pos_ev_rows', 0):,} ({_pct(f.get('pos_ev_pct'))})")
    print(f"  Executed trades        : {f.get('executed_trades', 0)}")
    print(f"  Open at end            : {f.get('open_at_end', 0)}")
    print(f"  Put assignments        : {f.get('put_assignments', 0)}")
    ratio = f.get("executed_over_ranked_positive")
    ratio_str = f"{ratio:.4f}" if ratio and math.isfinite(ratio) else "N/A"
    print(f"  Executed/ranked-pos    : {ratio_str}")

    print("\n--- 3. CALIBRATION (full friction) ---")
    print(f"  Spearman rho   : {_safe_fmt(c.get('spearman_rho'), '.4f')}")
    print(f"  Spearman p     : {_safe_fmt(c.get('spearman_p'), '.2e')}")
    print(
        f"  Hit rate       : {_pct(c.get('hit_rate', 0) * 100 if c.get('hit_rate') is not None else None)}"
    )
    print(f"  Mean realized  : ${_fmt(c.get('mean_realized'))}")
    print(f"  Settled rows   : {c.get('settled_rows', 0)}  unsettled: {c.get('unsettled_rows', 0)}")

    dec = c.get("ev_decile_table", [])
    if dec:
        print("\n  EV Decile Table (settled rows):")
        print(f"  {'Decile':>6}  {'N':>5}  {'Mean EV($)':>12}  {'Mean PnL($)':>12}")
        for d in dec:
            print(
                f"  {d['decile']:>6}  {d['n']:>5}  "
                f"{d['mean_predicted_ev']:>12.2f}  "
                f"{d['mean_realized_pnl']:>12.2f}"
            )

    pp = c.get("pp_reliability_table", [])
    if pp:
        print("\n  Prob-Profit Reliability Table (settled rows):")
        print(f"  {'Bin':>10}  {'Pred Mid':>10}  {'Realized Win%':>14}  {'N':>5}")
        for d in pp:
            print(
                f"  {d['bin']:>10}  "
                f"{d['predicted_midpoint'] * 100:>10.1f}%  "
                f"{d['realized_win_rate'] * 100:>14.1f}%  "
                f"{d['n']:>5}"
            )

    print("\n--- 4. FRICTION SENSITIVITY ---")
    print(f"  {'Friction':>10}  {'Final NAV':>12}  {'Trades':>7}  {'Hit Rate':>9}  {'Rho':>8}")
    for fs in a.get("friction_sensitivity", []):
        nav_s = _fmt(fs.get("final_nav"), 0)
        trades_s = str(fs.get("executed_trades") or "N/A")
        hr = fs.get("hit_rate")
        hr_s = _pct(hr * 100) if hr is not None else "N/A"
        rho = _safe_fmt(fs.get("spearman_rho"), ".4f")
        print(f"  {fs.get('friction', '?'):>10}  {nav_s:>12}  {trades_s:>7}  {hr_s:>9}  {rho:>8}")

    print("\n--- 5. CONCENTRATION AUDIT (running peaks) ---")
    print(
        f"  Max single-name % : {_pct(conc.get('max_single_name_pct'))} (R10 line: {R10_CAP * 100:.0f}%)"
    )
    print(f"  R10 fire days     : {conc.get('single_name_r10_fire_days', 0)}")
    print(
        f"  Max sector %      : {_pct(conc.get('max_sector_pct'))} (R9 line: {R9_CAP * 100:.0f}%)"
    )
    print(f"  R9 fire days      : {conc.get('sector_r9_fire_days', 0)}")
    unk = conc.get("unknown_sector_tickers", [])
    if unk:
        print(f"  Unknown-sector    : {', '.join(unk)}")
    else:
        print("  Unknown-sector    : none")

    print("\n--- 6. BENCHMARK ---")
    print(f"  EW B&H return  : {_pct(b.get('ew_return_pct'))}")
    print(f"  N tickers found: {b.get('n_tickers_in_universe', 0)}")
    if b.get("spy_available"):
        print(f"  SPY B&H return : {_pct(b.get('spy_return_pct'))}")
    else:
        print("  SPY            : SPY not in dataset")
    print(f"  Engine − BM    : {_pct(a.get('engine_minus_benchmark_pct'))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def discover_windows(root: Path) -> list[str]:
    """Return all subdirectory names that contain a full/ subfolder."""
    windows = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "full").is_dir():
            windows.append(child.name)
    return windows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--root",
        default=str(DEFAULT_ROOT),
        help=f"Root directory containing window subdirs (default: {DEFAULT_ROOT})",
    )
    ap.add_argument("--window", default=None, help="Analyze one specific window ID")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Analyze all windows under --root and write campaign_table.md",
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help=f"Run against {SELFTEST_ROOT} as a single pseudo-window named 'smoke'",
    )
    args = ap.parse_args()

    sector_map = _build_sector_map()
    if not sector_map:
        print("  WARN  sector map empty — all tickers will land in 'Unknown'")

    analyses: list[dict] = []

    if args.selftest:
        # Treat sim200k_smoke as one pseudo-window named 'smoke'
        smoke_root = SELFTEST_ROOT
        if not smoke_root.is_dir():
            print(f"ERROR  selftest dir not found: {smoke_root}")
            return 1
        # Wrap smoke root itself as the window_dir by making a temp parent
        # The structure is <smoke_root>/{none,bid_ask,full}/... (no window subdir)
        a = analyze_window(smoke_root, "smoke", sector_map)
        if a is None:
            print("ERROR  selftest failed — smoke data missing full/ subfolder")
            return 1
        analyses.append(a)
        print_analysis(a)

        # Write analysis.json alongside the smoke artifacts
        out_path = smoke_root / "analysis.json"
        out_path.write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
        print(f"\n  Wrote {out_path}")

        print("\n\n=== CAMPAIGN TABLE (smoke only) ===\n")
        print(render_summary_table(analyses))
        print("\n=== CALIBRATION ===")
        print(render_calibration_tables(analyses))
        print("\n=== CONCENTRATION ===")
        print(render_concentration_table(analyses))
        print("\n=== FRICTION SENSITIVITY ===")
        print(render_friction_table(analyses))
        return 0

    root = Path(args.root)
    if not root.is_dir():
        print(f"ERROR  root not found: {root}")
        return 1

    if args.window:
        window_dir = root / args.window
        a = analyze_window(window_dir, args.window, sector_map)
        if a is None:
            return 1
        analyses.append(a)
        print_analysis(a)
        out_path = window_dir / "analysis.json"
        out_path.write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
        print(f"\n  Wrote {out_path}")

    elif args.all:
        windows = discover_windows(root)
        if not windows:
            print(f"  No windows with full/ found under {root}")
            return 0
        print(f"  Found {len(windows)} window(s): {windows}")
        for wid in windows:
            window_dir = root / wid
            a = analyze_window(window_dir, wid, sector_map)
            if a is None:
                continue
            analyses.append(a)
            print_analysis(a)
            out_path = window_dir / "analysis.json"
            out_path.write_text(json.dumps(a, indent=2, default=str), encoding="utf-8")
            print(f"\n  Wrote {out_path}")

        # Write combined campaign_table.md
        md_lines = [
            "# sim200k Campaign Analysis\n",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Summary\n",
            render_summary_table(analyses),
            "\n## Calibration\n",
            render_calibration_tables(analyses),
            "\n## Concentration (R9/R10 audit)\n",
            render_concentration_table(analyses),
            "\n## Friction Sensitivity\n",
            render_friction_table(analyses),
        ]
        md_path = root / "campaign_table.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"\n  Wrote {md_path}")
    else:
        ap.print_help()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
