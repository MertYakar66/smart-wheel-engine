"""Per-window analysis for S41.

Consumes the artifacts the harness writes per window/friction-level
(``rank_log.csv``, ``metrics.json``, ``tracker_state.json``) and the
Bloomberg OHLCV CSV, and produces the analysis sections the S41
writeup needs:

  - Univ-EW passive baseline (equal-weighted buy-and-hold of the same
    100-ticker universe)
  - Per-year ρ + mean realized + executed count
  - Concentration: top-5 vs net realized P&L
  - Capital deployment (average across the window)
  - R9 / R10 fire-rate (post-hoc replay of the sector + single-name
    caps against the trade record; the harness runs without
    ``PortfolioContext`` so neither gate fires during execution)
  - §2 invariant counts

Outputs a JSON dict per window suitable for embedding in the writeup
tables. No engine state is mutated; this script is read-only.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import typer

# Universe alphabetical first-100 (matches S34 / S38 conventions).
from backtests.regression.universes import UNIVERSE_100

app = typer.Typer(add_completion=False, help=__doc__)


def _load_window(window_dir: Path) -> dict:
    """Load rank_log + metrics + tracker_state for each friction level."""
    data: dict = {"window_dir": str(window_dir), "per_friction": {}}
    summary = window_dir / "summary.json"
    if summary.exists():
        with open(summary, encoding="utf-8") as f:
            data["summary"] = json.load(f)
    for level in ("none", "bid_ask", "full"):
        d = window_dir / level
        entry: dict = {}
        if (d / "rank_log.csv").exists():
            entry["rank_log_path"] = str(d / "rank_log.csv")
            entry["rank_log"] = pd.read_csv(d / "rank_log.csv")
        if (d / "metrics.json").exists():
            with open(d / "metrics.json", encoding="utf-8") as f:
                entry["metrics"] = json.load(f)
        if (d / "tracker_state.json").exists():
            with open(d / "tracker_state.json", encoding="utf-8") as f:
                entry["tracker_state"] = json.load(f)
        data["per_friction"][level] = entry
    return data


def _univ_ew_return(start: str, end: str, ohlcv_path: Path) -> dict:
    """Compute the equal-weighted buy-and-hold return for the 100-ticker
    universe between ``start`` and ``end`` (close-to-close).

    Tickers without coverage on either bookend are excluded with a
    note (no fabrication). Returns the EW return and the per-ticker
    contribution.
    """
    df = pd.read_csv(ohlcv_path, usecols=["date", "ticker", "close"], parse_dates=["date"])
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    # Universe shape: connector returns 'XXX UN/UW Equity' format. Match
    # on the symbol-only prefix (split on space).
    df["sym"] = df["ticker"].str.split().str[0]

    # First & last close within window per ticker, restricted to UNIVERSE_100.
    # The connector normalises BF/B → BF-B etc. — match on the connector-
    # form (replace "/" with "-") and bare-form for safety.
    norm_universe = {t.replace("/", "-"): t for t in UNIVERSE_100} | {t: t for t in UNIVERSE_100}
    df = df[df["sym"].isin(norm_universe)].copy()

    in_window = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()

    per_ticker = []
    excluded = []
    for sym, grp in in_window.groupby("sym"):
        grp = grp.sort_values("date")
        if len(grp) < 2:
            excluded.append({"sym": sym, "reason": "<2 rows in window"})
            continue
        p0 = float(grp["close"].iloc[0])
        p1 = float(grp["close"].iloc[-1])
        if p0 <= 0 or p1 <= 0 or not math.isfinite(p0) or not math.isfinite(p1):
            excluded.append({"sym": sym, "reason": "bad price"})
            continue
        per_ticker.append(
            {
                "sym": sym,
                "start_close": p0,
                "end_close": p1,
                "ret": p1 / p0 - 1.0,
                "start_date": str(grp["date"].iloc[0].date()),
                "end_date": str(grp["date"].iloc[-1].date()),
            }
        )

    returns = np.array([t["ret"] for t in per_ticker])
    ew_ret = float(np.mean(returns)) if len(returns) else float("nan")
    median_ret = float(np.median(returns)) if len(returns) else float("nan")
    return {
        "ew_return_pct": ew_ret * 100,
        "median_return_pct": median_ret * 100,
        "n_tickers_included": len(per_ticker),
        "n_tickers_excluded": len(excluded),
        "excluded": excluded,
        "per_ticker": per_ticker,
    }


def _concentration(tracker_state: dict, ohlcv_path: Path) -> dict:
    """Top-5-vs-net realized P&L concentration analysis from the tracker
    closed_positions list.

    closed_positions contains only fully-closed trades (CC OTM-expire,
    CC called away, put assigned + CC sold). Put-assigned positions
    that haven't been called away yet remain in ``positions`` (active);
    their realized leg is the put leg's premium captured, not yet sold.
    """
    closed = tracker_state.get("closed_positions", [])
    if not closed:
        return {"n_closed": 0, "note": "no closed positions"}

    by_ticker: dict[str, dict] = defaultdict(lambda: {"trades": 0, "realized": 0.0})
    for rec in closed:
        t = str(rec.get("ticker", ""))
        # Realized P&L for closed positions: there are several shapes;
        # use total_pnl when present (the tracker computes net of premiums
        # in / out and stock leg).
        pnl = rec.get("total_pnl")
        if pnl is None:
            # Reconstruct from individual legs.
            put_p = float(rec.get("put_premium") or 0.0)
            call_p = float(rec.get("call_premium") or 0.0)
            put_buyback = float(rec.get("put_buyback") or 0.0)
            call_buyback = float(rec.get("call_buyback") or 0.0)
            stock_pnl = float(rec.get("stock_pnl") or 0.0)
            pnl = (put_p - put_buyback) * 100.0 + (call_p - call_buyback) * 100.0 + stock_pnl
        by_ticker[t]["trades"] += 1
        by_ticker[t]["realized"] += float(pnl)

    items = sorted(by_ticker.items(), key=lambda kv: kv[1]["realized"], reverse=True)
    total = sum(v["realized"] for v in by_ticker.values())
    top5 = sum(v["realized"] for _, v in items[:5])
    top10 = sum(v["realized"] for _, v in items[:10])
    negative = sum(v["realized"] for _, v in items if v["realized"] < 0)
    positive = sum(v["realized"] for _, v in items if v["realized"] > 0)

    return {
        "n_closed_positions": len(closed),
        "n_tickers_traded": len(by_ticker),
        "total_realized": total,
        "top5_realized": top5,
        "top5_share_of_net": (top5 / total) if abs(total) > 1e-6 else float("nan"),
        "top10_realized": top10,
        "negative_sum": negative,
        "positive_sum": positive,
        "by_ticker_sorted": [
            {"ticker": t, "trades": v["trades"], "realized": v["realized"]} for t, v in items
        ],
    }


def _refusal_rate(rank_log: pd.DataFrame, start: str, end: str) -> dict:
    """How often did the engine refuse (ev_dollars <= 0)?

    The harness opens only positions with ev_dollars > 0 (see
    ``_tracker_try_opens``). Refusal rate = share of ranked rows with
    ev_dollars <= 0 OR non-finite.
    """
    if rank_log.empty:
        return {"n_rows": 0}
    ev = rank_log["ev_dollars"].to_numpy(dtype=float)
    refused = int(np.sum((ev <= 0) | ~np.isfinite(ev)))
    return {
        "n_rows": int(len(rank_log)),
        "n_refused_by_ev": refused,
        "refusal_rate_pct": 100.0 * refused / len(rank_log),
    }


def _refusal_during_period(rank_log: pd.DataFrame, period_start: str, period_end: str) -> dict:
    """Refusal rate inside a specific period (used for COVID + 2022 bear)."""
    if rank_log.empty:
        return {"n_rows": 0, "period": [period_start, period_end]}
    rl = rank_log.copy()
    rl["date"] = pd.to_datetime(rl["date"])
    mask = (rl["date"] >= pd.to_datetime(period_start)) & (rl["date"] <= pd.to_datetime(period_end))
    sub = rl[mask]
    if sub.empty:
        return {"n_rows": 0, "period": [period_start, period_end]}
    ev = sub["ev_dollars"].to_numpy(dtype=float)
    refused = int(np.sum((ev <= 0) | ~np.isfinite(ev)))
    return {
        "n_rows": int(len(sub)),
        "n_refused_by_ev": refused,
        "refusal_rate_pct": 100.0 * refused / len(sub),
        "period": [period_start, period_end],
    }


def _section_2_scan(rank_log: pd.DataFrame) -> dict:
    """§2 invariant scan."""
    if rank_log.empty:
        return {"non_finite_ev": 0, "tradeable_with_negative_ev": 0, "passes": True}
    ev = rank_log["ev_dollars"].to_numpy(dtype=float)
    non_finite = int(np.sum(~np.isfinite(ev)))
    # The harness writes only ranked rows; the §2 invariant of *executed*
    # rows is enforced inline by ``_tracker_try_opens`` (filter ev>0).
    # The scan here is the R1a guard: no candidate emitted with NaN/inf EV.
    return {
        "non_finite_ev": non_finite,
        "rule_R1a_passes": non_finite == 0,
        "min_ev": float(np.nanmin(ev)),
        "max_ev": float(np.nanmax(ev)),
    }


def _r9_r10_audit(
    tracker_state: dict,
    sector_map: dict[str, str] | None,
    max_sector_pct: float = 0.25,
    max_single_name_pct: float = 0.10,
) -> dict:
    """Post-hoc replay: how often WOULD R9 (sector cap) or R10 (single-name
    cap) have fired against the realised trades?

    The harness ran without a ``PortfolioContext``, so neither gate
    actually fires during execution. This audit applies the same gate
    predicates after the fact to count would-be-fires per the published
    defaults (25% sector, 10% single name) using the tracker's
    actual open-position notional sequence.

    Without sector data this returns ``sector_audit_skipped``; the
    single-name audit is computable from tracker state alone.
    """
    closed = tracker_state.get("closed_positions", [])
    positions = tracker_state.get("positions", {})
    initial_capital = float(tracker_state.get("initial_capital", 0.0))

    # Reconstruct the time-ordered open-position notional. closed_positions
    # carry both entry and exit dates; we approximate the open set on each
    # entry by replaying chronologically. Notional = strike × 100 × contracts
    # (default contracts=1).
    events = []
    for rec in closed:
        t = str(rec.get("ticker", ""))
        ed = rec.get("put_entry_date") or rec.get("entry_date")
        xd = rec.get("exit_date") or rec.get("call_exit_date")
        strike = float(rec.get("put_strike") or rec.get("strike") or 0.0)
        if ed and strike > 0:
            events.append((str(ed), "open", t, strike))
        if xd and strike > 0:
            events.append((str(xd), "close", t, strike))
    # Open positions still present at end
    for t, pos in positions.items():
        strike = float(pos.get("put_strike") or 0.0)
        ed = pos.get("put_entry_date")
        if ed and strike > 0:
            events.append((str(ed), "open", t, strike))

    events.sort(key=lambda e: (e[0], e[1] == "close"))  # opens before closes on the same day

    open_by_ticker: dict[str, float] = {}
    open_by_sector: dict[str, float] = defaultdict(float)
    single_name_breaches = 0
    sector_breaches = 0
    n_opens = 0

    for evt in events:
        d, kind, t, strike = evt
        notional = strike * 100.0
        if kind == "open":
            n_opens += 1
            # Single-name check on THIS open
            new_single = open_by_ticker.get(t, 0.0) + notional
            if new_single > max_single_name_pct * initial_capital:
                single_name_breaches += 1
            open_by_ticker[t] = open_by_ticker.get(t, 0.0) + notional

            if sector_map:
                sec = sector_map.get(t)
                if sec:
                    new_sec = open_by_sector[sec] + notional
                    if new_sec > max_sector_pct * initial_capital:
                        sector_breaches += 1
                    open_by_sector[sec] += notional
        else:  # close
            open_by_ticker[t] = max(0.0, open_by_ticker.get(t, 0.0) - notional)
            if sector_map:
                sec = sector_map.get(t)
                if sec:
                    open_by_sector[sec] = max(0.0, open_by_sector[sec] - notional)

    return {
        "n_open_events": n_opens,
        "r10_single_name_would_fire_count": single_name_breaches,
        "r10_single_name_max_pct": (
            100.0 * max(open_by_ticker.values()) / initial_capital
            if open_by_ticker and initial_capital
            else 0.0
        ),
        "r9_sector_would_fire_count": sector_breaches if sector_map else None,
        "sector_audit_skipped": sector_map is None,
        "defaults": {
            "max_single_name_pct": max_single_name_pct,
            "max_sector_pct": max_sector_pct,
        },
    }


@app.command()
def analyze(window_dir: Path, ohlcv_path: Path = Path("data/bloomberg/sp500_ohlcv.csv")) -> None:
    """Full analysis of one window — outputs JSON to stdout."""
    data = _load_window(window_dir)
    summary = data.get("summary", {})
    start = summary.get("start") or "2018-01-03"
    end = summary.get("end") or "2022-12-30"

    report: dict = {
        "window_dir": str(window_dir),
        "window": {"start": start, "end": end},
        "univ_ew_baseline": _univ_ew_return(start, end, ohlcv_path),
        "per_friction": {},
    }

    for level, entry in data["per_friction"].items():
        rl = entry.get("rank_log")
        ts = entry.get("tracker_state")
        m = entry.get("metrics", {})

        sub = {
            "metrics_aggregate": m.get("aggregate", {}),
            "per_year": m.get("per_year", {}),
            "per_quartile": m.get("per_quartile", {}),
        }

        if rl is not None and len(rl):
            sub["section_2_scan"] = _section_2_scan(rl)
            sub["refusal_aggregate"] = _refusal_rate(rl, start, end)
            sub["refusal_covid"] = _refusal_during_period(rl, "2020-02-15", "2020-05-15")
            sub["refusal_2022_bear"] = _refusal_during_period(rl, "2022-01-01", "2022-10-31")

        if ts is not None:
            sub["concentration"] = _concentration(ts, ohlcv_path)
            sub["r9_r10_audit"] = _r9_r10_audit(ts, sector_map=None)

        report["per_friction"][level] = sub

    print(json.dumps(report, indent=2, default=str))


@app.command()
def all(
    root: Path = Path(os.environ.get("TEMP", "/tmp")) / "s41_backtest",
    ohlcv_path: Path = Path("data/bloomberg/sp500_ohlcv.csv"),
    out_path: Path | None = None,
) -> None:
    """Analyse all windows under ``root``."""
    reports = []
    for d in sorted(root.glob("w*_*")):
        if not d.is_dir():
            continue
        try:
            data = _load_window(d)
            summary = data.get("summary", {})
            start = summary.get("start") or "2018-01-03"
            end = summary.get("end") or "2022-12-30"

            report: dict = {
                "window_dir": str(d),
                "window_id": d.name,
                "window": {"start": start, "end": end},
                "univ_ew_baseline": _univ_ew_return(start, end, ohlcv_path),
                "per_friction": {},
            }
            for level, entry in data["per_friction"].items():
                rl = entry.get("rank_log")
                ts = entry.get("tracker_state")
                m = entry.get("metrics", {})
                sub = {
                    "metrics_aggregate": m.get("aggregate", {}),
                    "per_year": m.get("per_year", {}),
                    "per_quartile": m.get("per_quartile", {}),
                }
                if rl is not None and len(rl):
                    sub["section_2_scan"] = _section_2_scan(rl)
                    sub["refusal_aggregate"] = _refusal_rate(rl, start, end)
                    sub["refusal_covid"] = _refusal_during_period(rl, "2020-02-15", "2020-05-15")
                    sub["refusal_2022_bear"] = _refusal_during_period(
                        rl, "2022-01-01", "2022-10-31"
                    )
                if ts is not None:
                    sub["concentration"] = _concentration(ts, ohlcv_path)
                    sub["r9_r10_audit"] = _r9_r10_audit(ts, sector_map=None)
                report["per_friction"][level] = sub
            reports.append(report)
        except Exception as e:
            reports.append({"window_dir": str(d), "error": f"{type(e).__name__}: {e}"})

    blob = {"reports": reports}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2, default=str)
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        print(json.dumps(blob, indent=2, default=str))


if __name__ == "__main__":
    app()
