"""Reconstruct the executed-trade set from a rank_log when the harness's
tracker_state.json is unavailable.

Used for W1 of S43 — W1 launched before commit `daa2282` added the
tracker dump, so the only artifact on disk is `rank_log.csv` +
`metrics.json`. We replay the rank_log through the same filter logic
the harness uses in ``_tracker_try_opens`` /
``_tracker_step`` (`backtests/regression/_common.py`) to identify
which rank rows became actual opens:

  - ev_dollars > 0
  - ticker not currently in an open position
  - cash + (premium * 100) >= strike * 100  (buying-power check)
  - opens_today < max_new_per_day (3)

Cash is tracked starting at $1M. Opening a put:
  cash -= strike * 100; cash += premium * 100
  (i.e., cash -= (strike − premium) × 100 net change)
Settling a put at expiry:
  cash += max(0, strike − spot) × −100 (assignment loss)
  + (always implicit: premium already credited at open)
  But for tracking purposes we just add ``realized_pnl`` (from the
  rank_log row) at expiry — that already reconciles premium ± intrinsic.

The reconstructor cannot perfectly reproduce CC entries (the harness's
wheel-into-CC after assignment), because the rank_log only records
put candidates. The CC realized P&L is therefore EXCLUDED here.
This matches the put-only framing S38 uses for its "executed
realized P&L" line; the harness's tracker count of
``executed_trades`` is puts-only by the same convention.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False, help=__doc__)


@dataclass
class _OpenPosition:
    ticker: str
    strike: float
    premium: float
    entry_date: date
    expiration_date: date
    realized_pnl: float
    sequence_idx: int = 0


@dataclass
class ReconstructionResult:
    capital_initial: float
    n_rank_rows: int
    n_opens: int
    open_positions: list[dict] = field(default_factory=list)
    by_ticker: dict[str, dict] = field(default_factory=dict)
    cash_curve: list[dict] = field(default_factory=list)


def reconstruct_opens(
    rank_log: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    max_new_per_day: int = 3,
    contracts: int = 1,
) -> ReconstructionResult:
    """Replay the rank_log to identify which rows became opens.

    Only operates on a single friction level's rank_log.
    """
    if rank_log.empty:
        return ReconstructionResult(initial_capital, 0, 0)

    df = rank_log.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.date

    # Sort by date asc; within a day the harness uses the rank_log order
    # (which is already in rank order from rank_candidates_by_ev). The
    # rank_log row order is preserved by pandas.
    df = df.sort_values(["date"], kind="stable").reset_index(drop=True)

    cash = initial_capital
    opens: dict[str, _OpenPosition] = {}  # ticker → open position
    all_opens: list[_OpenPosition] = []
    cash_curve: list[dict] = []
    opens_per_day: dict[date, int] = defaultdict(int)
    seq = 0

    # Group by date to settle expirations first, then process opens.
    unique_dates = sorted(df["date"].unique())
    date_to_rows: dict[date, pd.DataFrame] = dict(df.groupby("date", sort=False).__iter__())

    for today in unique_dates:
        # Settle any expirations at today or before
        to_close = [t for t, p in opens.items() if p.expiration_date <= today]
        for t in to_close:
            p = opens.pop(t)
            # Cash flow on close:
            # The harness records realized_pnl = (premium - intrinsic) * 100
            # At open, we already adjusted cash for both collateral and premium.
            # At expiry, intrinsic (if any) was effectively a loss on the
            # collateral side. Net cash effect: cash += (strike * 100) [collateral returned]
            #   minus assignment loss (intrinsic * 100), if any.
            # But: realized_pnl already encodes (premium - intrinsic) * 100.
            # At open: cash -= (strike - premium) * 100.
            # At close: cash += strike * 100 - intrinsic * 100 = (strike - intrinsic) * 100
            # Total over the trade: -(strike - premium) + (strike - intrinsic) = (premium - intrinsic) = realized_pnl / 100
            # So the cash settles to original + realized_pnl.
            # We can model it as: at close, cash += strike * 100 - (intrinsic * 100)
            # where intrinsic = (strike * 100 - premium * 100 - realized_pnl) / 100
            # Simpler: cash += (strike + realized_pnl / 100 - premium) * 100
            cash += (p.strike + p.realized_pnl / 100.0 - p.premium) * 100.0 * contracts
            cash_curve.append({"date": str(today), "event": "close", "ticker": t, "cash": cash})

        # Process today's opens
        rows = date_to_rows.get(today)
        if rows is None:
            continue
        for _, row in rows.iterrows():
            if opens_per_day[today] >= max_new_per_day:
                break
            ev = float(row["ev_dollars"])
            if not np.isfinite(ev) or ev <= 0:
                continue
            t = str(row["ticker"])
            if t in opens:
                continue
            strike = float(row["strike"])
            premium = float(row["premium"])
            if premium <= 0 or strike <= 0:
                continue
            # BP check: same as harness — strike * 100 collateral needed
            if cash < strike * 100.0 * contracts:
                continue
            realized_pnl = float(row.get("realized_pnl") or 0.0)
            if not np.isfinite(realized_pnl):
                realized_pnl = 0.0
            # Cash flow on open: -collateral + premium
            cash -= (strike - premium) * 100.0 * contracts
            exp = row["expiration_date"]
            if not isinstance(exp, date):
                exp = pd.to_datetime(exp).date()
            opens[t] = _OpenPosition(
                ticker=t,
                strike=strike,
                premium=premium,
                entry_date=today,
                expiration_date=exp,
                realized_pnl=realized_pnl,
                sequence_idx=seq,
            )
            all_opens.append(opens[t])
            opens_per_day[today] += 1
            seq += 1
            cash_curve.append({"date": str(today), "event": "open", "ticker": t, "cash": cash})

    by_ticker: dict[str, dict] = defaultdict(lambda: {"trades": 0, "realized": 0.0})
    for p in all_opens:
        by_ticker[p.ticker]["trades"] += 1
        by_ticker[p.ticker]["realized"] += p.realized_pnl

    return ReconstructionResult(
        capital_initial=initial_capital,
        n_rank_rows=int(len(rank_log)),
        n_opens=len(all_opens),
        open_positions=[
            {
                "ticker": p.ticker,
                "strike": p.strike,
                "premium": p.premium,
                "entry_date": str(p.entry_date),
                "expiration_date": str(p.expiration_date),
                "realized_pnl": p.realized_pnl,
            }
            for p in all_opens
        ],
        by_ticker={t: dict(v) for t, v in by_ticker.items()},
        cash_curve=cash_curve,
    )


def _audit_r9_r10(
    opens: Iterable[dict],
    initial_capital: float,
    max_single_name_pct: float = 0.10,
    max_sector_pct: float = 0.25,
    sector_map: dict[str, str] | None = None,
) -> dict:
    """Post-hoc audit: would R9 (sector cap) or R10 (single-name cap)
    have refused any open under the published defaults?"""
    open_book_ticker: dict[str, float] = defaultdict(float)
    open_book_sector: dict[str, float] = defaultdict(float)
    events: list[tuple[str, str, str, float]] = []
    for p in opens:
        notional = p["strike"] * 100.0
        events.append((p["entry_date"], "open", p["ticker"], notional))
        events.append((p["expiration_date"], "close", p["ticker"], notional))
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1))

    r10_breaches = 0
    r9_breaches = 0
    n_opens = 0
    max_single = 0.0
    max_sector = 0.0

    for _d, kind, t, notional in events:
        if kind == "open":
            n_opens += 1
            new = open_book_ticker[t] + notional
            if new > max_single_name_pct * initial_capital:
                r10_breaches += 1
            open_book_ticker[t] = new
            max_single = max(max_single, new)
            if sector_map:
                sec = sector_map.get(t)
                if sec:
                    new_s = open_book_sector[sec] + notional
                    if new_s > max_sector_pct * initial_capital:
                        r9_breaches += 1
                    open_book_sector[sec] = new_s
                    max_sector = max(max_sector, new_s)
        else:
            open_book_ticker[t] = max(0.0, open_book_ticker[t] - notional)
            if sector_map:
                sec = sector_map.get(t)
                if sec:
                    open_book_sector[sec] = max(0.0, open_book_sector[sec] - notional)

    return {
        "n_opens": n_opens,
        "r10_single_name_would_fire_count": r10_breaches,
        "r10_single_name_max_pct_of_nav": 100.0 * max_single / initial_capital
        if initial_capital
        else 0.0,
        "r9_sector_would_fire_count": r9_breaches if sector_map else None,
        "r9_sector_max_pct_of_nav": 100.0 * max_sector / initial_capital
        if sector_map and initial_capital
        else None,
        "sector_audit_skipped": sector_map is None,
        "defaults": {
            "max_single_name_pct": max_single_name_pct,
            "max_sector_pct": max_sector_pct,
        },
    }


@app.command()
def run(
    window_dir: Path,
    friction: str = "full",
    initial_capital: float = 1_000_000.0,
) -> None:
    """Reconstruct opens from a window's rank_log + dump a JSON report."""
    rank_log_path = window_dir / friction / "rank_log.csv"
    if not rank_log_path.exists():
        raise typer.Exit(f"rank_log.csv missing at {rank_log_path}")
    df = pd.read_csv(rank_log_path)
    result = reconstruct_opens(df, initial_capital=initial_capital)

    # Concentration
    by_ticker = sorted(result.by_ticker.items(), key=lambda kv: kv[1]["realized"], reverse=True)
    total = sum(v["realized"] for v in result.by_ticker.values())
    top5 = sum(v["realized"] for _, v in by_ticker[:5])
    top10 = sum(v["realized"] for _, v in by_ticker[:10])

    # R9/R10 audit
    audit = _audit_r9_r10(result.open_positions, initial_capital)

    report = {
        "window_dir": str(window_dir),
        "friction": friction,
        "initial_capital": initial_capital,
        "n_rank_rows": result.n_rank_rows,
        "n_opens": result.n_opens,
        "total_realized": total,
        "top_5": [
            {"ticker": t, "trades": v["trades"], "realized": v["realized"]}
            for t, v in by_ticker[:5]
        ],
        "top_10_realized_sum": top10,
        "top_5_realized_sum": top5,
        "top_5_share_of_net": (top5 / total) if abs(total) > 1e-6 else float("nan"),
        "negative_tickers_sum": sum(v["realized"] for _, v in by_ticker if v["realized"] < 0),
        "positive_tickers_sum": sum(v["realized"] for _, v in by_ticker if v["realized"] > 0),
        "by_ticker_sorted": [
            {"ticker": t, "trades": v["trades"], "realized": v["realized"]} for t, v in by_ticker
        ],
        "r9_r10_audit": audit,
        "cash_curve_length": len(result.cash_curve),
        "cash_at_end": result.cash_curve[-1]["cash"] if result.cash_curve else initial_capital,
    }

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    app()
