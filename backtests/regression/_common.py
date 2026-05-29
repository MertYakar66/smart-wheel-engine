"""Shared driver, friction overlay, and snapshot I/O for the four
regression backtests.

Two drivers:
- :func:`run_backtest` — single friction level, used by S27.
- :func:`run_backtest_multi_friction` — N friction levels with a shared
  SP rank call per day (~N× cheaper than looping the single driver),
  used by S32/S34/S35.

Both route every candidate through
:meth:`engine.wheel_runner.WheelRunner.rank_candidates_by_ev`
end-to-end (no engine mocking, no §2 bypass), walk the ranked rows
through one or more :class:`engine.wheel_tracker.WheelTracker`
instances, wheel into covered calls on assignment, and compute the
metrics dict that maps 1:1 to the JSON snapshot schema.

Friction overlay matches the S32 doc method appendix:

==========================  =========================================
Component                   Model
==========================  =========================================
Bid/ask half-spread         ``max($0.05, 8 % of premium)`` per share
Commission (open / close)   ``$0.65 / contract``
Assignment slippage         ``10 bp × strike × 100`` (equity notional)
==========================  =========================================

Three friction levels are supported: ``"none"`` (frictionless),
``"bid_ask"`` (half-spread only), ``"full"`` (all three components).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Friction overlay
# ---------------------------------------------------------------------------

_FRICTION_LEVELS = ("none", "bid_ask", "full")


def friction_adjusted_premium(premium: float, friction_level: str) -> float:
    """Premium net of bid/ask half-spread (per share).

    Applies for ``bid_ask`` and ``full`` levels; passes through
    unchanged for ``none``.
    """
    if friction_level not in _FRICTION_LEVELS:
        raise ValueError(
            f"friction_level must be one of {_FRICTION_LEVELS}, got {friction_level!r}"
        )
    if friction_level == "none":
        return premium
    half_spread = max(0.05, 0.08 * premium)
    return max(0.0, premium - half_spread)


def friction_open_cost(contracts: int, friction_level: str) -> float:
    """Commission paid to open a position (dollars per contract)."""
    if friction_level not in _FRICTION_LEVELS:
        raise ValueError(
            f"friction_level must be one of {_FRICTION_LEVELS}, got {friction_level!r}"
        )
    if friction_level == "full":
        return 0.65 * contracts
    return 0.0


def friction_assignment_cost(strike: float, contracts: int, friction_level: str) -> float:
    """Slippage on assignment — 10 bp of equity notional.

    Applies only at ``full``; ``bid_ask`` skips assignment-slip because
    by construction it covers only the open-side spread.
    """
    if friction_level not in _FRICTION_LEVELS:
        raise ValueError(
            f"friction_level must be one of {_FRICTION_LEVELS}, got {friction_level!r}"
        )
    if friction_level == "full":
        return 0.0010 * strike * 100 * contracts + 0.65 * contracts
    return 0.0


# ---------------------------------------------------------------------------
# Data window assertion
# ---------------------------------------------------------------------------

_OHLCV_PATH = Path("data/bloomberg/sp500_ohlcv.csv")
_REFRESH_COMMAND = "python scripts/pull_ohlcv.py  # first edit hardcoded end_date"


def assert_data_window_available(start: str, end: str, ohlcv_path: Path | None = None) -> None:
    """Raise if the Bloomberg OHLCV CSV doesn't cover ``[start, end]``.

    Caller passes ISO date strings. Reads only the ``date`` column for
    head/tail efficiency on the 59 MB file.
    """
    path = ohlcv_path or _OHLCV_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"OHLCV CSV missing at {path}. Refresh per docs/DATA_POLICY.md §5: {_REFRESH_COMMAND}"
        )
    dates = pd.read_csv(path, usecols=["date"], parse_dates=["date"])["date"]
    earliest, latest = dates.min().date(), dates.max().date()
    req_start, req_end = date.fromisoformat(start), date.fromisoformat(end)
    if req_start < earliest or req_end > latest:
        raise RuntimeError(
            f"OHLCV CSV covers {earliest} → {latest}; backtest needs {req_start} → {req_end}. "
            f"Refresh: {_REFRESH_COMMAND}"
        )


def ohlcv_sha256(ohlcv_path: Path | None = None) -> str:
    """SHA-256 of the OHLCV CSV — pinned in snapshot fingerprints so a
    CSV refresh forces an explicit re-baseline."""
    path = ohlcv_path or _OHLCV_PATH
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Snapshot I/O
# ---------------------------------------------------------------------------

_SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def snapshot_path(snapshot_id: str) -> Path:
    return _SNAPSHOT_DIR / f"{snapshot_id}.json"


def load_snapshot(snapshot_id: str) -> dict:
    with open(snapshot_path(snapshot_id), encoding="utf-8") as f:
        return json.load(f)


def save_snapshot(snapshot_id: str, payload: dict) -> Path:
    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    p = snapshot_path(snapshot_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
        f.write("\n")
    return p


# ---------------------------------------------------------------------------
# Forward-replay and metrics
# ---------------------------------------------------------------------------


def _forward_replay_realized_pnl(strike: float, premium: float, spot_at_expiry: float) -> float:
    """Held-to-expiry P&L for a short cash-secured put, dollars per contract.

    OTM at expiry → keep full premium. ITM → assigned at strike; the
    long-stock leg is marked at ``spot_at_expiry`` (immediate sell
    convention used by S22 / S27).
    """
    intrinsic = max(0.0, strike - spot_at_expiry)
    return (premium - intrinsic) * 100.0


def _bucket_quartile(values: np.ndarray) -> np.ndarray:
    """Q0 (lowest) → Q3 (highest) by rank. ``pd.qcut`` with 4 bins; ties
    broken by first-occurrence (``duplicates='drop'`` is a no-op since
    we expect continuous floats)."""
    ranks = pd.Series(values).rank(method="first")
    return pd.qcut(ranks, q=4, labels=["Q0", "Q1", "Q2", "Q3"]).to_numpy()


def _spearman_rho_and_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Spearman ρ + two-sided p. ``scipy.stats.spearmanr`` returns NaN
    for n<2 or constant input — caller handles those branches."""
    from scipy.stats import spearmanr

    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    res = spearmanr(x, y, nan_policy="omit")
    return float(res.correlation), float(res.pvalue)


def _compute_metrics(rank_log: pd.DataFrame, tracker: Any) -> dict:
    """Reduce per-row rank log + tracker state to the snapshot dict.

    ``rank_log`` columns: ``date, ticker, ev_dollars, premium, strike,
    iv, prob_profit, expiration_date, spot_at_expiry, realized_pnl``.
    Rows with no realized_pnl (expiration outside window) are dropped
    before computing ρ / hit-rate.
    """
    replayed = rank_log.dropna(subset=["realized_pnl"]).copy()
    aggregate = {
        "row_count": int(len(replayed)),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "mean_realized": float("nan"),
        "hit_rate": float("nan"),
        "iv_mean": float("nan"),
        "ev_mean": float("nan"),
    }
    if len(replayed) >= 2:
        rho, p = _spearman_rho_and_p(
            replayed["ev_dollars"].to_numpy(), replayed["realized_pnl"].to_numpy()
        )
        aggregate.update(
            {
                "spearman_rho": rho,
                "spearman_p": p,
                "mean_realized": float(replayed["realized_pnl"].mean()),
                "hit_rate": float((replayed["realized_pnl"] > 0).mean()),
                "iv_mean": float(replayed["iv"].mean()),
                "ev_mean": float(replayed["ev_dollars"].mean()),
            }
        )

    per_year = {}
    if "date" in replayed.columns and len(replayed):
        replayed["year"] = pd.to_datetime(replayed["date"]).dt.year
        for year, grp in replayed.groupby("year"):
            rho, p = _spearman_rho_and_p(
                grp["ev_dollars"].to_numpy(), grp["realized_pnl"].to_numpy()
            )
            per_year[str(int(year))] = {
                "n": int(len(grp)),
                "rho": rho,
                "p": p,
                "mean_realized": float(grp["realized_pnl"].mean()),
                "hit_rate": float((grp["realized_pnl"] > 0).mean()),
                "iv_mean": float(grp["iv"].mean()),
            }

    per_quartile = {}
    if len(replayed) >= 4:
        replayed["quartile"] = _bucket_quartile(replayed["ev_dollars"].to_numpy())
        for q, grp in replayed.groupby("quartile", observed=True):
            per_quartile[str(q)] = {
                "n": int(len(grp)),
                "ev_mean": float(grp["ev_dollars"].mean()),
                "pnl_mean": float(grp["realized_pnl"].mean()),
                "hit": float((grp["realized_pnl"] > 0).mean()),
            }

    # Tracker state. closed_positions only appends on _finalize_position
    # (close_short_put / close_covered_call / handle_call_assignment — NOT
    # on handle_put_assignment, which transitions SHORT_PUT → STOCK_OWNED
    # without closing). So executed_trades = closed-with-put + still-open
    # with put-history. equity_curve records use key "portfolio_value".
    closed_with_put = sum(1 for r in tracker.closed_positions if (r.get("put_premium") or 0) > 0)
    open_with_put = sum(1 for p in tracker.positions.values() if (p.put_premium or 0) > 0)
    put_assigned_open = sum(
        1 for p in tracker.positions.values() if p.state.value in ("stock_owned", "covered_call")
    )
    put_assigned_closed = sum(
        1 for r in tracker.closed_positions if r.get("exit_reason") == "call_assigned"
    )
    final_pv = (
        float(tracker.equity_curve[-1].get("portfolio_value", tracker.cash))
        if tracker.equity_curve
        else float(tracker.cash)
    )
    tracker_metrics = {
        "final_cash": float(tracker.cash),
        "final_nav": final_pv,
        "executed_trades": int(closed_with_put + open_with_put),
        "put_assignments": int(put_assigned_open + put_assigned_closed),
        "open_at_end": int(
            sum(1 for p in tracker.positions.values() if p.state.value != "no_position")
        ),
    }

    return {
        "aggregate": {**aggregate, **tracker_metrics},
        "per_year": per_year,
        "per_quartile": per_quartile,
    }


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Outcome of :func:`run_backtest` for a single friction level."""

    metrics: dict
    rank_log: pd.DataFrame  # full per-row log; large — caller decides whether to persist
    fingerprint: dict


def _next_business_day(d: date) -> date:
    """Skip weekends. Holiday-aware calendars belong in PR2 if needed."""
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def _spot_on_or_after(
    conn: Any, ticker: str, target: date, max_lookahead_days: int = 7
) -> float | None:
    """Close price on ``target`` or the next available trading day within
    ``max_lookahead_days``. Returns ``None`` if no data."""
    try:
        df = conn.get_ohlcv(
            ticker,
            start_date=target.isoformat(),
            end_date=(target + timedelta(days=max_lookahead_days)).isoformat(),
        )
    except Exception:
        return None
    if df.empty:
        return None
    return float(df["close"].iloc[0])


def run_backtest(
    *,
    capital: float,
    tickers: Sequence[str],
    start: str,
    end: str,
    seed: int = 42,
    friction_level: str = "none",
    top_n: int = 10,
    max_new_per_day: int = 3,
    dte_target: int = 35,
    delta_target: float = 0.25,
    contracts: int = 1,
    output_dir: Path | str | None = None,
) -> BacktestResult:
    """End-to-end backtest driver. Routes the entire flow through
    ``WheelRunner.rank_candidates_by_ev`` — no engine mocking.

    Returns a :class:`BacktestResult` carrying the metrics dict (the
    snapshot's ``aggregate`` / ``per_year`` / ``per_quartile`` sections)
    and the full per-row rank log (useful for forensic deep-diff and
    optional CSV persistence to ``output_dir``).
    """
    from engine.wheel_runner import WheelRunner
    from engine.wheel_tracker import PositionState, WheelTracker

    if friction_level not in _FRICTION_LEVELS:
        raise ValueError(
            f"friction_level must be one of {_FRICTION_LEVELS}, got {friction_level!r}"
        )

    assert_data_window_available(start, end)
    np.random.default_rng(seed)  # placeholder for any downstream seed consumers

    runner = WheelRunner()
    conn = runner.connector
    tracker = WheelTracker(initial_capital=capital, connector=conn)

    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    tickers = list(tickers)
    rank_log_rows: list[dict] = []

    for today in trading_days:
        # 1) Mark-to-market open positions
        open_tickers = [
            t for t, p in tracker.positions.items() if p.state != PositionState.NO_POSITION
        ]
        prices_today: dict[str, float] = {}
        for t in open_tickers:
            spot = _spot_on_or_after(conn, t, today, max_lookahead_days=1)
            if spot is not None:
                prices_today[t] = spot
        if prices_today:
            tracker.mark_to_market(today, prices_today)

        # 2) Settle expirations — handle_put_expiration / handle_call_expiration
        # encapsulate the "ITM → assign, OTM → keep/close" branching cleanly.
        for t in list(tracker.positions.keys()):
            pos = tracker.positions[t]
            if (
                pos.state == PositionState.SHORT_PUT
                and pos.put_expiration_date
                and pos.put_expiration_date <= today
            ):
                spot = prices_today.get(t) or _spot_on_or_after(conn, t, today)
                if spot is None:
                    continue
                was_assigned = spot < (pos.put_strike or 0.0)
                tracker.handle_put_expiration(t, today, spot)
                if was_assigned and friction_level == "full":
                    tracker.cash -= friction_assignment_cost(
                        pos.put_strike or 0.0, contracts, friction_level
                    )
            elif (
                pos.state == PositionState.COVERED_CALL
                and pos.call_expiration_date
                and pos.call_expiration_date <= today
            ):
                spot = prices_today.get(t) or _spot_on_or_after(conn, t, today)
                if spot is None:
                    continue
                was_called_away = spot > (pos.call_strike or float("inf"))
                tracker.handle_call_expiration(t, today, spot)
                if was_called_away and friction_level == "full":
                    tracker.cash -= friction_assignment_cost(
                        pos.call_strike or 0.0, contracts, friction_level
                    )

        # 2.5) Wheel into covered calls on stock-owned tickers. Without
        # this, every assigned position locks its ticker out of future
        # rotation; S22 documented 16 CC entries off 7 put assignments
        # — the wheel's second leg is essential to the documented
        # execution count.
        expiration_default = _next_business_day(today + timedelta(days=dte_target))
        for t in list(tracker.positions.keys()):
            pos = tracker.positions[t]
            if pos.state != PositionState.STOCK_OWNED:
                continue
            try:
                cc_frame = runner.rank_covered_calls_by_ev(
                    ticker=t,
                    shares_held=100 * contracts,
                    as_of=today.isoformat(),
                    target_dtes=(dte_target,),
                    target_deltas=(delta_target,),
                    top_n=5,
                    min_ev_dollars=-1e9,
                    include_diagnostic_fields=True,
                )
            except Exception:
                continue
            if cc_frame is None or len(cc_frame) == 0:
                continue
            for _, cc_row in cc_frame.iterrows():
                if float(cc_row.get("ev_dollars", 0.0)) <= 0:
                    continue
                cc_strike = float(cc_row.get("strike", 0.0))
                cc_premium = friction_adjusted_premium(
                    float(cc_row.get("premium", 0.0)), friction_level
                )
                if cc_premium <= 0 or cc_strike <= 0:
                    continue
                # ``new_expiry`` is a Timestamp / str / date from the CC ranker
                raw_expiry = cc_row.get("new_expiry")
                if isinstance(raw_expiry, str):
                    cc_expiry = date.fromisoformat(raw_expiry[:10])
                elif hasattr(raw_expiry, "date"):
                    cc_expiry = raw_expiry.date()
                elif isinstance(raw_expiry, date):
                    cc_expiry = raw_expiry
                else:
                    cc_expiry = expiration_default
                opened_cc = tracker.open_covered_call(
                    ticker=t,
                    strike=cc_strike,
                    premium=cc_premium,
                    entry_date=today,
                    expiration_date=cc_expiry,
                    iv=float(cc_row.get("iv", 0.0)),
                )
                if opened_cc:
                    if friction_level == "full":
                        tracker.cash -= friction_open_cost(contracts, friction_level)
                    break  # one CC per ticker per day

        # 3) Rank candidates
        try:
            frame = runner.rank_candidates_by_ev(
                tickers=tickers,
                dte_target=dte_target,
                delta_target=delta_target,
                contracts=contracts,
                top_n=top_n,
                min_ev_dollars=-1e9,  # capture all ranked rows; filter on EV>0 at execution
                as_of=today.isoformat(),
                include_diagnostic_fields=True,
            )
        except Exception:
            continue
        if frame is None or len(frame) == 0:
            continue

        # 4) Persist rank log rows
        expiration_default = _next_business_day(today + timedelta(days=dte_target))
        for _, row in frame.iterrows():
            premium_raw = float(row.get("premium", 0.0))
            premium_adj = friction_adjusted_premium(premium_raw, friction_level)
            rank_log_rows.append(
                {
                    "date": today.isoformat(),
                    "ticker": str(row.get("ticker", "")),
                    "ev_dollars": float(row.get("ev_dollars", 0.0)),
                    "premium": premium_adj,
                    "premium_raw": premium_raw,
                    "strike": float(row.get("strike", 0.0)),
                    "iv": float(row.get("iv", 0.0)),
                    "prob_profit": float(row.get("prob_profit", float("nan"))),
                    "expiration_date": expiration_default.isoformat(),
                    "friction_level": friction_level,
                }
            )

        # 5) Open new positions (EV>0, BP available, no existing position, cap per day)
        opens_today = 0
        for _, row in frame.iterrows():
            if opens_today >= max_new_per_day:
                break
            if float(row.get("ev_dollars", 0.0)) <= 0:
                continue
            t = str(row.get("ticker", ""))
            if t in tracker.positions and tracker.positions[t].state != PositionState.NO_POSITION:
                continue
            strike = float(row.get("strike", 0.0))
            premium = friction_adjusted_premium(float(row.get("premium", 0.0)), friction_level)
            if premium <= 0 or strike <= 0:
                continue
            if tracker.available_buying_power() < strike * 100 * contracts:
                continue
            opened = tracker.open_short_put(
                ticker=t,
                strike=strike,
                premium=premium,
                entry_date=today,
                expiration_date=expiration_default,
                iv=float(row.get("iv", 0.0)),
            )
            if opened:
                if friction_level == "full":
                    tracker.cash -= friction_open_cost(contracts, friction_level)
                opens_today += 1

    # Forward-replay realized P&L on every ranked row
    rank_log = pd.DataFrame(rank_log_rows)
    if not rank_log.empty:
        realized = []
        spots_at_expiry = []
        for r in rank_log.itertuples(index=False):
            exp_dt = date.fromisoformat(r.expiration_date)
            spot = _spot_on_or_after(conn, r.ticker, exp_dt)
            if spot is None:
                realized.append(float("nan"))
                spots_at_expiry.append(float("nan"))
            else:
                realized.append(_forward_replay_realized_pnl(r.strike, r.premium, spot))
                spots_at_expiry.append(spot)
        rank_log["spot_at_expiry"] = spots_at_expiry
        rank_log["realized_pnl"] = realized

    metrics = _compute_metrics(rank_log, tracker)
    fingerprint = {
        "capital": capital,
        "tickers": list(tickers),
        "universe_size": len(tickers),
        "start": start,
        "end": end,
        "seed": seed,
        "friction_level": friction_level,
        "top_n": top_n,
        "max_new_per_day": max_new_per_day,
        "dte_target": dte_target,
        "delta_target": delta_target,
        "contracts": contracts,
        "data_csv_sha256": ohlcv_sha256(),
        "generated_at": datetime.now(UTC).isoformat(),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rank_log.to_csv(out / "rank_log.csv", index=False)
        with open(out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint, **metrics}, f, indent=2, default=str)
        # Additive: dump the tracker's serialised state so post-processors
        # (concentration analysis, R9/R10 fire-rate audit, cash-curve
        # reconstruction) don't have to replay the rank log. Pure additive
        # — no behaviour change; only writes a new file.
        try:
            with open(out / "tracker_state.json", "w", encoding="utf-8") as f:
                json.dump(tracker.to_dict(), f, indent=2, default=str)
        except Exception:
            pass

    return BacktestResult(metrics=metrics, rank_log=rank_log, fingerprint=fingerprint)


# ---------------------------------------------------------------------------
# Multi-friction driver — shares the SP rank call across friction levels
# ---------------------------------------------------------------------------
#
# The S32 throwaway harness ran 3 independent ``WheelTracker`` instances
# in parallel sharing one ``rank_candidates_by_ev`` call per day. This
# function reproduces that pattern: the SP rank (dominant cost) is
# computed once per day; each tracker then independently MTMs, settles
# expirations, wheels into CCs, and decides opens against the same
# ranked frame using its own friction-adjusted cash. Compute saving is
# ~3× on the dominant rank-call cost — material for S34 at 100 tickers.


def _tracker_step(
    *,
    tracker: Any,
    runner: Any,
    conn: Any,
    today: date,
    friction_level: str,
    contracts: int,
    dte_target: int,
    delta_target: float,
    expiration_default: date,
    PositionState: Any,
) -> dict[str, float]:
    """Per-tracker per-day pre-rank step: MTM, settle expirations, wheel
    into CCs. Returns the prices dict (callers reuse it for the open
    step). Mirrors run_backtest's inline steps 1-2.5 exactly."""
    open_tickers = [t for t, p in tracker.positions.items() if p.state != PositionState.NO_POSITION]
    prices_today: dict[str, float] = {}
    for t in open_tickers:
        spot = _spot_on_or_after(conn, t, today, max_lookahead_days=1)
        if spot is not None:
            prices_today[t] = spot
    if prices_today:
        tracker.mark_to_market(today, prices_today)

    # Settle expirations
    for t in list(tracker.positions.keys()):
        pos = tracker.positions[t]
        if (
            pos.state == PositionState.SHORT_PUT
            and pos.put_expiration_date
            and pos.put_expiration_date <= today
        ):
            spot = prices_today.get(t) or _spot_on_or_after(conn, t, today)
            if spot is None:
                continue
            was_assigned = spot < (pos.put_strike or 0.0)
            tracker.handle_put_expiration(t, today, spot)
            if was_assigned and friction_level == "full":
                tracker.cash -= friction_assignment_cost(
                    pos.put_strike or 0.0, contracts, friction_level
                )
        elif (
            pos.state == PositionState.COVERED_CALL
            and pos.call_expiration_date
            and pos.call_expiration_date <= today
        ):
            spot = prices_today.get(t) or _spot_on_or_after(conn, t, today)
            if spot is None:
                continue
            was_called_away = spot > (pos.call_strike or float("inf"))
            tracker.handle_call_expiration(t, today, spot)
            if was_called_away and friction_level == "full":
                tracker.cash -= friction_assignment_cost(
                    pos.call_strike or 0.0, contracts, friction_level
                )

    # Wheel into CCs on stock-owned tickers
    for t in list(tracker.positions.keys()):
        pos = tracker.positions[t]
        if pos.state != PositionState.STOCK_OWNED:
            continue
        try:
            cc_frame = runner.rank_covered_calls_by_ev(
                ticker=t,
                shares_held=100 * contracts,
                as_of=today.isoformat(),
                target_dtes=(dte_target,),
                target_deltas=(delta_target,),
                top_n=5,
                min_ev_dollars=-1e9,
                include_diagnostic_fields=True,
            )
        except Exception:
            continue
        if cc_frame is None or len(cc_frame) == 0:
            continue
        for _, cc_row in cc_frame.iterrows():
            if float(cc_row.get("ev_dollars", 0.0)) <= 0:
                continue
            cc_strike = float(cc_row.get("strike", 0.0))
            cc_premium = friction_adjusted_premium(
                float(cc_row.get("premium", 0.0)), friction_level
            )
            if cc_premium <= 0 or cc_strike <= 0:
                continue
            raw_expiry = cc_row.get("new_expiry")
            if isinstance(raw_expiry, str):
                cc_expiry = date.fromisoformat(raw_expiry[:10])
            elif hasattr(raw_expiry, "date"):
                cc_expiry = raw_expiry.date()
            elif isinstance(raw_expiry, date):
                cc_expiry = raw_expiry
            else:
                cc_expiry = expiration_default
            opened_cc = tracker.open_covered_call(
                ticker=t,
                strike=cc_strike,
                premium=cc_premium,
                entry_date=today,
                expiration_date=cc_expiry,
                iv=float(cc_row.get("iv", 0.0)),
            )
            if opened_cc:
                if friction_level == "full":
                    tracker.cash -= friction_open_cost(contracts, friction_level)
                break

    return prices_today


def _tracker_try_opens(
    *,
    tracker: Any,
    frame: pd.DataFrame,
    today: date,
    friction_level: str,
    max_new_per_day: int,
    contracts: int,
    expiration_default: date,
    PositionState: Any,
) -> None:
    """Mirrors run_backtest's inline 'open new positions' step."""
    opens_today = 0
    for _, row in frame.iterrows():
        if opens_today >= max_new_per_day:
            break
        if float(row.get("ev_dollars", 0.0)) <= 0:
            continue
        t = str(row.get("ticker", ""))
        if t in tracker.positions and tracker.positions[t].state != PositionState.NO_POSITION:
            continue
        strike = float(row.get("strike", 0.0))
        premium = friction_adjusted_premium(float(row.get("premium", 0.0)), friction_level)
        if premium <= 0 or strike <= 0:
            continue
        if tracker.available_buying_power() < strike * 100 * contracts:
            continue
        opened = tracker.open_short_put(
            ticker=t,
            strike=strike,
            premium=premium,
            entry_date=today,
            expiration_date=expiration_default,
            iv=float(row.get("iv", 0.0)),
        )
        if opened:
            if friction_level == "full":
                tracker.cash -= friction_open_cost(contracts, friction_level)
            opens_today += 1


def run_backtest_multi_friction(
    *,
    capital: float,
    tickers: Sequence[str],
    start: str,
    end: str,
    friction_levels: Sequence[str] = ("none", "bid_ask", "full"),
    seed: int = 42,
    top_n: int = 10,
    max_new_per_day: int = 3,
    dte_target: int = 35,
    delta_target: float = 0.25,
    contracts: int = 1,
    output_dir: Path | str | None = None,
) -> dict[str, BacktestResult]:
    """Run N friction levels with one shared SP rank call per day.

    Compared to looping ``run_backtest`` N times, this saves N× on the
    dominant ``rank_candidates_by_ev`` cost. Per-tracker CC ranks and
    forward-replay spot lookups are unique per friction-level (the
    rank_log differs in ``premium`` / ``friction_level``) and remain
    independent.

    Returns ``{level: BacktestResult}`` for downstream payload assembly.
    """
    from engine.wheel_runner import WheelRunner
    from engine.wheel_tracker import PositionState, WheelTracker

    for level in friction_levels:
        if level not in _FRICTION_LEVELS:
            raise ValueError(f"friction_level must be one of {_FRICTION_LEVELS}, got {level!r}")

    assert_data_window_available(start, end)
    np.random.default_rng(seed)

    runner = WheelRunner()
    conn = runner.connector
    trackers = {
        level: WheelTracker(initial_capital=capital, connector=conn) for level in friction_levels
    }
    rank_log_rows = {level: [] for level in friction_levels}

    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    tickers = list(tickers)

    _total_days = len(trading_days)
    _progress_every = max(1, _total_days // 25)  # ~25 prints across the run
    _t_start = time.time()

    for _day_idx, today in enumerate(trading_days):
        if _day_idx > 0 and _day_idx % _progress_every == 0:
            _elapsed = time.time() - _t_start
            _rate = _day_idx / _elapsed if _elapsed > 0 else 0.0
            _eta = (_total_days - _day_idx) / _rate if _rate > 0 else 0.0
            print(
                f"[multi_friction] day {_day_idx:4d}/{_total_days} ({100 * _day_idx / _total_days:5.1f}%) "
                f"elapsed {_elapsed / 60:6.1f}min  ETA {_eta / 60:6.1f}min  ({_rate:5.2f} day/s)",
                flush=True,
            )
        expiration_default = _next_business_day(today + timedelta(days=dte_target))

        # Per-tracker pre-rank steps (independent)
        for level, tracker in trackers.items():
            _tracker_step(
                tracker=tracker,
                runner=runner,
                conn=conn,
                today=today,
                friction_level=level,
                contracts=contracts,
                dte_target=dte_target,
                delta_target=delta_target,
                expiration_default=expiration_default,
                PositionState=PositionState,
            )

        # ONE shared SP rank call per day — friction-independent
        try:
            frame = runner.rank_candidates_by_ev(
                tickers=tickers,
                dte_target=dte_target,
                delta_target=delta_target,
                contracts=contracts,
                top_n=top_n,
                min_ev_dollars=-1e9,
                as_of=today.isoformat(),
                include_diagnostic_fields=True,
            )
        except Exception:
            continue
        if frame is None or len(frame) == 0:
            continue

        # Per-tracker: persist rank log + try opens
        for level, tracker in trackers.items():
            for _, row in frame.iterrows():
                premium_raw = float(row.get("premium", 0.0))
                premium_adj = friction_adjusted_premium(premium_raw, level)
                rank_log_rows[level].append(
                    {
                        "date": today.isoformat(),
                        "ticker": str(row.get("ticker", "")),
                        "ev_dollars": float(row.get("ev_dollars", 0.0)),
                        "premium": premium_adj,
                        "premium_raw": premium_raw,
                        "strike": float(row.get("strike", 0.0)),
                        "iv": float(row.get("iv", 0.0)),
                        "prob_profit": float(row.get("prob_profit", float("nan"))),
                        "expiration_date": expiration_default.isoformat(),
                        "friction_level": level,
                    }
                )
            _tracker_try_opens(
                tracker=tracker,
                frame=frame,
                today=today,
                friction_level=level,
                max_new_per_day=max_new_per_day,
                contracts=contracts,
                expiration_default=expiration_default,
                PositionState=PositionState,
            )

    # Forward-replay + metrics per tracker. Spot lookups for the same
    # ``(ticker, expiration_date)`` are identical across levels, so cache.
    spot_cache: dict[tuple[str, str], float | None] = {}
    results: dict[str, BacktestResult] = {}
    for level, tracker in trackers.items():
        rank_log = pd.DataFrame(rank_log_rows[level])
        if not rank_log.empty:
            realized = []
            spots_at_expiry = []
            for r in rank_log.itertuples(index=False):
                key = (r.ticker, r.expiration_date)
                if key not in spot_cache:
                    exp_dt = date.fromisoformat(r.expiration_date)
                    spot_cache[key] = _spot_on_or_after(conn, r.ticker, exp_dt)
                spot = spot_cache[key]
                if spot is None:
                    realized.append(float("nan"))
                    spots_at_expiry.append(float("nan"))
                else:
                    realized.append(_forward_replay_realized_pnl(r.strike, r.premium, spot))
                    spots_at_expiry.append(spot)
            rank_log["spot_at_expiry"] = spots_at_expiry
            rank_log["realized_pnl"] = realized

        metrics = _compute_metrics(rank_log, tracker)
        fingerprint = {
            "capital": capital,
            "tickers": list(tickers),
            "universe_size": len(tickers),
            "start": start,
            "end": end,
            "seed": seed,
            "friction_level": level,
            "top_n": top_n,
            "max_new_per_day": max_new_per_day,
            "dte_target": dte_target,
            "delta_target": delta_target,
            "contracts": contracts,
            "data_csv_sha256": ohlcv_sha256(),
            "generated_at": datetime.now(UTC).isoformat(),
        }

        if output_dir is not None:
            out = Path(output_dir) / level
            out.mkdir(parents=True, exist_ok=True)
            rank_log.to_csv(out / "rank_log.csv", index=False)
            with open(out / "metrics.json", "w", encoding="utf-8") as f:
                json.dump({"fingerprint": fingerprint, **metrics}, f, indent=2, default=str)
            # Additive: dump the tracker's serialised state so post-processors
            # (concentration analysis, R9/R10 fire-rate audit, cash-curve
            # reconstruction) don't have to replay the rank log. Pure additive
            # — no behaviour change; only writes a new file.
            try:
                with open(out / "tracker_state.json", "w", encoding="utf-8") as f:
                    json.dump(tracker.to_dict(), f, indent=2, default=str)
            except Exception:
                pass

        results[level] = BacktestResult(metrics=metrics, rank_log=rank_log, fingerprint=fingerprint)

    return results
