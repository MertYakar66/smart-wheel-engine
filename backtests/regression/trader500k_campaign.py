"""TRADER-500K reliability campaign driver — issue #517.

Simulates a $500k book that trades ONLY the engine's ranked output over
eleven 18-month windows (2020->2026), measuring calibration (forecast vs
realized) and economics (P&L vs passive benchmarks, net of frictions).

Thin wrapper over the existing replay harness building blocks in
``backtests/regression/_common.py`` — reused verbatim, not rebuilt:

- daily lifecycle via ``_tracker_step``   (mark -> settle -> covered calls)
- put entries via ``_tracker_try_opens``  (EV>0 only, BP gate, R10 armed)
- rail pin-off via ``_option_premium_rail_pinned_off`` + neutralization assert
- realized-P&L convention via ``_forward_replay_realized_pnl``
- harness metrics via ``_compute_metrics`` (Spearman/hit-rate/quartiles)

Campaign-specific divergences from ``run_backtest_multi_friction`` (all at
the loop level; the harness has no hooks for them — see issue #517 BRINGUP
and the SANDBOX FIX-REQUEST):

- WEEKLY put-rank cadence (first trading day of each ISO week); marks,
  settlements and covered-call wheeling stay DAILY (position lifecycle).
- REAL exchange calendar: trading days are the distinct dates present in
  the OHLCV monolith within the window — not ``pd.bdate_range`` — so no
  rank/mark/settle ever lands on an exchange holiday where the 1-day
  spot lookahead would leak the next session's close into a decision-day
  mark (driver-review finding, 2026-07-22).
- Capture wide, execute narrow (#517 FIX-REQUEST): the rank call uses
  ``top_n=len(universe)`` (compute-free — the ranker evaluates the full
  universe regardless and only truncates the returned frame) and the FULL
  frame goes to ``calibration_rows.csv.gz``; execution slices
  ``frame.head(execution_slice=20)`` FIRST, then applies the locked book
  semantics (``ev_dollars > 0``, max 5 opens, BP gate, R10) — byte-
  equivalent to the locked top-20 book, rows 21+ can never execute.
- Entry cutoff: no new put entries after ``window_end - 40`` calendar days
  (W11 overrides earlier, per the grid) so every 35-DTE put resolves
  inside the window. Covered calls on assigned stock continue after the
  cutoff (wheeling out of stock is lifecycle, not a new rank entry);
  unresolved CCs are marked, not settled, at the final NAV.
- End-of-day re-mark: after settlement and opens, each tracker is marked
  once more with the same day's prices, so the committed daily NAV is
  POST-settlement / post-open (the harness's mark-before-settle entry
  understates expiry-day intrinsic losses — driver-review finding). The
  committed curve uses this EOD mark; days where any open position lacks
  a same-day price fall back to carry-forward and are counted in
  ``n_partial_mark_days``.
- Full-universe scan: the universe is resolved ONCE per window via
  ``conn.get_universe()`` (identical to the ranker's ``tickers=None``
  path) and recorded in the fingerprint.
- ``use_credit_regime=False`` passed explicitly at the put-rank call
  (offline determinism; the harness rides the ranker default True).
- Captures ``frame.attrs["drops"]`` gate tallies (harness discards them).
- Refuses to run with ``SWE_DEEP_HISTORY`` set (#517 FIX-REQUEST): the
  campaign is specced on the 2018+ monolith; deep-ON would silently
  change forward-distribution support and diverge from recorded SHAs.

§2 stays intact: every candidate flows through
``rank_candidates_by_ev`` -> ``EVEngine.evaluate``; the decision trio is
never imported for mutation, only called.

Per-window committed bundle (docs/verification_artifacts/trader500k/Wnn/):
``summary.json``, ``equity_curve.csv.gz``, ``trades.csv.gz``,
``calibration_rows.csv.gz``. Large debug artifacts (per-level rank logs,
tracker states, raw drops) go to a local uncommitted work dir.

P&L conventions:
- ``realized_pnl_synth`` (calibration rows): frictionless hold-to-expiry
  ``(premium_raw - max(0, strike - spot_at_expiry)) * 100`` — the harness
  ``_forward_replay_realized_pnl`` convention.
- trades ledger ``realized_pnl``: same formula on the friction-adjusted
  premium, minus open cost (full only) and assignment cost (full only,
  when assigned). Hold-to-expiry attribution of the put leg; the wheel's
  post-assignment stock/CC economics live in the equity curve, not here.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import typer

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtests.regression._common import (  # noqa: E402
    _assert_rail_neutralized,
    _compute_metrics,
    _forward_replay_realized_pnl,
    _next_business_day,
    _option_premium_rail_pinned_off,
    _spot_on_or_after,
    _tracker_step,
    _tracker_try_opens,
    assert_data_window_available,
    connector_data_sha256,
    friction_adjusted_premium,
    friction_assignment_cost,
    friction_open_cost,
    ohlcv_sha256,
    treasury_sha256,
    vol_iv_sha256,
)

# ---------------------------------------------------------------------------
# Campaign spec (locked in issue #517 — do not tune here)
# ---------------------------------------------------------------------------

WINDOWS: dict[str, tuple[str, str, str]] = {
    "W01": ("2020-01-02", "2021-06-30", "earliest rankable (504-bar gate)"),
    "W02": ("2020-07-01", "2021-12-31", ""),
    "W03": ("2021-01-04", "2022-06-30", ""),
    "W04": ("2021-07-01", "2022-12-30", ""),
    "W05": ("2022-01-03", "2023-06-30", "PILOT — full 2022 bear, hardest test"),
    "W06": ("2022-07-01", "2023-12-29", ""),
    "W07": ("2023-01-03", "2024-06-28", ""),
    "W08": ("2023-07-03", "2024-12-31", ""),
    "W09": ("2024-01-02", "2025-06-30", ""),
    "W10": ("2024-07-01", "2025-12-31", ""),
    "W11": ("2025-01-02", "2026-06-30", "entry cutoff 2026-05-15 (frontier 2026-07-02)"),
}
ENTRY_CUTOFF_OVERRIDE: dict[str, str] = {"W11": "2026-05-15"}
ENTRY_CUTOFF_CAL_DAYS = 40
FRICTION_LEVELS = ("none", "bid_ask", "full")

CONFIG: dict = {
    "capital": 500_000.0,
    "capture_top_n": "universe_size",  # resolved to len(universe) at run time
    "execution_slice": 20,
    "max_new_per_rank_day": 5,
    "dte_target": 35,
    "delta_target": 0.25,
    "contracts": 1,
    "seed": 42,  # fingerprint-only; the replay path consumes no RNG
    "enforce_single_name_cap": True,
    "use_credit_regime": False,
    "min_ev_dollars_capture": -1e9,
    "rank_cadence": "weekly_first_trading_day",
    "trading_calendar": "ohlcv_distinct_dates",
    "cc_cadence": "daily_lifecycle",
    "entry_cutoff_cal_days": ENTRY_CUTOFF_CAL_DAYS,
    "option_premium_rail": "pinned_off",
}

ARTIFACT_ROOT = _REPO_ROOT / "docs" / "verification_artifacts" / "trader500k"
MACRO_CSV = _REPO_ROOT / "data" / "bloomberg" / "sp500_macro.csv"
OHLCV_CSV = _REPO_ROOT / "data" / "bloomberg" / "sp500_ohlcv.csv"

app = typer.Typer(add_completion=False, help=__doc__)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def spx_price_return(start: str, end: str) -> dict:
    """SPY-proxy: SPX close-to-close price return (ex-dividend) from the
    committed macro monolith (``instrument == 'spx'``), plus the
    ``spx_tr_approx`` total-return approximation required by the #517
    FIX-REQUEST (price return + 1.5%/yr accrued over the covered span)."""
    df = pd.read_csv(MACRO_CSV, usecols=["date", "close", "instrument"])
    df = df[df["instrument"] == "spx"].copy()
    df["date"] = pd.to_datetime(df["date"])
    win = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
    if len(win) < 2:
        return {"return_pct": float("nan"), "n_rows": len(win), "label": "price_return_ex_div"}
    first, last = float(win["close"].iloc[0]), float(win["close"].iloc[-1])
    first_d, last_d = win["date"].iloc[0].date(), win["date"].iloc[-1].date()
    ret_pct = (last / first - 1.0) * 100.0
    years = (last_d - first_d).days / 365.25
    return {
        "label": "price_return_ex_div",
        "return_pct": ret_pct,
        "spx_tr_approx_return_pct": ret_pct + 1.5 * years,
        "tr_approx_note": "price return + 1.5%/yr accrued — APPROXIMATION (#517 FIX-REQUEST)",
        "first_date": first_d.isoformat(),
        "last_date": last_d.isoformat(),
        "first_close": first,
        "last_close": last,
        "coverage_truncated": last_d.isoformat() < end,
        "note": "SPX close-to-close from sp500_macro.csv",
    }


def ew_passive_return(start: str, end: str, universe: list[str]) -> dict:
    """Equal-weight buy-and-hold over the campaign universe — the
    ``s43_analyze._univ_ew_return`` construction, applied to the resolved
    universe. Symbols match on the exchange-suffix-stripped form of BOTH
    sides (no slash->dash transform: the connector universe keeps 'BRK/B'
    and so does the OHLCV ticker column — driver-review finding)."""
    df = pd.read_csv(OHLCV_CSV, usecols=["date", "ticker", "close"])
    df["symbol"] = df["ticker"].str.split().str[0]
    keep = {u.split()[0] for u in universe}
    df = df[df["symbol"].isin(keep)]
    df["date"] = pd.to_datetime(df["date"])
    win = df[(df["date"] >= start) & (df["date"] <= end)]
    rets: list[float] = []
    excluded: list[str] = []
    for sym, g in win.groupby("symbol"):
        g = g.sort_values("date")
        if len(g) < 2:
            excluded.append(sym)
            continue
        first, last = float(g["close"].iloc[0]), float(g["close"].iloc[-1])
        if first <= 0 or not np.isfinite(first) or not np.isfinite(last):
            excluded.append(sym)
            continue
        rets.append(last / first - 1.0)
    missing = sorted(keep - set(win["symbol"].unique()))
    excluded = sorted(set(excluded) | set(missing))
    return {
        "label": "price_return_ex_div",
        "ew_return_pct": float(np.mean(rets)) * 100.0 if rets else float("nan"),
        "median_return_pct": float(np.median(rets)) * 100.0 if rets else float("nan"),
        "n_tickers_included": len(rets),
        "n_tickers_excluded": len(excluded),
        "excluded": excluded,
        "note": "EW buy-and-hold, first->last close in window (split- but not dividend-adjusted), per s43_analyze construction",
    }


# ---------------------------------------------------------------------------
# Curve metrics (equity-curve level — the regression lane doesn't compute these)
# ---------------------------------------------------------------------------


def curve_metrics(nav: pd.Series, initial_capital: float) -> dict:
    """CAGR / daily Sharpe / Sortino / max drawdown over a complete daily
    NAV series (business days)."""
    from engine.performance_metrics import (
        calculate_max_drawdown,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
    )

    nav = nav.astype(float)
    n_days = len(nav)
    total_return = float(nav.iloc[-1] / initial_capital - 1.0)
    cagr = float((1.0 + total_return) ** (252.0 / max(n_days, 1)) - 1.0)
    daily_ret = nav.pct_change().dropna()
    eq_df = pd.DataFrame({"portfolio_value": nav.values})
    max_dd, dd_days = calculate_max_drawdown(eq_df)
    return {
        "final_nav": float(nav.iloc[-1]),
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe_daily": float(calculate_sharpe_ratio(daily_ret)),
        "sortino_daily": float(calculate_sortino_ratio(daily_ret)),
        "max_drawdown_pct": float(max_dd) * 100.0,
        "max_drawdown_days": int(dd_days),
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# Window runner
# ---------------------------------------------------------------------------


def _exchange_trading_days(start: str, end: str) -> list[date]:
    """Actual exchange sessions in the window: the distinct dates present
    in the OHLCV monolith. Unlike ``pd.bdate_range`` this excludes
    exchange holidays, so no mark/rank/settle day can pull a next-session
    close through the 1-day spot lookahead."""
    df = pd.read_csv(OHLCV_CSV, usecols=["date"])
    days = sorted({d for d in df["date"] if start <= d <= end})
    return [date.fromisoformat(d) for d in days]


def _weekly_rank_days(trading_days: list[date]) -> set[date]:
    """First trading day of each ISO week present in the window."""
    rank_days: set[date] = set()
    seen: set[tuple[int, int]] = set()
    for d in trading_days:
        wk = d.isocalendar()[:2]
        if wk not in seen:
            seen.add(wk)
            rank_days.add(d)
    return rank_days


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    from scipy.stats import spearmanr

    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan"), float("nan")
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p)


def run_window(
    window_id: str,
    *,
    friction_levels: tuple[str, ...] = FRICTION_LEVELS,
    out_root: Path | None = None,
    work_root: Path | None = None,
    rail_on: bool = False,
    _smoke_universe_size: int | None = None,
    _smoke_window: tuple[str, str] | None = None,
) -> dict:
    from engine.wheel_runner import WheelRunner
    from engine.wheel_tracker import PositionState, WheelTracker

    assert not os.environ.get("SWE_DEEP_HISTORY"), (
        "SWE_DEEP_HISTORY is set — the campaign is specced on the 2018+ "
        "monolith and deep-ON would diverge from the recorded SHAs (#517 "
        "FIX-REQUEST). Unset it and re-run."
    )

    if _smoke_window is not None:
        start, end, note = *_smoke_window, "SMOKE — plumbing check only, not a campaign window"
    else:
        if window_id not in WINDOWS:
            raise typer.BadParameter(
                f"unknown window {window_id!r}; expected one of {sorted(WINDOWS)}"
            )
        start, end, note = WINDOWS[window_id]
    assert_data_window_available(start, end)

    t0 = time.time()
    if rail_on:
        # Phase C: the rail must be LIVE — real EOD mids served where the
        # (ticker, expiry, date) point coheres, synthetic-BSM elsewhere,
        # per-row premium_source provenance (#435/#463). Different premium
        # provenance from Phase A/B: outputs land in Wnn_rail/, never mixed.
        runner = WheelRunner()
        conn = runner.connector
        probe = conn._load_option_premium("AAPL")
        if probe is None or len(probe) == 0:
            raise RuntimeError(
                "--rail-on requested but the option-premium rail is not serving data "
                "(probe ticker AAPL returned empty)"
            )
    else:
        with _option_premium_rail_pinned_off():
            runner = WheelRunner()
            conn = runner.connector
        _assert_rail_neutralized(conn)
    rail_label = "on" if rail_on else "pinned_off"
    bundle_id = window_id + ("_rail" if rail_on else "")

    universe = list(conn.get_universe())
    if _smoke_universe_size is not None:
        universe = universe[:_smoke_universe_size]
    universe_sha = sha256("\n".join(sorted(universe)).encode()).hexdigest()

    trackers = {
        lvl: WheelTracker(initial_capital=CONFIG["capital"], connector=conn)
        for lvl in friction_levels
    }

    trading_days = _exchange_trading_days(start, end)
    if not trading_days:
        raise RuntimeError(f"no exchange trading days found in [{start}, {end}]")
    end_d = date.fromisoformat(end)
    cutoff = end_d - timedelta(days=ENTRY_CUTOFF_CAL_DAYS)
    if window_id in ENTRY_CUTOFF_OVERRIDE:
        cutoff = min(cutoff, date.fromisoformat(ENTRY_CUTOFF_OVERRIDE[window_id]))
    rank_days = _weekly_rank_days(trading_days)

    print(
        f"[trader500k] {window_id} {start}->{end}  universe={len(universe)} "
        f"days={len(trading_days)} rank_weeks={sum(1 for d in rank_days if d <= cutoff)} "
        f"cutoff={cutoff} frictions={friction_levels}",
        flush=True,
    )

    rank_log_rows: dict[str, list[dict]] = {lvl: [] for lvl in friction_levels}
    trade_rows: dict[str, list[dict]] = {lvl: [] for lvl in friction_levels}
    calib_frames: list[pd.DataFrame] = []
    daily_nav: dict[str, list[dict]] = {lvl: [] for lvl in friction_levels}
    partial_mark_days: Counter = Counter()
    premium_source_tally: Counter = Counter()
    drop_tally: Counter = Counter()
    drop_raw: list[dict] = []
    rank_errors: list[dict] = []
    n_rank_days_run = 0

    for day_idx, today in enumerate(trading_days):
        expiration_default = _next_business_day(today + timedelta(days=CONFIG["dte_target"]))

        step_prices: dict[str, dict[str, float]] = {}
        for lvl, tracker in trackers.items():
            step_prices[lvl] = _tracker_step(
                tracker=tracker,
                runner=runner,
                conn=conn,
                today=today,
                friction_level=lvl,
                contracts=CONFIG["contracts"],
                dte_target=CONFIG["dte_target"],
                delta_target=CONFIG["delta_target"],
                expiration_default=expiration_default,
                PositionState=PositionState,
            )

        frame = None
        if today in rank_days and today <= cutoff:
            n_rank_days_run += 1
            try:
                frame = runner.rank_candidates_by_ev(
                    tickers=universe,
                    dte_target=CONFIG["dte_target"],
                    delta_target=CONFIG["delta_target"],
                    contracts=CONFIG["contracts"],
                    top_n=len(universe),
                    min_ev_dollars=CONFIG["min_ev_dollars_capture"],
                    as_of=today.isoformat(),
                    include_diagnostic_fields=True,
                    use_credit_regime=CONFIG["use_credit_regime"],
                )
            except Exception as exc:  # noqa: BLE001 — record, keep replaying
                rank_errors.append({"date": today.isoformat(), "error": repr(exc)})

            if frame is not None:
                for drop in frame.attrs.get("drops", []):
                    drop_tally[str(drop.get("gate", "unknown"))] += 1
                    drop_raw.append({"date": today.isoformat(), **drop})
                if len(frame) == 0:
                    frame = None

            if frame is not None:
                snap = frame.copy()
                snap["as_of"] = today.isoformat()
                snap["expiration_date"] = expiration_default.isoformat()
                calib_frames.append(snap)
                if "premium_source" in frame.columns:
                    premium_source_tally.update(frame["premium_source"].astype(str))

        # #517 FIX-REQUEST: execution consumes head(execution_slice) ONLY;
        # the full frame above is capture-only (rows 21+ can never execute).
        exec_frame = frame.head(CONFIG["execution_slice"]) if frame is not None else None

        for lvl, tracker in trackers.items():
            opened_spots: dict[str, float] = {}
            if frame is not None:
                for _, row in frame.iterrows():
                    premium_raw = float(row.get("premium", 0.0))
                    rank_log_rows[lvl].append(
                        {
                            "date": today.isoformat(),
                            "ticker": str(row.get("ticker", "")),
                            "ev_dollars": float(row.get("ev_dollars", 0.0)),
                            "premium": friction_adjusted_premium(premium_raw, lvl),
                            "premium_raw": premium_raw,
                            "strike": float(row.get("strike", 0.0)),
                            "iv": float(row.get("iv", 0.0)),
                            "prob_profit": float(row.get("prob_profit", float("nan"))),
                            "expiration_date": expiration_default.isoformat(),
                            "friction_level": lvl,
                        }
                    )
            if exec_frame is not None:
                held_before = {
                    t
                    for t, p in tracker.positions.items()
                    if p.state == PositionState.SHORT_PUT
                }
                _tracker_try_opens(
                    tracker=tracker,
                    frame=exec_frame,
                    today=today,
                    friction_level=lvl,
                    max_new_per_day=CONFIG["max_new_per_rank_day"],
                    contracts=CONFIG["contracts"],
                    expiration_default=expiration_default,
                    PositionState=PositionState,
                    enforce_single_name_cap=CONFIG["enforce_single_name_cap"],
                )
                by_ticker = {str(r["ticker"]): r for _, r in exec_frame.iterrows()}
                for t, p in tracker.positions.items():
                    if (
                        p.state == PositionState.SHORT_PUT
                        and t not in held_before
                        and p.put_entry_date == today
                    ):
                        src = by_ticker.get(t, {})
                        spot_val = float(src.get("spot", float("nan")))
                        if np.isfinite(spot_val) and spot_val > 0:
                            opened_spots[t] = spot_val
                        trade_rows[lvl].append(
                            {
                                "entry_date": today.isoformat(),
                                "ticker": t,
                                "strike": float(p.put_strike or 0.0),
                                "dte": (expiration_default - today).days,
                                "expiration_date": expiration_default.isoformat(),
                                "premium": float(p.put_premium or 0.0),
                                "premium_raw": float(src.get("premium", float("nan"))),
                                "ev_dollars": float(src.get("ev_dollars", float("nan"))),
                                "prob_profit": float(src.get("prob_profit", float("nan"))),
                                "iv": float(src.get("iv", float("nan"))),
                                "premium_source": str(src.get("premium_source", "")),
                                "friction_level": lvl,
                            }
                        )

            # End-of-day re-mark: POST-settlement / post-open NAV. The
            # _tracker_step mark precedes settlement, so the expiry-day
            # intrinsic of ITM puts is missing from that entry; this one
            # is what the committed curve consumes.
            remark_prices = dict(step_prices[lvl])
            remark_prices.update(opened_spots)
            if remark_prices:
                tracker.mark_to_market(today, remark_prices)

            eq = tracker.equity_curve
            open_tickers_now = [
                t for t, p in tracker.positions.items() if p.state != PositionState.NO_POSITION
            ]
            n_open = len(open_tickers_now)
            mark_ok = (
                bool(eq)
                and eq[-1]["date"] == today
                and all(t in remark_prices for t in open_tickers_now)
                and np.isfinite(float(eq[-1]["portfolio_value"]))
            )
            if n_open == 0:
                nav = float(tracker.cash)
            elif mark_ok:
                nav = float(eq[-1]["portfolio_value"])
            elif daily_nav[lvl]:
                nav = float(daily_nav[lvl][-1]["nav"])  # carry-forward guard
                partial_mark_days[lvl] += 1
            else:
                nav = float(tracker.cash)
            daily_nav[lvl].append(
                {
                    "date": today.isoformat(),
                    "friction_level": lvl,
                    "nav": nav,
                    "cash": float(tracker.cash),
                    "n_open_positions": n_open,
                }
            )

        if day_idx % 20 == 0 or day_idx == len(trading_days) - 1:
            elapsed = time.time() - t0
            rate = (day_idx + 1) / elapsed if elapsed > 0 else 0.0
            eta_min = (len(trading_days) - day_idx - 1) / rate / 60 if rate > 0 else float("inf")
            navs = " ".join(
                f"{lvl}={daily_nav[lvl][-1]['nav']:,.0f}" for lvl in friction_levels
            )
            print(
                f"[trader500k] {window_id} day {day_idx + 1}/{len(trading_days)} {today} "
                f"NAV {navs}  elapsed {elapsed / 60:.1f}min ETA {eta_min:.0f}min",
                flush=True,
            )

    # ------------------------------------------------------------------
    # Realized outcomes (hold-to-expiry attribution, harness convention)
    # ------------------------------------------------------------------
    spot_cache: dict[tuple[str, str], float | None] = {}

    def spot_at(ticker: str, expiry_iso: str) -> float | None:
        key = (ticker, expiry_iso)
        if key not in spot_cache:
            spot_cache[key] = _spot_on_or_after(conn, ticker, date.fromisoformat(expiry_iso))
        return spot_cache[key]

    calib = pd.concat(calib_frames, ignore_index=True) if calib_frames else pd.DataFrame()
    if not calib.empty:
        spots, realized, otm, exact = [], [], [], []
        for r in calib.itertuples(index=False):
            s = spot_at(str(r.ticker), str(r.expiration_date))
            prem = float(getattr(r, "premium", float("nan")))
            strike = float(getattr(r, "strike", float("nan")))
            if s is None or not np.isfinite(prem) or not np.isfinite(strike):
                spots.append(float("nan"))
                realized.append(float("nan"))
                otm.append(float("nan"))
                exact.append(float("nan"))
            else:
                spots.append(s)
                realized.append(_forward_replay_realized_pnl(strike, prem, s))
                otm.append(float(s >= strike))
                exact.append(float(s > strike - prem))
        calib["spot_at_expiry"] = spots
        calib["realized_pnl_synth"] = realized
        calib["otm_expire"] = otm
        calib["engine_exact"] = exact

    trades_frames = []
    for lvl in friction_levels:
        tdf = pd.DataFrame(trade_rows[lvl])
        if tdf.empty:
            trades_frames.append(tdf)
            continue
        spots, outcomes, realized = [], [], []
        for r in tdf.itertuples(index=False):
            s = spot_at(str(r.ticker), str(r.expiration_date))
            if s is None:
                spots.append(float("nan"))
                outcomes.append("unresolved_no_spot")
                realized.append(float("nan"))
                continue
            assigned = s < r.strike
            pnl = _forward_replay_realized_pnl(r.strike, r.premium, s)
            pnl -= friction_open_cost(CONFIG["contracts"], lvl)
            if assigned:
                pnl -= friction_assignment_cost(r.strike, CONFIG["contracts"], lvl)
            spots.append(s)
            outcomes.append("assigned" if assigned else "expired_otm")
            realized.append(pnl)
        tdf["spot_at_expiry"] = spots
        tdf["outcome"] = outcomes
        tdf["realized_pnl"] = realized
        trades_frames.append(tdf)
    trades = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    benchmarks = {
        "spy_proxy_spx": spx_price_return(start, end),
        "ew_passive": ew_passive_return(start, end, universe),
    }

    per_friction: dict[str, dict] = {}
    for lvl, tracker in trackers.items():
        rank_log = pd.DataFrame(rank_log_rows[lvl])
        if not rank_log.empty and not calib.empty:
            # Realized P&L on the friction-adjusted premium, aligned to the
            # shared calibration spots (identical spot per ticker+expiry).
            realized_lvl = []
            for r in rank_log.itertuples(index=False):
                s = spot_at(str(r.ticker), str(r.expiration_date))
                realized_lvl.append(
                    float("nan")
                    if s is None
                    else _forward_replay_realized_pnl(r.strike, r.premium, s)
                )
            rank_log["realized_pnl"] = realized_lvl
        try:
            harness_metrics = _compute_metrics(rank_log, tracker)
        except Exception as exc:  # noqa: BLE001 — an all-empty window must still emit its bundle
            harness_metrics = {
                "aggregate": {"error": repr(exc), "row_count": int(len(rank_log))},
                "per_year": {},
                "per_quartile": {},
            }
        harness_metrics["aggregate"]["r10_refusals"] = int(
            getattr(tracker, "_sim_r10_refusals", 0)
        )

        nav_df = pd.DataFrame(daily_nav[lvl])
        curve = curve_metrics(nav_df["nav"], CONFIG["capital"])

        tdf = trades[trades["friction_level"] == lvl] if not trades.empty else pd.DataFrame()
        resolved = (
            tdf[tdf["outcome"].isin(["assigned", "expired_otm"])] if not tdf.empty else tdf
        )
        n_trades = int(len(tdf))
        n_assigned = int((tdf["outcome"] == "assigned").sum()) if not tdf.empty else 0
        win_rate = (
            float((resolved["realized_pnl"] > 0).mean()) if len(resolved) else float("nan")
        )
        trade_stats = {
            "n_trades": n_trades,
            "n_assignments": n_assigned,
            "n_unresolved": int(n_trades - len(resolved)),
            "win_rate": win_rate,
            "mean_realized_per_trade": (
                float(resolved["realized_pnl"].mean()) if len(resolved) else float("nan")
            ),
            "median_realized_per_trade": (
                float(resolved["realized_pnl"].median()) if len(resolved) else float("nan")
            ),
        }
        if len(resolved) >= 3:
            rho_ev, p_ev = _spearman(resolved["ev_dollars"], resolved["realized_pnl"])
            rho_pp, p_pp = _spearman(resolved["prob_profit"], resolved["realized_pnl"])
            trade_stats["spearman_ev_realized_executed"] = {"rho": rho_ev, "p": p_ev}
            trade_stats["spearman_prob_realized_executed"] = {"rho": rho_pp, "p": p_pp}

        if not tdf.empty and "premium_source" in tdf.columns:
            n_mid = int((tdf["premium_source"] == "market_mid").sum())
            trade_stats["market_mid_fraction_executed"] = n_mid / max(len(tdf), 1)
            trade_stats["n_market_mid_executed"] = n_mid

        deployed_days = int((nav_df["n_open_positions"] > 0).sum())
        per_friction[lvl] = {
            "curve": curve,
            "trades": trade_stats,
            "time_deployed_pct": deployed_days / max(len(nav_df), 1) * 100.0,
            "mean_open_positions": float(nav_df["n_open_positions"].mean()),
            "n_partial_mark_days": int(partial_mark_days.get(lvl, 0)),
            "harness_aggregate": harness_metrics["aggregate"],
            "harness_per_year": harness_metrics["per_year"],
            "harness_per_quartile": harness_metrics["per_quartile"],
        }

    calib_stats: dict = {"n_rows": int(len(calib))}
    if not calib.empty and "premium_source" in calib.columns:
        n_mid_pool = int((calib["premium_source"] == "market_mid").sum())
        calib_stats["market_mid_fraction_pool"] = n_mid_pool / max(len(calib), 1)
        calib_stats["n_market_mid_pool"] = n_mid_pool
    if not calib.empty:
        rho_ev, p_ev = _spearman(calib["ev_dollars"], calib["realized_pnl_synth"])
        rho_pp, p_pp = _spearman(calib["prob_profit"], calib["realized_pnl_synth"])
        calib_stats.update(
            {
                "n_positive_ev": int((calib["ev_dollars"] > 0).sum()),
                "spearman_ev_realized_pool": {"rho": rho_ev, "p": p_ev},
                "spearman_prob_realized_pool": {"rho": rho_pp, "p": p_pp},
            }
        )
        try:
            from engine.paper_book import reliability

            ok = calib["prob_profit"].notna() & calib["engine_exact"].notna()
            rows, brier, ece = reliability(
                calib.loc[ok, "prob_profit"].tolist(),
                calib.loc[ok, "engine_exact"].tolist(),
            )
            calib_stats["reliability_engine_exact"] = {
                "bins": rows,
                "brier": brier,
                "ece": ece,
            }
        except Exception as exc:  # noqa: BLE001 — diagnostics only
            calib_stats["reliability_error"] = repr(exc)

    fingerprint = {
        "window_id": window_id,
        "start": start,
        "end": end,
        "note": note,
        "entry_cutoff": cutoff.isoformat(),
        **{k: v for k, v in CONFIG.items()},
        "option_premium_rail": rail_label,
        "capture_top_n_resolved": len(universe),
        "friction_levels": list(friction_levels),
        "universe_size": len(universe),
        "universe_sha256": universe_sha,
        "universe": universe,
        "n_trading_days": len(trading_days),
        "n_rank_days_run": n_rank_days_run,
        "data_csv_sha256": ohlcv_sha256(),
        "vol_iv_sha256": vol_iv_sha256(),
        "treasury_sha256": treasury_sha256(),
        "connector_data_sha256": connector_data_sha256(),
        "generated_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    summary = {
        "window_id": window_id,
        "start": start,
        "end": end,
        "note": note,
        "entry_cutoff": cutoff.isoformat(),
        "config": {**CONFIG, "option_premium_rail": rail_label, "capture_top_n_resolved": len(universe)},
        "benchmarks": benchmarks,
        "per_friction": per_friction,
        "calibration": calib_stats,
        "drop_gate_tallies": dict(drop_tally),
        "premium_source_counts": dict(premium_source_tally),
        "rank_errors": rank_errors,
        "caveats": [
            "Synthetic-BSM premiums (option-premium rail pinned OFF): absolute P&L "
            "optimistic; friction band + calibration are the trustworthy signals "
            "(SIM_200K_RELIABILITY convention).",
            "Survivorship: universe = current members of the committed 2026 monolith "
            "(conn.get_universe(), no PIT membership). Early windows trade only names "
            "known to survive; EW-passive benchmark shares this universe (internally "
            "consistent), the SPX proxy does not.",
            "Benchmarks are price returns ex-dividends (see labels); spx_tr_approx "
            "adds 1.5%/yr as a marked approximation.",
            "Hold-to-expiry is a modeling choice; no rolls or profit-targets.",
            "Committed daily NAV is the post-settlement/post-open end-of-day re-mark; "
            "covered calls opened after the entry cutoff (lifecycle wheeling) may be "
            "open at window end and are marked, not settled, in the final NAV.",
            "Trades-ledger realized_pnl is hold-to-expiry attribution of the put leg; "
            "post-assignment stock/CC economics live in the equity curve.",
        ],
        "fingerprint": fingerprint,
    }

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    out_dir = (out_root or ARTIFACT_ROOT) / bundle_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
        f.write("\n")
    eq_all = pd.concat(
        [pd.DataFrame(daily_nav[lvl]) for lvl in friction_levels], ignore_index=True
    )
    eq_all.to_csv(out_dir / "equity_curve.csv.gz", index=False, compression="gzip")
    trades.to_csv(out_dir / "trades.csv.gz", index=False, compression="gzip")
    calib.to_csv(out_dir / "calibration_rows.csv.gz", index=False, compression="gzip")

    work_dir = (
        work_root or Path(os.environ.get("TMPDIR", "/tmp")) / "trader500k_work"
    ) / bundle_id
    work_dir.mkdir(parents=True, exist_ok=True)
    for lvl, tracker in trackers.items():
        pd.DataFrame(rank_log_rows[lvl]).to_csv(work_dir / f"rank_log_{lvl}.csv", index=False)
        try:
            with open(work_dir / f"tracker_state_{lvl}.json", "w", encoding="utf-8") as f:
                json.dump(tracker.to_dict(), f, indent=2, default=str)
        except Exception:
            pass
    pd.DataFrame(drop_raw).to_csv(work_dir / "drops_raw.csv.gz", index=False, compression="gzip")

    print(
        f"[trader500k] {window_id} DONE in {(time.time() - t0) / 60:.1f}min — bundle at {out_dir}",
        flush=True,
    )
    for lvl in friction_levels:
        c = per_friction[lvl]["curve"]
        t = per_friction[lvl]["trades"]
        print(
            f"[trader500k] {window_id} {lvl:8s} NAV {c['final_nav']:>12,.0f} "
            f"({c['total_return_pct']:+.2f}%) maxDD {c['max_drawdown_pct']:.2f}% "
            f"trades {t['n_trades']} assigned {t['n_assignments']} win {t['win_rate']:.2%}"
            if t["n_trades"]
            else f"[trader500k] {window_id} {lvl:8s} NAV {c['final_nav']:>12,.0f} — no trades",
            flush=True,
        )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def one(
    window: str = typer.Argument(..., help="Window id, e.g. W05"),
    out_root: Path = typer.Option(None, help="Override bundle root (default: committed artifacts dir)"),
    rail_on: bool = typer.Option(False, "--rail-on", help="Phase C: option-premium rail LIVE; bundle lands in Wnn_rail/"),
) -> None:
    """Run a single campaign window with all three friction levels."""
    run_window(window.upper(), out_root=out_root, rail_on=rail_on)


@app.command()
def smoke(
    n_tickers: int = typer.Option(12, help="Universe subset size"),
    start: str = typer.Option("2022-01-03", help="Smoke window start"),
    end: str = typer.Option("2022-03-31", help="Smoke window end"),
    rail_on: bool = typer.Option(False, "--rail-on", help="Smoke with the premium rail LIVE"),
) -> None:
    """Plumbing smoke: tiny universe, short window, output to a temp dir.

    Never writes into the committed artifacts dir — smoke bundles land
    under $TMPDIR/trader500k_smoke/ for inspection only.
    """
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "trader500k_smoke"
    run_window(
        "SMOKE",
        out_root=tmp / "bundle",
        work_root=tmp / "work",
        rail_on=rail_on,
        _smoke_universe_size=n_tickers,
        _smoke_window=(start, end),
    )


@app.command()
def info() -> None:
    """Print the locked window grid and config."""
    print(json.dumps({"windows": WINDOWS, "config": CONFIG}, indent=2))


if __name__ == "__main__":
    app()
