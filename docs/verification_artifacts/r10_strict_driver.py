"""HT-D — R10 strict-mode at $1M/100t scale driver (2026-05-30).

Every prior canonical backtest (S27 / S32 / S34 / S35 / S38 / S40 / S43 / S44)
ran with ``require_ev_authority=False`` on the ``WheelTracker``. That meant
the **R10 single-name notional cap** (`engine.portfolio_risk_gates.check_single_name_cap`,
the doc-designated *load-bearing magnitude guard* for the engine's
``prob_profit`` top-bin over-confidence per
``docs/PROB_PROFIT_CALIBRATION_2026-05-28.md`` + ``docs/F4_TAIL_RISK_DIAGNOSTIC.md`` §10
+ ``docs/PRODUCTION_READINESS.md`` §3 B1) has **never** been exercised in a backtest.
S44 explicitly flagged this open question in its AI handoff (§7,
"For future research: test R10 in strict mode on the S38 setup").

This driver runs the canonical S38 setup ($1M / 100 first-alphanumeric SP500
tickers / 2020-01-02 → 2024-12-31 / 35-DTE 25-Δ short puts / wheel into CC
on assignment) under BOTH:

  * ``tracker_loose``  — ``require_ev_authority=False``, identical to S38/S44.
  * ``tracker_strict`` — ``require_ev_authority=True``, ``min_nav_for_trading=0``,
    with token issuance via ``WheelTracker.consume_ranker_row`` so the D17
    portfolio-risk hard-blocks (R9 sector cap, R10 single-name cap, R-delta
    portfolio-delta cap, R-kelly Kelly cap) fire at every open attempt.

Both trackers share **one daily SP rank call** (the ``run_backtest_multi_friction``
pattern from ``backtests.regression._common``) and both run at ``full`` friction
— the canonical S38/S44 headline level. Single friction is enough because
R10 cares about per-name notional aggregation, not entry-cost shape (an
independent dimension already characterised by S38/S44 across all 3 levels).

The driver also captures, for each strict-tracker open attempt:
  * the EV row that was ranked (date, ticker, strike, premium, ev_dollars),
  * the consume outcome (success / refuse + reason from ``tracker._ev_authority_log``),
  * the NAV-as-of-attempt (`_compute_live_nav` source: live_mark_to_market
    when connector available),
  * the per-name short-option notional immediately PRIOR to the attempt
    (so a `single_name_breach` reject can be cross-referenced against the
    existing exposure that triggered it).

Output (under ``--out-dir``):

  * ``rank_log_loose.csv``         — every ranked row, loose tracker, with
                                     realised P&L forward-replayed at expiry.
  * ``rank_log_strict.csv``        — same, strict tracker.
  * ``open_attempts_strict.csv``   — every open attempt for the strict tracker
                                     (success + refusals with reasons).
  * ``open_attempts_loose.csv``    — same, loose tracker.
  * ``equity_curve_loose.csv``     — daily NAV.
  * ``equity_curve_strict.csv``    — same.
  * ``tracker_loose_state.json``   — full tracker state (positions, closed_positions,
                                     ev_authority_log).
  * ``tracker_strict_state.json``  — same.
  * ``summary.json``               — headline metrics + R10/R9 bind rates.

Non-§2; READ-ONLY on engine/ (no engine edits — this is observation only,
per HT-D card rules). The driver imports the canonical helpers from
``backtests/regression/_common.py`` (friction overlay, spot lookup,
forward-replay, business-day arithmetic) and the canonical universe from
``backtests/regression/universes.py`` to keep the comparison apples-to-
apples with S38/S44.

Usage:

    # Pilot — short window to validate correctness + measure R10 bind rate
    python docs/verification_artifacts/r10_strict_driver.py \\
        --start 2020-01-02 --end 2020-03-31 \\
        --out-dir <some_temp_dir>/r10_pilot

    # Full 5y heavy run (the canonical HT-D measurement)
    python docs/verification_artifacts/r10_strict_driver.py \\
        --start 2020-01-02 --end 2024-12-31 \\
        --out-dir <some_temp_dir>/r10_full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtests.regression._common import (  # noqa: E402
    _forward_replay_realized_pnl,
    _next_business_day,
    _spot_on_or_after,
    assert_data_window_available,
    friction_adjusted_premium,
    friction_assignment_cost,
    friction_open_cost,
    ohlcv_sha256,
)
from backtests.regression.universes import UNIVERSE_100  # noqa: E402
from engine.portfolio_risk_gates import take_snapshot  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402
from engine.wheel_tracker import (  # noqa: E402
    EVAuthorityRefused,
    PositionState,
    WheelTracker,
)

# -----------------------------------------------------------------------------
# Per-attempt logging helpers
# -----------------------------------------------------------------------------


def _last_log_entry_for(tracker: WheelTracker, ticker: str) -> dict:
    """Return the most recent ``tracker._ev_authority_log`` entry whose
    ``ticker`` matches, or an empty dict.

    The log grows with every issue / consume / reject; D17 ``reject``
    entries are appended by ``_evaluate_d17_hard_blocks`` and carry the
    ``reason`` (e.g. ``"single_name_breach"``, ``"sector_cap_breach"``,
    ``"portfolio_delta_breach"``) plus the gate's details bag (e.g.
    ``post_open_name_pct``, ``current_name_notional``, ``name_limit_pct``).
    """
    for entry in reversed(tracker._ev_authority_log):
        if entry.get("ticker") == ticker:
            return dict(entry)
    return {}


def _per_name_short_notional(tracker: WheelTracker, ticker: str) -> float:
    """Sum of strike × 100 × contracts across all HELD short option legs
    for ``ticker`` (puts AND covered-call legs)."""
    snap = take_snapshot(tracker.positions)
    total = 0.0
    for p in snap.option_positions:
        if str(p.get("symbol", "")).upper() != ticker.upper():
            continue
        if not bool(p.get("is_short", False)):
            continue
        try:
            total += float(p.get("strike", 0.0)) * 100.0 * int(p.get("contracts", 0))
        except (TypeError, ValueError):
            continue
    return total


def _per_sector_short_notional(tracker: WheelTracker) -> dict[str, float]:
    """Sum of short-option notional per GICS sector. Uses the engine's
    ``DEFAULT_SECTOR_MAP`` so the answer matches what
    ``check_sector_cap`` would aggregate against.

    Pure observability — not part of the gate path.
    """
    from engine.risk_manager import SectorExposureManager

    mgr = SectorExposureManager()
    snap = take_snapshot(tracker.positions)
    out: dict[str, float] = {}
    for p in snap.option_positions:
        if not bool(p.get("is_short", False)):
            continue
        try:
            strike = float(p.get("strike", 0.0))
            contracts = int(p.get("contracts", 0))
            sym = str(p.get("symbol", "")).upper()
        except (TypeError, ValueError):
            continue
        sec = mgr.get_sector(sym)
        out[sec] = out.get(sec, 0.0) + strike * 100.0 * contracts
    return out


# -----------------------------------------------------------------------------
# Per-tracker daily steps
# -----------------------------------------------------------------------------


def _tracker_settle_and_wheel(
    *,
    tracker: WheelTracker,
    runner: WheelRunner,
    conn: Any,
    today: date,
    friction_level: str,
    contracts: int,
    dte_target: int,
    delta_target: float,
    expiration_default: date,
    strict: bool,
    cc_attempts_log: list[dict],
) -> dict[str, float]:
    """MTM, settle put/call expirations, wheel into covered calls.

    Mirrors `_common._tracker_step` exactly, except the CC open call uses
    strict-mode token issuance when ``strict=True``.
    """
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
            cc_ev = float(cc_row.get("ev_dollars", 0.0))
            if cc_ev <= 0:
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

            pre_attempt_log_len = len(tracker._ev_authority_log)
            if strict:
                # Strict: issue a CC token, then open_covered_call with token
                # + current_ev_dollars. D17 hard-blocks (sector + R10 +
                # delta — Kelly skipped on CC leg) fire inside open.
                cc_row_dict = {
                    "ticker": t,
                    "strike": cc_strike,
                    "premium": cc_premium,
                    "dte": (cc_expiry - today).days,
                    "ev_dollars": cc_ev,
                    "prob_profit": float(cc_row.get("prob_profit", 0.0) or 0.0),
                    "distribution_source": str(cc_row.get("distribution_source", "")),
                }
                try:
                    token = tracker.issue_ev_authority_token(cc_row_dict)
                except EVAuthorityRefused:
                    cc_attempts_log.append(
                        {
                            "date": today.isoformat(),
                            "ticker": t,
                            "leg": "call",
                            "ev_dollars": cc_ev,
                            "strike": cc_strike,
                            "outcome": "refuse_issue",
                            "reason": "non_positive_ev",
                            "name_notional_pre": _per_name_short_notional(tracker, t),
                        }
                    )
                    continue
                opened_cc = tracker.open_covered_call(
                    ticker=t,
                    strike=cc_strike,
                    premium=cc_premium,
                    entry_date=today,
                    expiration_date=cc_expiry,
                    iv=float(cc_row.get("iv", 0.0)),
                    ev_authority_token=token,
                    current_ev_dollars=cc_ev,
                )
            else:
                opened_cc = tracker.open_covered_call(
                    ticker=t,
                    strike=cc_strike,
                    premium=cc_premium,
                    entry_date=today,
                    expiration_date=cc_expiry,
                    iv=float(cc_row.get("iv", 0.0)),
                )

            # Capture attempt — outcome + reason if refused.
            if opened_cc:
                if friction_level == "full":
                    tracker.cash -= friction_open_cost(contracts, friction_level)
                cc_attempts_log.append(
                    {
                        "date": today.isoformat(),
                        "ticker": t,
                        "leg": "call",
                        "ev_dollars": cc_ev,
                        "strike": cc_strike,
                        "outcome": "opened",
                        "reason": None,
                        "name_notional_pre": _per_name_short_notional(tracker, t)
                        - cc_strike * 100.0 * contracts,
                    }
                )
                break  # one CC per ticker per day
            else:
                # Refused — find the last reject entry in the log to surface
                # the reason. May be empty in loose mode (open_covered_call
                # logs nothing in non-strict mode beyond the existing failure
                # paths — duplicate ticker, wrong state — neither of which
                # apply here since we just iterated only STOCK_OWNED).
                reject_entry: dict = {}
                if strict and len(tracker._ev_authority_log) > pre_attempt_log_len:
                    for entry in reversed(tracker._ev_authority_log[pre_attempt_log_len:]):
                        if entry.get("action") == "reject":
                            reject_entry = entry
                            break
                cc_attempts_log.append(
                    {
                        "date": today.isoformat(),
                        "ticker": t,
                        "leg": "call",
                        "ev_dollars": cc_ev,
                        "strike": cc_strike,
                        "outcome": "refused",
                        "reason": reject_entry.get("reason", "unknown"),
                        "name_notional_pre": _per_name_short_notional(tracker, t),
                        "post_open_name_pct": reject_entry.get("post_open_name_pct"),
                        "post_open_sector_pct": reject_entry.get("post_open_sector_pct"),
                        "nav": reject_entry.get("nav"),
                    }
                )
                # Don't break — try the next CC row in case the first was
                # gated. Matches the canonical harness's break-on-success
                # but allows the strict tracker to surface multiple refusals.

    return prices_today


def _tracker_try_opens(
    *,
    tracker: WheelTracker,
    runner: WheelRunner,
    frame: pd.DataFrame,
    today: date,
    friction_level: str,
    max_new_per_day: int,
    contracts: int,
    expiration_default: date,
    strict: bool,
    put_attempts_log: list[dict],
) -> None:
    """Mirrors _common._tracker_try_opens but with strict-mode token plumbing
    and per-attempt capture."""
    opens_today = 0
    for _, row in frame.iterrows():
        if opens_today >= max_new_per_day:
            break
        ev_dollars = float(row.get("ev_dollars", 0.0))
        if ev_dollars <= 0:
            continue
        t = str(row.get("ticker", ""))
        if t in tracker.positions and tracker.positions[t].state != PositionState.NO_POSITION:
            continue
        strike = float(row.get("strike", 0.0))
        premium = friction_adjusted_premium(float(row.get("premium", 0.0)), friction_level)
        if premium <= 0 or strike <= 0:
            continue
        # Canonical harness BP gate — keep for parity with S38/S44.
        if tracker.available_buying_power() < strike * 100 * contracts:
            put_attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "leg": "put",
                    "ev_dollars": ev_dollars,
                    "strike": strike,
                    "outcome": "refused",
                    "reason": "insufficient_bp",
                    "name_notional_pre": _per_name_short_notional(tracker, t),
                }
            )
            continue

        pre_attempt_log_len = len(tracker._ev_authority_log)
        if strict:
            # Strict path: build the canonical ranker row dict and use the
            # production helper ``consume_ranker_row`` so D16 (token issue
            # → consume with current_ev_dollars re-check) + D17 (sector +
            # R10 single-name + delta + Kelly) all fire.
            row_dict = {
                "ticker": t,
                "strike": strike,
                "premium": premium,
                "dte": (expiration_default - today).days,
                "ev_dollars": ev_dollars,
                "prob_profit": float(row.get("prob_profit", 0.0) or 0.0),
                "distribution_source": str(row.get("distribution_source", "")),
                "iv": float(row.get("iv", 0.0)),
            }
            try:
                opened = tracker.consume_ranker_row(
                    row_dict, entry_date=today, expiration_date=expiration_default
                )
            except EVAuthorityRefused:
                put_attempts_log.append(
                    {
                        "date": today.isoformat(),
                        "ticker": t,
                        "leg": "put",
                        "ev_dollars": ev_dollars,
                        "strike": strike,
                        "outcome": "refuse_issue",
                        "reason": "non_positive_ev",
                        "name_notional_pre": _per_name_short_notional(tracker, t),
                    }
                )
                continue
        else:
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
            put_attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "leg": "put",
                    "ev_dollars": ev_dollars,
                    "strike": strike,
                    "outcome": "opened",
                    "reason": None,
                    "name_notional_pre": _per_name_short_notional(tracker, t)
                    - strike * 100.0 * contracts,
                }
            )
            opens_today += 1
        else:
            reject_entry: dict = {}
            if strict and len(tracker._ev_authority_log) > pre_attempt_log_len:
                for entry in reversed(tracker._ev_authority_log[pre_attempt_log_len:]):
                    if entry.get("action") == "reject":
                        reject_entry = entry
                        break
            put_attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "leg": "put",
                    "ev_dollars": ev_dollars,
                    "strike": strike,
                    "outcome": "refused",
                    "reason": reject_entry.get("reason", "unknown"),
                    "name_notional_pre": _per_name_short_notional(tracker, t),
                    "post_open_name_pct": reject_entry.get("post_open_name_pct"),
                    "post_open_sector_pct": reject_entry.get("post_open_sector_pct"),
                    "nav": reject_entry.get("nav"),
                }
            )


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------


def run_strict_vs_loose(
    *,
    capital: float,
    tickers: Sequence[str],
    start: str,
    end: str,
    friction_level: str = "full",
    top_n: int = 10,
    max_new_per_day: int = 3,
    dte_target: int = 35,
    delta_target: float = 0.25,
    contracts: int = 1,
    out_dir: Path,
    progress_every_days: int = 25,
) -> dict[str, Any]:
    """The HT-D driver. Runs loose + strict trackers in parallel sharing
    one daily SP rank call. Persists per-tracker artifacts to ``out_dir``
    and returns a summary metrics dict.
    """
    assert_data_window_available(start, end)
    np.random.default_rng(42)  # placeholder; deterministic engine, no random opens

    runner = WheelRunner()
    conn = runner.connector
    print(f"[r10_strict_driver] connector: {type(conn).__name__}", flush=True)
    print(f"[r10_strict_driver] worktree:  {WORKTREE}", flush=True)
    print(
        f"[r10_strict_driver] universe size={len(tickers)} window={start}->{end} "
        f"capital=${capital:,.0f} friction={friction_level}",
        flush=True,
    )

    tracker_loose = WheelTracker(
        initial_capital=capital, connector=conn, require_ev_authority=False
    )
    tracker_strict = WheelTracker(
        initial_capital=capital,
        connector=conn,
        require_ev_authority=True,
        min_nav_for_trading=0.0,
    )

    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    tickers = list(tickers)

    rank_log_rows_loose: list[dict] = []
    rank_log_rows_strict: list[dict] = []
    put_attempts_loose: list[dict] = []
    put_attempts_strict: list[dict] = []
    cc_attempts_loose: list[dict] = []
    cc_attempts_strict: list[dict] = []
    daily_state: list[dict] = []  # per-day: NAV both trackers, n_positions, max-name-pct

    _total_days = len(trading_days)
    _progress_every = max(1, _total_days // progress_every_days)
    _t_start = time.time()
    _checkpoint_every = max(progress_every_days * 4, 100)  # write CSVs every ~100 days

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[r10_strict_driver] out_dir: {out_dir}", flush=True)

    for _day_idx, today in enumerate(trading_days):
        if _day_idx > 0 and _day_idx % _progress_every == 0:
            _elapsed = time.time() - _t_start
            _rate = _day_idx / _elapsed if _elapsed > 0 else 0.0
            _eta = (_total_days - _day_idx) / _rate if _rate > 0 else 0.0
            try:
                _nav_loose = (
                    float(tracker_loose.equity_curve[-1].get("portfolio_value", tracker_loose.cash))
                    if tracker_loose.equity_curve
                    else float(tracker_loose.cash)
                )
                _nav_strict = (
                    float(
                        tracker_strict.equity_curve[-1].get("portfolio_value", tracker_strict.cash)
                    )
                    if tracker_strict.equity_curve
                    else float(tracker_strict.cash)
                )
            except Exception:
                _nav_loose = _nav_strict = float("nan")
            print(
                f"[r10_strict] day {_day_idx:4d}/{_total_days} "
                f"({100 * _day_idx / _total_days:5.1f}%) "
                f"elapsed {_elapsed / 60:6.1f}min  ETA {_eta / 60:6.1f}min  "
                f"({_rate:5.2f} day/s)  "
                f"NAV_loose=${_nav_loose:,.0f}  NAV_strict=${_nav_strict:,.0f}  "
                f"n_loose={len(tracker_loose.positions)} n_strict={len(tracker_strict.positions)}",
                flush=True,
            )

        # Periodic checkpoint write — survives compaction / crash without losing data.
        if _day_idx > 0 and _day_idx % _checkpoint_every == 0:
            try:
                pd.DataFrame(put_attempts_loose).to_csv(
                    out_dir / "open_attempts_loose.csv", index=False
                )
                pd.DataFrame(put_attempts_strict).to_csv(
                    out_dir / "open_attempts_strict.csv", index=False
                )
                pd.DataFrame(cc_attempts_loose).to_csv(
                    out_dir / "cc_attempts_loose.csv", index=False
                )
                pd.DataFrame(cc_attempts_strict).to_csv(
                    out_dir / "cc_attempts_strict.csv", index=False
                )
                pd.DataFrame(daily_state).to_csv(out_dir / "daily_state.csv", index=False)
            except Exception as e:
                print(f"[r10_strict] checkpoint write failed: {e}", flush=True)

        expiration_default = _next_business_day(today + timedelta(days=dte_target))

        # Per-tracker MTM + settle + wheel-into-CC
        _tracker_settle_and_wheel(
            tracker=tracker_loose,
            runner=runner,
            conn=conn,
            today=today,
            friction_level=friction_level,
            contracts=contracts,
            dte_target=dte_target,
            delta_target=delta_target,
            expiration_default=expiration_default,
            strict=False,
            cc_attempts_log=cc_attempts_loose,
        )
        _tracker_settle_and_wheel(
            tracker=tracker_strict,
            runner=runner,
            conn=conn,
            today=today,
            friction_level=friction_level,
            contracts=contracts,
            dte_target=dte_target,
            delta_target=delta_target,
            expiration_default=expiration_default,
            strict=True,
            cc_attempts_log=cc_attempts_strict,
        )

        # ONE shared SP rank call per day — friction-level independent.
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
        except Exception as exc:
            print(f"[r10_strict] rank failed {today}: {exc}", flush=True)
            continue
        if frame is None or len(frame) == 0:
            continue

        # Persist rank log + try opens — per tracker
        for _, row in frame.iterrows():
            premium_raw = float(row.get("premium", 0.0))
            premium_adj = friction_adjusted_premium(premium_raw, friction_level)
            common_log_row = {
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
            rank_log_rows_loose.append(dict(common_log_row))
            rank_log_rows_strict.append(dict(common_log_row))

        _tracker_try_opens(
            tracker=tracker_loose,
            runner=runner,
            frame=frame,
            today=today,
            friction_level=friction_level,
            max_new_per_day=max_new_per_day,
            contracts=contracts,
            expiration_default=expiration_default,
            strict=False,
            put_attempts_log=put_attempts_loose,
        )
        _tracker_try_opens(
            tracker=tracker_strict,
            runner=runner,
            frame=frame,
            today=today,
            friction_level=friction_level,
            max_new_per_day=max_new_per_day,
            contracts=contracts,
            expiration_default=expiration_default,
            strict=True,
            put_attempts_log=put_attempts_strict,
        )

        # End-of-day per-tracker snapshot for the time-series view.
        for label, tr in (("loose", tracker_loose), ("strict", tracker_strict)):
            try:
                nav = (
                    float(tr.equity_curve[-1].get("portfolio_value", tr.cash))
                    if tr.equity_curve
                    else float(tr.cash)
                )
            except Exception:
                nav = float("nan")
            sector_nots = _per_sector_short_notional(tr)
            max_sector_pct = (max(sector_nots.values()) / nav) if (nav > 0 and sector_nots) else 0.0
            # Per-name maxima
            snap = take_snapshot(tr.positions)
            name_nots: dict[str, float] = {}
            for p in snap.option_positions:
                if not bool(p.get("is_short", False)):
                    continue
                try:
                    sym = str(p.get("symbol", ""))
                    name_nots[sym] = name_nots.get(sym, 0.0) + float(
                        p.get("strike", 0.0)
                    ) * 100.0 * int(p.get("contracts", 0))
                except (TypeError, ValueError):
                    continue
            max_name_pct = (max(name_nots.values()) / nav) if (nav > 0 and name_nots) else 0.0
            daily_state.append(
                {
                    "date": today.isoformat(),
                    "tracker": label,
                    "nav": nav,
                    "cash": float(tr.cash),
                    "n_positions": len(tr.positions),
                    "max_sector_pct": max_sector_pct,
                    "max_name_pct": max_name_pct,
                    "num_names_held": len(name_nots),
                }
            )

    # Forward-replay realized P&L on every ranked row
    spot_cache: dict[tuple[str, str], float | None] = {}

    def _attach_realized(rank_log_rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rank_log_rows)
        if df.empty:
            return df
        realized = []
        spots_at_expiry = []
        for r in df.itertuples(index=False):
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
        df["spot_at_expiry"] = spots_at_expiry
        df["realized_pnl"] = realized
        return df

    rank_log_loose = _attach_realized(rank_log_rows_loose)
    rank_log_strict = _attach_realized(rank_log_rows_strict)

    # Final writes
    rank_log_loose.to_csv(out_dir / "rank_log_loose.csv", index=False)
    rank_log_strict.to_csv(out_dir / "rank_log_strict.csv", index=False)
    pd.DataFrame(put_attempts_loose).to_csv(out_dir / "open_attempts_loose.csv", index=False)
    pd.DataFrame(put_attempts_strict).to_csv(out_dir / "open_attempts_strict.csv", index=False)
    pd.DataFrame(cc_attempts_loose).to_csv(out_dir / "cc_attempts_loose.csv", index=False)
    pd.DataFrame(cc_attempts_strict).to_csv(out_dir / "cc_attempts_strict.csv", index=False)
    pd.DataFrame(daily_state).to_csv(out_dir / "daily_state.csv", index=False)

    pd.DataFrame(tracker_loose.equity_curve).to_csv(out_dir / "equity_curve_loose.csv", index=False)
    pd.DataFrame(tracker_strict.equity_curve).to_csv(
        out_dir / "equity_curve_strict.csv", index=False
    )

    with open(out_dir / "tracker_loose_state.json", "w", encoding="utf-8") as f:
        json.dump(tracker_loose.to_dict(), f, indent=2, default=str)
    with open(out_dir / "tracker_strict_state.json", "w", encoding="utf-8") as f:
        json.dump(tracker_strict.to_dict(), f, indent=2, default=str)

    # Headline summary
    summary = _build_summary(
        tracker_loose=tracker_loose,
        tracker_strict=tracker_strict,
        rank_log_loose=rank_log_loose,
        rank_log_strict=rank_log_strict,
        put_attempts_loose=put_attempts_loose,
        put_attempts_strict=put_attempts_strict,
        cc_attempts_loose=cc_attempts_loose,
        cc_attempts_strict=cc_attempts_strict,
        capital=capital,
        start=start,
        end=end,
        friction_level=friction_level,
        universe_size=len(tickers),
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("[r10_strict_driver] DONE", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  loose NAV:  ${summary['loose']['final_nav']:,.2f}", flush=True)
    print(f"  strict NAV: ${summary['strict']['final_nav']:,.2f}", flush=True)
    print(
        f"  R10 bind count (strict puts): {summary['strict']['put_refuse_by_reason'].get('single_name_breach', 0)}",
        flush=True,
    )
    print(
        f"  R9  bind count (strict puts): {summary['strict']['put_refuse_by_reason'].get('sector_cap_breach', 0)}",
        flush=True,
    )
    return summary


def _build_summary(
    *,
    tracker_loose: WheelTracker,
    tracker_strict: WheelTracker,
    rank_log_loose: pd.DataFrame,
    rank_log_strict: pd.DataFrame,
    put_attempts_loose: list[dict],
    put_attempts_strict: list[dict],
    cc_attempts_loose: list[dict],
    cc_attempts_strict: list[dict],
    capital: float,
    start: str,
    end: str,
    friction_level: str,
    universe_size: int,
) -> dict[str, Any]:
    """Headline metrics + R10/R9 bind rates for the findings doc."""

    def _tracker_metrics(
        tr: WheelTracker, rl: pd.DataFrame, attempts: list[dict], cc_attempts: list[dict]
    ) -> dict:
        final_nav = (
            float(tr.equity_curve[-1].get("portfolio_value", tr.cash))
            if tr.equity_curve
            else float(tr.cash)
        )
        replayed = rl.dropna(subset=["realized_pnl"]) if not rl.empty else rl
        executed_put_returns: list[float] = []
        for cp in tr.closed_positions:
            if (cp.get("put_premium") or 0) > 0 and cp.get("exit_reason") not in (
                None,
                "open",
            ):
                executed_put_returns.append(float(cp.get("realized_pnl", 0.0)))

        closed_with_put = sum(1 for r in tr.closed_positions if (r.get("put_premium") or 0) > 0)
        open_with_put = sum(1 for p in tr.positions.values() if (p.put_premium or 0) > 0)
        put_assigned_open = sum(
            1 for p in tr.positions.values() if p.state.value in ("stock_owned", "covered_call")
        )
        put_assigned_closed = sum(
            1 for r in tr.closed_positions if r.get("exit_reason") == "call_assigned"
        )

        put_outcomes = pd.DataFrame(attempts) if attempts else pd.DataFrame()
        put_refuse_by_reason = (
            put_outcomes[put_outcomes["outcome"].isin(["refused", "refuse_issue"])]
            .groupby("reason")
            .size()
            .to_dict()
            if not put_outcomes.empty
            else {}
        )
        # Make reasons JSON-clean (ints).
        put_refuse_by_reason = {str(k): int(v) for k, v in put_refuse_by_reason.items()}
        n_put_opened = (
            int((put_outcomes["outcome"] == "opened").sum()) if not put_outcomes.empty else 0
        )
        n_put_attempts = int(len(put_outcomes))

        cc_outcomes = pd.DataFrame(cc_attempts) if cc_attempts else pd.DataFrame()
        cc_refuse_by_reason = (
            cc_outcomes[cc_outcomes["outcome"].isin(["refused", "refuse_issue"])]
            .groupby("reason")
            .size()
            .to_dict()
            if not cc_outcomes.empty
            else {}
        )
        cc_refuse_by_reason = {str(k): int(v) for k, v in cc_refuse_by_reason.items()}
        n_cc_opened = (
            int((cc_outcomes["outcome"] == "opened").sum()) if not cc_outcomes.empty else 0
        )

        # Per-year executed-put mean realized
        per_year: dict[str, dict] = {}
        if not replayed.empty:
            tmp = replayed.copy()
            tmp["year"] = pd.to_datetime(tmp["date"]).dt.year
            for year, grp in tmp.groupby("year"):
                from scipy.stats import spearmanr

                rho_res = spearmanr(
                    grp["ev_dollars"].to_numpy(),
                    grp["realized_pnl"].to_numpy(),
                    nan_policy="omit",
                )
                per_year[str(int(year))] = {
                    "n_candidates": int(len(grp)),
                    "rho": float(rho_res.correlation)
                    if not np.isnan(rho_res.correlation)
                    else None,
                    "mean_realized_all_candidates": float(grp["realized_pnl"].mean()),
                    "hit_rate_all_candidates": float((grp["realized_pnl"] > 0).mean()),
                }

        # Aggregate rho across full window
        if not replayed.empty and len(replayed) >= 2:
            from scipy.stats import spearmanr

            rho_res = spearmanr(
                replayed["ev_dollars"].to_numpy(),
                replayed["realized_pnl"].to_numpy(),
                nan_policy="omit",
            )
            agg_rho = float(rho_res.correlation) if not np.isnan(rho_res.correlation) else None
            agg_mean_realized = float(replayed["realized_pnl"].mean())
            agg_hit_rate = float((replayed["realized_pnl"] > 0).mean())
        else:
            agg_rho = None
            agg_mean_realized = None
            agg_hit_rate = None

        return {
            "final_nav": final_nav,
            "final_cash": float(tr.cash),
            "return_pct": (final_nav - capital) / capital,
            "n_put_attempts": n_put_attempts,
            "n_put_opened": n_put_opened,
            "n_put_assignments": int(put_assigned_open + put_assigned_closed),
            "n_cc_attempts": int(len(cc_outcomes)),
            "n_cc_opened": n_cc_opened,
            "executed_trades_put_metric": int(closed_with_put + open_with_put),
            "put_refuse_by_reason": put_refuse_by_reason,
            "cc_refuse_by_reason": cc_refuse_by_reason,
            "spearman_rho_all_candidates": agg_rho,
            "mean_realized_all_candidates": agg_mean_realized,
            "hit_rate_all_candidates": agg_hit_rate,
            "per_year": per_year,
            "executed_realized_total": float(sum(executed_put_returns)),
            "executed_realized_mean": (
                float(np.mean(executed_put_returns)) if executed_put_returns else None
            ),
            "n_closed_positions": len(tr.closed_positions),
            "n_open_positions_at_end": len(tr.positions),
        }

    summary = {
        "setup": {
            "capital": capital,
            "universe_size": universe_size,
            "start": start,
            "end": end,
            "friction_level": friction_level,
            "dte_target": 35,
            "delta_target": 0.25,
            "contracts": 1,
            "ohlcv_csv_sha256": ohlcv_sha256(),
        },
        "loose": _tracker_metrics(
            tracker_loose, rank_log_loose, put_attempts_loose, cc_attempts_loose
        ),
        "strict": _tracker_metrics(
            tracker_strict, rank_log_strict, put_attempts_strict, cc_attempts_strict
        ),
    }
    # Delta block
    summary["delta_strict_minus_loose"] = {
        "final_nav": summary["strict"]["final_nav"] - summary["loose"]["final_nav"],
        "return_pp": (summary["strict"]["return_pct"] - summary["loose"]["return_pct"]) * 100,
        "executed_trades_put_metric": (
            summary["strict"]["executed_trades_put_metric"]
            - summary["loose"]["executed_trades_put_metric"]
        ),
        "n_put_opened": summary["strict"]["n_put_opened"] - summary["loose"]["n_put_opened"],
        "n_cc_opened": summary["strict"]["n_cc_opened"] - summary["loose"]["n_cc_opened"],
        "n_put_assignments": (
            summary["strict"]["n_put_assignments"] - summary["loose"]["n_put_assignments"]
        ),
    }
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HT-D R10 strict-vs-loose backtest driver.")
    ap.add_argument("--start", default="2020-01-02", help="Window start ISO date.")
    ap.add_argument("--end", default="2024-12-31", help="Window end ISO date.")
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory (will be created if missing). Persist CSVs / JSONs here.",
    )
    ap.add_argument(
        "--capital", type=float, default=1_000_000.0, help="Starting capital. Default $1M."
    )
    ap.add_argument(
        "--friction",
        default="full",
        choices=("none", "bid_ask", "full"),
        help="Friction level. Default 'full'.",
    )
    ap.add_argument("--top-n", type=int, default=10, help="Top N candidates per day. Default 10.")
    ap.add_argument(
        "--max-new-per-day",
        type=int,
        default=3,
        help="Max new positions per day. Default 3.",
    )
    ap.add_argument(
        "--universe",
        default="100",
        choices=("24", "100"),
        help="Universe size. Default '100' = UNIVERSE_100.",
    )
    args = ap.parse_args(argv)

    if args.universe == "100":
        tickers = UNIVERSE_100
    else:
        from backtests.regression.universes import UNIVERSE_24

        tickers = UNIVERSE_24

    out_dir = Path(args.out_dir).resolve()
    run_strict_vs_loose(
        capital=args.capital,
        tickers=tickers,
        start=args.start,
        end=args.end,
        friction_level=args.friction,
        top_n=args.top_n,
        max_new_per_day=args.max_new_per_day,
        out_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
