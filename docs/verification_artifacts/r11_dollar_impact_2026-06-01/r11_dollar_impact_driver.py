"""R11 dollar-impact backtest — post-ship validation of the elevated-vol
top-bin size-down (heavy-verify 2026-05-31 I11; shipped #306/#307).

R11 (``engine/candidate_dossier.py`` :534-566) downgrades a *proceed*
candidate to *review* when the market-wide VIX *level* is elevated
(``vix > R11_VIX_THRESHOLD`` = 25.0) AND the candidate is a high-confidence
top-bin pick (``prob_profit > R11_TOP_BIN_PROB`` = 0.90). It shipped on the
I11 leave-one-crisis-out evidence (2020 +$86k / 2022 +$3.5k averted-vs-
forgone, per-contract). It has **never** been measured in a full-scale
dollar backtest. This driver does that.

WHY A NEW DRIVER (and not the canonical multi-friction harness verbatim):
R11 lives in the ``EnginePhaseReviewer`` dossier chain, NOT in
``rank_candidates_by_ev``. The canonical S38/S43/S44 backtest
(``backtests/regression/_common.run_backtest_multi_friction``) routes through
``rank_candidates_by_ev`` and opens on ``ev_dollars > 0`` — it never invokes
the reviewer, so R11 is dormant in it. To measure R11's dollar impact we run
TWO arms over the SAME daily rank:

  * ``suppressed`` — the literal pre-R11 engine: open on EV>0, top-N, BP-gated,
    cap ``max_new_per_day``. Identical to S38/S43/S44's open policy.
  * ``active``     — same, but a candidate is NOT opened when R11's exact gate
    fires (``ev_dollars > MIN_PROCEED_EV_DOLLARS`` AND
    ``prob_profit > R11_TOP_BIN_PROB`` AND ``vix(as_of) > R11_VIX_THRESHOLD``).
    The day's open quota BACKFILLS to the next eligible candidate — the
    realistic "size the top bin down, redeploy elsewhere" behaviour.

The gate is REPLICATED from the engine's own constants
(``R11_TOP_BIN_PROB``, ``R11_VIX_THRESHOLD``, ``MIN_PROCEED_EV_DOLLARS``
imported from ``engine.candidate_dossier`` so they cannot drift) and the
engine's own PIT VIX source (``connector.get_vix_regime(as_of)["vix"]`` —
the exact call ``wheel_runner`` makes at :3360). We replicate rather than
call the full reviewer on purpose: the reviewer's R2 (chart-missing → review)
would downgrade EVERY candidate and swamp R11. Replicating R11 alone keeps the
suppressed arm equal to the documented pre-R11 baseline and isolates R11's
marginal effect.

SCOPE: R11 gates PUT ENTRIES (its design intent — top-bin put assignment
over-confidence). The covered-call leg is identical between arms. Both arms
use ``require_ev_authority=False`` (the S38/S43/S44 baseline) so D17 R9/R10/
delta/Kelly stay dormant and R11 is the ONLY difference between the arms.

§2: every candidate routes through ``EVEngine.evaluate`` via
``rank_candidates_by_ev``. R11 only ever REMOVES an open (downgrade-only);
it never rescues a non-tradeable candidate. No ``engine/`` file is modified —
this is observation only. The driver imports canonical helpers from
``backtests/regression/_common.py`` and the canonical universe from
``backtests/regression/universes.py`` to keep the comparison apples-to-apples
with S38/S44.

Outputs (under ``--out-dir``):
  * ``rank_log.csv``            — every ranked row (shared across arms) with
                                  realised P&L forward-replayed at expiry.
  * ``open_attempts_active.csv``    — every put open attempt, active arm
                                      (opened / r11_blocked / skipped reasons).
  * ``open_attempts_suppressed.csv``— same, suppressed arm.
  * ``r11_blocked.csv``         — the candidates R11 removed in the active arm,
                                  with forward-replayed counterfactual P&L +
                                  assignment flag (the averted-vs-forgone set).
  * ``daily_nav.csv``           — per-day NAV both arms (Sharpe input).
  * ``tracker_active_state.json`` / ``tracker_suppressed_state.json``.
  * ``summary.json``            — headline Δ + blocked-set + per-regime +
                                  per-VIX-bucket + §2 scan.

Usage::

    # Pilot — COVID quarter, validates R11 fires + blocked-set sign
    python docs/verification_artifacts/r11_dollar_impact_2026-06-01/r11_dollar_impact_driver.py \\
        --start 2020-02-01 --end 2020-05-31 --out-dir <tmp>/r11_pilot

    # Full window (all four regimes: 2020 crash, 2021 bull, 2022 bear, 2023-24 calm)
    python .../r11_dollar_impact_driver.py \\
        --start 2020-01-02 --end 2024-12-31 --out-dir <tmp>/r11_w3

    # Post-hoc analysis (re-runnable without the backtest)
    python .../r11_dollar_impact_driver.py --analyze --out-dir <tmp>/r11_w3
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

# Worktree on sys.path[0] BEFORE engine imports — defeats the user-site .pth
# shadow that would otherwise import engine.* from the older primary clone
# (memory: sys-path-worktree-shadow).
WORKTREE = Path(__file__).resolve().parents[3]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

# Windows console is cp1252; our progress/■ output uses non-latin1 glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtests.regression._common import (  # noqa: E402
    _forward_replay_realized_pnl,
    _next_business_day,
    _spot_on_or_after,
    _tracker_step,
    assert_data_window_available,
    friction_adjusted_premium,
    friction_open_cost,
    ohlcv_sha256,
)
from backtests.regression.universes import UNIVERSE_100  # noqa: E402

# The engine's own R11 constants — imported so the replicated gate cannot
# drift from the production reviewer's thresholds.
from engine.candidate_dossier import (  # noqa: E402
    MIN_PROCEED_EV_DOLLARS,
    R11_TOP_BIN_PROB,
    R11_VIX_THRESHOLD,
)
from engine.wheel_runner import WheelRunner  # noqa: E402
from engine.wheel_tracker import PositionState, WheelTracker  # noqa: E402

ARMS = ("suppressed", "active")


# -----------------------------------------------------------------------------
# R11 gate — replicated from candidate_dossier.py :546-553 exactly.
# -----------------------------------------------------------------------------


def r11_fires(*, ev_dollars: float, prob_profit: float, vix_level: float | None) -> bool:
    """True iff R11 would downgrade this *proceed* candidate to *review*.

    Mirrors ``engine.candidate_dossier.EnginePhaseReviewer`` R11 (:546-553):
    only candidates that would be ``proceed`` (EV above the proceed floor)
    are eligible; the gate is ``prob_profit > R11_TOP_BIN_PROB`` AND
    ``vix > R11_VIX_THRESHOLD`` (strict ``>`` on both). ``vix_level is None``
    → no-op (missing-evidence semantics).
    """
    if vix_level is None:
        return False
    # "verdict == proceed" precondition: in the dossier, proceed requires
    # EV above MIN_PROCEED_EV_DOLLARS. Candidates with 0 < EV <= floor are
    # opened by the harness but are NOT proceed, so R11 never touches them.
    if ev_dollars <= MIN_PROCEED_EV_DOLLARS:
        return False
    try:
        pp = float(prob_profit or 0.0)
        vix_f = float(vix_level)
    except (TypeError, ValueError):
        return False
    return pp > R11_TOP_BIN_PROB and vix_f > R11_VIX_THRESHOLD


def regime_of(d: date) -> str:
    """Calendar regime bucket (the four the JOB-1 spec names + spillover).

    Objective VIX-at-entry bucketing is provided separately in the summary;
    this calendar map is the human-readable secondary slice.
    """
    if date(2020, 2, 15) <= d <= date(2020, 4, 30):
        return "crash_2020"
    if date(2021, 1, 1) <= d <= date(2021, 12, 31):
        return "bull_2021"
    if date(2022, 1, 1) <= d <= date(2022, 12, 31):
        return "bear_2022"
    # 2020 H2 recovery, 2023, 2024, 2025 — the low-vol "calm" remainder.
    return "calm"


def vix_bucket(vix: float | None) -> str:
    """Objective VIX-at-entry bucket — the slice R11's mechanism keys on."""
    if vix is None or not np.isfinite(vix):
        return "unknown"
    if vix <= 15:
        return "vix<=15"
    if vix <= 25:
        return "15-25"
    if vix <= 35:
        return "25-35"
    return "vix>35"


# -----------------------------------------------------------------------------
# Open step — mirrors _common._tracker_try_opens, adds the R11 gate (active arm)
# -----------------------------------------------------------------------------


def _try_opens(
    *,
    tracker: WheelTracker,
    frame: pd.DataFrame,
    today: date,
    vix_today: float | None,
    friction_level: str,
    max_new_per_day: int,
    contracts: int,
    expiration_default: date,
    apply_r11: bool,
    attempts_log: list[dict],
    blocked_log: list[dict],
) -> None:
    """One arm's daily open step. ``apply_r11=False`` reproduces the canonical
    S38/S43 open policy exactly. ``apply_r11=True`` skips (backfills past)
    candidates the R11 gate fires on, logging them to ``blocked_log``.
    """
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

        pp = float(row.get("prob_profit", float("nan")))
        # R11 gate — active arm only. Backfill: a blocked candidate does NOT
        # consume the day's quota; we move to the next eligible candidate.
        if apply_r11 and r11_fires(ev_dollars=ev_dollars, prob_profit=pp, vix_level=vix_today):
            blocked_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "vix": vix_today,
                    "strike": strike,
                    "premium": premium,
                    "premium_raw": float(row.get("premium", 0.0)),
                    "iv": float(row.get("iv", 0.0)),
                    "expiration_date": expiration_default.isoformat(),
                    "regime": regime_of(today),
                    "vix_bucket": vix_bucket(vix_today),
                }
            )
            attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "strike": strike,
                    "vix": vix_today,
                    "outcome": "r11_blocked",
                }
            )
            continue

        if tracker.available_buying_power() < strike * 100 * contracts:
            attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "strike": strike,
                    "vix": vix_today,
                    "outcome": "insufficient_bp",
                }
            )
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
            attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "strike": strike,
                    "vix": vix_today,
                    "outcome": "opened",
                }
            )
            opens_today += 1


# -----------------------------------------------------------------------------
# Top-level driver
# -----------------------------------------------------------------------------


def run_r11_ab(
    *,
    capital: float,
    tickers: Sequence[str],
    start: str,
    end: str,
    out_dir: Path,
    friction_level: str = "full",
    top_n: int = 10,
    max_new_per_day: int = 3,
    dte_target: int = 35,
    delta_target: float = 0.25,
    contracts: int = 1,
    progress_every_days: int = 25,
) -> dict[str, Any]:
    """Run the suppressed + active arms in parallel sharing one daily rank
    call. Persist artifacts to ``out_dir`` and return the summary dict."""
    assert_data_window_available(start, end)

    runner = WheelRunner()
    conn = runner.connector
    print(f"[r11_ab] connector: {type(conn).__name__}", flush=True)
    print(f"[r11_ab] worktree:  {WORKTREE}", flush=True)
    print(
        f"[r11_ab] universe={len(tickers)}t window={start}->{end} "
        f"capital=${capital:,.0f} friction={friction_level} "
        f"R11(vix>{R11_VIX_THRESHOLD}, pp>{R11_TOP_BIN_PROB}, ev>{MIN_PROCEED_EV_DOLLARS})",
        flush=True,
    )

    trackers = {
        arm: WheelTracker(initial_capital=capital, connector=conn, require_ev_authority=False)
        for arm in ARMS
    }

    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    tickers = list(tickers)

    rank_log_rows: list[dict] = []
    attempts: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    blocked_log: list[dict] = []
    daily_nav: list[dict] = []
    vix_cache: dict[str, float | None] = {}

    _total = len(trading_days)
    _every = max(1, _total // progress_every_days)
    _ckpt = max(progress_every_days * 4, 100)
    _t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[r11_ab] out_dir: {out_dir}", flush=True)

    def _nav(tr: WheelTracker) -> float:
        try:
            return (
                float(tr.equity_curve[-1].get("portfolio_value", tr.cash))
                if tr.equity_curve
                else float(tr.cash)
            )
        except Exception:
            return float("nan")

    for _i, today in enumerate(trading_days):
        if _i > 0 and _i % _every == 0:
            el = time.time() - _t0
            rate = _i / el if el > 0 else 0.0
            eta = (_total - _i) / rate if rate > 0 else 0.0
            print(
                f"[r11_ab] day {_i:4d}/{_total} ({100 * _i / _total:5.1f}%) "
                f"elapsed {el / 60:6.1f}min ETA {eta / 60:6.1f}min ({rate:5.2f} day/s) "
                f"NAV_supp=${_nav(trackers['suppressed']):,.0f} "
                f"NAV_act=${_nav(trackers['active']):,.0f} "
                f"blocked={len(blocked_log)}",
                flush=True,
            )
        if _i > 0 and _i % _ckpt == 0:
            _checkpoint(out_dir, rank_log_rows, attempts, blocked_log, daily_nav)

        expiration_default = _next_business_day(today + timedelta(days=dte_target))

        # PIT VIX once per day — the exact call wheel_runner.py:3360 makes.
        as_of = today.isoformat()
        if as_of not in vix_cache:
            try:
                _v = conn.get_vix_regime(as_of).get("vix")
                vix_cache[as_of] = float(_v) if _v is not None and np.isfinite(_v) else None
            except Exception:
                vix_cache[as_of] = None
        vix_today = vix_cache[as_of]

        # Per-arm pre-rank step (MTM + settle + wheel-into-CC) — canonical
        # _common._tracker_step, identical between arms.
        for arm in ARMS:
            _tracker_step(
                tracker=trackers[arm],
                runner=runner,
                conn=conn,
                today=today,
                friction_level=friction_level,
                contracts=contracts,
                dte_target=dte_target,
                delta_target=delta_target,
                expiration_default=expiration_default,
                PositionState=PositionState,
            )

        # ONE shared SP rank per day.
        try:
            frame = runner.rank_candidates_by_ev(
                tickers=tickers,
                dte_target=dte_target,
                delta_target=delta_target,
                contracts=contracts,
                top_n=top_n,
                min_ev_dollars=-1e9,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
        except Exception as exc:
            print(f"[r11_ab] rank failed {today}: {exc}", flush=True)
            continue
        if frame is None or len(frame) == 0:
            continue

        # Shared rank log (one copy; arms share the same ranked frame).
        for _, row in frame.iterrows():
            premium_raw = float(row.get("premium", 0.0))
            rank_log_rows.append(
                {
                    "date": as_of,
                    "ticker": str(row.get("ticker", "")),
                    "ev_dollars": float(row.get("ev_dollars", 0.0)),
                    "premium": friction_adjusted_premium(premium_raw, friction_level),
                    "premium_raw": premium_raw,
                    "strike": float(row.get("strike", 0.0)),
                    "iv": float(row.get("iv", 0.0)),
                    "prob_profit": float(row.get("prob_profit", float("nan"))),
                    "vix": vix_today,
                    "expiration_date": expiration_default.isoformat(),
                }
            )

        # Open step per arm.
        _try_opens(
            tracker=trackers["suppressed"],
            frame=frame,
            today=today,
            vix_today=vix_today,
            friction_level=friction_level,
            max_new_per_day=max_new_per_day,
            contracts=contracts,
            expiration_default=expiration_default,
            apply_r11=False,
            attempts_log=attempts["suppressed"],
            blocked_log=[],  # never used in suppressed arm
        )
        _try_opens(
            tracker=trackers["active"],
            frame=frame,
            today=today,
            vix_today=vix_today,
            friction_level=friction_level,
            max_new_per_day=max_new_per_day,
            contracts=contracts,
            expiration_default=expiration_default,
            apply_r11=True,
            attempts_log=attempts["active"],
            blocked_log=blocked_log,
        )

        daily_nav.append(
            {
                "date": as_of,
                "vix": vix_today,
                "nav_suppressed": _nav(trackers["suppressed"]),
                "nav_active": _nav(trackers["active"]),
                "cash_suppressed": float(trackers["suppressed"].cash),
                "cash_active": float(trackers["active"].cash),
                "n_pos_suppressed": len(trackers["suppressed"].positions),
                "n_pos_active": len(trackers["active"].positions),
            }
        )

    # Forward-replay realised P&L on rank_log + r11_blocked (counterfactual).
    spot_cache: dict[tuple[str, str], float | None] = {}

    def _replay(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        realized, spots, assigned = [], [], []
        for r in df.itertuples(index=False):
            key = (r.ticker, r.expiration_date)
            if key not in spot_cache:
                spot_cache[key] = _spot_on_or_after(
                    conn, r.ticker, date.fromisoformat(r.expiration_date)
                )
            spot = spot_cache[key]
            if spot is None:
                realized.append(float("nan"))
                spots.append(float("nan"))
                assigned.append(None)
            else:
                realized.append(_forward_replay_realized_pnl(r.strike, r.premium, spot))
                spots.append(spot)
                assigned.append(bool(spot < r.strike))
        df["spot_at_expiry"] = spots
        df["realized_pnl"] = realized
        df["assigned"] = assigned
        return df

    rank_log = _replay(rank_log_rows)
    blocked = _replay(blocked_log)

    rank_log.to_csv(out_dir / "rank_log.csv", index=False)
    blocked.to_csv(out_dir / "r11_blocked.csv", index=False)
    pd.DataFrame(attempts["active"]).to_csv(out_dir / "open_attempts_active.csv", index=False)
    pd.DataFrame(attempts["suppressed"]).to_csv(
        out_dir / "open_attempts_suppressed.csv", index=False
    )
    pd.DataFrame(daily_nav).to_csv(out_dir / "daily_nav.csv", index=False)
    for arm in ARMS:
        with open(out_dir / f"tracker_{arm}_state.json", "w", encoding="utf-8") as f:
            json.dump(trackers[arm].to_dict(), f, indent=2, default=str)

    summary = _build_summary(
        trackers=trackers,
        attempts=attempts,
        blocked=blocked,
        daily_nav=pd.DataFrame(daily_nav),
        capital=capital,
        start=start,
        end=end,
        friction_level=friction_level,
        universe_size=len(tickers),
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n[r11_ab] DONE", flush=True)
    print(f"  suppressed NAV: ${summary['suppressed']['final_nav']:,.2f}", flush=True)
    print(f"  active     NAV: ${summary['active']['final_nav']:,.2f}", flush=True)
    print(
        f"  Δ NAV (active-suppressed): ${summary['delta']['final_nav']:+,.2f}",
        flush=True,
    )
    bk = summary["r11_blocked"]
    print(
        f"  R11 blocked {bk['n']} opens; counterfactual realised "
        f"${bk['realized_total']:+,.2f} (mean ${bk['realized_mean'] or 0:+,.2f}/contract, "
        f"assignment {100 * (bk['assignment_rate'] or 0):.1f}%)",
        flush=True,
    )
    return summary


def _checkpoint(
    out_dir: Path,
    rank_log_rows: list[dict],
    attempts: dict[str, list[dict]],
    blocked_log: list[dict],
    daily_nav: list[dict],
) -> None:
    """Best-effort partial dump — survives compaction/crash without re-running."""
    try:
        pd.DataFrame(attempts["active"]).to_csv(out_dir / "open_attempts_active.csv", index=False)
        pd.DataFrame(attempts["suppressed"]).to_csv(
            out_dir / "open_attempts_suppressed.csv", index=False
        )
        pd.DataFrame(blocked_log).to_csv(out_dir / "r11_blocked_partial.csv", index=False)
        pd.DataFrame(daily_nav).to_csv(out_dir / "daily_nav.csv", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[r11_ab] checkpoint failed: {e}", flush=True)


def _sharpe(nav: pd.Series) -> float:
    """Annualised Sharpe of a daily NAV series (rf=0). NaN if < 2 points."""
    nav = nav.dropna()
    if len(nav) < 3:
        return float("nan")
    rets = nav.pct_change().dropna()
    if rets.std(ddof=1) == 0 or len(rets) < 2:
        return float("nan")
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))


def _build_summary(
    *,
    trackers: dict[str, WheelTracker],
    attempts: dict[str, list[dict]],
    blocked: pd.DataFrame,
    daily_nav: pd.DataFrame,
    capital: float,
    start: str,
    end: str,
    friction_level: str,
    universe_size: int,
) -> dict[str, Any]:
    def _arm_metrics(tr: WheelTracker, atts: list[dict], nav_col: str) -> dict:
        final_nav = (
            float(tr.equity_curve[-1].get("portfolio_value", tr.cash))
            if tr.equity_curve
            else float(tr.cash)
        )
        df = pd.DataFrame(atts)
        n_opened = int((df["outcome"] == "opened").sum()) if not df.empty else 0
        n_blocked = int((df["outcome"] == "r11_blocked").sum()) if not df.empty else 0
        executed = [
            float(cp.get("realized_pnl", 0.0))
            for cp in tr.closed_positions
            if (cp.get("put_premium") or 0) > 0 and cp.get("exit_reason") not in (None, "open")
        ]
        put_assigned_open = sum(
            1 for p in tr.positions.values() if p.state.value in ("stock_owned", "covered_call")
        )
        put_assigned_closed = sum(
            1 for r in tr.closed_positions if r.get("exit_reason") == "call_assigned"
        )
        sharpe = _sharpe(daily_nav[nav_col]) if (nav_col in daily_nav.columns) else float("nan")
        return {
            "final_nav": final_nav,
            "final_cash": float(tr.cash),
            "return_pct": (final_nav - capital) / capital,
            "sharpe": sharpe,
            "n_put_opened": n_opened,
            "n_r11_blocked": n_blocked,
            "n_put_assignments": int(put_assigned_open + put_assigned_closed),
            "executed_realized_total": float(sum(executed)) if executed else 0.0,
            "executed_realized_mean": float(np.mean(executed)) if executed else None,
            "n_closed_positions": len(tr.closed_positions),
            "n_open_at_end": len(tr.positions),
        }

    supp = _arm_metrics(trackers["suppressed"], attempts["suppressed"], "nav_suppressed")
    act = _arm_metrics(trackers["active"], attempts["active"], "nav_active")

    # R11-blocked set — the averted-vs-forgone counterfactual.
    bl = blocked.dropna(subset=["realized_pnl"]) if not blocked.empty else blocked
    blocked_block = {
        "n": int(len(blocked)),
        "n_with_realized": int(len(bl)),
        "realized_total": float(bl["realized_pnl"].sum()) if not bl.empty else 0.0,
        "realized_mean": float(bl["realized_pnl"].mean()) if not bl.empty else None,
        "assignment_rate": float(bl["assigned"].mean()) if not bl.empty else None,
        "premium_forgone_total": float((bl["premium"] * 100).sum()) if not bl.empty else 0.0,
    }
    # Per-regime + per-VIX-bucket on the blocked set.
    by_regime: dict[str, dict] = {}
    by_vixb: dict[str, dict] = {}
    if not bl.empty:
        for key, grp in bl.groupby("regime"):
            by_regime[str(key)] = {
                "n": int(len(grp)),
                "realized_total": float(grp["realized_pnl"].sum()),
                "realized_mean": float(grp["realized_pnl"].mean()),
                "assignment_rate": float(grp["assigned"].mean()),
            }
        for key, grp in bl.groupby("vix_bucket"):
            by_vixb[str(key)] = {
                "n": int(len(grp)),
                "realized_total": float(grp["realized_pnl"].sum()),
                "realized_mean": float(grp["realized_pnl"].mean()),
                "assignment_rate": float(grp["assigned"].mean()),
            }

    # §2 scan — any opened put with ev_dollars<=0 in either arm.
    s2 = {}
    for arm in ARMS:
        df = pd.DataFrame(attempts[arm])
        opened = df[df["outcome"] == "opened"] if not df.empty else df
        s2[arm] = {
            "n_opened": int(len(opened)),
            "n_opened_nonpositive_ev": int((opened["ev_dollars"] <= 0).sum())
            if not opened.empty
            else 0,
        }

    return {
        "setup": {
            "capital": capital,
            "universe_size": universe_size,
            "start": start,
            "end": end,
            "friction_level": friction_level,
            "dte_target": 35,
            "delta_target": 0.25,
            "contracts": 1,
            "r11_vix_threshold": R11_VIX_THRESHOLD,
            "r11_top_bin_prob": R11_TOP_BIN_PROB,
            "min_proceed_ev_dollars": MIN_PROCEED_EV_DOLLARS,
            "ohlcv_csv_sha256": ohlcv_sha256(),
        },
        "suppressed": supp,
        "active": act,
        "delta": {
            "final_nav": act["final_nav"] - supp["final_nav"],
            "return_pp": (act["return_pct"] - supp["return_pct"]) * 100,
            "sharpe": (act["sharpe"] - supp["sharpe"])
            if (np.isfinite(act["sharpe"]) and np.isfinite(supp["sharpe"]))
            else float("nan"),
            "n_put_opened": act["n_put_opened"] - supp["n_put_opened"],
            "n_put_assignments": act["n_put_assignments"] - supp["n_put_assignments"],
            "executed_realized_total": act["executed_realized_total"]
            - supp["executed_realized_total"],
        },
        "r11_blocked": blocked_block,
        "r11_blocked_by_regime": by_regime,
        "r11_blocked_by_vix_bucket": by_vixb,
        "section2_scan": s2,
    }


# -----------------------------------------------------------------------------
# Post-hoc analyzer
# -----------------------------------------------------------------------------


def analyze(out_dir: Path) -> None:
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    setup, supp, act, delta = (
        summary["setup"],
        summary["suppressed"],
        summary["active"],
        summary["delta"],
    )
    bk = summary["r11_blocked"]

    print("=" * 78)
    print(f"R11 DOLLAR-IMPACT ANALYSIS — {out_dir.name}")
    print(f"Window: {setup['start']} -> {setup['end']}  Capital: ${setup['capital']:,.0f}")
    print(f"Universe: {setup['universe_size']}t  Friction: {setup['friction_level']}")
    print(
        f"R11 gate: vix > {setup['r11_vix_threshold']}  AND  "
        f"prob_profit > {setup['r11_top_bin_prob']}  AND  ev > ${setup['min_proceed_ev_dollars']}"
    )
    print("=" * 78)
    print()

    print("## Headline — active (R11 on) vs suppressed (pre-R11 baseline)")
    print()
    print("| Metric | Suppressed | Active | Δ (active − suppressed) |")
    print("|---|---|---|---|")
    print(
        f"| Final NAV | ${supp['final_nav']:,.0f} | ${act['final_nav']:,.0f} | "
        f"${delta['final_nav']:+,.0f} |"
    )
    print(
        f"| Return | {supp['return_pct'] * 100:+.2f}% | {act['return_pct'] * 100:+.2f}% | "
        f"{delta['return_pp']:+.2f}pp |"
    )
    print(
        f"| Sharpe (daily NAV, ann.) | {supp['sharpe']:.3f} | {act['sharpe']:.3f} | "
        f"{delta['sharpe']:+.3f} |"
    )
    print(
        f"| Puts opened | {supp['n_put_opened']:,} | {act['n_put_opened']:,} | "
        f"{delta['n_put_opened']:+,} |"
    )
    print(
        f"| Put assignments | {supp['n_put_assignments']:,} | {act['n_put_assignments']:,} | "
        f"{delta['n_put_assignments']:+,} |"
    )
    print(
        f"| Executed realised total | ${supp['executed_realized_total']:+,.0f} | "
        f"${act['executed_realized_total']:+,.0f} | ${delta['executed_realized_total']:+,.0f} |"
    )
    print(f"| R11 blocked opens | 0 | {act['n_r11_blocked']:,} | {act['n_r11_blocked']:+,} |")
    print()

    print("## R11-blocked set — counterfactual (held-to-expiry P&L of the removed opens)")
    print()
    print(
        f"- Blocked opens (active arm): **{bk['n']:,}** ({bk['n_with_realized']:,} with expiry data)"
    )
    if bk["realized_mean"] is not None:
        print(
            f"- Counterfactual realised P&L if KEPT: **${bk['realized_total']:+,.0f}** total, "
            f"**${bk['realized_mean']:+,.2f}/contract** mean"
        )
        print(f"- Assignment rate of blocked set: **{100 * bk['assignment_rate']:.1f}%**")
        print(
            f"- Premium forgone (gross, if all expired worthless): ${bk['premium_forgone_total']:,.0f}"
        )
        verdict = (
            "R11 AVERTED net losses on the blocked set"
            if bk["realized_total"] < 0
            else "R11 FORWENT net premium on the blocked set"
        )
        print(f"- **Direction: {verdict}.**")
    print()

    print("## R11-blocked set by calendar regime")
    print()
    print("| Regime | n blocked | Counterfactual realised total | mean/contract | assignment % |")
    print("|---|---|---|---|---|")
    for reg, m in sorted(summary.get("r11_blocked_by_regime", {}).items()):
        print(
            f"| {reg} | {m['n']:,} | ${m['realized_total']:+,.0f} | ${m['realized_mean']:+,.2f} | "
            f"{100 * m['assignment_rate']:.1f}% |"
        )
    print()

    print("## R11-blocked set by VIX-at-entry bucket (the mechanism slice)")
    print()
    print(
        "| VIX bucket | n blocked | Counterfactual realised total | mean/contract | assignment % |"
    )
    print("|---|---|---|---|---|")
    for vb, m in sorted(summary.get("r11_blocked_by_vix_bucket", {}).items()):
        print(
            f"| {vb} | {m['n']:,} | ${m['realized_total']:+,.0f} | ${m['realized_mean']:+,.2f} | "
            f"{100 * m['assignment_rate']:.1f}% |"
        )
    print()

    print("## §2 invariant scan")
    print()
    for arm, s in summary.get("section2_scan", {}).items():
        ok = "§2 OK" if s["n_opened_nonpositive_ev"] == 0 else "§2 BREACH"
        print(
            f"- **{arm}**: opened {s['n_opened']:,} puts; with ev_dollars<=0: "
            f"{s['n_opened_nonpositive_ev']}. {ok}."
        )
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R11 dollar-impact A/B backtest driver.")
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--friction", default="full", choices=("none", "bid_ask", "full"))
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-new-per-day", type=int, default=3)
    ap.add_argument("--universe", default="100", choices=("24", "100"))
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir).resolve()
    if args.analyze:
        analyze(out_dir)
        return 0

    if args.universe == "100":
        tickers = UNIVERSE_100
    else:
        from backtests.regression.universes import UNIVERSE_24

        tickers = UNIVERSE_24

    run_r11_ab(
        capital=args.capital,
        tickers=tickers,
        start=args.start,
        end=args.end,
        out_dir=out_dir,
        friction_level=args.friction,
        top_n=args.top_n,
        max_new_per_day=args.max_new_per_day,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
