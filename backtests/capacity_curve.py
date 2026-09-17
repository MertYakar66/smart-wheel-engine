"""Capacity-curve validation (V4) — where is the knee of edge vs deployed dollars?

`docs/VALIDATION_PHASE_PLAN.md` section 7 is the pre-registered design this
module implements.  Three discovered constraints shape it (section 7.0):
the tracker is contract-blind (so `run_backtest(contracts>1)` is
internally inconsistent — never exercised beyond 1 in any locked study);
a no-impact capacity curve is a straight line by construction; and no
option-volume data exists in this checkout, so the deliverable is the knee
AS A FUNCTION of the stock-ADV proxy assumption, never a point estimate.

Method (section 7.1):

* **Capital-equivalent ladder** — ladder point N runs a plain 1-contract
  `WheelTracker` at ``capital = BASE/N``.  Every cash flow in the
  1-contract book is exactly 1/N of the N-contract world under the linear
  cost model, so returns and gating are identical; the size-dependence
  enters ONLY through the impact overlay, priced at the TRUE order size N.
* **Shared rank** — one daily `rank_candidates_by_ev` (and a per-(ticker,
  day) covered-call rank cache) serves every ladder point: the rank is
  capital- and impact-independent, so the whole grid costs ~1x the
  dominant rank bill.
* **Engine-native impact overlay** — on every option fill, on top of
  ``full`` friction: `calculate_slippage(mid, bid_ask_spread=0, "sell",
  num_contracts=N, adv_contracts=proxy)` — zero spread isolates the
  engine's own dormant Almgren-Chriss sqrt term (``k*mid*sqrt(N/adv)``,
  shipped k=0.10); the spread cost stays with the ``full`` overlay (no
  double count).  Participation cap: a fill is refused when
  ``N > 0.10 * adv_contracts``.  Proxy: ``adv_contracts = r x
  avg_vol_30d`` (shares), r swept over {1e-5, 1e-4, 1e-3}.
* **Linearity control** — the same ladder with the overlay disabled must
  produce ``return_pct`` exactly PROPORTIONAL to N while no BP refusal has
  fired (same 1-contract book on 1/N the capital), with any deviation
  from proportionality coinciding exactly with the first BP refusal (the
  A/A of the study; a deviation without a BP refusal is a harness bug).

Scope / invariants (CLAUDE.md section 2): measurement-only.  A sibling
driver in the sanctioned `r10_strict_driver.py` copy-the-driver pattern —
imports `_common` helpers, zero engine changes, rail pinned off, loose
mode (strict mode excluded per section 7.0(3)); nothing feeds back; any
"better" sizing implied by results is a reported finding, never shipped.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-registered constants (plan doc section 7.1)
# ---------------------------------------------------------------------------

LADDER: tuple[int, ...] = (1, 5, 10, 25)
PROXY_RATIOS: tuple[float, ...] = (1e-5, 1e-4, 1e-3)
IMPACT_K = 0.10  # the engine's shipped impact_coefficient (worklist B5 owns its calibration)
PARTICIPATION_CAP = 0.10  # max fraction of proxied option ADV per fill


# ---------------------------------------------------------------------------
# ADV proxy lookup (PIT: last observation at or before the fill date)
# ---------------------------------------------------------------------------


class AdvLookup:
    """Per-ticker cached ``avg_vol_30d`` series with PIT as-of reads.

    One full `get_liquidity` fetch per ticker (the connector serves an
    inclusive date-range slice with no as_of of its own — PIT correctness
    is this caller's job, per the section-7 recon).
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self._cache: dict[str, pd.Series | None] = {}

    def adv_shares(self, ticker: str, as_of: date) -> float | None:
        if ticker not in self._cache:
            try:
                df = self._conn.get_liquidity(ticker)
                self._cache[ticker] = (
                    df["avg_vol_30d"].dropna() if df is not None and not df.empty else None
                )
            except Exception:  # noqa: BLE001 — missing liquidity degrades, never aborts
                self._cache[ticker] = None
        series = self._cache[ticker]
        if series is None or series.empty:
            return None
        sliced = series.loc[: pd.Timestamp(as_of)]
        if sliced.empty:
            return None
        v = float(sliced.iloc[-1])
        return v if np.isfinite(v) and v > 0 else None

    def adv_contracts(self, ticker: str, as_of: date, ratio: float) -> float | None:
        shares = self.adv_shares(ticker, as_of)
        return None if shares is None else ratio * shares


# ---------------------------------------------------------------------------
# Impact overlay (pure decision logic; engine-native math)
# ---------------------------------------------------------------------------


def impact_per_share(
    mid: float, n_contracts: int, adv_contracts: float, *, k: float = IMPACT_K
) -> float:
    """The engine's own Almgren-Chriss sqrt size term, isolated.

    ``bid_ask_spread=0`` zeroes the spread component of
    `calculate_slippage`, leaving exactly ``k * mid * sqrt(N/adv)`` — the
    spread cost stays with the ``full`` friction overlay upstream, so the
    two compose without double counting.
    """
    from engine.transaction_costs import calculate_slippage

    return float(
        calculate_slippage(
            mid,
            0.0,
            "sell",
            num_contracts=n_contracts,
            adv_contracts=adv_contracts,
            impact_coefficient=k,
        )
    )


def decide_fill(
    premium_frictioned: float,
    premium_raw: float,
    n_contracts: int,
    adv_contracts: float | None,
    *,
    participation_cap: float = PARTICIPATION_CAP,
    k: float = IMPACT_K,
) -> tuple[str, float, float]:
    """Overlay decision for one option fill at true order size N.

    Returns ``(verdict, premium_net, impact_share)``:

    * ``"control"`` — overlay disabled upstream (``adv_contracts`` is the
      sentinel ``float('inf')``): premium passes through untouched.
    * ``"no_adv"`` — no ADV observation for this (ticker, date): the
      missing-evidence convention (Q3) — no impact applied, counted
      loudly so vacuity is visible.
    * ``"part_refused"`` — ``N > participation_cap * adv``: refused.
    * ``"priced_out"`` — impact consumes the entire frictioned premium.
    * ``"filled"`` — impact-net premium.
    """
    if adv_contracts is not None and np.isinf(adv_contracts):
        return "control", premium_frictioned, 0.0
    if adv_contracts is None or adv_contracts <= 0:
        return "no_adv", premium_frictioned, 0.0
    if n_contracts > participation_cap * adv_contracts:
        return "part_refused", 0.0, 0.0
    imp = impact_per_share(premium_raw, n_contracts, adv_contracts, k=k)
    net = premium_frictioned - imp
    if net <= 0.0:
        return "priced_out", 0.0, imp
    return "filled", net, imp


# ---------------------------------------------------------------------------
# The ladder driver (sibling of _common.run_backtest; shared rank)
# ---------------------------------------------------------------------------


@dataclass
class LadderPoint:
    """One (N, ratio) arm. ``ratio=None`` is the linearity control."""

    n_contracts: int
    ratio: float | None
    tracker: Any = None
    opens: int = 0
    cc_opens: int = 0
    bp_refused: int = 0
    part_refused: int = 0
    priced_out: int = 0
    no_adv_fills: int = 0
    impact_dollars_scaled: float = 0.0  # BASE-world dollars: imp/share x 100 x N
    gross_premium_scaled: float = 0.0  # BASE-world dollars: frictioned premium x 100 x N
    fills: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        r = "control" if self.ratio is None else f"{self.ratio:g}"
        return f"N{self.n_contracts}_r{r}"


def run_capacity_ladder(
    *,
    base_capital: float,
    tickers: Sequence[str],
    start: str,
    end: str,
    ladder: Sequence[int] = LADDER,
    ratios: Sequence[float] = PROXY_RATIOS,
    include_control: bool = True,
    top_n: int = 10,
    max_new_per_day: int = 3,
    dte_target: int = 35,
    delta_target: float = 0.25,
    participation_cap: float = PARTICIPATION_CAP,
    impact_k: float = IMPACT_K,
    progress: bool = True,
) -> dict[str, Any]:
    """Run the full (N x ratio [+ control]) grid behind one shared rank.

    Mirrors `_common.run_backtest`'s daily flow (MTM -> settle -> CC
    re-entry -> rank -> opens) with a tracker per grid point at
    ``capital = base_capital / N``, ``full`` friction throughout, and the
    impact overlay applied per point at true order size N.
    """
    from backtests.regression._common import (
        _forward_replay_realized_pnl,  # noqa: F401 — reserved for fill-level replay in analyze
        _next_business_day,
        _option_premium_rail_pinned_off,
        _spot_on_or_after,
        friction_adjusted_premium,
        friction_assignment_cost,
        friction_open_cost,
    )
    from engine.wheel_runner import WheelRunner
    from engine.wheel_tracker import PositionState, WheelTracker

    with _option_premium_rail_pinned_off():
        runner = WheelRunner()
        conn = runner.connector
    adv = AdvLookup(conn)

    points: list[LadderPoint] = []
    if include_control:
        points += [LadderPoint(n, None) for n in ladder]
    points += [LadderPoint(n, r) for r in ratios for n in ladder]
    for p in points:
        p.tracker = WheelTracker(initial_capital=base_capital / p.n_contracts, connector=conn)

    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    cc_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    t0 = time.time()

    for i, today in enumerate(trading_days):
        if progress and i and i % max(1, len(trading_days) // 20) == 0:
            el = time.time() - t0
            eta = (len(trading_days) - i) / (i / el)
            print(
                f"[capacity_curve] {i:4d}/{len(trading_days)} ({100 * i / len(trading_days):5.1f}%) "
                f"elapsed {el / 60:5.1f}m ETA {eta / 60:5.1f}m",
                flush=True,
            )
        # Shared spot fetches for the union of open tickers.
        open_union = {
            t
            for p in points
            for t, pos in p.tracker.positions.items()
            if pos.state != PositionState.NO_POSITION
        }
        spots_today: dict[str, float] = {}
        for t in open_union:
            spot = _spot_on_or_after(conn, t, today, max_lookahead_days=1)
            if spot is not None:
                spots_today[t] = spot

        for p in points:
            tracker = p.tracker
            prices = {
                t: s
                for t, s in spots_today.items()
                if t in tracker.positions
                and tracker.positions[t].state != PositionState.NO_POSITION
            }
            if prices:
                tracker.mark_to_market(today, prices)
            # Settle expirations (the _common flow, contracts=1 in the scaled book).
            for t in list(tracker.positions.keys()):
                pos = tracker.positions[t]
                if (
                    pos.state == PositionState.SHORT_PUT
                    and pos.put_expiration_date
                    and pos.put_expiration_date <= today
                ):
                    spot = prices.get(t) or _spot_on_or_after(conn, t, today)
                    if spot is None:
                        continue
                    was_assigned = spot < (pos.put_strike or 0.0)
                    tracker.handle_put_expiration(t, today, spot)
                    if was_assigned:
                        tracker.cash -= friction_assignment_cost(pos.put_strike or 0.0, 1, "full")
                elif (
                    pos.state == PositionState.COVERED_CALL
                    and pos.call_expiration_date
                    and pos.call_expiration_date <= today
                ):
                    spot = prices.get(t) or _spot_on_or_after(conn, t, today)
                    if spot is None:
                        continue
                    was_called = spot > (pos.call_strike or float("inf"))
                    tracker.handle_call_expiration(t, today, spot)
                    if was_called:
                        tracker.cash -= friction_assignment_cost(pos.call_strike or 0.0, 1, "full")

        # Covered-call re-entry — per-(ticker, day) rank shared across points.
        expiration_default = _next_business_day(today + timedelta(days=dte_target))
        for p in points:
            tracker = p.tracker
            for t in list(tracker.positions.keys()):
                if tracker.positions[t].state != PositionState.STOCK_OWNED:
                    continue
                ck = (t, today.isoformat())
                if ck not in cc_cache:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            cc_cache[ck] = runner.rank_covered_calls_by_ev(
                                ticker=t,
                                shares_held=100,
                                as_of=today.isoformat(),
                                target_dtes=(dte_target,),
                                target_deltas=(delta_target,),
                                top_n=5,
                                min_ev_dollars=-1e9,
                                include_diagnostic_fields=True,
                            )
                    except Exception:  # noqa: BLE001 — a bad CC rank skips, never aborts
                        cc_cache[ck] = None
                cc_frame = cc_cache[ck]
                if cc_frame is None or len(cc_frame) == 0:
                    continue
                for _, cc_row in cc_frame.iterrows():
                    if float(cc_row.get("ev_dollars", 0.0)) <= 0:
                        continue
                    cc_strike = float(cc_row.get("strike", 0.0))
                    cc_raw = float(cc_row.get("premium", 0.0))
                    cc_fric = friction_adjusted_premium(cc_raw, "full")
                    if cc_fric <= 0 or cc_strike <= 0:
                        continue
                    adv_c = (
                        float("inf") if p.ratio is None else adv.adv_contracts(t, today, p.ratio)
                    )
                    verdict, cc_net, imp = decide_fill(
                        cc_fric,
                        cc_raw,
                        p.n_contracts,
                        adv_c,
                        participation_cap=participation_cap,
                        k=impact_k,
                    )
                    if verdict == "part_refused":
                        p.part_refused += 1
                        continue
                    if verdict == "priced_out":
                        p.priced_out += 1
                        continue
                    if verdict == "no_adv":
                        p.no_adv_fills += 1
                    raw_expiry = cc_row.get("new_expiry")
                    if isinstance(raw_expiry, str):
                        cc_expiry = date.fromisoformat(raw_expiry[:10])
                    elif hasattr(raw_expiry, "date"):
                        cc_expiry = raw_expiry.date()
                    elif isinstance(raw_expiry, date):
                        cc_expiry = raw_expiry
                    else:
                        cc_expiry = expiration_default
                    if tracker.open_covered_call(
                        ticker=t,
                        strike=cc_strike,
                        premium=cc_net,
                        entry_date=today,
                        expiration_date=cc_expiry,
                        iv=float(cc_row.get("iv", 0.0)),
                    ):
                        tracker.cash -= friction_open_cost(1, "full")
                        p.cc_opens += 1
                        p.impact_dollars_scaled += imp * 100.0 * p.n_contracts
                        p.gross_premium_scaled += cc_fric * 100.0 * p.n_contracts
                        break

        # One shared SP rank.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frame = runner.rank_candidates_by_ev(
                    tickers=list(tickers),
                    dte_target=dte_target,
                    delta_target=delta_target,
                    contracts=1,
                    top_n=top_n,
                    min_ev_dollars=-1e9,
                    as_of=today.isoformat(),
                    include_diagnostic_fields=True,
                )
        except Exception:  # noqa: BLE001 — a bad day must not abort the grid
            continue
        if frame is None or len(frame) == 0:
            continue

        for p in points:
            tracker = p.tracker
            opens_today = 0
            for _, row in frame.iterrows():
                if opens_today >= max_new_per_day:
                    break
                if float(row.get("ev_dollars", 0.0)) <= 0:
                    continue
                t = str(row.get("ticker", ""))
                if (
                    t in tracker.positions
                    and tracker.positions[t].state != PositionState.NO_POSITION
                ):
                    continue
                strike = float(row.get("strike", 0.0))
                raw = float(row.get("premium", 0.0))
                fric = friction_adjusted_premium(raw, "full")
                if fric <= 0 or strike <= 0:
                    continue
                if tracker.available_buying_power() < strike * 100.0:
                    p.bp_refused += 1
                    continue
                adv_c = float("inf") if p.ratio is None else adv.adv_contracts(t, today, p.ratio)
                verdict, net, imp = decide_fill(
                    fric,
                    raw,
                    p.n_contracts,
                    adv_c,
                    participation_cap=participation_cap,
                    k=impact_k,
                )
                if verdict == "part_refused":
                    p.part_refused += 1
                    continue
                if verdict == "priced_out":
                    p.priced_out += 1
                    continue
                if verdict == "no_adv":
                    p.no_adv_fills += 1
                if tracker.open_short_put(
                    ticker=t,
                    strike=strike,
                    premium=net,
                    entry_date=today,
                    expiration_date=expiration_default,
                    iv=float(row.get("iv", 0.0)),
                ):
                    tracker.cash -= friction_open_cost(1, "full")
                    p.opens += 1
                    p.impact_dollars_scaled += imp * 100.0 * p.n_contracts
                    p.gross_premium_scaled += fric * 100.0 * p.n_contracts
                    p.fills.append(
                        {
                            "date": today.isoformat(),
                            "ticker": t,
                            "strike": strike,
                            "premium_net": net,
                            "impact_per_share": imp,
                            "n_contracts": p.n_contracts,
                            "ratio": p.ratio,
                        }
                    )
                    opens_today += 1

    return {"points": {p.key: _point_metrics(p, base_capital) for p in points}}


def _point_metrics(p: LadderPoint, base_capital: float) -> dict[str, Any]:
    tracker = p.tracker
    scaled_capital = base_capital / p.n_contracts
    curve = getattr(tracker, "equity_curve", []) or []
    navs = [float(e.get("portfolio_value", scaled_capital)) for e in curve]
    final_nav = navs[-1] if navs else scaled_capital
    mean_open_positions = (
        float(np.mean([float(e.get("num_positions", 0)) for e in curve])) if curve else 0.0
    )
    gross = p.gross_premium_scaled
    return {
        "n_contracts": p.n_contracts,
        "ratio": p.ratio,
        "final_nav_scaled": final_nav * p.n_contracts,
        "return_pct": (final_nav / scaled_capital - 1.0) * 100.0,
        "opens": p.opens,
        "cc_opens": p.cc_opens,
        "bp_refused": p.bp_refused,
        "part_refused": p.part_refused,
        "priced_out": p.priced_out,
        "no_adv_fills": p.no_adv_fills,
        "gross_premium_scaled": gross,
        "impact_dollars_scaled": p.impact_dollars_scaled,
        "impact_share_of_premium": (p.impact_dollars_scaled / gross) if gross > 0 else 0.0,
        "mean_open_positions": mean_open_positions,
        "n_fills_logged": len(p.fills),
    }


# ---------------------------------------------------------------------------
# Report helpers (pure)
# ---------------------------------------------------------------------------


def linearity_check(points: dict[str, dict], *, rel_tol: float = 1e-9) -> dict[str, Any]:
    """The pre-registered A/A (as corrected pre-run, plan section 7.2(1)).

    While no BP refusal has fired, a control point's ``return_pct`` must be
    exactly proportional to N (same 1-contract book on 1/N the capital) —
    so ``return_pct / N`` is constant across unthrottled control points,
    and any deviation from proportionality must coincide with a nonzero
    ``bp_refused`` count.  A deviation WITHOUT a BP refusal is a harness
    bug (FAIL); throttled deviations are the BP knee, reported not failed.
    """
    controls = sorted(
        (v for v in points.values() if v["ratio"] is None), key=lambda v: v["n_contracts"]
    )
    if len(controls) < 2:
        return {"n_controls": len(controls), "verdict": "INSUFFICIENT"}
    unthrottled = [v for v in controls if v["bp_refused"] == 0]
    per_n = [v["return_pct"] / v["n_contracts"] for v in unthrottled]
    spread = (max(per_n) - min(per_n)) if per_n else 0.0
    ref = max(abs(x) for x in per_n) if per_n else 1.0
    ok = len(unthrottled) < 2 or spread <= rel_tol * max(ref, 1e-12)
    return {
        "n_controls": len(controls),
        "n_unthrottled": len(unthrottled),
        "return_per_n_spread": float(spread),
        "throttled_points": {
            f"N{v['n_contracts']}": v["bp_refused"] for v in controls if v["bp_refused"] > 0
        },
        "verdict": "PASS" if ok else "FAIL",
    }


def knee_table(points: dict[str, dict]) -> list[dict[str, Any]]:
    """BASE-world return per (r, N) + the knee (argmax return) per r."""
    out: list[dict[str, Any]] = []
    ratios = sorted({v["ratio"] for v in points.values() if v["ratio"] is not None})
    for r in ratios:
        arm = sorted(
            (v for v in points.values() if v["ratio"] == r), key=lambda v: v["n_contracts"]
        )
        best = max(arm, key=lambda v: v["return_pct"])
        out.append(
            {
                "ratio": r,
                "per_n": {
                    str(v["n_contracts"]): {
                        "return_pct": v["return_pct"],
                        "impact_share": v["impact_share_of_premium"],
                        "part_refused": v["part_refused"],
                        "bp_refused": v["bp_refused"],
                        "opens": v["opens"],
                    }
                    for v in arm
                },
                "knee_n": best["n_contracts"],
            }
        )
    return out
