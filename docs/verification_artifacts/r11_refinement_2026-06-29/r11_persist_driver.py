"""R11 persistence-N study (research card: r11-onset-aware-trigger).

Extends the shipped r11_dollar_impact harness with persistence arms: fire R11
only when VIX>25 has held for >=N CONSECUTIVE trading days. Hypothesis (D23):
this keeps the 2022 sustained-grind protection while skipping the 2020 acute
spike (whose entries ride the V-recovery). All arms share ONE daily rank; only
the open-step gate differs. Read-only on the engine (gate replicated from the
engine's own R11 constants). No engine/ file modified.

Arms: suppressed (pre-R11), active (live R11: vix>25 & pp>0.90), and
persist{5,10,20} (active AND vix>25 held >=N consecutive trading days).

Usage:
  py -3.12 r11_persist_driver.py --start 2020-01-02 --end 2024-12-31 --out-dir <d>
  py -3.12 r11_persist_driver.py --analyze --out-dir <d>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

WORKTREE = (
    Path(__file__).resolve().parents[3]
)  # repo root (docs/verification_artifacts/<dir>/<file>)
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass
warnings.filterwarnings("ignore")

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
from engine.candidate_dossier import (  # noqa: E402
    MIN_PROCEED_EV_DOLLARS,
    R11_TOP_BIN_PROB,
    R11_VIX_THRESHOLD,
)
from engine.wheel_runner import WheelRunner  # noqa: E402
from engine.wheel_tracker import PositionState, WheelTracker  # noqa: E402

# arm -> required consecutive-run-length of VIX>25 (0 = base R11 level trigger)
PERSIST_N = {"persist5": 5, "persist10": 10, "persist20": 20}
ARMS = ("suppressed", "active", "persist5", "persist10", "persist20")


def r11_base_fires(ev_dollars: float, prob_profit: float, vix_level: float | None) -> bool:
    if vix_level is None or ev_dollars <= MIN_PROCEED_EV_DOLLARS:
        return False
    try:
        return float(prob_profit or 0.0) > R11_TOP_BIN_PROB and float(vix_level) > R11_VIX_THRESHOLD
    except (TypeError, ValueError):
        return False


def regime_of(d: date) -> str:
    if date(2020, 2, 15) <= d <= date(2020, 4, 30):
        return "crash_2020"
    if date(2021, 1, 1) <= d <= date(2021, 12, 31):
        return "bull_2021"
    if date(2022, 1, 1) <= d <= date(2022, 12, 31):
        return "bear_2022"
    return "calm"


def vix_bucket(vix: float | None) -> str:
    if vix is None or not np.isfinite(vix):
        return "unknown"
    if vix <= 15:
        return "vix<=15"
    if vix <= 25:
        return "15-25"
    if vix <= 35:
        return "25-35"
    return "vix>35"


def arm_blocks(arm: str, *, ev: float, pp: float, vix: float | None, vix_run: int) -> bool:
    """Does this arm's gate block (size down) the candidate?"""
    if arm == "suppressed":
        return False
    base = r11_base_fires(ev, pp, vix)
    if arm == "active":
        return base
    return base and vix_run >= PERSIST_N[arm]


def _try_opens(
    *,
    tracker,
    frame,
    today,
    vix_today,
    vix_run,
    friction_level,
    max_new_per_day,
    contracts,
    expiration_default,
    arm,
    attempts_log,
    blocked_log,
):
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
        if arm_blocks(arm, ev=ev_dollars, pp=pp, vix=vix_today, vix_run=vix_run):
            blocked_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "vix": vix_today,
                    "vix_run": vix_run,
                    "strike": strike,
                    "premium": premium,
                    "expiration_date": expiration_default.isoformat(),
                    "regime": regime_of(today),
                    "vix_bucket": vix_bucket(vix_today),
                }
            )
            attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "outcome": "r11_blocked",
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "vix": vix_today,
                }
            )
            continue
        if tracker.available_buying_power() < strike * 100 * contracts:
            attempts_log.append(
                {
                    "date": today.isoformat(),
                    "ticker": t,
                    "outcome": "insufficient_bp",
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "vix": vix_today,
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
                    "outcome": "opened",
                    "ev_dollars": ev_dollars,
                    "prob_profit": pp,
                    "vix": vix_today,
                }
            )
            opens_today += 1


def _nav(tr) -> float:
    try:
        return (
            float(tr.equity_curve[-1].get("portfolio_value", tr.cash))
            if tr.equity_curve
            else float(tr.cash)
        )
    except Exception:
        return float("nan")


def _sharpe(nav: pd.Series) -> float:
    nav = nav.dropna()
    if len(nav) < 3:
        return float("nan")
    rets = nav.pct_change().dropna()
    if len(rets) < 2 or rets.std(ddof=1) == 0:
        return float("nan")
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))


def run(
    *,
    capital,
    tickers,
    start,
    end,
    out_dir,
    friction_level="full",
    top_n=10,
    max_new_per_day=3,
    dte_target=35,
    delta_target=0.25,
    contracts=1,
):
    assert_data_window_available(start, end)
    runner = WheelRunner()
    conn = runner.connector
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[persist] connector={type(conn).__name__} arms={ARMS} window={start}->{end} "
        f"universe={len(tickers)}t out={out_dir}",
        flush=True,
    )

    trackers = {
        a: WheelTracker(initial_capital=capital, connector=conn, require_ev_authority=False)
        for a in ARMS
    }
    attempts = {a: [] for a in ARMS}
    blocked = {a: [] for a in ARMS}
    daily_nav = []
    vix_cache: dict[str, float | None] = {}
    vix_run = 0  # consecutive trading days with VIX>25 (incl today)

    trading_days = [d.date() for d in pd.bdate_range(start, end)]
    tickers = list(tickers)
    _t0 = time.time()
    _total = len(trading_days)
    _every = max(1, _total // 30)

    for _i, today in enumerate(trading_days):
        if _i > 0 and _i % _every == 0:
            el = time.time() - _t0
            rate = _i / el if el > 0 else 0
            navs = " ".join(f"{a[:4]}=${_nav(trackers[a]):,.0f}" for a in ARMS)
            print(
                f"[persist] {_i:4d}/{_total} ({100 * _i / _total:4.1f}%) {el / 60:5.1f}min "
                f"ETA {((_total - _i) / rate / 60 if rate else 0):5.1f}min  {navs}  "
                f"blk_act={len(blocked['active'])} blk_p10={len(blocked['persist10'])}",
                flush=True,
            )
            # checkpoint
            try:
                pd.DataFrame(daily_nav).to_csv(out_dir / "daily_nav.csv", index=False)
            except Exception:
                pass

        expiration_default = _next_business_day(today + timedelta(days=dte_target))
        as_of = today.isoformat()
        if as_of not in vix_cache:
            try:
                _v = conn.get_vix_regime(as_of).get("vix")
                vix_cache[as_of] = float(_v) if _v is not None and np.isfinite(_v) else None
            except Exception:
                vix_cache[as_of] = None
        vix_today = vix_cache[as_of]
        # update consecutive run-length of VIX>25
        if vix_today is not None and vix_today > R11_VIX_THRESHOLD:
            vix_run += 1
        else:
            vix_run = 0

        for a in ARMS:
            _tracker_step(
                tracker=trackers[a],
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
            print(f"[persist] rank failed {today}: {exc}", flush=True)
            continue
        if frame is None or len(frame) == 0:
            continue

        for a in ARMS:
            _try_opens(
                tracker=trackers[a],
                frame=frame,
                today=today,
                vix_today=vix_today,
                vix_run=vix_run,
                friction_level=friction_level,
                max_new_per_day=max_new_per_day,
                contracts=contracts,
                expiration_default=expiration_default,
                arm=a,
                attempts_log=attempts[a],
                blocked_log=blocked[a],
            )

        row = {"date": as_of, "vix": vix_today, "vix_run": vix_run}
        for a in ARMS:
            row[f"nav_{a}"] = _nav(trackers[a])
        daily_nav.append(row)

    # forward-replay blocked sets (counterfactual)
    spot_cache: dict[tuple, float | None] = {}

    def _replay(rows):
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        rz, asg = [], []
        for r in df.itertuples(index=False):
            key = (r.ticker, r.expiration_date)
            if key not in spot_cache:
                spot_cache[key] = _spot_on_or_after(
                    conn, r.ticker, date.fromisoformat(r.expiration_date)
                )
            spot = spot_cache[key]
            if spot is None:
                rz.append(float("nan"))
                asg.append(None)
            else:
                rz.append(_forward_replay_realized_pnl(r.strike, r.premium, spot))
                asg.append(bool(spot < r.strike))
        df["realized_pnl"] = rz
        df["assigned"] = asg
        return df

    nav_df = pd.DataFrame(daily_nav)
    nav_df.to_csv(out_dir / "daily_nav.csv", index=False)

    summary: dict[str, Any] = {
        "setup": {
            "capital": capital,
            "universe_size": len(tickers),
            "start": start,
            "end": end,
            "friction_level": friction_level,
            "r11_vix_threshold": R11_VIX_THRESHOLD,
            "r11_top_bin_prob": R11_TOP_BIN_PROB,
            "persist_N": PERSIST_N,
            "ohlcv_csv_sha256": ohlcv_sha256(),
        },
        "arms": {},
        "blocked": {},
    }

    supp_nav = _nav(trackers["suppressed"])
    for a in ARMS:
        tr = trackers[a]
        fn = _nav(tr)
        bl = _replay(blocked[a])
        blv = bl.dropna(subset=["realized_pnl"]) if not bl.empty else bl
        by_regime = {}
        if not blv.empty:
            for k, g in blv.groupby("regime"):
                by_regime[str(k)] = {
                    "n": int(len(g)),
                    "realized_total": float(g["realized_pnl"].sum()),
                    "realized_mean": float(g["realized_pnl"].mean()),
                    "assignment_rate": float(g["assigned"].mean()),
                }
        df = pd.DataFrame(attempts[a])
        summary["arms"][a] = {
            "final_nav": fn,
            "return_pct": (fn - capital) / capital,
            "delta_nav_vs_suppressed": fn - supp_nav,
            "sharpe": _sharpe(nav_df[f"nav_{a}"]) if f"nav_{a}" in nav_df else float("nan"),
            "n_opened": int((df["outcome"] == "opened").sum()) if not df.empty else 0,
            "n_blocked": int((df["outcome"] == "r11_blocked").sum()) if not df.empty else 0,
            "blocked_realized_total": float(blv["realized_pnl"].sum()) if not blv.empty else 0.0,
            "blocked_realized_mean": float(blv["realized_pnl"].mean()) if not blv.empty else None,
            "blocked_by_regime": by_regime,
        }
        if not bl.empty:
            bl.to_csv(out_dir / f"blocked_{a}.csv", index=False)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print("\n[persist] DONE", flush=True)
    for a in ARMS:
        m = summary["arms"][a]
        print(
            f"  {a:<11} NAV=${m['final_nav']:,.0f} Δvs_supp=${m['delta_nav_vs_suppressed']:+,.0f} "
            f"Sharpe={m['sharpe']:.3f} opened={m['n_opened']} blocked={m['n_blocked']} "
            f"blk_realized=${m['blocked_realized_total']:+,.0f}",
            flush=True,
        )
    return summary


def analyze(out_dir: Path):
    s = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    setup = s["setup"]
    print("=" * 80)
    print(
        f"R11 PERSISTENCE-N — {out_dir.name}  {setup['start']}->{setup['end']} "
        f"{setup['universe_size']}t {setup['friction_level']}"
    )
    print("=" * 80)
    print(
        f"{'arm':<11} {'final_nav':>14} {'Δvs_supp':>12} {'Sharpe':>8} {'opened':>7} "
        f"{'blocked':>8} {'blk_realized':>13}"
    )
    for a in ARMS:
        m = s["arms"][a]
        print(
            f"{a:<11} ${m['final_nav']:>13,.0f} ${m['delta_nav_vs_suppressed']:>+11,.0f} "
            f"{m['sharpe']:>8.3f} {m['n_opened']:>7} {m['n_blocked']:>8} "
            f"${m['blocked_realized_total']:>+12,.0f}"
        )
    print(
        "\nBlocked-set counterfactual by regime (negative total = R11 AVERTED loss; "
        "positive = FORGONE gain):"
    )
    for a in ("active", "persist5", "persist10", "persist20"):
        print(f"\n  {a}:")
        for reg, mm in sorted(s["arms"][a]["blocked_by_regime"].items()):
            print(
                f"    {reg:<12} n={mm['n']:>4} realized=${mm['realized_total']:>+11,.0f} "
                f"mean=${mm['realized_mean']:>+9,.2f} assign={100 * mm['assignment_rate']:.0f}%"
            )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-02")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--friction", default="full")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir).resolve()
    if a.analyze:
        analyze(out)
        return 0
    run(
        capital=a.capital,
        tickers=UNIVERSE_100,
        start=a.start,
        end=a.end,
        out_dir=out,
        friction_level=a.friction,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
