"""I7 — Roll-management economics: does suggest_rolls beat hold-to-assignment?

CAPITAL-TRUST QUESTION
----------------------
The wheel is a MANAGEMENT strategy. When a short put goes CHALLENGED
(underlying at/below strike near expiry), does following
``engine.wheel_tracker.WheelTracker.suggest_rolls`` produce a better
realized forward outcome than simply holding to assignment?

This driver is OBSERVE-AND-DOCUMENT ONLY. It does not modify engine/.

METHODOLOGY (PIT-correct, apples-to-apples)
-------------------------------------------
Population
    Monthly full-universe ranked 25-delta / 35-DTE put snapshots
    (snapshots/put_*.parquet). Each row is a put the ENTRY ranker would
    have surfaced at ``as_of``.

Roll-decision date
    ROLL_DATE = entry as_of + 28 calendar days (~7 trading days before
    the 35-DTE expiry). A put is CHALLENGED iff the PIT underlying close
    on/after ROLL_DATE is at/below the strike (the short put is ITM).

PIT state at the roll date
    spot   = OHLCV close at/after ROLL_DATE (campaign_lib.close_series)
    iv     = hist_put_imp_vol / 100 from sp500_vol_iv_full.csv, asof
             ROLL_DATE (PIT — no look-ahead)
    rf     = campaign_lib.risk_free_for(ROLL_DATE)

Two realized forward outcomes, BOTH measured from the roll moment forward
(the already-collected entry premium is sunk and identical in both arms,
so it is excluded — this is exactly how suggest_rolls defines hold_ev /
roll_ev: "expected dollar change in account value from this decision
moment forward").

(a) HOLD to assignment — keep the original put to its own expiry
    (entry + 35d). Forward P&L from the roll moment:
        hold_realized = -intrinsic_old_at_old_expiry * 100
    (you collect no new premium; you owe the terminal intrinsic).
    Equivalently realize_short_put(strike, entry, 35, premium=0).

(b) ROLL per suggest_rolls — buy back the old put at the roll moment,
    sell the engine's TOP-RANKED new put, realize it to ITS expiry:
        roll_realized = (new_premium - new_intrinsic_at_new_expiry) * 100
                        - buyback_cost_of_old_put
    where buyback_cost = BSM value of the old put at the roll moment
    (T = remaining DTE) * 100 (the per-share buyback_cost column * 100;
    transaction costs excluded for a clean BSM-vs-BSM comparison, stated).
    The new put strike/dte/premium come straight from suggest_rolls
    (which prices the new put at BSM fair value and ranks via
    EVEngine.evaluate). We realize the new put against the actual
    underlying path with campaign_lib.realize_short_put.

Comparison
    Per put: roll_realized - hold_realized (positive = roll won),
    win/assignment-avoidance rates, paired t / bootstrap CI, and a
    breakdown by HMM regime, by the engine's ``recommend`` flag, and by
    moneyness at the roll date. Adversarial check: does the roll reduce
    realized loss or merely DEFER it (compare the new put's terminal
    moneyness — a roll that ends ITM again has just postponed assignment)?
"""

from __future__ import annotations

import os
import sys
import glob
import math
from datetime import date, datetime, timedelta

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # so `import campaign_lib` resolves; it inserts ROOT itself
import campaign_lib as L  # noqa: E402  (inserts ROOT on sys.path on import)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.wheel_tracker import WheelTracker  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

RNG = np.random.default_rng(7)

# --------------------------------------------------------------------------- #
# PIT IV lookup (asof, no look-ahead) straight from the vol CSV
# --------------------------------------------------------------------------- #
_IV_RAW = pd.read_csv(L.IV_CSV)
_IV_RAW["t"] = _IV_RAW["ticker"].str.split(" ").str[0].str.upper()
_IV_RAW["date"] = pd.to_datetime(_IV_RAW["date"])
_IV_BY_T: dict[str, pd.Series] = {}
for t, g in _IV_RAW.groupby("t"):
    _IV_BY_T[t] = g.set_index("date")["hist_put_imp_vol"].sort_index()


def pit_put_iv(ticker: str, as_of) -> float | None:
    """hist_put_imp_vol/100 asof as_of (decimal). None if unavailable."""
    s = _IV_BY_T.get(L._norm(ticker))
    if s is None or s.empty:
        return None
    v = s.asof(pd.Timestamp(as_of))
    if pd.isna(v):
        return None
    return float(v) / 100.0


def pit_spot_on_or_after(ticker: str, as_of) -> tuple[float, pd.Timestamp] | None:
    """First OHLCV close on/after as_of (PIT spot at the roll date)."""
    s = L.close_series(ticker)
    if s.empty:
        return None
    fwd = s[s.index >= pd.Timestamp(as_of)]
    if fwd.empty:
        return None
    return float(fwd.iloc[0]), fwd.index[0]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ENTRY_DTE = 35           # snapshot puts are 35-DTE
ROLL_OFFSET_DAYS = 28    # roll decision ~7 trading days before expiry
DIV_YLD = 0.0            # match suggest_rolls default; price-return realization


def regime_for(as_of_date: date) -> str:
    for name, (s, e) in L.REGIMES.items():
        if pd.Timestamp(s).date() <= as_of_date <= pd.Timestamp(e).date():
            return name
    return "other"


def moneyness_bucket(spot: float, strike: float) -> str:
    """How deep ITM the short put is at the roll date (spot/strike)."""
    r = spot / strike
    if r >= 0.99:
        return "atm_0to1pct"
    if r >= 0.95:
        return "itm_1to5pct"
    if r >= 0.90:
        return "itm_5to10pct"
    return "itm_gt10pct"


# --------------------------------------------------------------------------- #
# Core per-put evaluation
# --------------------------------------------------------------------------- #
# Module-level shared connector (created ONCE) — the original built a fresh
# WheelRunner() per challenged put, reloading the connector each time (the real
# bottleneck). The connector is read-only data access; reusing it is safe and the
# per-put WheelTracker is still fresh. STRIDE samples 1-in-N challenged puts
# (unbiased across the chronological scan) so the suggest_rolls grid runs a few
# hundred times instead of thousands. I7_STRIDE=1 reproduces the full population.
_RUNNER = WheelRunner()
_CONN = _RUNNER.connector
STRIDE = int(os.environ.get("I7_STRIDE", "6"))
_CH_SEEN = 0


def evaluate_put(row) -> dict | None:
    global _CH_SEEN
    ticker = L._norm(row["ticker"])
    strike = float(row["strike"])
    entry_premium = float(row["premium"])
    as_of = pd.Timestamp(row["as_of"]).date()
    entry_iv = float(row["iv"])
    hmm = row.get("hmm_regime", "unknown")

    roll_date = as_of + timedelta(days=ROLL_OFFSET_DAYS)

    # PIT spot at the roll date
    spot_info = pit_spot_on_or_after(ticker, roll_date)
    if spot_info is None:
        return None
    spot, spot_dt = spot_info

    # CHALLENGED filter: short put ITM at the roll date
    if spot > strike:
        return None  # not challenged

    # Unbiased stride-sample of challenged puts (keeps the suggest_rolls grid cost bounded)
    _CH_SEEN += 1
    if _CH_SEEN % STRIDE != 0:
        return None

    # PIT IV at the roll date
    iv = pit_put_iv(ticker, roll_date)
    if iv is None or not (0.0 < iv <= 3.0):
        return None
    rf = L.risk_free_for(roll_date)

    # Remaining DTE of the original put at the roll date
    expiry_date = as_of + timedelta(days=ENTRY_DTE)
    dte_remaining = (expiry_date - roll_date).days
    if dte_remaining <= 0:
        return None

    # ---------- HOLD outcome: original put to its expiry ----------
    hold = L.realize_short_put(ticker, strike, as_of, ENTRY_DTE, premium=0.0)
    if hold is None:
        return None
    hold_realized = hold["realized_pnl_synth"]  # = -intrinsic_old * 100
    old_assigned = hold["assigned"]
    old_intrinsic = hold["intrinsic"]
    old_expiry_close = hold["realized_close"]

    # ---------- ROLL outcome via suggest_rolls ----------
    tracker = WheelTracker(connector=_CONN)
    ok = tracker.open_short_put(
        ticker=ticker,
        strike=strike,
        premium=entry_premium,
        entry_date=as_of,
        expiration_date=expiry_date,
        iv=entry_iv,
    )
    if not ok:
        return None
    try:
        rolls = tracker.suggest_rolls(
            ticker=ticker,
            as_of=roll_date,
            current_spot=spot,
            current_iv=iv,
            risk_free_rate=rf,
            dividend_yield=DIV_YLD,
        )
        # Permissive arm: allow rescue debit rolls so we can see what the
        # engine's roll MECHANICS produce when forced to act (isolates the
        # value of the engine's refusal/discipline from the roll itself).
        rolls_resc = tracker.suggest_rolls(
            ticker=ticker,
            as_of=roll_date,
            current_spot=spot,
            current_iv=iv,
            risk_free_rate=rf,
            dividend_yield=DIV_YLD,
            min_net_credit=-1e9,
        )
    except Exception as ex:  # noqa: BLE001
        return {"ticker": ticker, "as_of": as_of.isoformat(),
                "error": f"{type(ex).__name__}: {ex}"}

    n_candidates = len(rolls)

    # ----- permissive (rescue-allowed) realized outcome -----
    resc_realized = None
    resc_recommend = None
    resc_new_assigned = None
    if len(rolls_resc) > 0:
        rt = rolls_resc.iloc[0]
        r_new_strike = float(rt["new_strike"])
        r_new_dte = int(rt["new_dte"])
        r_new_premium = float(rt["new_premium"])
        r_buyback = float(rt["buyback_cost"]) * 100.0
        rr = L.realize_short_put(ticker, r_new_strike, roll_date, r_new_dte, premium=r_new_premium)
        if rr is not None:
            resc_realized = rr["realized_pnl_synth"] - r_buyback
            resc_recommend = bool(rt["recommend"])
            resc_new_assigned = rr["assigned"]
    buyback_per_share = None
    hold_ev_engine = None
    if n_candidates > 0:
        hold_ev_engine = float(rolls.iloc[0]["hold_ev"])
        buyback_per_share = float(rolls.iloc[0]["buyback_cost"])

    # No surviving credit-roll candidate -> engine offers no roll; this is
    # itself a finding (rolling not actionable). Record and move on.
    if n_candidates == 0:
        return {
            "ticker": ticker, "as_of": as_of.isoformat(), "roll_date": roll_date.isoformat(),
            "regime": regime_for(as_of), "hmm_regime": hmm,
            "moneyness": moneyness_bucket(spot, strike),
            "strike": strike, "spot_at_roll": spot, "spot_over_strike": spot / strike,
            "dte_remaining": dte_remaining, "iv_roll": iv,
            "hold_realized": hold_realized, "old_assigned": old_assigned,
            "old_intrinsic": old_intrinsic, "old_expiry_close": old_expiry_close,
            "roll_offered": False, "recommend": None,
            "roll_realized": None, "delta_roll_minus_hold": None,
            "new_strike": None, "new_dte": None, "new_premium": None,
            "buyback_per_share": None, "roll_ev": None, "hold_ev": None,
            "new_assigned": None, "new_expiry_close": None,
            "n_candidates": 0,
            "resc_realized": resc_realized, "resc_recommend": resc_recommend,
            "resc_new_assigned": resc_new_assigned,
            "resc_delta": (None if resc_realized is None else resc_realized - hold_realized),
        }

    # Engine's headline pick = top row (sorted by roll_ev desc).
    top = rolls.iloc[0]
    new_strike = float(top["new_strike"])
    new_dte = int(top["new_dte"])
    new_premium = float(top["new_premium"])
    roll_ev = float(top["roll_ev"])
    recommend = bool(top["recommend"])
    new_prob_otm = float(top["prob_otm"])

    # Realize the NEW put against the actual path, from the roll date.
    new_real = L.realize_short_put(ticker, new_strike, roll_date, new_dte, premium=new_premium)
    if new_real is None:
        return None  # new put expiry beyond data end
    new_put_pnl = new_real["realized_pnl_synth"]  # (new_premium - new_intrinsic)*100
    new_assigned = new_real["assigned"]
    new_expiry_close = new_real["realized_close"]

    # Buyback cost of the OLD put at the roll moment (BSM principal * 100;
    # txn costs excluded for clean BSM-vs-BSM — same basis as new_put_pnl).
    buyback_cost = (buyback_per_share or 0.0) * 100.0
    roll_realized = new_put_pnl - buyback_cost

    return {
        "ticker": ticker, "as_of": as_of.isoformat(), "roll_date": roll_date.isoformat(),
        "regime": regime_for(as_of), "hmm_regime": hmm,
        "moneyness": moneyness_bucket(spot, strike),
        "strike": strike, "spot_at_roll": spot, "spot_over_strike": spot / strike,
        "dte_remaining": dte_remaining, "iv_roll": iv,
        "hold_realized": hold_realized, "old_assigned": old_assigned,
        "old_intrinsic": old_intrinsic, "old_expiry_close": old_expiry_close,
        "roll_offered": True, "recommend": recommend,
        "roll_realized": roll_realized, "delta_roll_minus_hold": roll_realized - hold_realized,
        "new_strike": new_strike, "new_dte": new_dte, "new_premium": new_premium,
        "buyback_per_share": buyback_per_share, "buyback_cost": buyback_cost,
        "new_put_pnl": new_put_pnl,
        "roll_ev": roll_ev, "hold_ev": hold_ev_engine, "new_prob_otm": new_prob_otm,
        "new_assigned": new_assigned, "new_expiry_close": new_expiry_close,
        "n_candidates": n_candidates,
        "resc_realized": resc_realized, "resc_recommend": resc_recommend,
        "resc_new_assigned": resc_new_assigned,
        "resc_delta": (None if resc_realized is None else resc_realized - hold_realized),
    }


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def boot_mean_ci(x, n_boot=5000, alpha=0.05):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return (None, None, None, 0)
    if len(x) == 1:
        return (float(x[0]), None, None, 1)
    means = np.array([RNG.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(x.mean()), float(lo), float(hi), len(x))


def paired_summary(rows):
    """Paired hold vs roll over puts where a roll was offered."""
    paired = [r for r in rows if r.get("roll_offered") and r.get("roll_realized") is not None]
    if not paired:
        return None
    d = np.array([r["delta_roll_minus_hold"] for r in paired], dtype=float)
    hold = np.array([r["hold_realized"] for r in paired], dtype=float)
    roll = np.array([r["roll_realized"] for r in paired], dtype=float)
    m, lo, hi, n = boot_mean_ci(d)
    win = float((d > 0).mean())
    # paired t
    if n > 1 and d.std(ddof=1) > 0:
        tstat = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    else:
        tstat = float("nan")
    return {
        "n": n, "mean_hold": float(hold.mean()), "mean_roll": float(roll.mean()),
        "mean_delta": m, "delta_ci": (lo, hi), "roll_win_rate": win, "paired_t": tstat,
        "median_delta": float(np.median(d)),
    }


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "  n/a "
    return f"{v:>7.{nd}f}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("I7 — ROLL-MANAGEMENT ECONOMICS: suggest_rolls vs hold-to-assignment")
    print("=" * 78)
    print(f"ROOT={L.ROOT}")
    print(f"provider={os.environ.get('SWE_DATA_PROVIDER', '<unset>')}  "
          f"connector={type(WheelRunner().connector).__name__}")
    print(f"ENTRY_DTE={ENTRY_DTE}  ROLL_OFFSET_DAYS={ROLL_OFFSET_DAYS}  div_yld={DIV_YLD}")
    print("CONVENTION: both arms measure forward P&L from the roll moment "
          "(entry premium is sunk, excluded in both).")
    print("  HOLD = -intrinsic_old_at_expiry*100")
    print("  ROLL = (new_premium - new_intrinsic_at_new_expiry)*100 - buyback_cost(old, BSM*100)")
    print()

    snaps = sorted(glob.glob(os.path.join(HERE, "snapshots", "put_*.parquet")))
    print(f"snapshots found: {len(snaps)}")

    all_rows = []
    n_scanned = 0
    n_challenged = 0
    errors = []
    for sp in snaps:
        df = pd.read_parquet(sp)
        for _, row in df.iterrows():
            n_scanned += 1
            res = evaluate_put(row)
            if res is None:
                continue
            if "error" in res:
                errors.append(res)
                continue
            n_challenged += 1
            all_rows.append(res)

    print(f"puts scanned: {n_scanned}")
    print(f"CHALLENGED puts (ITM at roll date, valid PIT data): {n_challenged}")
    print(f"errors from suggest_rolls: {len(errors)}")
    for e in errors[:10]:
        print("   ERR", e["ticker"], e["as_of"], e["error"])
    print()

    if not all_rows:
        print("NO CHALLENGED PUTS — cannot evaluate. STOP.")
        return

    rdf = pd.DataFrame(all_rows)

    # --- roll availability ---
    n_offered = int(rdf["roll_offered"].sum())
    print("-" * 78)
    print("ROLL AVAILABILITY")
    print("-" * 78)
    print(f"challenged puts: {len(rdf)}")
    print(f"  engine offered >=1 credit-roll candidate: {n_offered} "
          f"({100*n_offered/len(rdf):.1f}%)")
    print(f"  engine offered NO credit-roll (default min_net_credit=0): "
          f"{len(rdf)-n_offered} ({100*(len(rdf)-n_offered)/len(rdf):.1f}%)")
    print()

    # --- headline paired comparison ---
    print("-" * 78)
    print("HEADLINE: HOLD vs ROLL (paired, over puts where a roll was offered)")
    print("-" * 78)
    ps = paired_summary(all_rows)
    if ps is None:
        print("No paired observations (no roll offered on any challenged put).")
    else:
        ci = ps["delta_ci"]
        print(f"n paired                 : {ps['n']}")
        print(f"mean HOLD realized ($)   : {fmt(ps['mean_hold'])}")
        print(f"mean ROLL realized ($)   : {fmt(ps['mean_roll'])}")
        print(f"mean (ROLL - HOLD) ($)   : {fmt(ps['mean_delta'])}  "
              f"95% CI [{fmt(ci[0])}, {fmt(ci[1])}]")
        print(f"median (ROLL - HOLD) ($) : {fmt(ps['median_delta'])}")
        print(f"roll-wins rate           : {ps['roll_win_rate']*100:.1f}%")
        print(f"paired t-stat            : {fmt(ps['paired_t'], 2)}")
        verdict = "ROLL ADDS VALUE" if (ci[0] is not None and ci[0] > 0) else (
            "ROLL DESTROYS VALUE" if (ci[1] is not None and ci[1] < 0) else
            "INCONCLUSIVE (CI straddles 0)")
        print(f"VERDICT (headline)       : {verdict}")
    print()

    # --- permissive (rescue-allowed) arm over ALL challenged puts ---
    print("-" * 78)
    print("PERMISSIVE ARM: min_net_credit=-1e9 (rescue debit rolls ALLOWED)")
    print("  Isolates roll MECHANICS (engine forced to act on every challenged")
    print("  put) from the engine's default discipline (refusing to roll).")
    print("-" * 78)
    resc = rdf[rdf["resc_realized"].notna()].copy()
    print(f"challenged puts where a rescue-roll was produced: {len(resc)} / {len(rdf)}")
    if len(resc):
        d = resc["resc_delta"].values
        m, lo, hi, n = boot_mean_ci(d)
        print(f"  mean HOLD realized ($)   : {fmt(resc['hold_realized'].mean())}")
        print(f"  mean RESCUE-ROLL ($)     : {fmt(resc['resc_realized'].mean())}")
        print(f"  mean (RESCUE - HOLD) ($) : {fmt(m)}  95% CI [{fmt(lo)}, {fmt(hi)}]")
        print(f"  median (RESCUE - HOLD)   : {fmt(float(np.median(d)))}")
        print(f"  rescue-wins rate         : {100*(d>0).mean():.1f}%")
        rverdict = "RESCUE-ROLL ADDS VALUE" if (lo is not None and lo > 0) else (
            "RESCUE-ROLL DESTROYS VALUE" if (hi is not None and hi < 0) else
            "INCONCLUSIVE (CI straddles 0)")
        print(f"  VERDICT (mechanics)      : {rverdict}")
        # deferral: does the rescue put just end ITM again?
        ra = resc["resc_new_assigned"].mean()
        print(f"  rescue NEW put ends ITM again (deferred): {ra*100:.1f}%")
    print()

    # --- assignment-avoidance ---
    print("-" * 78)
    print("ASSIGNMENT / DEFERRAL ANALYSIS  (adversarial: roll defers vs reduces?)")
    print("-" * 78)
    off = rdf[rdf["roll_offered"]].copy()
    if len(off):
        old_assigned = off["old_assigned"].mean()
        new_assigned = off["new_assigned"].mean()
        print(f"of {len(off)} rolled puts:")
        print(f"  HOLD would be ASSIGNED at old expiry : {old_assigned*100:.1f}%")
        print(f"  ROLL's NEW put ALSO ends ITM (deferred, not avoided): "
              f"{new_assigned*100:.1f}%")
        # of the rolls that avoided the immediate assignment, how many just re-challenged?
        print(f"  -> roll AVOIDED terminal assignment in: "
              f"{(1-new_assigned)*100:.1f}% of rolled puts")
        # mean buyback cost paid (the real cash drag of rolling)
        print(f"  mean buyback cost paid at roll moment : "
              f"${off['buyback_cost'].mean():.2f}")
    print()

    # --- is `recommend` predictive? ---
    print("-" * 78)
    print("IS THE ENGINE'S `recommend` FLAG PREDICTIVE OF WHEN TO ROLL?")
    print("-" * 78)
    if len(off):
        for flag in [True, False]:
            sub = off[off["recommend"] == flag]
            if len(sub) == 0:
                print(f"  recommend={flag}: n=0")
                continue
            d = sub["delta_roll_minus_hold"].values
            m, lo, hi, n = boot_mean_ci(d)
            print(f"  recommend={str(flag):5s}: n={n:3d}  "
                  f"mean(ROLL-HOLD)={fmt(m)}  95%CI[{fmt(lo)},{fmt(hi)}]  "
                  f"win={100*(d>0).mean():.1f}%")
    print()

    # --- by regime ---
    print("-" * 78)
    print("BY CALENDAR REGIME (paired ROLL-HOLD over offered rolls)")
    print("-" * 78)
    print(f"  {'regime':<22s} {'n':>4s} {'meanHOLD':>9s} {'meanROLL':>9s} "
          f"{'ROLL-HOLD':>10s} {'CI_lo':>8s} {'CI_hi':>8s} {'win%':>6s}")
    for reg in list(L.REGIMES.keys()) + ["other"]:
        sub = off[off["regime"] == reg]
        if len(sub) == 0:
            continue
        d = sub["delta_roll_minus_hold"].values
        m, lo, hi, n = boot_mean_ci(d)
        print(f"  {reg:<22s} {n:>4d} {fmt(sub['hold_realized'].mean())} "
              f"{fmt(sub['roll_realized'].mean())} {fmt(m)} {fmt(lo)} {fmt(hi)} "
              f"{100*(d>0).mean():>5.1f}")
    print()

    # --- by moneyness at roll ---
    print("-" * 78)
    print("BY MONEYNESS AT ROLL DATE (how deep ITM the short put is)")
    print("-" * 78)
    print(f"  {'bucket':<16s} {'n':>4s} {'meanHOLD':>9s} {'meanROLL':>9s} "
          f"{'ROLL-HOLD':>10s} {'CI_lo':>8s} {'CI_hi':>8s} {'win%':>6s}")
    for b in ["atm_0to1pct", "itm_1to5pct", "itm_5to10pct", "itm_gt10pct"]:
        sub = off[off["moneyness"] == b]
        if len(sub) == 0:
            continue
        d = sub["delta_roll_minus_hold"].values
        m, lo, hi, n = boot_mean_ci(d)
        print(f"  {b:<16s} {n:>4d} {fmt(sub['hold_realized'].mean())} "
              f"{fmt(sub['roll_realized'].mean())} {fmt(m)} {fmt(lo)} {fmt(hi)} "
              f"{100*(d>0).mean():>5.1f}")
    print()

    # --- raw sample dump (first 25 offered rolls) ---
    print("-" * 78)
    print("SAMPLE ROWS (first 25 offered rolls)")
    print("-" * 78)
    cols = ["ticker", "as_of", "regime", "moneyness", "spot_over_strike",
            "strike", "new_strike", "new_dte", "recommend",
            "hold_realized", "roll_realized", "delta_roll_minus_hold",
            "old_assigned", "new_assigned"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(off[cols].head(25).to_string(index=False))
    print()

    # --- distribution sanity ---
    print("-" * 78)
    print("DISTRIBUTION SANITY (offered rolls)")
    print("-" * 78)
    print(f"  ROLL-HOLD  min={off['delta_roll_minus_hold'].min():.0f}  "
          f"p25={off['delta_roll_minus_hold'].quantile(.25):.0f}  "
          f"med={off['delta_roll_minus_hold'].median():.0f}  "
          f"p75={off['delta_roll_minus_hold'].quantile(.75):.0f}  "
          f"max={off['delta_roll_minus_hold'].max():.0f}")
    print(f"  HOLD       mean={off['hold_realized'].mean():.0f}  "
          f"min={off['hold_realized'].min():.0f}  max={off['hold_realized'].max():.0f}")
    print(f"  ROLL       mean={off['roll_realized'].mean():.0f}  "
          f"min={off['roll_realized'].min():.0f}  max={off['roll_realized'].max():.0f}")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
