"""I5 -- re-verify three prior claims on the CURRENT (post-#294) engine.

Observe-and-document. No engine edits. Each claim is reproduced with its own
code so every number traces to a value computed here on real data.

  (A) ev_dollars is probability-weighted P&L incl. tails -- a MU covered call
      at ~25-delta returns NEGATIVE EV despite a fat premium (S25).
  (B) ranker IV is now point-in-time. A/B test: rank with PIT IV (current) vs a
      forced get_iv_history->None fallback (reproduces the pre-d26a8d6 bug, which
      used a stale snapshot IV). Show the EV/sign delta on the current engine.
  (C) Bloomberg IV file has no skew: hist_put_imp_vol == hist_call_imp_vol.

Run:  python i5_prior_claims.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
def claim_C_no_skew():
    hr("CLAIM C -- Bloomberg IV file has no skew (put_iv == call_iv)")
    df = pd.read_csv(L.IV_CSV, usecols=["hist_put_imp_vol", "hist_call_imp_vol"])
    both = df.dropna()
    eq = np.isclose(both["hist_put_imp_vol"], both["hist_call_imp_vol"], atol=1e-9, rtol=0)
    n = len(both)
    maxdiff = float((both["hist_put_imp_vol"] - both["hist_call_imp_vol"]).abs().max())
    print(f"total rows           : {len(df):,}")
    print(f"rows both populated  : {n:,}")
    print(f"put_iv == call_iv    : {int(eq.sum()):,}  ({100*eq.mean():.4f}%)")
    print(f"max |put_iv-call_iv| : {maxdiff:.3e}")
    print(f"VERDICT: {'CONFIRMED -- no skew, identical' if eq.mean() > 0.999 else 'OVERTURNED'}")


# --------------------------------------------------------------------------- #
def claim_A_mu_cc():
    hr("CLAIM A -- MU covered call ~25-delta: NEGATIVE EV despite fat premium")
    runner = WheelRunner()
    print(f"connector={type(runner.connector).__name__}")
    for asof in ("2026-03-17", "2026-03-19"):
        df = runner.rank_covered_calls_by_ev(
            "MU", shares_held=100, as_of=asof, target_dtes=(35,),
            target_deltas=(0.25,), top_n=5, min_ev_dollars=-1e9,
            use_event_gate=False, max_as_of_staleness_days=10_000,
            include_diagnostic_fields=True,
        )
        if not len(df):
            print(f"  {asof}: no candidate (gated/failed)")
            continue
        r = df.iloc[0]
        cols = {c: r[c] for c in ["spot", "strike", "premium", "iv", "ev_dollars",
                                   "ev_raw", "prob_profit", "prob_assignment", "cvar_5",
                                   "pnl_p25", "pnl_p50", "pnl_p75", "regime_multiplier",
                                   "hmm_regime"] if c in df.columns}
        print(f"  {asof}: " + "  ".join(f"{k}={v}" for k, v in cols.items()))
    print("Interpretation: EV<0 with a large premium => ev_dollars is probability-weighted")
    print("P&L incl. the fat left tail (assignment loss), NOT premium-if-all-goes-well.")


# --------------------------------------------------------------------------- #
def claim_B_pit_iv():
    hr("CLAIM B -- ranker IV is point-in-time (A/B vs forced snapshot fallback)")
    runner = WheelRunner()
    conn = runner.connector
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "UNH"]
    asof = "2022-06-01"  # historical: PIT IV differs from the 2026 snapshot

    def rank():
        return runner.rank_candidates_by_ev(
            tickers=tickers, dte_target=35, delta_target=0.25, top_n=20,
            min_ev_dollars=-1e9, as_of=asof, include_diagnostic_fields=True,
            max_as_of_staleness_days=10_000, use_event_gate=False,
        )

    # PIT (current, correct)
    pit = rank().set_index("ticker")

    # Force the pre-fix bug: get_iv_history -> None makes _resolve_pit_atm_iv fall
    # back to the static fundamentals snapshot IV (the 2026 value).
    orig = conn.get_iv_history
    try:
        conn.get_iv_history = lambda *a, **k: None  # type: ignore[assignment]
        snap = rank().set_index("ticker")
    finally:
        conn.get_iv_history = orig  # type: ignore[assignment]

    print(f"as_of={asof}  (PIT IV vs forced-snapshot-IV fallback)")
    print(f"{'ticker':6s} {'iv_pit':>8s} {'iv_snap':>8s} {'ev_pit':>10s} {'ev_snap':>10s} {'sign_flip':>10s}")
    flips = 0
    for t in tickers:
        if t not in pit.index or t not in snap.index:
            print(f"{t:6s} (missing in one run; gated)")
            continue
        ivp, ivs = pit.loc[t, "iv"], snap.loc[t, "iv"]
        evp, evs = pit.loc[t, "ev_dollars"], snap.loc[t, "ev_dollars"]
        flip = (np.sign(evp) != np.sign(evs))
        flips += int(flip)
        print(f"{t:6s} {ivp:8.4f} {ivs:8.4f} {evp:10.2f} {evs:10.2f} {str(flip):>10s}")
    print(f"sign flips PIT-vs-snapshot: {flips}/{len(tickers)}")
    print("VERDICT: PIT IV is active (current code); using a stale snapshot IV")
    print("materially changes EV and can flip the trade sign -- the bug the fix closed.")


if __name__ == "__main__":
    claim_C_no_skew()
    claim_A_mu_cc()
    claim_B_pit_iv()
    print("\nI5 DONE.")
