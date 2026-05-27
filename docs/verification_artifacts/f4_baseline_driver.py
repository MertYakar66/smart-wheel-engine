"""F4 tail-risk pre-fix baseline driver (2026-05-26).

Captures the engine's current predictions on the two canonical F4
failure cases — COST 2022-04-04 (35d realized -23.89%) and
UNH 2024-11-11 (35d realized -19.31%) — and compares to the realized
35-day forward returns straight from the Bloomberg OHLCV.

Purpose: when Terminal A's `claude/fix-f4-regime-conditioned-widening`
branch lands, re-run this driver and diff the output. A successful
F4 fix should:

  1. Drop `prob_profit` materially below the pre-fix baseline on both
     cases (current 0.83-0.90 -> ideally <= 0.70 to reflect the
     -20-30% realized drops).
  2. Widen `cvar_5` (absolute value) to reflect the realized tail.
  3. Either flip the `heavy_tail` flag to True OR move `prob_loss`
     into the >0.20 band.
  4. NOT regress on the AAPL 2026-03-20 no-loss control (engine
     should not become spuriously bearish on a normal-regime case).

Read-only client of the production ranker. NOT a §2 surface change.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKTREE = Path(r"C:\Users\merty\Desktop\swe-terminal-b").resolve()
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

import pandas as pd  # noqa: E402  (sys.path bootstrap above)

from engine.wheel_runner import WheelRunner  # noqa: E402

# Same canonical anchor dates as tests/test_f4_tail_risk_gap.py
CASES = [
    {
        "ticker": "COST",
        "as_of": "2022-04-04",
        "documented_realized_35d_pct": -23.89,
        "note": "F4 canonical case from PR #178; realized verified from CSV",
    },
    {
        "ticker": "UNH",
        "as_of": "2024-11-11",
        "documented_realized_35d_pct": -19.31,
        "note": "F4 second canonical case; realized verified from CSV",
    },
    {
        "ticker": "AAPL",
        "as_of": "2026-02-13",
        "documented_realized_35d_pct": None,
        "note": "No-loss control — normal-regime date; engine should NOT become spuriously bearish post-fix",
    },
]

print("=" * 78)
print("F4 TAIL-RISK PRE-FIX BASELINE — 2026-05-26")
print("=" * 78)
print("Engine: origin/main @ 9f0afaf (pre-F4-fix)")
print()
print("Each row is the production-ranker output for a known F4 case.")
print("Compare to: documented realized 35-day forward return (where known)")
print("and to the same row produced AFTER Terminal A's F4 fix lands.")
print()

runner = WheelRunner()


def compute_realized_35d_pct(ticker: str, as_of_iso: str) -> float | None:
    """Compute the realised 35-calendar-day forward return from the
    Bloomberg OHLCV CSV. Independent of the engine — this is the
    ground-truth answer to what prob_profit *should* have been
    predicting."""
    ohlcv = runner.connector.get_ohlcv(ticker)
    if ohlcv is None or ohlcv.empty:
        return None
    as_of = pd.Timestamp(as_of_iso)
    before = ohlcv.loc[ohlcv.index <= as_of]
    after = ohlcv.loc[ohlcv.index >= as_of + pd.Timedelta(days=35)]
    if before.empty or after.empty:
        return None
    spot_t0 = float(before["close"].iloc[-1])
    spot_t35 = float(after["close"].iloc[0])
    return 100.0 * (spot_t35 / spot_t0 - 1.0)


for case in CASES:
    ticker = case["ticker"]
    as_of = case["as_of"]
    print(f"--- {ticker} as_of={as_of} ---")
    print(f"  note: {case['note']}")
    if case["documented_realized_35d_pct"] is not None:
        print(
            f"  documented realized 35d: {case['documented_realized_35d_pct']:+.2f}% "
            "(from tests/test_f4_tail_risk_gap.py)"
        )

    # Ground-truth realized return — recompute from OHLCV
    realized_pct = compute_realized_35d_pct(ticker, as_of)
    if realized_pct is not None:
        print(f"  recomputed realized 35d: {realized_pct:+.2f}%")
    else:
        print("  recomputed realized 35d: N/A (out-of-data-window)")

    # Production ranker call — same path as the test fixtures
    try:
        df = runner.rank_candidates_by_ev(
            tickers=[ticker],
            as_of=as_of,
            top_n=1,
            min_ev_dollars=-1e9,
            use_event_gate=False,  # bypass earnings lockout for reproducibility
            include_diagnostic_fields=True,
        )
        if df.empty:
            drops = df.attrs.get("drops", [])
            print(f"  ranker returned empty; drops={drops}")
            print()
            continue
        row = df.iloc[0]
        print("  Engine output (pre-F4-fix):")
        print(f"    strike       = {row['strike']:.2f}")
        print(f"    premium      = {row['premium']:.4f}")
        print(f"    iv           = {row['iv']:.6f}")
        print(f"    ev_dollars   = {row['ev_dollars']:+.2f}")
        print(f"    prob_profit  = {row['prob_profit']:.4f}")
        # Diagnostic fields (may be None depending on connector state)
        for col, label in [
            ("cvar_5", "cvar_5      "),
            ("cvar_99_evt", "cvar_99_evt "),
            ("heavy_tail", "heavy_tail  "),
            ("regime_multiplier", "regime_mult "),
        ]:
            val = row.get(col)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                print(f"    {label}= (n/a)")
            elif isinstance(val, float):
                print(f"    {label}= {val:.4f}")
            else:
                print(f"    {label}= {val}")
        # F4 gap quantification
        if realized_pct is not None and realized_pct < -10.0:
            # We were in a heavy-tail event. Engine should have flagged it.
            print(
                f"  F4 GAP: realized {realized_pct:+.2f}% but "
                f"engine prob_profit = {row['prob_profit']:.4f} "
                f"(should be <= ~0.70 post-fix)"
            )
        elif realized_pct is not None:
            print(
                f"  No-loss control: realized {realized_pct:+.2f}%; "
                f"engine prob_profit = {row['prob_profit']:.4f} (should stay similar post-fix)"
            )
    except Exception as exc:
        print(f"  Engine call FAILED: {exc!r}")
    print()

print("=" * 78)
print("BASELINE CAPTURED")
print("=" * 78)
print()
print("To validate F4 fix (when Terminal A's branch lands):")
print("  1. Re-run this driver against the post-fix engine.")
print("  2. Diff the prob_profit / cvar_5 / heavy_tail columns vs this baseline.")
print("  3. SUCCESS = prob_profit drops materially on the two F4 cases,")
print("     control case AAPL stays roughly unchanged.")
print("  4. FAILURE = prob_profit drops on the control (over-correction)")
print("     OR stays high on F4 cases (under-correction).")
