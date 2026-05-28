"""S41 F4 fix validation driver (post-#260, 2026-05-28).

Re-probes the canonical F4 cases on the post-#260 engine to validate
the realized-vol-ratio widening claims in PR #260's body. Three
probe sets:

  1. **Layer 1 — named cases.** COST 2022-04-04 (F4 canonical case 1),
     UNH 2024-11-11 (F4 canonical case 2), AAPL 2026-02-13 (no-loss
     control). Diffs `prob_profit` / `ev_dollars` / `cvar_5` /
     `tail_widening_factor` against the pre-fix baseline captured in
     `f4_baseline_2026-05-26_raw_output.txt` (PR #245, engine SHA
     `9f0afaf`).

  2. **Layer 1c — COST 2022-04 unfolding window.** All 10 dates from
     `docs/F4_TAIL_RISK_DIAGNOSTIC.md` §2.1 (2022-04-01 → 2022-04-14).
     Validates the F4 doc claim that `prob_profit = [0.8333] × 10`
     stays constant through the entire 14-day unfolding event, and
     that the new `tail_widening_factor` correctly stays at 1.0 because
     pre-event `rv30/rv252` is below the firing threshold.

  3. **Layer 3 — calm-regime controls.** AAPL/MSFT at three calm
     2023-2024 dates. Confirms the fix does NOT introduce spurious
     widening on calm tickers (`factor = 1.0` everywhere).

Companion to `docs/ENGINE_BACKTEST_S41_F4_FIX_VALIDATION.md`. The
2022-2024 backtest (Layer 2) runs through
`backtests.regression.s27_ivpit_24t_100k` and produces a separate
artifact (rank_log.csv + metrics.json) under %TEMP% per the Sn
convention.

Read-only client of the production ranker. NOT a §2 surface change.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from engine.forward_distribution import (  # noqa: E402
    realized_vol_ratio,
    realized_vol_widening_factor,
)
from engine.wheel_runner import WheelRunner  # noqa: E402

runner = WheelRunner()
conn = runner.connector
print(f"connector: {type(conn).__name__}")
print(f"worktree:  {WORKTREE}")
print()


def fmt(v, w=10, p=4):
    if v is None or v != v:
        return f"{'n/a':>{w}}"
    return f"{float(v):>{w}.{p}f}"


# ---------------------------------------------------------------------------
# Layer 1 — three canonical named cases
# ---------------------------------------------------------------------------
print("=" * 78)
print("LAYER 1 — named F4 cases (engine output diffed against pre-fix baseline)")
print("=" * 78)

L1_CASES = [
    ("COST", "2022-04-04", "F4 canonical case 1 (calm pre-drawdown)"),
    ("UNH", "2024-11-11", "F4 canonical case 2 (vol-cluster firing)"),
    ("AAPL", "2026-02-13", "No-loss control (must NOT widen)"),
]

for ticker, as_of, note in L1_CASES:
    df = runner.rank_candidates_by_ev(
        tickers=[ticker],
        as_of=as_of,
        top_n=1,
        min_ev_dollars=-1e12,
        include_diagnostic_fields=True,
    )
    if df.empty:
        print(f"  {ticker} {as_of}  EMPTY drops={df.attrs.get('drops', [])}")
        continue
    r = df.iloc[0]
    print(f"\n--- {ticker} as_of={as_of} ---")
    print(f"  note: {note}")
    print(f"  spot      = {float(r['spot']):.2f}")
    print(f"  strike    = {float(r['strike']):.2f}")
    print(f"  premium   = {float(r['premium']):.4f}")
    print(f"  iv        = {float(r['iv']):.6f}")
    print(f"  ev_dollars       = {float(r['ev_dollars']):+.2f}")
    print(f"  prob_profit      = {float(r['prob_profit']):.4f}")
    print(f"  cvar_5           = {float(r['cvar_5']):+.2f}")
    print(f"  tail_widening_factor = {float(r['tail_widening_factor']):.4f}")
    print(f"  hmm_regime       = {r.get('hmm_regime')}")
    print(f"  distribution     = {r.get('distribution_source')}")


# ---------------------------------------------------------------------------
# Layer 1c — COST 2022-04 unfolding-event window (10 dates)
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("LAYER 1c — COST 2022-04 unfolding-event window (F4 doc §2.1 reproduction)")
print("=" * 78)
COST_DATES = [
    "2022-04-01",
    "2022-04-04",
    "2022-04-05",
    "2022-04-06",
    "2022-04-07",
    "2022-04-08",
    "2022-04-11",
    "2022-04-12",
    "2022-04-13",
    "2022-04-14",
]
print()
print(
    f"{'as_of':12} {'spot':>8} {'strike':>8} {'prem':>7} {'iv':>7} "
    f"{'pp':>7} {'ev':>9} {'rv30/rv252':>12} {'factor':>8} {'hmm':>14}"
)
running_ev = 0.0
running_pp = 0.0
running_ratio = 0.0
running_factor = 0.0
n = 0
for as_of in COST_DATES:
    ohlcv = conn.get_ohlcv("COST", end_date=as_of)
    ratio = realized_vol_ratio(ohlcv, as_of=as_of)
    df = runner.rank_candidates_by_ev(
        tickers=["COST"],
        as_of=as_of,
        top_n=1,
        min_ev_dollars=-1e12,
        include_diagnostic_fields=True,
    )
    if df.empty:
        print(f"  {as_of:12} EMPTY")
        continue
    r = df.iloc[0]
    print(
        f"  {as_of:12} {float(r['spot']):>8.2f} {float(r['strike']):>8.2f} "
        f"{float(r['premium']):>7.3f} {float(r['iv']):>7.4f} "
        f"{float(r['prob_profit']):>7.4f} {float(r['ev_dollars']):>+9.2f} "
        f"{ratio:>12.4f} {float(r['tail_widening_factor']):>8.4f} "
        f"{str(r.get('hmm_regime')):>14}"
    )
    running_ev += float(r["ev_dollars"])
    running_pp += float(r["prob_profit"])
    running_ratio += ratio
    running_factor += float(r["tail_widening_factor"])
    n += 1

if n > 0:
    print(f"\n  mean ev_dollars       = {running_ev / n:+.2f}")
    print(f"  mean prob_profit      = {running_pp / n:.4f}")
    print(f"  mean rv30/rv252       = {running_ratio / n:.4f}")
    print(f"  mean tail_widening    = {running_factor / n:.4f}")
    print("  cross-check S27 doc:  pre-#260 mean EV = +$127.35; mean pp = 0.8333")


# ---------------------------------------------------------------------------
# Layer 3 — calm-regime controls
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("LAYER 3 — calm-regime controls (must have factor = 1.0)")
print("=" * 78)
L3_CASES = [
    ("AAPL", "2023-06-12", "calm bull mid-2023"),
    ("MSFT", "2023-06-12", "calm bull mid-2023"),
    ("AAPL", "2024-06-10", "calm bull mid-2024"),
    ("MSFT", "2024-06-10", "calm bull mid-2024"),
    ("AAPL", "2024-09-09", "calm late-2024"),
    ("MSFT", "2024-09-09", "calm late-2024"),
]
print()
print(
    f"  {'ticker':<5} {'as_of':12} {'factor':>8} {'prob_profit':>12} "
    f"{'ev':>10} {'iv':>8} {'hmm':>14}  note"
)
non_unity = 0
for ticker, as_of, note in L3_CASES:
    df = runner.rank_candidates_by_ev(
        tickers=[ticker],
        as_of=as_of,
        top_n=1,
        min_ev_dollars=-1e12,
        include_diagnostic_fields=True,
    )
    if df.empty:
        drops = df.attrs.get("drops", [])
        print(f"  {ticker:<5} {as_of:12}  EMPTY  drops={drops}")
        continue
    r = df.iloc[0]
    factor = float(r["tail_widening_factor"])
    if abs(factor - 1.0) > 1e-9:
        non_unity += 1
    print(
        f"  {ticker:<5} {as_of:12} {factor:>8.4f} "
        f"{float(r['prob_profit']):>12.4f} {float(r['ev_dollars']):>+10.2f} "
        f"{float(r['iv']):>8.4f} {str(r.get('hmm_regime')):>14}  {note}"
    )
if non_unity == 0:
    print("\n  [OK] All calm-regime control rows have factor = 1.0 (no spurious caution).")
else:
    print(f"\n  [FAIL] {non_unity} calm-regime control rows had factor != 1.0")


# ---------------------------------------------------------------------------
# Layer 0 — direct rv30/rv252 + widening factor probe
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("LAYER 0 — direct rv30/rv252 + widening factor (PR #260 signal table check)")
print("=" * 78)
L0_CASES = [
    ("COST", "2022-04-04", "F4 case 1 — PR #260: rv30/rv252=0.96, factor=1.00"),
    ("UNH", "2024-11-11", "F4 case 2 — PR #260: rv30/rv252=1.36, factor=1.012"),
    ("AAPL", "2026-02-13", "control — PR #260: rv30/rv252=0.85, factor=1.00"),
    ("META", "2022-02-02", "HMM-missed earnings — PR #260: rv30/rv252=1.15, factor=1.00"),
]
print()
print(f"  {'ticker':<5} {'as_of':12} {'rv30/rv252':>12} {'factor':>8}  note")
for ticker, as_of, note in L0_CASES:
    ohlcv = conn.get_ohlcv(ticker, end_date=as_of)
    ratio = realized_vol_ratio(ohlcv, as_of=as_of)
    factor = realized_vol_widening_factor(ohlcv, as_of=as_of)
    print(f"  {ticker:<5} {as_of:12} {ratio:>12.4f} {factor:>8.4f}  {note}")

print()
print("=" * 78)
print("S41 F4 FIX VALIDATION DRIVER — COMPLETE")
print("=" * 78)
