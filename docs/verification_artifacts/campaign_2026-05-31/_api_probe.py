"""API-surface probe for the 2026-05-31 heavy-verification campaign.

Read-only. Grounds the exact engine API the realizer + simulator will drive:
  - WheelRunner connector type (CLAUDE.md bring-up step 1).
  - 5-ticker smoke at a HISTORICAL as_of (step 2, PIT mode).
  - The full set of columns rank_candidates_by_ev emits (so the realizer/sim
    reference real field names, not guesses).
  - connector.get_ohlcv signature/sample + get_universe size.

Run:  python docs/verification_artifacts/campaign_2026-05-31/_api_probe.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

from engine.wheel_runner import WheelRunner  # noqa: E402

AS_OF = "2022-06-01"  # historical, inside the 504-day-gate-feasible window

print("=" * 78)
print("1. Connector / provider")
runner = WheelRunner()
print(f"   connector type: {type(runner.connector).__name__}")
print(f"   SWE_DATA_PROVIDER env: {os.environ.get('SWE_DATA_PROVIDER', '(unset->bloomberg default)')}")

print("=" * 78)
print(f"2. 5-ticker PIT smoke @ as_of={AS_OF}")
df = runner.rank_candidates_by_ev(
    tickers=["AAPL", "MSFT", "JPM", "XOM", "UNH"],
    top_n=10,
    min_ev_dollars=-1e9,
    as_of=AS_OF,
    include_diagnostic_fields=True,
    max_as_of_staleness_days=10_000,
)
print(f"   rows returned: {len(df)}")
print(f"   COLUMNS ({len(df.columns)}): {df.columns.tolist()}")
if len(df):
    cols = [c for c in ["ticker", "strike", "premium", "iv", "delta", "dte",
                        "expiration", "ev_dollars", "ev_raw", "prob_profit",
                        "prob_otm", "prob_assignment", "cvar", "regime_multiplier",
                        "collateral", "roc", "contracts"] if c in df.columns]
    print(df[cols].to_string(index=False))
    print("\n   --- FULL first row (all fields) ---")
    for k, v in df.iloc[0].items():
        print(f"     {k!r}: {v!r}")

print("=" * 78)
print("3. connector.get_ohlcv sample (AAPL, faithful rename applied by connector)")
try:
    o = runner.connector.get_ohlcv("AAPL", start_date="2022-05-25", end_date="2022-06-03")
    print(f"   type={type(o).__name__} cols={list(getattr(o, 'columns', []))}")
    print(o.tail(6).to_string() if hasattr(o, "tail") else o)
except Exception as e:  # noqa: BLE001
    print(f"   get_ohlcv error: {type(e).__name__}: {e}")

print("=" * 78)
print("4. connector.get_universe size")
try:
    u = runner.connector.get_universe()
    u = list(u)
    print(f"   universe size: {len(u)}  sample: {u[:8]}")
except Exception as e:  # noqa: BLE001
    print(f"   get_universe error: {type(e).__name__}: {e}")

print("=" * 78)
print("DONE.")
