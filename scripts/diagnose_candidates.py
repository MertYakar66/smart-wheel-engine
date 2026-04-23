"""
Diagnose why the wheel EV ranker is returning zero candidates.

Runs the ranker on the first 50 tickers with each gate turned off one
at a time, then prints a funnel report showing where tickers get
dropped. Use this whenever ``/api/candidates`` returns an empty
``trades`` list.

Usage
-----
    $env:SWE_DATA_PROVIDER = "theta"
    python scripts/diagnose_candidates.py

The script is read-only — it doesn't write files or call any network
endpoints beyond what the ranker would normally touch.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from engine.wheel_runner import WheelRunner  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _run(runner: WheelRunner, label: str, **kwargs) -> int:
    """Run the ranker with the given kwargs and return row count."""
    print(f"\n=== {label} ===")
    try:
        df = runner.rank_candidates_by_ev(
            tickers=None,
            dte_target=35,
            delta_target=0.25,
            top_n=100,
            min_ev_dollars=-1e9,  # accept negative EV for diagnostics
            include_diagnostic_fields=True,
            **kwargs,
        )
    except Exception as exc:
        print(f"  RAISED: {exc}")
        return 0
    n = 0 if df is None or df.empty else len(df)
    print(f"  candidates returned: {n}")
    if n > 0:
        cols = [c for c in ("ticker", "ev_dollars", "ev_per_day", "iv", "premium") if c in df.columns]
        print(df[cols].head(10).to_string())
    return n


def main() -> int:
    provider = os.environ.get("SWE_DATA_PROVIDER", "bloomberg")
    print(f"Data provider: {provider}")
    runner = WheelRunner()

    print(
        "\nThis tool turns off one gate at a time and reports whether candidates\n"
        "appear. The first config that produces non-zero is the culprit."
    )

    # Baseline: all gates on (same as /api/candidates endpoint)
    n_baseline = _run(runner, "Baseline (all gates on, default API)")

    # 1. Event gate off
    n_no_event = _run(runner, "Event gate OFF (earnings lockout disabled)",
                      use_event_gate=False)

    # 2. Chain quality gate off
    n_no_chain = _run(runner, "Chain quality gate OFF",
                      enforce_chain_quality_gate=False)

    # 3. History gate off
    n_no_hist = _run(runner, "History gate OFF (< 504 trading days allowed)",
                     enforce_history_gate=False)

    # 4. Dealer positioning off
    n_no_dealer = _run(runner, "Dealer positioning OFF",
                       use_dealer_positioning=False)

    # 5. All gates OFF (pure EV math)
    n_raw = _run(runner, "ALL GATES OFF (pure EV math, no filters)",
                 use_event_gate=False,
                 enforce_chain_quality_gate=False,
                 enforce_history_gate=False,
                 use_dealer_positioning=False,
                 use_skew_dynamics=False,
                 use_news_sentiment=False,
                 use_credit_regime=False)

    # --- Report ---
    print("\n" + "=" * 60)
    print("FUNNEL:")
    print(f"  baseline (all gates on):           {n_baseline}")
    print(f"  event gate off:                    {n_no_event}")
    print(f"  chain quality gate off:            {n_no_chain}")
    print(f"  history gate off:                  {n_no_hist}")
    print(f"  dealer positioning off:            {n_no_dealer}")
    print(f"  ALL gates off (raw EV only):       {n_raw}")
    print("=" * 60)

    if n_raw == 0:
        print(
            "\nEven with every gate off, zero candidates. The problem is inside\n"
            "the EV math path — likely IV/delta-solver/premium floor. Check:\n"
            "  - Are IVs loading (not all NaN)?\n"
            "  - Is the BSM put-delta solver bracketing (0.5*spot, 0.99*spot)?\n"
            "  - Is the synthetic premium > $0.05 floor?"
        )
    elif n_no_event > n_baseline:
        print(
            "\nEvent gate is filtering the universe. We're in Q1 earnings season\n"
            "(late April 2026); most SP500 tickers have earnings within 35 days.\n"
            "Options: (a) drop dte_target to 14-21, (b) set use_event_gate=False,\n"
            "         (c) accept fewer candidates during earnings season."
        )
    elif n_no_chain > n_baseline:
        print("\nChain quality gate is the blocker — a more aggressive filter needed.")
    elif n_no_hist > n_baseline:
        print("\nHistory gate is blocking — tickers missing enough OHLCV history.")
    elif n_baseline > 0:
        print("\nThe ranker IS producing candidates. Check /api/candidates again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
