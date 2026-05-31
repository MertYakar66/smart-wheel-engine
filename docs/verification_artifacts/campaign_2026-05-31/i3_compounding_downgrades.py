"""I3-C: COMPOUNDING / FIRST-MATCH DOWNGRADES in EnginePhaseReviewer.

Build a positive-EV row and stack downgrade signals to confirm:
  - reviewer is downgrade-only (never upgrades; chart/dealer/portfolio
    can shift proceed->review/skip and negative-EV stays blocked, but
    nothing rescues a negative or non-finite EV);
  - rules fire FIRST-MATCH in fixed order R1..R10 (the first triggered
    rule returns; later rules never see the candidate);
  - a single positive-EV row gives different verdicts as we add signals.

READ-ONLY.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = L.ROOT


def make_chart(ticker, spot, *, ok=True, price=None, phase=None):
    from engine.chart_context import ChartContext

    vi = {"phase": phase} if phase else {}
    return ChartContext(
        ticker=ticker, timeframe="1D", captured_at=datetime(2021, 6, 1),
        screenshot_path=(Path("fake.png") if ok else None),
        visible_price=price, visible_indicators=vi, source="probe",
        error=("" if ok else "capture_failed"),
    )


def make_market_structure(ticker, spot, *, regime, put_wall_strike):
    from engine.dealer_positioning import GammaWall, MarketStructure, DealerAssumption
    from datetime import date

    ms = MarketStructure(
        ticker=ticker, as_of=datetime(2021, 6, 1), spot=spot, expiry=date(2021, 7, 16),
        assumption=DealerAssumption.LONG_CALLS_SHORT_PUTS, regime=regime,
    )
    ms.nearest_put_wall = GammaWall(strike=put_wall_strike, distance_pct=-2.0,
                                    net_gex=-1e9, side="put")
    return ms


def review(ev_row, *, chart=None, ms=None, ctx=None):
    from engine.candidate_dossier import CandidateDossier, EnginePhaseReviewer

    d = CandidateDossier(ticker=ev_row["ticker"], ev_row=ev_row, chart_context=chart,
                         market_structure=ms, portfolio_context=ctx)
    return EnginePhaseReviewer().review(d)


def main() -> None:
    print("=" * 78)
    print("I3-C  COMPOUNDING / FIRST-MATCH DOWNGRADES")
    print("=" * 78)

    snap = pd.read_parquet(os.path.join(HERE, "snapshots", "put_2021-06-01.parquet"))
    base = snap[snap["ev_dollars"] > 0].sort_values("ev_dollars", ascending=False).iloc[0].to_dict()
    tkr, spot, strike, ev = base["ticker"], float(base["spot"]), float(base["strike"]), float(base["ev_dollars"])
    print(f"\nBase row: {tkr} spot={spot:.2f} strike={strike:.2f} ev_dollars={ev:.2f} (positive)")
    print(f"Put wall placed at strike {strike:.2f} (== candidate strike, so 'at/above') for R6.\n")

    scenarios = [
        ("clean proceed (good chart, price matches)",
         dict(chart=make_chart(tkr, spot, ok=True, price=spot))),
        ("R2: chart MISSING -> review",
         dict(chart=make_chart(tkr, spot, ok=False))),
        ("R3: spot mismatch >2% -> skip (chart present)",
         dict(chart=make_chart(tkr, spot, ok=True, price=spot * 1.05))),
        ("R6: short-gamma + strike at/above put wall -> review",
         dict(chart=make_chart(tkr, spot, ok=True, price=spot),
              ms=make_market_structure(tkr, spot, regime="short_gamma_amplifying",
                                       put_wall_strike=strike))),
        ("STACK R2+R3+R6: chart missing AND mismatch AND short-gamma",
         dict(chart=make_chart(tkr, spot, ok=False, price=spot * 1.05),
              ms=make_market_structure(tkr, spot, regime="short_gamma_amplifying",
                                       put_wall_strike=strike))),
        ("STACK R3+R6: mismatch AND short-gamma (chart present)",
         dict(chart=make_chart(tkr, spot, ok=True, price=spot * 1.05),
              ms=make_market_structure(tkr, spot, regime="short_gamma_amplifying",
                                       put_wall_strike=strike))),
    ]

    for label, kw in scenarios:
        v, r, notes = review(dict(base), **kw)
        print(f"  [{label}]")
        print(f"     verdict={v} reason={r}")
        print(f"     notes={notes}")
        print()

    # ---- Can ANY signal rescue a negative / non-finite EV? ---------------
    print("--- RESCUE TEST: can a perfect chart upgrade a negative / +inf EV? ---")
    neg = dict(base)
    neg["ev_dollars"] = -123.45
    v, r, notes = review(neg, chart=make_chart(tkr, spot, ok=True, price=spot),
                         ms=make_market_structure(tkr, spot, regime="long_gamma_dampening",
                                                  put_wall_strike=strike * 0.5))
    print(f"  negative EV (-123.45) + perfect chart + benign dealer: verdict={v} reason={r}")
    assert v == "blocked" and r == "negative_ev", "RESCUE LEAK on negative EV!"

    inf = dict(base)
    inf["ev_dollars"] = math.inf
    v, r, notes = review(inf, chart=make_chart(tkr, spot, ok=True, price=spot))
    print(f"  +inf EV + perfect chart: verdict={v} reason={r}")
    assert v == "blocked" and r == "ev_non_finite", "RESCUE LEAK on +inf EV!"

    nan = dict(base)
    nan["ev_dollars"] = math.nan
    v, r, notes = review(nan, chart=make_chart(tkr, spot, ok=True, price=spot))
    print(f"  NaN EV + perfect chart: verdict={v} reason={r}")
    assert v == "blocked" and r == "ev_non_finite", "RESCUE LEAK on NaN EV!"

    # ---- First-match ordering proof: R1 beats everything ----------------
    print("\n--- FIRST-MATCH ORDERING: negative EV short-circuits before R2 ---")
    neg2 = dict(base)
    neg2["ev_dollars"] = -50.0
    # chart missing (would be R2 review) AND negative EV -> R1 wins (blocked)
    v, r, notes = review(neg2, chart=make_chart(tkr, spot, ok=False))
    print(f"  negative EV + missing chart: verdict={v} reason={r}  "
          f"(R1 blocked beats R2 review)")
    assert v == "blocked", "ordering violated"

    print("\nALL RESCUE / ORDERING ASSERTIONS PASSED.")
    print("DONE.")


if __name__ == "__main__":
    main()
