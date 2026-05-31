"""I3-D: EARNINGS LOCKOUT + POINT-IN-TIME CORRECTNESS.

Two parts:
  (D1) Attribute the 2022-07-01 candidate collapse (~57 rows vs ~442 in June)
       across the three gates -- earnings event lockout vs 504-day history gate
       vs as_of staleness -- by re-ranking a ticker subset with use_event_gate
       True vs False and reading .attrs['drops'] gate tags.
  (D2) CRITICAL: is the event gate's earnings source POINT-IN-TIME? The gate is
       fed by conn.get_next_earnings(ticker, as_of), which filters a STATIC CSV
       of REALIZED announcement_dates to announcement_date > as_of. Determine
       whether that peeks at earnings dates that would not have been *scheduled/
       known* at the historical as_of.

READ-ONLY (re-ranks via rank_candidates_by_ev with gate flags -- allowed:
the prompt permits re-rank when a probe needs gate on/off).
"""
from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = L.ROOT


def attribute_drops(df) -> Counter:
    drops = df.attrs.get("drops", []) if df is not None else []
    return Counter(d.get("gate", "?") for d in drops)


def main() -> None:
    print("=" * 78)
    print("I3-D  EARNINGS LOCKOUT + PIT-CORRECTNESS")
    print("=" * 78)

    from engine.wheel_runner import WheelRunner
    wr = WheelRunner()
    print(f"\nConnector: {type(wr.connector).__name__}")

    # Pick a subset spanning sectors with known Q2 (July) earnings clustering.
    subset = ["AAPL", "MSFT", "JPM", "XOM", "UNH", "PG", "JNJ", "KO", "MCD",
              "CAT", "BA", "GS", "NVDA", "HD", "VZ", "T", "PFE", "MRK",
              "ABBV", "CVX", "WMT", "DIS", "INTC", "CSCO", "ORCL"]
    print(f"Subset: {len(subset)} tickers")

    # D1: same date, gate ON vs OFF, plus the June baseline for contrast.
    for as_of in ("2022-06-01", "2022-07-01"):
        print(f"\n{'=' * 60}\nas_of={as_of}\n{'=' * 60}")
        for gate in (False, True):
            df = wr.rank_candidates_by_ev(
                tickers=subset, as_of=as_of, top_n=10**6, min_ev_dollars=-1e9,
                use_event_gate=gate, include_diagnostic_fields=False,
            )
            n = 0 if df is None else len(df)
            dc = attribute_drops(df)
            print(f"  use_event_gate={gate}: survivors={n}  drops_by_gate={dict(dc)}")
            if gate:
                # Show which event-gated drops there were (earnings reasons).
                ev_drops = [d for d in df.attrs.get("drops", []) if d.get("gate") == "event"]
                for d in ev_drops[:12]:
                    print(f"        EVENT-DROP {d['ticker']}: {d['reason']}")

    # D2: PIT look-ahead probe -------------------------------------------
    print(f"\n{'=' * 60}\nD2: POINT-IN-TIME CORRECTNESS OF EARNINGS SOURCE\n{'=' * 60}")
    conn = wr.connector

    # The earnings CSV is a single static file of REALIZED announcement_dates.
    # get_next_earnings(as_of) returns the first announcement_date > as_of.
    # Probe 1: does it return FUTURE-OF-TODAY dates? (i.e. the file contains
    #          dates that are realized-but-future, proving it's not a
    #          point-in-time scheduled calendar).
    print("\n  [Probe 1] earnings rows dated AFTER today (2026-05-31) in the CSV:")
    e = pd.read_csv(os.path.join(ROOT, "data", "bloomberg", "sp500_earnings.csv"))
    e["announcement_date"] = pd.to_datetime(e["announcement_date"])
    future_today = e[e["announcement_date"] > pd.Timestamp("2026-05-31")]
    print(f"     rows: {len(future_today)} (max date {e['announcement_date'].max().date()})")
    print(f"     -> the file is a FIXED calendar of dates; for a historical "
          f"as_of it\n        returns the date as it ENDED UP being, not as it "
          f"was scheduled then.")

    # Probe 2: for a historical as_of, how far ahead is the 'next earnings'
    # the gate would lock out, and is that date plausibly KNOWN at as_of?
    # Companies pre-announce earnings ~2-5 weeks ahead. A 'next earnings'
    # 60-90+ days out is one a trader could NOT have known on as_of.
    print("\n  [Probe 2] horizon of the 'next earnings' the gate uses, per "
          "as_of (days from as_of):")
    test_asofs = ["2022-06-01", "2022-06-25", "2022-07-01", "2023-01-15", "2024-09-15"]
    rows = []
    for ao in test_asofs:
        for t in subset[:12]:
            ne = conn.get_next_earnings(t, ao)
            if not ne:
                continue
            d = ne["announcement_date"]
            d = d.date() if hasattr(d, "date") else d
            horizon = (pd.Timestamp(d) - pd.Timestamp(ao)).days
            rows.append((ao, t, str(d), horizon))
    hdf = pd.DataFrame(rows, columns=["as_of", "ticker", "next_earn", "horizon_d"])
    for ao in test_asofs:
        sub = hdf[hdf["as_of"] == ao]
        if sub.empty:
            continue
        beyond_30 = int((sub["horizon_d"] > 30).sum())
        beyond_45 = int((sub["horizon_d"] > 45).sum())
        print(f"     as_of={ao}: n={len(sub)} median_horizon={sub['horizon_d'].median():.0f}d "
              f"max={sub['horizon_d'].max()}d  | #beyond 30d={beyond_30} #beyond 45d={beyond_45}")
    # Worst-case examples: long horizons the gate is using.
    far = hdf[hdf["horizon_d"] > 60].sort_values("horizon_d", ascending=False)
    print(f"\n     'next earnings' the gate locked out that were >60 days ahead "
          f"of as_of\n     (a trader could NOT have known these on as_of -> if the "
          f"buffer\n      reached them it would be look-ahead; mostly it does NOT "
          f"because the\n      5-day buffer only bites near-dated ones):")
    for _, r in far.head(8).iterrows():
        print(f"        as_of={r['as_of']} {r['ticker']}: next_earn={r['next_earn']} "
              f"(+{r['horizon_d']}d)")

    # Probe 3: the ACTUAL look-ahead mechanism. The gate buffer is +-5d. For a
    # candidate opened on as_of with dte=35, the trade window is
    # [as_of, as_of+35]. The gate blocks if a REALIZED earnings date falls in
    # [as_of-5, as_of+35+5]. Because the realized date is used (not the
    # scheduled-as-of-as_of date), if a company's realized report slipped vs
    # its originally-announced date, the gate uses hindsight. Quantify how
    # often the realized next-earnings lands INSIDE a 35d+5 window -> those are
    # the ones whose EXACT date had to be known to gate correctly.
    print("\n  [Probe 3] for as_of=2022-06-25, dte=35 (+/-5d buffer): which "
          "subset names\n     get earnings-blocked, and is the blocking date "
          "within the trade window?")
    ao = "2022-06-25"
    win_lo = pd.Timestamp(ao) - pd.Timedelta(days=5)
    win_hi = pd.Timestamp(ao) + pd.Timedelta(days=35 + 5)
    for t in subset[:15]:
        ne = conn.get_next_earnings(t, ao)
        if not ne:
            continue
        d = ne["announcement_date"]
        d = pd.Timestamp(d.date() if hasattr(d, "date") else d)
        inside = win_lo <= d <= win_hi
        flag = "  <-- BLOCKS (date must be known)" if inside else ""
        print(f"     {t}: next_earn={d.date()} in_window[{win_lo.date()}..{win_hi.date()}]"
              f"={inside}{flag}")

    print("\n  INTERPRETATION:")
    print("     The gate sources REALIZED announcement dates from a static CSV and")
    print("     filters announcement_date > as_of. It is therefore NOT a true")
    print("     point-in-time scheduled-earnings calendar: for any near-dated")
    print("     earnings whose realized date differs from what was scheduled/known")
    print("     on as_of, the gate decides using hindsight. The PRACTICAL exposure")
    print("     is bounded by the 5d buffer (only earnings within ~as_of+40d can")
    print("     bite), and most S&P names DO pre-announce within that horizon, so")
    print("     the realized vs scheduled gap is usually small -- but the gate has")
    print("     no mechanism to use only as-of-known dates. See verdict in report.")
    print("\nDONE.")


if __name__ == "__main__":
    main()
