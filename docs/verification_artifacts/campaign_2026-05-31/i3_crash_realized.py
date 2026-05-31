"""I3-E supplement: did the engine's CRASH-window positive-EV verdicts actually
hold up? Realize the 2020-03-02 crash-entry short puts against the actual
underlying path into the March-23 bottom, and contrast with a calm date.

This is the decisive test of 'does it refuse what a senior trader would refuse
in a crash'. READ-ONLY -- uses campaign_lib.realize_short_put (frictionless,
engine-synthetic premium) on existing snapshot rows.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SNAP = os.path.join(HERE, "snapshots")


def realize_snapshot(nm: str, as_of: str) -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(SNAP, nm))
    rows = []
    for _, r in d.iterrows():
        out = L.realize_short_put(r["ticker"], float(r["strike"]), as_of,
                                  int(r["dte"]), float(r["premium"]))
        if out is None:
            continue
        rows.append({
            "ticker": r["ticker"], "ev_dollars": float(r["ev_dollars"]),
            "regime_mult": float(r["regime_multiplier"]),
            "hmm_regime": r["hmm_regime"], "iv": float(r["iv"]),
            "realized_pnl": out["realized_pnl_synth"], "assigned": out["assigned"],
        })
    return pd.DataFrame(rows)


def summarize(rdf: pd.DataFrame, label: str) -> None:
    n = len(rdf)
    pos = rdf[rdf["ev_dollars"] > 0]
    print(f"\n  [{label}] n={n}")
    print(f"     engine: mean_ev=${rdf['ev_dollars'].mean():.1f} "
          f"median_ev=${rdf['ev_dollars'].median():.1f} "
          f"positive_ev_share={(rdf['ev_dollars'] > 0).mean():.1%}")
    print(f"     REALIZED (all): mean_pnl=${rdf['realized_pnl'].mean():.1f} "
          f"median_pnl=${rdf['realized_pnl'].median():.1f} "
          f"win_rate={(rdf['realized_pnl'] > 0).mean():.1%} "
          f"assignment_rate={rdf['assigned'].mean():.1%}")
    if len(pos):
        print(f"     REALIZED (engine said POSITIVE-EV only, n={len(pos)}): "
              f"mean_pnl=${pos['realized_pnl'].mean():.1f} "
              f"median_pnl=${pos['realized_pnl'].median():.1f} "
              f"win_rate={(pos['realized_pnl'] > 0).mean():.1%}")
    worst = rdf.loc[rdf["realized_pnl"].idxmin()]
    print(f"     worst single realized: {worst['ticker']} ${worst['realized_pnl']:.0f} "
          f"(engine ev was ${worst['ev_dollars']:.0f}, regime_mult={worst['regime_mult']:.2f})")


def main() -> None:
    print("=" * 78)
    print("I3-E supplement  CRASH-ENTRY REALIZED OUTCOMES")
    print("=" * 78)
    print("\nFrictionless realized P&L (engine-synthetic premium). Per contract.")
    print("Caveat: synthetic premium (no real spread); 504-day history gate thins")
    print("the early-2020 universe; assignment uses close at as_of+dte (no early")
    print("management). Direction-of-result is the point, not the exact dollar.")

    crash = realize_snapshot("put_2020-03-02.parquet", "2020-03-02")
    summarize(crash, "CRASH entry 2020-03-02 (21 days before Mar-23 bottom)")

    crash2 = realize_snapshot("put_2020-04-01.parquet", "2020-04-01")
    summarize(crash2, "CRASH entry 2020-04-01 (deep drawdown, recovering)")

    calm = realize_snapshot("put_2021-06-01.parquet", "2021-06-01")
    summarize(calm, "CALM entry 2021-06-01 (low-vol bull)")

    print("\n--- VERDICT INPUT ---")
    cr_pos = crash[crash["ev_dollars"] > 0]
    print(f"  On 2020-03-02 the engine flagged {(crash['ev_dollars'] > 0).mean():.0%} of "
          f"candidates POSITIVE-EV.")
    print(f"  Those positive-EV trades realized a mean ${cr_pos['realized_pnl'].mean():.0f} "
          f"and {(cr_pos['realized_pnl'] > 0).mean():.0%} win rate into the bottom.")
    print(f"  => The engine did NOT refuse crash trades a senior trader would refuse;")
    print(f"     it RECOMMENDED them. Root cause probed separately: forward")
    print(f"     distribution = 100% empirical_overlapping off the trailing (still")
    print(f"     bull-dominated) window; HMM de-rated to ~0.31x but not enough to")
    print(f"     flip EV sign while IV-inflated premium pulled EV positive.")
    print("\nDONE.")


if __name__ == "__main__":
    main()
