"""I3-E: CRASH-WINDOW TAIL BEHAVIOR.

At 2020 COVID-crash entry dates, does the engine widen tail risk (cvar_5,
cvar_99_evt, tail_xi, heavy_tail) and push ev_dollars appropriately negative
-- i.e. does it REFUSE trades a senior trader would refuse in a crash --
vs a calm date?

Compares the per-snapshot distributions of cvar_5 / ev_dollars / iv /
prob_assignment for crash dates vs calm dates. The 504-day history gate
thins the early-2020 universe; we note the rankable-n caveat throughout.

READ-ONLY -- consumes existing snapshots only.
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


def load(nm):
    d = pd.read_parquet(os.path.join(SNAP, nm))
    return d if "ev_dollars" in d.columns else None


def stat_block(d: pd.DataFrame, label: str) -> dict:
    n = len(d)
    pos_ev = int((d["ev_dollars"] > 0).sum())
    # cvar_5 is a per-contract dollar tail loss (more negative = worse).
    def q(s, p):
        s = s.dropna()
        return float(np.percentile(s, p)) if len(s) else float("nan")
    block = {
        "label": label,
        "n_rankable": n,
        "pct_positive_ev": pos_ev / n if n else float("nan"),
        "median_ev_dollars": float(d["ev_dollars"].median()),
        "median_iv": float(d["iv"].median()),
        "median_prob_assignment": float(d["prob_assignment"].median()),
        "cvar5_median": float(d["cvar_5"].median()),
        "cvar5_p5_worst": q(d["cvar_5"], 5),   # 5th pct = deepest losses
        "cvar5_p95": q(d["cvar_5"], 95),
        "cvar99_evt_filled_pct": float(d["cvar_99_evt"].notna().mean()),
        "cvar99_evt_median": float(d["cvar_99_evt"].median()),
        "tail_xi_median": float(d["tail_xi"].median()),
        "heavy_tail_n": int(d["heavy_tail"].sum()),
        "heavy_tail_pct": float(d["heavy_tail"].mean()),
    }
    return block


def main() -> None:
    print("=" * 78)
    print("I3-E  CRASH-WINDOW TAIL BEHAVIOR")
    print("=" * 78)

    # Crash entries: 2020-03-02 is the cleanest pre-bottom crash entry with a
    # populated universe (378 rows); 2020-02-03 is pre-crash-but-elevated;
    # 2020-09-01 is the post-recovery snapshot. Calm: 2021-06-01 / 2021-11-01.
    sets = {
        "CRASH 2020-03-02 (pre-Mar-23 bottom)": "put_2020-03-02.parquet",
        "CRASH 2020-04-01 (deep drawdown)": "put_2020-04-01.parquet",
        "PRE-CRASH 2020-02-03": "put_2020-02-03.parquet",
        "CALM 2021-06-01 (low-vol bull)": "put_2021-06-01.parquet",
        "CALM 2021-11-01": "put_2021-11-01.parquet",
    }
    blocks = []
    for lbl, nm in sets.items():
        d = load(nm)
        if d is None:
            print(f"  {lbl}: EMPTY snapshot, skipped")
            continue
        blocks.append(stat_block(d, lbl))

    cols = ["label", "n_rankable", "pct_positive_ev", "median_ev_dollars",
            "median_iv", "median_prob_assignment", "cvar5_median",
            "cvar5_p5_worst", "cvar99_evt_filled_pct", "tail_xi_median",
            "heavy_tail_pct"]
    bdf = pd.DataFrame(blocks)[cols]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n--- per-snapshot distribution summary ---")
    print(bdf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    # Direct crash-vs-calm contrast on the cleanest pair.
    crash = load("put_2020-03-02.parquet")
    calm = load("put_2021-06-01.parquet")
    print("\n--- CRASH(2020-03-02) vs CALM(2021-06-01) head-to-head ---")
    def med(d, c):
        return float(d[c].median())
    for c in ["iv", "ev_dollars", "cvar_5", "prob_assignment", "prob_profit"]:
        cr, ca = med(crash, c), med(calm, c)
        print(f"  median {c:18s}: crash={cr:12.4f}  calm={ca:12.4f}  "
              f"crash/calm={cr / ca if ca else float('nan'):.2f}x")
    # Does the engine refuse MORE in the crash? Compare positive-EV share.
    cr_pos = (crash["ev_dollars"] > 0).mean()
    ca_pos = (calm["ev_dollars"] > 0).mean()
    print(f"\n  share of candidates with POSITIVE ev_dollars: "
          f"crash={cr_pos:.1%}  calm={ca_pos:.1%}")
    print(f"  -> engine {'REFUSES MORE' if cr_pos < ca_pos else 'REFUSES LESS'} "
          f"in the crash window (lower positive-EV share = more refusals)")

    # Tail-fatness: are cvar_5 losses deeper (more negative) in the crash?
    cr_cvar = crash["cvar_5"].median()
    ca_cvar = calm["cvar_5"].median()
    print(f"\n  median cvar_5 (per-contract $ tail loss; more negative = fatter "
          f"left tail):\n     crash={cr_cvar:.1f}  calm={ca_cvar:.1f}  "
          f"-> crash tail is {'DEEPER' if cr_cvar < ca_cvar else 'SHALLOWER'} "
          f"({cr_cvar / ca_cvar:.2f}x)")

    # Heavy-tail flag rate.
    print(f"\n  heavy_tail flag rate: crash={crash['heavy_tail'].mean():.1%} "
          f"calm={calm['heavy_tail'].mean():.1%}")

    # IV-driven premium illusion check: in a crash IV is high -> premium high
    # -> a NAIVE 'premium harvested' view looks great, but does ev_dollars
    # (probability-weighted incl. tails) stay disciplined?
    print("\n  --- premium-illusion discipline check (crash 2020-03-02) ---")
    cr = crash.copy()
    hi_iv = cr[cr["iv"] > cr["iv"].median()]
    print(f"     high-IV half (median iv={hi_iv['iv'].median():.2f}): "
          f"median premium=${hi_iv['premium'].median():.2f} "
          f"BUT median ev_dollars=${hi_iv['ev_dollars'].median():.1f}, "
          f"positive-EV share={ (hi_iv['ev_dollars'] > 0).mean():.1%}")
    print(f"     -> if median ev_dollars is modest/negative despite fat premiums,")
    print(f"        the engine is NOT fooled by crash-inflated premium (good).")

    print("\nDONE.")


if __name__ == "__main__":
    main()
