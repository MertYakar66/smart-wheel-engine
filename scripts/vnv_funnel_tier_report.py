#!/usr/bin/env python
"""Reliability/realism diagnostic: candidate funnel + forward-tier coverage.

Runs ``WheelRunner.rank_candidates_by_ev`` over a universe at a fixed PIT
``as_of`` and reports, as pure observability (no extra ``EVEngine.evaluate``
calls beyond the ranker's own):

1. **Funnel** — how many tickers entered, how many emitted a candidate, and a
   breakdown of *why* the rest dropped, by ``gate`` (data / history / event /
   strike / premium / chain_quality / ev_threshold) with example reasons. The
   drop records come straight from ``df.attrs["drops"]`` (the HT-A transparency
   surface), so "silent filtering" becomes an auditable table.
2. **Forward-tier distribution** — the count of each ``distribution_source``
   across the survivors. This is the *coverage* of the prob_profit Wilson
   sampling-CI feature: the CI is emitted only on the IID
   ``empirical_non_overlapping`` tier (``is_iid_forward_source``), so this
   table says how often a trader actually sees the interval vs. a suppressed
   (null) one.

§2: strictly read-only. Routes every candidate through the authoritative
ranker; surfaces, never overrides.

Usage:
    python scripts/vnv_funnel_tier_report.py --as-of 2026-03-20
    python scripts/vnv_funnel_tier_report.py --as-of 2026-03-20 --limit 60 \
        --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# The Windows console defaults to cp1252, which chokes on stray unicode in
# printed output; force UTF-8 so the report renders identically everywhere.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Bootstrap: allow ``python scripts/vnv_funnel_tier_report.py`` (direct
# invocation does not add the repo root to sys.path the way pytest does).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from engine.wheel_runner import WheelRunner  # noqa: E402

# The IID forward tier is the only one where prob_profit's Wilson sampling CI is
# an honest independent-trial interval. Mirrors
# ``engine.forward_distribution.is_iid_forward_source`` (PR #317); inlined here
# so this diagnostic also runs on plain ``main`` before that feature merges.
_IID_FORWARD_SOURCES = frozenset({"empirical_non_overlapping"})


def is_iid_forward_source(source: object) -> bool:
    return source in _IID_FORWARD_SOURCES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default="2026-03-20", help="PIT cutoff YYYY-MM-DD")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="universe_limit (None = full S&P 500 universe)",
    )
    ap.add_argument(
        "--min-ev",
        type=float,
        default=-1e9,
        help="min_ev_dollars (default -1e9 = keep all so the funnel shows EV-threshold drops)",
    )
    ap.add_argument("--json", default=None, help="optional path to write the report JSON")
    args = ap.parse_args()

    runner = WheelRunner()
    provider = type(runner.connector).__name__
    print(f"provider={provider}  as_of={args.as_of}  limit={args.limit}")

    t0 = time.time()
    df = runner.rank_candidates_by_ev(
        tickers=None,
        universe_limit=args.limit,
        top_n=10_000,
        min_ev_dollars=args.min_ev,
        as_of=args.as_of,
        include_diagnostic_fields=True,
    )
    dt = time.time() - t0

    drops = list(df.attrs.get("drops", []))
    n_survivors = len(df)
    # Universe size = survivors + uniquely-dropped tickers (a ticker can drop
    # at multiple grid cells; count distinct tickers for the entry tally).
    dropped_tickers = {d.get("ticker") for d in drops}
    survivor_tickers = set(df["ticker"]) if n_survivors else set()
    n_tickers_in = len(dropped_tickers | survivor_tickers)

    gate_counts = Counter(d.get("gate", "?") for d in drops)
    # one representative reason per gate
    gate_example: dict[str, str] = {}
    for d in drops:
        g = d.get("gate", "?")
        if g not in gate_example:
            gate_example[g] = f"{d.get('ticker')}: {d.get('reason')}"

    src_counts = Counter(df["distribution_source"]) if n_survivors else Counter()
    iid_rows = sum(v for k, v in src_counts.items() if is_iid_forward_source(k))
    ci_coverage = (iid_rows / n_survivors) if n_survivors else 0.0

    print(f"\nelapsed={dt:.1f}s")
    print("=== FUNNEL ===")
    print(
        f"tickers_in~{n_tickers_in}  survivor_rows={n_survivors}  "
        f"survivor_tickers={len(survivor_tickers)}  drop_records={len(drops)}"
    )
    for gate, c in gate_counts.most_common():
        print(f"  drop[{gate:>13}] = {c:5d}   e.g. {gate_example.get(gate)}")

    print("\n=== FORWARD-TIER DISTRIBUTION (Wilson-CI coverage) ===")
    for src, c in src_counts.most_common():
        flag = "CI-shown" if is_iid_forward_source(src) else "CI-suppressed"
        print(f"  {src:>28} = {c:5d}  ({100 * c / max(1, n_survivors):5.1f}%)  [{flag}]")
    print(f"  -> Wilson CI shown on {iid_rows}/{n_survivors} survivors ({100 * ci_coverage:.1f}%)")

    report = {
        "provider": provider,
        "as_of": args.as_of,
        "limit": args.limit,
        "elapsed_s": round(dt, 1),
        "tickers_in": n_tickers_in,
        "survivor_rows": n_survivors,
        "survivor_tickers": len(survivor_tickers),
        "drops_by_gate": dict(gate_counts),
        "drop_examples": gate_example,
        "tier_distribution": dict(src_counts),
        "ci_coverage_pct": round(100 * ci_coverage, 1),
    }
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
