"""SIM-200K forward-distribution tier probe (Opus review Required Change 1).

For each campaign window, ranks the full universe at three representative
dates (start+5bd, midpoint, end-5bd) and tabulates the per-row
``distribution_source`` the engine actually used — surfacing which windows
ran on the full-quality ``empirical_non_overlapping`` tier vs the degraded
``empirical_overlapping`` / bootstrap / HAR fallbacks. Read-only probe; the
campaign rank logs do not persist this column, so it is re-derived here from
the same deterministic engine + data.

Usage::

    python scripts/sim200k_tier_probe.py            # all eight windows
    python scripts/sim200k_tier_probe.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from backtests.regression.sim200k_reliability import CANONICAL, WINDOWS  # noqa: E402
from backtests.regression.universes import UNIVERSE_100  # noqa: E402


def _probe_dates(start: str, end: str) -> list[str]:
    days = pd.bdate_range(start, end)
    return [
        days[5].date().isoformat(),
        days[len(days) // 2].date().isoformat(),
        days[-5].date().isoformat(),
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="optional output JSON path")
    args = ap.parse_args()

    from engine.wheel_runner import WheelRunner

    runner = WheelRunner()
    print(f"connector: {type(runner.connector).__name__}", flush=True)

    out: dict[str, dict] = {}
    for window_id, start, end, _note in WINDOWS:
        per_date: dict[str, dict] = {}
        for as_of in _probe_dates(start, end):
            frame = runner.rank_candidates_by_ev(
                tickers=list(UNIVERSE_100),
                dte_target=CANONICAL["dte_target"],
                delta_target=CANONICAL["delta_target"],
                contracts=CANONICAL["contracts"],
                top_n=CANONICAL["top_n"],
                min_ev_dollars=-1e9,
                as_of=as_of,
                include_diagnostic_fields=True,
            )
            if frame is None or len(frame) == 0:
                per_date[as_of] = {"rows": 0, "tiers": {}}
                continue
            col = next(
                (c for c in frame.columns if "distribution_source" in c or c == "dist_source"),
                None,
            )
            tiers = (
                dict(Counter(str(v) for v in frame[col]))
                if col
                else {"<column missing>": len(frame)}
            )
            per_date[as_of] = {"rows": int(len(frame)), "tiers": tiers}
            print(f"{window_id} {as_of}: {per_date[as_of]}", flush=True)
        out[window_id] = per_date

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
