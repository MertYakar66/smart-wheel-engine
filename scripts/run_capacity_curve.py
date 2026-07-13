"""Driver for the V4 capacity-curve harness (plan doc section 7).

    python scripts/run_capacity_curve.py run --config 24t     # pilot, ~30-45 min
    python scripts/run_capacity_curve.py run --config 100t    # terminal-scale, ~4-5 h

One shared daily rank serves the whole (N x proxy-ratio [+ control]) grid;
outputs land under ``$SWE_VALIDATION_DIR`` or the gitignored
``data_processed/validation/capacity_curve/``.  Measurement-only
(CLAUDE.md section 2): loose-mode trackers, rail pinned off, zero engine
changes; the knee is reported AS A FUNCTION of the stock-ADV proxy ratio,
never as a point estimate (section 7.0(4)).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backtests import capacity_curve as cc  # noqa: E402

DEFAULT_OUT_DIR = (
    Path(os.environ.get("SWE_VALIDATION_DIR", str(_REPO_ROOT / "data_processed" / "validation")))
    / "capacity_curve"
)

#: Pilot mirrors the S32 config shape (24t / $1M / 2022-2024); 100t mirrors S34.
CONFIGS = {
    "24t": {"universe": "UNIVERSE_24", "top_n": 10},
    "100t": {"universe": "UNIVERSE_100", "top_n": 15},
}
BASE_CAPITAL = 1_000_000.0
START, END = "2022-01-03", "2024-12-31"
MAX_NEW_PER_DAY = 3


def _universe(name: str) -> list[str]:
    from backtests.regression.universes import UNIVERSE_24, UNIVERSE_100

    return list(UNIVERSE_24 if name == "UNIVERSE_24" else UNIVERSE_100)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = CONFIGS[args.config]
    print(
        f"[capacity_curve] config={args.config} base=${BASE_CAPITAL:,.0f} "
        f"window={START}..{END} ladder={list(cc.LADDER)} ratios={list(cc.PROXY_RATIOS)} "
        f"+ control",
        flush=True,
    )
    result = cc.run_capacity_ladder(
        base_capital=BASE_CAPITAL,
        tickers=_universe(cfg["universe"]),
        start=START,
        end=END,
        top_n=cfg["top_n"],
        max_new_per_day=MAX_NEW_PER_DAY,
    )
    points = result["points"]
    report = {
        "meta": {
            "config": args.config,
            "base_capital": BASE_CAPITAL,
            "window": [START, END],
            "ladder": list(cc.LADDER),
            "ratios": list(cc.PROXY_RATIOS),
            "impact_k": cc.IMPACT_K,
            "participation_cap": cc.PARTICIPATION_CAP,
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "points": points,
        "linearity_control": cc.linearity_check(points),
        "knee_table": cc.knee_table(points),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"capacity_report_{args.config}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[capacity_curve] wrote {out}", flush=True)

    lin = report["linearity_control"]
    print(f"\n=== V4 capacity curve ({args.config}) ===")
    print(
        f"  linearity control: {lin['verdict']} "
        f"(return/N spread {lin.get('return_per_n_spread', float('nan')):.2e} over "
        f"{lin.get('n_unthrottled', 0)} unthrottled points; throttled: "
        f"{lin.get('throttled_points', {})})"
    )
    print("  arm      N   return%   impact-share  part_ref  bp_ref  opens")
    for k in sorted(points):
        v = points[k]
        tag = "ctrl" if v["ratio"] is None else f"r={v['ratio']:g}"
        print(
            f"  {tag:>7} {v['n_contracts']:3d}  {v['return_pct']:+8.2f}  "
            f"{v['impact_share_of_premium']:12.4f}  {v['part_refused']:8d}  "
            f"{v['bp_refused']:6d}  {v['opens']:5d}"
        )
    for row in report["knee_table"]:
        print(f"  knee @ r={row['ratio']:g}: N* = {row['knee_n']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["run"])
    p.add_argument("--config", choices=sorted(CONFIGS), default="24t")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
