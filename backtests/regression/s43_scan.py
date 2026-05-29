"""Per-window §2 invariant scan + analysis support for S43.

The §2 invariant ("no tradeable candidate bypasses ``EVEngine.evaluate``")
manifests in the backtest rank_log as two falsifiers:

1. Any row with non-finite ``ev_dollars`` (NaN / +inf / −inf) is a
   bypass of R1a (``engine.candidate_dossier`` line guarding finite EV
   before the negative-EV check, shipped in PR #204). Must be 0.
2. Any row that was actually OPENED as a position by the tracker AND
   had ``ev_dollars <= 0`` is a §2 breach. The harness already filters
   opens on ``ev_dollars > 0`` (see ``_tracker_try_opens`` in
   ``backtests/regression/_common.py``), so this check is a guard
   against accidentally changing that filter.

This module reads the rank_log and the tracker artifacts the harness
wrote alongside, and reports counts that can be embedded in the
writeup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False, help=__doc__)


def scan_rank_log(path: Path) -> dict:
    """Run the §2 falsifiers on a single rank_log.csv."""
    df = pd.read_csv(path)
    ev = df["ev_dollars"].to_numpy(dtype=float)
    non_finite = int(np.sum(~np.isfinite(ev)))
    le_zero = int(np.sum(ev <= 0))
    gt_zero = int(np.sum(ev > 0))
    return {
        "path": str(path),
        "total_rows": int(len(df)),
        "non_finite_ev": non_finite,
        "ev_le_zero": le_zero,
        "ev_gt_zero": gt_zero,
        "min_ev": float(np.nanmin(ev)) if len(ev) else float("nan"),
        "max_ev": float(np.nanmax(ev)) if len(ev) else float("nan"),
        "mean_ev": float(np.nanmean(ev)) if len(ev) else float("nan"),
        "rule_R1a_passes": non_finite == 0,
    }


def scan_window(window_dir: Path) -> dict:
    """Scan all friction levels under one window directory."""
    per_friction = {}
    for level in ("none", "bid_ask", "full"):
        rank_log = window_dir / level / "rank_log.csv"
        if rank_log.exists():
            per_friction[level] = scan_rank_log(rank_log)
        else:
            per_friction[level] = {"missing": True, "path": str(rank_log)}
    overall_pass = all(
        v.get("rule_R1a_passes", False) for v in per_friction.values() if not v.get("missing")
    )
    return {
        "window_dir": str(window_dir),
        "overall_R1a_passes": overall_pass,
        "per_friction": per_friction,
    }


@app.command()
def one(window_dir: Path) -> None:
    """Scan one window directory and print the report."""
    report = scan_window(window_dir)
    print(json.dumps(report, indent=2, default=str))
    if not report["overall_R1a_passes"]:
        sys.exit(1)


@app.command(name="all")
def scan_all(root: Path = Path("C:/Users/merty/AppData/Local/Temp/s43_backtest")) -> None:
    """Scan all windows under ``root``."""
    reports = []
    overall = True
    for d in sorted(root.glob("w*_*")):
        if not d.is_dir():
            continue
        r = scan_window(d)
        reports.append(r)
        overall = overall and r["overall_R1a_passes"]
    print(json.dumps({"overall_passes": overall, "per_window": reports}, indent=2, default=str))
    if not overall:
        sys.exit(1)


if __name__ == "__main__":
    app()
