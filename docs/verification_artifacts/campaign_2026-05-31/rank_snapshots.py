"""Produce monthly full-universe PUT ranked snapshots (shared by I1 + I2).

This is the expensive shared computation: rank the full 503-name universe
point-in-time at the first business day of each month, 2020-01 .. 2026-02, and
persist the complete ranked DataFrame per date. I1 (calibration) and I2 (P&L)
both consume these snapshots, so the engine is invoked once, identically.

PIT correctness is delegated to the engine: as_of plumbs PIT IV (d26a8d6),
PIT OHLCV (504-day survivorship gate), and HMM-on-history-up-to-as_of. Dealer/
skew/news/credit multipliers are structurally 1.0 on Bloomberg (no chain).
Event gate ON (realistic tradeable set).

Usage:
    python rank_snapshots.py --chunk 0 4      # process dates[0::4]
    python rank_snapshots.py --list           # just print the date list + counts
    python rank_snapshots.py --force          # re-rank even if snapshot exists
"""
from __future__ import annotations

import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import campaign_lib as L  # noqa: E402

import pandas as pd  # noqa: E402
from engine.wheel_runner import WheelRunner  # noqa: E402

SNAP_DIR = os.path.join(HERE, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

DTE_TARGET = 35
DELTA_TARGET = 0.25
TOP_N = 600                # > 503 so nothing is truncated
MIN_EV = -1e9             # keep negative-EV rows (needed for full-range calibration)


def monthly_dates() -> list[str]:
    """First business day of each month, 2020-01 .. 2026-02 (realize <= 2026-03-20)."""
    d = pd.date_range("2020-01-01", "2026-02-28", freq="BMS")
    return [x.date().isoformat() for x in d]


def snap_path(asof: str) -> str:
    return os.path.join(SNAP_DIR, f"put_{asof}.parquet")


def run_chunk(idx: int, stride: int, force: bool) -> None:
    dates = monthly_dates()
    mine = dates[idx::stride] if stride > 1 else dates
    runner = WheelRunner()
    uni = list(L.universe_503())
    print(f"[chunk {idx}/{stride}] {len(mine)} dates, universe={len(uni)}, "
          f"connector={type(runner.connector).__name__}", flush=True)
    logf = os.path.join(SNAP_DIR, f"_chunklog_{idx}_{stride}.txt")
    for asof in mine:
        sp = snap_path(asof)
        if os.path.exists(sp) and not force:
            print(f"  {asof}: exists, skip", flush=True)
            continue
        t = time.time()
        try:
            df = runner.rank_candidates_by_ev(
                tickers=uni, dte_target=DTE_TARGET, delta_target=DELTA_TARGET,
                top_n=TOP_N, min_ev_dollars=MIN_EV, as_of=asof,
                include_diagnostic_fields=True, max_as_of_staleness_days=10_000,
                use_event_gate=True,
            )
            df = df.copy()
            df["as_of"] = asof
            df.to_parquet(sp, index=False)
            dt = time.time() - t
            n_pos = int((df["ev_dollars"] > 0).sum()) if len(df) else 0
            line = f"  {asof}: rows={len(df)} ev>0={n_pos} elapsed={dt:.1f}s"
            print(line, flush=True)
            with open(logf, "a") as fh:
                fh.write(line + "\n")
        except Exception as ex:  # noqa: BLE001
            line = f"  {asof}: ERROR {type(ex).__name__}: {ex}"
            print(line, flush=True)
            with open(logf, "a") as fh:
                fh.write(line + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", nargs=2, type=int, metavar=("IDX", "STRIDE"), default=[0, 1])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        d = monthly_dates()
        done = sum(os.path.exists(snap_path(x)) for x in d)
        print(f"{len(d)} monthly dates {d[0]}..{d[-1]}; snapshots present: {done}/{len(d)}")
        return 0
    run_chunk(a.chunk[0], a.chunk[1], a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
