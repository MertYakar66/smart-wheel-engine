#!/usr/bin/env python3
"""
Backfill the feature store for every ticker in the universe.

Until this runs, only AAPL has computed features and the wheel runner ranks
blind for 499 / 500 names. This script calls ``FeaturePipeline.compute_all``
for each ticker, in parallel, and writes a CSV summary you can grep.

Usage::

    python scripts/backfill_features.py                 # full S&P
    python scripts/backfill_features.py --workers 8     # more parallelism
    python scripts/backfill_features.py --tickers AAPL MSFT NVDA   # subset
    python scripts/backfill_features.py --force         # recompute even if fresh
    python scripts/backfill_features.py --limit 20      # smoke N tickers
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Make stdout UTF-8-safe on Windows so printing the ✓ check mark doesn't
# crash the whole run. This must happen before any feature-pipeline import
# that might do its own print.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _compute_one(ticker: str, force: bool) -> tuple[str, bool, str, float]:
    """Run the feature pipeline for a single ticker in a worker process."""
    import time as _t

    try:
        from data.feature_pipeline import FeaturePipeline
    except Exception as e:  # import errors bubble up from worker
        return ticker, False, f"import: {e}", 0.0

    t0 = _t.perf_counter()
    try:
        pipe = FeaturePipeline()
        res = pipe.compute_all(ticker, force=force)
        ok = getattr(res, "success", True)
        detail = (
            getattr(res, "message", "")
            or f"{getattr(res, 'n_categories', '?')} categories"
        )
        return ticker, ok, detail, (_t.perf_counter() - t0) * 1000.0
    except Exception as e:
        return ticker, False, f"{type(e).__name__}: {e}", (_t.perf_counter() - t0) * 1000.0


def load_universe(path: str) -> list[str]:
    import pandas as pd

    df = pd.read_csv(path)
    # Accept either a "ticker" column or a single-column file.
    col = "ticker" if "ticker" in df.columns else df.columns[0]
    tickers = [str(t).strip().upper() for t in df[col].dropna().tolist()]
    # Drop tickers with suffixes like "BRK.B" that often don't have option data.
    # Keep them; let the pipeline itself fail on missing data.
    return [t for t in tickers if t and t != "NAN"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill feature store for S&P 500")
    ap.add_argument("--tickers", nargs="+", help="Explicit ticker subset")
    ap.add_argument(
        "--universe",
        default="data_raw/sp500_constituents_current.csv",
        help="Universe CSV (default: %(default)s)",
    )
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--force", action="store_true", help="Recompute even if fresh")
    ap.add_argument("--limit", type=int, default=0, help="Only run first N tickers")
    ap.add_argument(
        "--log-csv",
        default="data/features/_backfill_log.csv",
        help="Where to write per-ticker results",
    )
    args = ap.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_universe(args.universe)
    if args.limit:
        tickers = tickers[: args.limit]

    print(f"Backfilling features for {len(tickers)} tickers "
          f"with {args.workers} workers  force={args.force}")

    t0 = time.perf_counter()
    results: list[tuple[str, bool, str, float]] = []
    n_done = 0
    n_ok = 0
    n_fail = 0

    if args.workers <= 1:
        for t in tickers:
            res = _compute_one(t, args.force)
            results.append(res)
            n_done += 1
            if res[1]:
                n_ok += 1
            else:
                n_fail += 1
            print(f"  [{n_done:>3}/{len(tickers)}] {res[0]:<6} "
                  f"{'OK' if res[1] else 'FAIL':<4} {res[3]:>7.0f}ms  {res[2][:70]}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_compute_one, t, args.force): t for t in tickers}
            for fut in as_completed(futures):
                res = fut.result()
                results.append(res)
                n_done += 1
                if res[1]:
                    n_ok += 1
                else:
                    n_fail += 1
                print(f"  [{n_done:>3}/{len(tickers)}] {res[0]:<6} "
                      f"{'OK' if res[1] else 'FAIL':<4} {res[3]:>7.0f}ms  {res[2][:70]}",
                      flush=True)

    elapsed = time.perf_counter() - t0

    # Persist log
    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with log_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "ok", "detail", "ms"])
        for t, ok, d, ms in sorted(results):
            w.writerow([t, int(ok), d, f"{ms:.1f}"])

    print()
    print("=" * 70)
    print(f" Done in {elapsed:.1f}s  |  {n_ok} OK  |  {n_fail} FAIL")
    print(f" Log: {log_path}")
    print("=" * 70)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
