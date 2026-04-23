#!/usr/bin/env python3
"""
Pull daily treasury-yield curve from Yahoo Finance and write it out in the
schema ``data/consolidated_loader.py`` already expects.

Yahoo Finance publishes CBOE-calculated Treasury yield indices under:
    ^IRX  — 13-week  (3M)
    ^FVX  — 5-year
    ^TNX  — 10-year
    ^TYX  — 30-year

Yahoo does NOT publish the 6M and 2Y tenors directly. We approximate the
2Y by linear interpolation between ^IRX (0.25y) and ^FVX (5y) — good
enough for a risk-free-rate lookup.

Output schema matches ``data/bloomberg/treasury_yields.csv``::

    date, rate_3m, rate_6m, rate_2y, rate_10y

We interpolate ``rate_6m`` from (3m, 2y) and leave the existing file's
30y tenor out to stay 1:1 with the consumer schema.

Usage
-----
    python scripts/pull_treasury_yields_yf.py --years 10
    python scripts/pull_treasury_yields_yf.py --incremental
    python scripts/pull_treasury_yields_yf.py --out data/bloomberg/treasury_yields.csv
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

OUT_CSV = _ROOT / "data" / "bloomberg" / "treasury_yields.csv"

YIELD_SYMBOLS = {
    "rate_3m": "^IRX",
    "rate_5y": "^FVX",
    "rate_10y": "^TNX",
    "rate_30y": "^TYX",
}


def _fetch(sym: str, start: date, end: date) -> pd.Series:
    h = yf.Ticker(sym).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=False,
    )
    if h is None or h.empty:
        return pd.Series(dtype=float)
    h = h.reset_index().rename(columns={"Date": "date", "Close": "close"})
    h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize()
    # Yahoo returns these as percentage, e.g. 4.25 = 4.25%; downstream code
    # expects the same (rate > 1 triggers the "/100" branch in get_risk_free_rate).
    return h.set_index("date")["close"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=10.0)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--incremental", action="store_true",
                    help="Append only rows newer than the last saved date")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    end_d = date.fromisoformat(args.end) if args.end else date.today()
    start_d = (
        date.fromisoformat(args.start) if args.start
        else end_d - timedelta(days=int(args.years * 365))
    )

    out_path = Path(args.out)
    if args.incremental and out_path.exists():
        existing = pd.read_csv(out_path, parse_dates=["date"])
        if not existing.empty:
            last = existing["date"].max().date()
            start_d = max(start_d, last + timedelta(days=1))
            if start_d > end_d:
                print(f"Already up to date (last={last}). Nothing to do.")
                return 0

    print(f"Pulling treasury yields  range={start_d}..{end_d}")
    t0 = time.perf_counter()
    series: dict[str, pd.Series] = {}
    for col, sym in YIELD_SYMBOLS.items():
        s = _fetch(sym, start_d, end_d)
        if s.empty:
            print(f"  {col:<8} ({sym}) EMPTY")
            continue
        series[col] = s
        print(f"  {col:<8} ({sym}) rows={len(s):<5}  {s.index.min().date()} to {s.index.max().date()}")

    if not series:
        print("No data fetched. Yahoo may be throttling — retry in a minute.")
        return 1

    df = pd.DataFrame(series).sort_index()
    # rate_6m: linear between 3m (0.25y tenor) and 5y.
    if "rate_3m" in df and "rate_5y" in df:
        # 6m at 0.5y -> weight 0.053 on 3m, 0.947 on 5y... actually 6m is just
        # 0.5y, so interpolate along tenor axis:
        #   t(3m)=0.25, t(5y)=5 → 6m at 0.5 → w3m = (5-0.5)/(5-0.25) = 0.947
        w3 = (5.0 - 0.5) / (5.0 - 0.25)
        df["rate_6m"] = w3 * df["rate_3m"] + (1 - w3) * df["rate_5y"]
    # rate_2y: interpolate between 3m and 5y at tenor 2y.
    if "rate_3m" in df and "rate_5y" in df:
        w3 = (5.0 - 2.0) / (5.0 - 0.25)
        df["rate_2y"] = w3 * df["rate_3m"] + (1 - w3) * df["rate_5y"]

    df = df.reset_index().rename(columns={"index": "date", "date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Columns we ship (compatible with the existing loader)
    final_cols = ["date", "rate_3m", "rate_6m", "rate_2y", "rate_10y"]
    df = df[[c for c in final_cols if c in df.columns]]

    if args.incremental and out_path.exists():
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df], ignore_index=True)
        df = df.drop_duplicates(subset="date", keep="last").sort_values("date")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {len(df)} rows → {out_path}")
    print(f"Done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
