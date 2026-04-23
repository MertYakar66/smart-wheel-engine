#!/usr/bin/env python3
"""
Pull upcoming + recent earnings dates from yfinance.

Activates the event gate's earnings-lockout path for the full universe
(today ``sp500_earnings.csv`` only carries Bloomberg's historical rows,
and the event gate needs *upcoming* dates to block trades).

yfinance's ``Ticker.earnings_dates`` returns both past-and-upcoming
earnings with EPS estimates and surprises. We normalise the schema to
match the existing loader::

    year/period (synthesised)
    announcement_date      → date
    announcement_time      → "BMO" / "AMC" / "" (from hour component)
    earnings_eps           → Reported EPS
    comparable_eps         → (left NaN — not in yfinance)
    estimate_eps           → EPS Estimate
    ticker                 → "AAPL US Equity"

Output
------
``data/bloomberg/sp500_earnings_yf.csv``  (parallel to sp500_earnings.csv)

Run
---
    python scripts/pull_earnings_yf.py             # full S&P
    python scripts/pull_earnings_yf.py --tickers AAPL MSFT NVDA
    python scripts/pull_earnings_yf.py --workers 6
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

logger = logging.getLogger(__name__)
OUT_CSV = _ROOT / "data" / "bloomberg" / "sp500_earnings_yf.csv"


def load_universe(pit_date: str | None = None) -> list[str]:
    from data.consolidated_loader import get_bloomberg_loader
    L = get_bloomberg_loader()
    u = L.get_universe_as_of(pit_date)
    return sorted({t for t in u if all(c.isalpha() or c == "." for c in t)})


def _time_label(ts: pd.Timestamp) -> str:
    """Classify an earnings timestamp as BMO (before open), DMT (during),
    or AMC (after close). Based on NYSE 9:30-16:00 local."""
    # yfinance timestamps are tz-aware in US/Eastern. Convert if needed.
    if ts.tz is not None:
        ts_et = ts.tz_convert("America/New_York")
    else:
        ts_et = ts
    h = ts_et.hour
    if h < 9 or (h == 9 and ts_et.minute < 30):
        return "BMO"
    if h >= 16:
        return "AMC"
    return "DMT"


def _pull_one(ticker: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        ed = t.earnings_dates
    except Exception as e:
        return pd.DataFrame({"_error": [f"{type(e).__name__}: {e}"[:100]]})
    if ed is None or not hasattr(ed, "empty") or ed.empty:
        return pd.DataFrame()
    df = ed.reset_index().rename(
        columns={
            "Earnings Date": "ts",
            "EPS Estimate": "estimate_eps",
            "Reported EPS": "earnings_eps",
            "Surprise(%)": "surprise_pct",
        }
    )
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    df["announcement_date"] = df["ts"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    df["announcement_time"] = df["ts"].map(_time_label)
    # Fiscal period — approximate as "YYYY:QN" based on the calendar date.
    cal = df["ts"].dt.tz_convert("America/New_York")
    df["year/period"] = cal.dt.year.astype(str) + ":Q" + ((cal.dt.month - 1) // 3 + 1).astype(str)
    df["comparable_eps"] = np.nan
    df["ticker"] = f"{ticker} US Equity"
    keep = ["year/period", "announcement_date", "announcement_time",
            "earnings_eps", "comparable_eps", "estimate_eps", "ticker"]
    return df[keep]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+")
    ap.add_argument("--pit-date")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT_CSV))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tickers = (
        [t.upper() for t in args.tickers]
        if args.tickers else
        load_universe(args.pit_date)
    )
    print(f"Earnings-calendar pull  tickers={len(tickers)}  workers={args.workers}")

    t0 = time.perf_counter()
    n_done = n_err = n_empty = 0
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_pull_one, t): t for t in tickers}
        for fut in as_completed(futs):
            df = fut.result()
            n_done += 1
            if df.empty:
                n_empty += 1
            elif "_error" in df.columns:
                n_err += 1
            else:
                frames.append(df)
            if n_done % 50 == 0:
                print(f"  [{n_done:>4}/{len(tickers)}]  OK={len(frames)} empty={n_empty} err={n_err}", flush=True)

    if not frames:
        print("No earnings data fetched. Yahoo may be throttling — retry in a minute.")
        return 1

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values(["ticker", "announcement_date"]).drop_duplicates(
        subset=["ticker", "announcement_date"], keep="last"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)

    today = pd.Timestamp.now(tz="America/New_York").date()
    upcoming = all_df[pd.to_datetime(all_df["announcement_date"]).dt.date >= today]

    elapsed = time.perf_counter() - t0
    print()
    print(f"Wrote {len(all_df)} rows → {out}")
    print(f"Upcoming earnings (≥ today): {len(upcoming)} across {upcoming['ticker'].nunique()} tickers")
    print(f"Done in {elapsed:.1f}s  |  {n_err} errors  |  {n_empty} empties")
    # Partial success is still success — yfinance rate-limits aggressively
    # and a re-run later in the day will pick up the ones we missed. Only
    # fail if we got literally no data.
    return 0 if len(all_df) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
