"""
Refresh + deep-backfill the Treasury yield curve -> data/bloomberg/treasury_yields.csv

The connector reads this file for the risk-free rate (get_risk_free_rate). The
prior file (yfinance) started 2021-05-07, so get_risk_free_rate returned NaN
before then -> Black-Scholes NaN -> every pre-2021 candidate R1a-blocked. This
backfills to 1994 so pre-2021 backtests become runnable, and refreshes the tail.

Schema preserved EXACTLY (no extra columns, same order):
  date,rate_3m,rate_6m,rate_2y,rate_10y

Tenor -> BBG ticker; field PX_LAST returns the yield ALREADY IN PERCENT:
  rate_1m  <- USGG1M Index    (hist floor ~2001)
  rate_3m  <- USGG3M Index
  rate_6m  <- USGG6M Index
  rate_2y  <- USGG2YR Index
  rate_5y  <- USGG5YR Index
  rate_10y <- USGG10YR Index
  rate_30y <- USGG30YR Index
  sofr     <- SOFRRATE Index  (Secured Overnight Financing Rate, %, floor 2018-04)

SCALE (load-bearing, D20): PX_LAST is already PERCENT (e.g. 5.2 == 5.2%). Write
it through UNCHANGED -- do NOT divide by 100 / convert to decimal; the connector
divides by 100 itself. A wrong scale here silently 100x's the risk-free rate.

rate_2y/rate_10y use the SAME tickers as sp500_macro us_2y/us_10y, so they must
match that file exactly on overlapping dates (a built-in cross-check).

Env knobs:
  SWE_TSY_SMOKE=1   bdp PX_LAST only (current levels), print, NO write.
  SWE_PULL_NO_WRITE pull+print, skip the write.
  SWE_PULL_FLOOR    backfill floor (default 1994-01-01).
  SWE_PULL_END      forward end   (default 2026-06-05 = today).
"""
from __future__ import annotations

import functools
import io
import os
import sys

import pandas as pd
from xbbg import blp

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8", errors="replace")

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg")
OUT = os.path.join(DATA, "treasury_yields.csv")

# output column -> BBG ticker (order is the committed column order: logical tenor order)
TREASURY_MAP = {
    "rate_1m": "USGG1M Index",
    "rate_3m": "USGG3M Index",
    "rate_6m": "USGG6M Index",
    "rate_2y": "USGG2YR Index",
    "rate_5y": "USGG5YR Index",
    "rate_10y": "USGG10YR Index",
    "rate_30y": "USGG30YR Index",
    "sofr": "SOFRRATE Index",
}
OUT_COLS = ["date"] + list(TREASURY_MAP)


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def pull_one(col: str, ticker: str, floor: str, end: str) -> pd.DataFrame:
    raw = to_native(blp.bdh(tickers=ticker, flds=["PX_LAST"], start_date=floor, end_date=end))
    if raw is None or len(raw) == 0:
        print(f"  [{col}] {ticker}: EMPTY")
        return pd.DataFrame(columns=["date", col])
    raw.columns = [c.lower() for c in raw.columns]
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    out = raw[["date", "value"]].rename(columns={"value": col}).dropna(subset=[col])
    print(f"  [{col}] {ticker}: {len(out):,} rows  {out['date'].min()} -> {out['date'].max()}")
    return out


def main():
    smoke = bool(os.environ.get("SWE_TSY_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    floor = os.environ.get("SWE_PULL_FLOOR", "1994-01-01")
    end = os.environ.get("SWE_PULL_END", "2026-06-05")

    print("=== smoke: bdp PX_LAST (current levels, must be percent) ===", flush=True)
    bdp = to_native(blp.bdp(tickers=list(TREASURY_MAP.values()), flds=["PX_LAST"]))
    bdp.columns = [c.lower() for c in bdp.columns]
    for col, tk in TREASURY_MAP.items():
        row = bdp[bdp["ticker"] == tk]
        v = row["value"].iloc[0] if len(row) else None
        print(f"  {col:8s} {tk:16s} PX_LAST={v}")
    if smoke:
        print("\nSWE_TSY_SMOKE -> stopping before historical pull.")
        return

    print(f"\n=== bdh PX_LAST {floor} -> {end} ===", flush=True)
    frames = [pull_one(col, tk, floor, end) for col, tk in TREASURY_MAP.items()]
    wide = functools.reduce(lambda a, b: pd.merge(a, b, on="date", how="outer"), frames)
    wide = wide[OUT_COLS].sort_values("date").reset_index(drop=True)

    print(f"\nmerged: {len(wide):,} rows  {wide['date'].min()} -> {wide['date'].max()}")
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written.")
        print(wide.head(3).to_string(index=False))
        print(wide.tail(3).to_string(index=False))
        return
    wide.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(wide):,} rows)")


if __name__ == "__main__":
    main()
