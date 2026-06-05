"""
Refresh + backfill SPDR sector ETFs -> data/bloomberg/sp500_sector_etfs.csv

Sector-rotation context. Refreshes to current and backfills each ETF to its
inception (the original 9 SPDRs launched Dec 1998; XLRE 2015-10, XLC 2018-06).

Schema preserved EXACTLY:
  date,open,high,low,close,volume,etf
Straight map PX_OPEN/HIGH/LOW/LAST/VOLUME -> open/high/low/close/volume (NOT
rotated). ETF label is the bare symbol (e.g. "XLF").

Env knobs:
  SWE_ETF_SMOKE=1   pull 2 ETFs, print head/tail, NO write.
  SWE_PULL_NO_WRITE pull+print, skip write.
  SWE_PULL_FLOOR    floor (default 1998-01-01).  SWE_PULL_END (default today).
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
from xbbg import blp

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8", errors="replace")

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg")
OUT = os.path.join(DATA, "sp500_sector_etfs.csv")

ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
FMAP = {"PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low", "PX_LAST": "close", "PX_VOLUME": "volume"}
OUT_COLS = ["date", "open", "high", "low", "close", "volume", "etf"]


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def pull_one(etf: str, floor: str, end: str) -> pd.DataFrame | None:
    raw = to_native(blp.bdh(tickers=etf + " US Equity", flds=FIELDS, start_date=floor, end_date=end))
    if raw is None or len(raw) == 0:
        print(f"  [{etf}] EMPTY"); return None
    raw.columns = [c.lower() for c in raw.columns]
    w = raw.pivot_table(index="date", columns="field", values="value", aggfunc="first").reset_index()
    w.columns.name = None
    rn = {c: FMAP[c.upper()] for c in w.columns if c.upper() in FMAP}
    w = w.rename(columns=rn)
    for c in ("open", "high", "low", "close", "volume"):
        if c not in w.columns:
            w[c] = pd.NA
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    w["etf"] = etf
    out = w[OUT_COLS].sort_values("date").reset_index(drop=True)
    print(f"  [{etf}] {len(out):,} rows  {out['date'].min()} -> {out['date'].max()}")
    return out


def main():
    smoke = bool(os.environ.get("SWE_ETF_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    floor = os.environ.get("SWE_PULL_FLOOR", "1998-01-01")
    end = os.environ.get("SWE_PULL_END", pd.Timestamp.today().strftime("%Y-%m-%d"))
    etfs = ETFS[:2] if smoke else ETFS
    print(f"{len(etfs)} ETFs; {floor} -> {end}" + ("  [SMOKE]" if smoke else ""))

    frames = [pull_one(e, floor, end) for e in etfs]
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    df = df.drop_duplicates(subset=["date", "etf"], keep="last")
    df = df.sort_values(["etf", "date"]).reset_index(drop=True)
    print(f"\nrows={len(df):,}  etfs={df['etf'].nunique()}  {df['date'].min()} -> {df['date'].max()}")
    print("per-ETF span:")
    for e in sorted(df["etf"].unique()):
        s = df[df["etf"] == e]
        print(f"  {e:5s} {s['date'].min()} -> {s['date'].max()}  ({len(s):,})")

    if smoke:
        print(df.head(2).to_string(index=False)); print(df.tail(2).to_string(index=False)); return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return
    df.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
