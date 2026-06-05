"""
Refresh sp500_vix_full.csv tail (VIX complex close history).

Long format: date,close,instrument. Six instruments (refresh to current; the
existing 2015 floor is kept -- deep VIX history lives in vix_term_structure):
  vix    <- VIX Index      vvix  <- VVIX Index
  vix3m  <- VIX3M Index    vix6m <- VIX6M Index
  vx1    <- UX1 Index      vx2   <- UX2 Index    (front two generic VIX futures)

Re-pulls each instrument from the existing floor to today and merges keep-last
(additive: extends the tail, reproduces overlap). Schema unchanged.

Env knobs:
  SWE_VIX_SMOKE=1   pull 2 instruments recent window, print, NO write.
  SWE_PULL_NO_WRITE pull+print, skip write.
  SWE_PULL_FLOOR    floor (default 2015-01-01).  SWE_PULL_END (default today).
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
OUT = os.path.join(DATA, "sp500_vix_full.csv")

VIX_MAP = {
    "vix": "VIX Index", "vvix": "VVIX Index",
    "vix3m": "VIX3M Index", "vix6m": "VIX6M Index",
    "vx1": "UX1 Index", "vx2": "UX2 Index",
}
OUT_COLS = ["date", "close", "instrument"]


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def pull_one(inst: str, ticker: str, floor: str, end: str) -> pd.DataFrame | None:
    raw = to_native(blp.bdh(tickers=ticker, flds=["PX_LAST"], start_date=floor, end_date=end))
    if raw is None or len(raw) == 0:
        print(f"  [{inst}] {ticker}: EMPTY"); return None
    raw.columns = [c.lower() for c in raw.columns]
    out = raw[["date", "value"]].rename(columns={"value": "close"}).dropna(subset=["close"])
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out["instrument"] = inst
    print(f"  [{inst}] {ticker}: {len(out):,} rows  {out['date'].min()} -> {out['date'].max()}")
    return out[OUT_COLS]


def main():
    smoke = bool(os.environ.get("SWE_VIX_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    floor = os.environ.get("SWE_PULL_FLOOR", "2015-01-01")
    end = os.environ.get("SWE_PULL_END", pd.Timestamp.today().strftime("%Y-%m-%d"))

    existing = None
    if os.path.exists(OUT) and os.path.getsize(OUT) > 50:
        existing = pd.read_csv(OUT, dtype={"date": str})
        print(f"existing: {len(existing):,} rows, {existing['date'].min()} -> {existing['date'].max()}")

    items = list(VIX_MAP.items())
    if smoke:
        items = [("vix", "VIX Index"), ("vx1", "UX1 Index")]
        floor = "2026-05-20"
        print(f"SMOKE: {[i[0] for i in items]} from {floor}")

    fresh = [pull_one(i, t, floor, end) for i, t in items]
    fresh = pd.concat([f for f in fresh if f is not None], ignore_index=True)

    if smoke:
        print(fresh.groupby("instrument")["date"].agg(["min", "max", "count"]).to_string()); return

    if existing is not None:
        combined = pd.concat([existing[OUT_COLS], fresh], ignore_index=True)
    else:
        combined = fresh
    combined = combined.drop_duplicates(subset=["date", "instrument"], keep="last")
    combined = combined.sort_values(["instrument", "date"]).reset_index(drop=True)
    print(f"\nrows={len(combined):,}  {combined['date'].min()} -> {combined['date'].max()}")
    for inst in sorted(combined["instrument"].unique()):
        s = combined[combined["instrument"] == inst]
        print(f"  {inst:6s} {s['date'].min()} -> {s['date'].max()}  ({len(s):,})")
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return
    combined.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
