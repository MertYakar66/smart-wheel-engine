"""
Refresh + deep-backfill the macro OHLC panel -> data/bloomberg/sp500_macro.csv

This is a *cross-asset* panel (not SPX members), so it does NOT use the
per-name engine in _bbg_panel.py. Six fixed instruments, each a different
asset class, are pulled via blp.bdh and stacked long with an `instrument`
label. The connector (data/consolidated_loader.py get_macro) keys on that
`instrument` string + date, so the labels and the OHLC schema are load-bearing
and preserved exactly.

Schema (committed, preserved EXACTLY):
  date,open,high,low,close,instrument
Straight map: PX_OPEN->open, PX_HIGH->high, PX_LOW->low, PX_LAST->close.
(NOT rotated -- unlike the per-name sp500_ohlcv monolith.)

Instrument -> BBG ticker map, VERIFIED 2026-06-05 by reproducing the committed
2026-03-19/20 OHLC exactly (overlap match, all six):
  us_10y  -> USGG10YR Index      (10Y CMT yield OHLC)
  us_2y   -> USGG2YR Index       (2Y CMT yield OHLC)
  spx     -> SPX Index           (S&P 500 price)
  dxy     -> DXY Curncy          (US dollar index)
  gold    -> XAU Curncy          (spot gold; GC1 futures differ)
  wti_oil -> CL1 Comdty          (front WTI future; USCRWTIC spot is flat)

The refresh is purely ADDITIVE: existing rows are merged keep-last with the
fresh pull (which reproduces them identically on overlap), so the forward gap
[2026-03-21 -> END] and the deep backfill [FLOOR -> 2014] are filled while no
existing row is dropped. No fill is applied -- each instrument keeps its own
trading calendar (rates/fx/cmdty/equity differ), matching the committed file.

Env knobs:
  SWE_MACRO_SMOKE=1   pull ONE instrument (spx) full-range, print head/tail, NO write.
  SWE_MACRO_ONLY=<inst>  restrict to one instrument (e.g. wti_oil).
  SWE_PULL_NO_WRITE   pull+print, skip the write.
  SWE_PULL_FLOOR      backfill floor (default 1990-01-01).
  SWE_PULL_END        forward end   (default 2026-06-04, matches the rest of the refresh).
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
OUT = os.path.join(DATA, "sp500_macro.csv")

MACRO_MAP = {
    "us_10y": "USGG10YR Index",
    "us_2y": "USGG2YR Index",
    "spx": "SPX Index",
    "dxy": "DXY Curncy",
    "gold": "XAU Curncy",
    "wti_oil": "CL1 Comdty",
}

FLDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"]
FIELD_MAP = {"PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low", "PX_LAST": "close"}
OUT_COLS = ["date", "open", "high", "low", "close", "instrument"]


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def pull_instrument(inst: str, ticker: str, floor: str, end: str) -> pd.DataFrame | None:
    print(f"\n[{inst}] bdh {ticker}  {floor}..{end}", flush=True)
    raw = to_native(blp.bdh(tickers=ticker, flds=FLDS, start_date=floor, end_date=end))
    if raw is None or len(raw) == 0:
        print("  EMPTY -> skipping")
        return None
    raw.columns = [c.lower() for c in raw.columns]
    wide = raw.pivot_table(index="date", columns="field", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={k.lower() if k.lower() in wide.columns else k: v
                                for k, v in FIELD_MAP.items()})
    # field names come back upper-cased; map case-insensitively
    rn = {}
    for c in list(wide.columns):
        cu = c.upper()
        if cu in FIELD_MAP:
            rn[c] = FIELD_MAP[cu]
    wide = wide.rename(columns=rn)
    for col in ("open", "high", "low", "close"):
        if col not in wide.columns:
            wide[col] = pd.NA
    wide["date"] = pd.to_datetime(wide["date"]).dt.strftime("%Y-%m-%d")
    wide["instrument"] = inst
    out = wide[OUT_COLS].sort_values("date").reset_index(drop=True)
    print(f"  pulled {len(out):,} rows  {out['date'].min()} -> {out['date'].max()}")
    return out


def main():
    smoke = bool(os.environ.get("SWE_MACRO_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    only = os.environ.get("SWE_MACRO_ONLY")
    floor = os.environ.get("SWE_PULL_FLOOR", "1990-01-01")
    end = os.environ.get("SWE_PULL_END", "2026-06-04")

    existing = None
    if os.path.exists(OUT) and os.path.getsize(OUT) > 100:
        existing = pd.read_csv(OUT, dtype={"date": str})
        print(f"Existing sp500_macro.csv: {len(existing):,} rows, "
              f"{existing['date'].min()} -> {existing['date'].max()}, "
              f"instruments={sorted(existing['instrument'].unique())}")

    items = list(MACRO_MAP.items())
    if smoke:
        items = [("spx", MACRO_MAP["spx"])]
        print("SMOKE: spx only, full-range, NO write")
    elif only:
        items = [(only, MACRO_MAP[only])]

    fresh = []
    for inst, ticker in items:
        df = pull_instrument(inst, ticker, floor, end)
        if df is not None:
            fresh.append(df)

    if not fresh:
        print("Nothing pulled.")
        return
    fresh_all = pd.concat(fresh, ignore_index=True)

    if smoke:
        print("\n--- spx head ---");  print(fresh_all.head(3).to_string(index=False))
        print("\n--- spx tail ---");  print(fresh_all.tail(3).to_string(index=False))
        return

    # additive merge: existing first, fresh appended -> keep-last lets fresh win
    # ties (identical on overlap), while existing-only rows (e.g. holiday flats)
    # survive. Restrict existing to the instruments we just refreshed.
    if existing is not None and not only:
        combined = pd.concat([existing[OUT_COLS], fresh_all], ignore_index=True)
    elif existing is not None and only:
        keep = existing[existing["instrument"] != only]
        combined = pd.concat([keep[OUT_COLS], fresh_all], ignore_index=True)
    else:
        combined = fresh_all
    combined = combined.drop_duplicates(subset=["date", "instrument"], keep="last")
    combined = combined.sort_values(["instrument", "date"]).reset_index(drop=True)

    print("\n=== per-instrument coverage (post-merge) ===")
    for inst in sorted(combined["instrument"].unique()):
        s = combined[combined["instrument"] == inst]
        print(f"  {inst:9s} {s['date'].min()} -> {s['date'].max()}  ({len(s):,} rows)")
    print(f"  TOTAL {len(combined):,} rows")

    if no_write:
        print("\nSWE_PULL_NO_WRITE -> not written.")
        return
    combined.to_csv(OUT, index=False)
    print(f"\nWROTE {os.path.normpath(OUT)} ({len(combined):,} rows)")


if __name__ == "__main__":
    main()
