"""
Backfill point-in-time SPX membership -> data/bloomberg/sp500_index_membership.csv
and derive the delisted/missing universe -> data/bloomberg/_delisted_universe.csv

Survivorship fix: the prior file was quarterly 2015-2026 only. blp.bds with
INDX_MWEIGHT_HIST + END_DATE_OVERRIDE returns the AS-OF constituent list (incl.
point-in-time delisted Bloomberg IDs like "0111145D UN") back to 1990-04-01
(verified floor: 1990-01-01 -> 0 names, 1990-04-01 -> 500). Pulled QUARTERLY at
quarter-start as_of (matching the existing cadence) and OVERWRITTEN.

Schema preserved EXACTLY:
  member_ticker_and_exchange_code,percentage_weight,as_of_date
NOTE: percentage_weight is the all-zeros sentinel (~ -2.4e-14) at every date --
historical index weights are NOT entitled at this tier (same as the prior file and
INDX_MWEIGHT). The MEMBERSHIP (names + as_of) is the valuable, available part.

Delisted universe: distinct historical members MINUS the tickers we already have
OHLCV for (sp500_ohlcv.csv) -> _delisted_universe.csv with first/last in-index
quarter, snapshot count, and (best-effort) NAME. This drives Phase 2.

Env knobs:
  SWE_MEM_SMOKE=1   pull 3 quarters only, print, NO write.
  SWE_PULL_NO_WRITE pull+print, skip writes.
  SWE_MEM_FLOOR     first as_of quarter (default 1990-04-01).
  SWE_MEM_END       last  as_of quarter (default 2026-04-01).
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
OUT = os.path.join(DATA, "sp500_index_membership.csv")
DELISTED = os.path.join(DATA, "_delisted_universe.csv")
OHLCV = os.path.join(DATA, "sp500_ohlcv.csv")
OUT_COLS = ["member_ticker_and_exchange_code", "percentage_weight", "as_of_date"]


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def quarter_grid(floor: str, end: str) -> list[str]:
    qs = pd.date_range(start=floor, end=end, freq="QS")  # quarter starts
    return [d.strftime("%Y-%m-%d") for d in qs]


def pull_snapshot(as_of: str) -> pd.DataFrame | None:
    r = to_native(blp.bds("SPX Index", "INDX_MWEIGHT_HIST",
                          END_DATE_OVERRIDE=as_of.replace("-", "")))
    if r is None or len(r) == 0:
        return None
    mcol = [c for c in r.columns if "member" in c.lower()][0]
    wcol = [c for c in r.columns if "weight" in c.lower() or "percent" in c.lower()]
    out = pd.DataFrame({
        "member_ticker_and_exchange_code": r[mcol],
        "percentage_weight": r[wcol[0]] if wcol else pd.NA,
        "as_of_date": as_of,
    })
    return out[OUT_COLS]


def best_effort_names(tickers: list[str]) -> dict:
    """bdp NAME for delisted tickers; tolerate BAD_SEC (backend skips them)."""
    names = {}
    for i in range(0, len(tickers), 50):
        ck = [t + " Equity" for t in tickers[i:i + 50]]
        try:
            r = to_native(blp.bdp(tickers=ck, flds=["NAME"]))
            if r is not None and len(r):
                r.columns = [c.lower() for c in r.columns]
                for _, row in r.iterrows():
                    names[str(row["ticker"]).replace(" Equity", "")] = row["value"]
        except Exception:
            pass
    return names


def main():
    smoke = bool(os.environ.get("SWE_MEM_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    floor = os.environ.get("SWE_MEM_FLOOR", "1990-04-01")
    end = os.environ.get("SWE_MEM_END", "2026-04-01")
    grid = quarter_grid(floor, end)
    if smoke:
        grid = grid[:2] + grid[-1:]
        print(f"SMOKE quarters: {grid}")
    print(f"{len(grid)} quarterly snapshots {grid[0]} -> {grid[-1]}", flush=True)

    frames, empties = [], []
    for i, q in enumerate(grid, 1):
        s = pull_snapshot(q)
        if s is None or len(s) == 0:
            empties.append(q)
        else:
            frames.append(s)
        if i % 20 == 0:
            print(f"  {i}/{len(grid)} ({q})", flush=True)
    if empties:
        print(f"  (no data for {len(empties)} early quarters: {empties[:4]}...)")

    mem = pd.concat(frames, ignore_index=True)
    mem = mem.drop_duplicates(subset=["member_ticker_and_exchange_code", "as_of_date"], keep="last")
    mem = mem.sort_values(["as_of_date", "member_ticker_and_exchange_code"]).reset_index(drop=True)
    snaps = sorted(mem["as_of_date"].unique())
    per = mem.groupby("as_of_date")["member_ticker_and_exchange_code"].nunique()
    distinct = mem["member_ticker_and_exchange_code"].nunique()
    print(f"\nMEMBERSHIP: {len(mem):,} rows, {len(snaps)} snapshots {snaps[0]} -> {snaps[-1]}, "
          f"{distinct} distinct names ever; names/snapshot min={per.min()} max={per.max()}")

    # --- delisted universe diff ---
    have = set()
    if os.path.exists(OHLCV):
        oh = pd.read_csv(OHLCV, usecols=["ticker"])
        # OHLCV stores "A UN Equity"; membership is "A UN" -> strip suffix to compare
        have = set(oh["ticker"].str.replace(" Equity", "", regex=False).unique())
    agg = mem.groupby("member_ticker_and_exchange_code")["as_of_date"].agg(
        first_in_index="min", last_in_index="max", n_snapshots="count").reset_index()
    agg = agg.rename(columns={"member_ticker_and_exchange_code": "ticker"})
    missing = agg[~agg["ticker"].isin(have)].copy()
    print(f"DELISTED/MISSING: {len(missing)} of {distinct} distinct names have NO OHLCV "
          f"(have prices for {len(have)} tickers)")

    if smoke:
        print(mem.head(3).to_string(index=False))
        print(missing.head(5).to_string(index=False))
        return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return

    mem.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(mem):,} rows)")

    names = best_effort_names(missing["ticker"].tolist())
    missing.insert(1, "name", missing["ticker"].map(names))
    missing = missing.sort_values(["last_in_index", "ticker"]).reset_index(drop=True)
    missing.to_csv(DELISTED, index=False)
    print(f"WROTE {os.path.normpath(DELISTED)} ({len(missing)} delisted/missing names; "
          f"{missing['name'].notna().sum()} resolved a NAME)")


if __name__ == "__main__":
    main()
