"""
Refresh full dividend history -> data/bloomberg/sp500_dividends.csv

The connector reads this for covered-call timing (get_next_dividend). The prior
file had dividends declared only through 2026-03-20 and just ~10 forward ex-dates.
This re-pulls the FULL declared history for every current SPX member via the
dividend-history bulk field and overwrites the file (history is small).

Source: blp.bds(ticker, "EQY_DVD_HIST_ALL") -- returns ALL declared dividend
events (incl. forward-declared ex-dates, splits, specials) back to inception.
Columns returned: Declared Date, Ex-Date, Record Date, Payable Date,
Dividend Amount, Dividend Frequency, Dividend Type.

Schema preserved EXACTLY (committed order):
  declared_date,ex_date,record_date,payable_date,dividend_amount,
  dividend_frequency,dividend_type,ticker
Ticker format "A UN" (SPX member code, no " Equity" suffix). Dates YYYY-MM-DD
(blank where Bloomberg has none). Dedupe on (ticker, ex_date) keep-last.

Env knobs:
  SWE_DVD_SMOKE=1   pull 5 tickers, print, NO write.
  SWE_PULL_NO_WRITE pull+print, skip the write.
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
OUT = os.path.join(DATA, "sp500_dividends.csv")

COLMAP = {
    "Declared Date": "declared_date",
    "Ex-Date": "ex_date",
    "Record Date": "record_date",
    "Payable Date": "payable_date",
    "Dividend Amount": "dividend_amount",
    "Dividend Frequency": "dividend_frequency",
    "Dividend Type": "dividend_type",
}
DATE_COLS = ["declared_date", "ex_date", "record_date", "payable_date"]
OUT_COLS = [
    "declared_date", "ex_date", "record_date", "payable_date",
    "dividend_amount", "dividend_frequency", "dividend_type", "ticker",
]


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def get_members():
    m = to_native(blp.bds("SPX Index", "INDX_MWEIGHT"))
    col = [c for c in m.columns if "member" in c.lower() and "ticker" in c.lower()][0]
    return m[col].tolist()  # e.g. "A UN", "AAPL UW"


def pull_one(member: str) -> pd.DataFrame | None:
    df = to_native(blp.bds(member + " Equity", "EQY_DVD_HIST_ALL"))
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns=COLMAP)
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ticker"] = member
    keep = [c for c in OUT_COLS if c in df.columns]
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OUT_COLS]


def main():
    smoke = bool(os.environ.get("SWE_DVD_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    members = get_members()
    print(f"{len(members)} SPX members")
    if smoke:
        members = ["KO UN", "AAPL UW", "IBM UN", "JPM UN", "NVDA UW"]
        print(f"SMOKE: {members}")

    frames, payers = [], 0
    for i, mb in enumerate(members, 1):
        d = pull_one(mb)
        if d is not None and len(d):
            frames.append(d)
            payers += 1
        if i % 50 == 0:
            print(f"  {i}/{len(members)} processed, {payers} payers", flush=True)

    alld = pd.concat(frames, ignore_index=True)
    alld = alld.drop_duplicates(subset=["ticker", "ex_date"], keep="last")
    alld = alld.sort_values(["ticker", "ex_date"]).reset_index(drop=True)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    fwd = (alld["ex_date"] > today).sum()
    print(f"\nrows={len(alld):,}  payers={payers}  "
          f"ex_date {alld['ex_date'].min()} -> {alld['ex_date'].max()}  "
          f"declared_date max={alld['declared_date'].max()}  forward_ex(>{today})={fwd}")

    if smoke:
        print(alld.groupby("ticker")["ex_date"].agg(["min", "max", "count"]).to_string())
        return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written.")
        return
    alld.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(alld):,} rows)")


if __name__ == "__main__":
    main()
