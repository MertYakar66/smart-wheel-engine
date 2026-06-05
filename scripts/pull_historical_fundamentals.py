"""
Backfill quarterly fundamentals -> data/bloomberg/sp500_historical_fundamentals.csv

Rewritten for xbbg 1.2.6 (old df.stack(level=0) path breaks on long narwhals
frames). Backfills to ~1990 at QUARTERLY cadence, refreshing the tail.

Schema preserved EXACTLY:
  date,ticker,pe_ratio,eps,revenue,ebitda,book_value_per_share
Field map (same as the prior file):
  pe_ratio<-PE_RATIO  eps<-IS_EPS  revenue<-SALES_REV_TURN
  ebitda<-EBITDA      book_value_per_share<-BOOK_VAL_PER_SH
bdh Per=Q Fill=P. Ticker "A UN" (no " Equity").

Env knobs:
  SWE_HF_SMOKE=1    pull 3 tickers, print, NO write (sanity-check field scale).
  SWE_PULL_NO_WRITE pull+print, skip write.
  SWE_PULL_FLOOR    floor (default 1990-01-01).  SWE_PULL_END (default today).
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
OUT = os.path.join(DATA, "sp500_historical_fundamentals.csv")
FMAP = {
    "PE_RATIO": "pe_ratio", "IS_EPS": "eps", "SALES_REV_TURN": "revenue",
    "EBITDA": "ebitda", "BOOK_VAL_PER_SH": "book_value_per_share",
}
FIELDS = list(FMAP)
OUT_COLS = ["date", "ticker", "pe_ratio", "eps", "revenue", "ebitda", "book_value_per_share"]
CHUNK = 25


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def get_members():
    m = to_native(blp.bds("SPX Index", "INDX_MWEIGHT"))
    col = [c for c in m.columns if "member" in c.lower() and "ticker" in c.lower()][0]
    return m[col].tolist()


def reshape(raw: pd.DataFrame) -> pd.DataFrame:
    raw.columns = [c.lower() for c in raw.columns]
    w = raw.pivot_table(index=["date", "ticker"], columns="field", values="value",
                        aggfunc="first").reset_index()
    w.columns.name = None
    w = w.rename(columns={c: FMAP[c.upper()] for c in w.columns if c.upper() in FMAP})
    w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    for c in FMAP.values():
        if c not in w.columns:
            w[c] = pd.NA
    return w[OUT_COLS]


def main():
    smoke = bool(os.environ.get("SWE_HF_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    floor = os.environ.get("SWE_PULL_FLOOR", "1990-01-01")
    end = os.environ.get("SWE_PULL_END", pd.Timestamp.today().strftime("%Y-%m-%d"))
    members = get_members()
    print(f"{len(members)} SPX members; {floor} -> {end} (Per=Q)")
    if smoke:
        members = ["A UN", "AAPL UW", "JPM UN"]
        print(f"SMOKE: {members}")

    chunks = []
    for i in range(0, len(members), CHUNK):
        ck = [m + " Equity" for m in members[i:i + CHUNK]]
        print(f"  {i + 1}-{min(i + CHUNK, len(members))}/{len(members)}", flush=True)
        try:
            raw = to_native(blp.bdh(tickers=ck, flds=FIELDS, start_date=floor,
                                    end_date=end, Per="Q", Fill="P"))
            if raw is not None and len(raw):
                chunks.append(reshape(raw))
        except Exception as e:
            print(f"    ERROR chunk {i}: {e}", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=list(FMAP.values()), how="all")
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"\nrows={len(df):,}  tickers={df['ticker'].nunique()}  {df['date'].min()} -> {df['date'].max()}")
    print("non-null per field:")
    for c in FMAP.values():
        print(f"  {c:22s} {df[c].notna().sum():,}")

    if smoke:
        for tk in ["AAPL UW"]:
            s = df[df["ticker"] == tk]
            print(f"\n--- {tk} head/tail ---")
            print(s.head(3).to_string(index=False)); print(s.tail(3).to_string(index=False))
        return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return
    df.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
