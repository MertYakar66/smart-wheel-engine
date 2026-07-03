"""
Backfill corporate actions -> data/bloomberg/sp500_corporate_actions.csv

Same source as sp500_dividends (blp.bds EQY_DVD_HIST_ALL) reshaped into the
corporate-actions schema. Verified: the prior committed corp_actions file is a
2015+ window of exactly this field (99.6% of rows match a dividend event, 100%
equal values, announcement_date == declared_date). This backfills to the earliest
available (~1962) and captures ALL events (no (ticker,ex_date) dedup, so a special
+ regular sharing an ex-date are both kept).

Schema preserved EXACTLY:
  announcement_date,effective_date,action_type,ratio,amount,ticker
Mapping from EQY_DVD_HIST_ALL:
  announcement_date <- Declared Date     effective_date <- Ex-Date
  action_type       <- Dividend Type
  ratio  <- Dividend Amount  IF action_type is a share-ratio action
  amount <- Dividend Amount  otherwise (cash / value-per-share)
RATIO actions (verified vs committed file): Stock Split, Stock Dividend, Split-Off
(and any "*Split*" variant). Everything else -> amount. Ticker "A UN".

Env knobs:
  SWE_CA_SMOKE=1    pull 5 tickers, print, NO write.
  SWE_PULL_NO_WRITE pull+print, skip write.
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
OUT = os.path.join(DATA, "sp500_corporate_actions.csv")
OUT_COLS = ["announcement_date", "effective_date", "action_type", "ratio", "amount", "ticker"]
RATIO_TYPES = {"Stock Split", "Stock Dividend", "Split-Off"}


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def is_ratio(t) -> bool:
    if not isinstance(t, str):
        return False
    return t in RATIO_TYPES or "split" in t.lower()


def get_members():
    m = to_native(blp.bds("SPX Index", "INDX_MWEIGHT"))
    col = [c for c in m.columns if "member" in c.lower() and "ticker" in c.lower()][0]
    return m[col].tolist()


def pull_one(member: str) -> pd.DataFrame | None:
    df = to_native(blp.bds(member + " Equity", "EQY_DVD_HIST_ALL"))
    if df is None or len(df) == 0:
        return None
    df = df.rename(columns={
        "Declared Date": "announcement_date", "Ex-Date": "effective_date",
        "Dividend Type": "action_type", "Dividend Amount": "_val",
    })
    for c in ("announcement_date", "effective_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.strftime("%Y-%m-%d")
    val = pd.to_numeric(df.get("_val"), errors="coerce")
    rt = df["action_type"].apply(is_ratio)
    df["ratio"] = val.where(rt)
    df["amount"] = val.where(~rt)
    df["ticker"] = member
    for c in OUT_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    return df[OUT_COLS]


def main():
    smoke = bool(os.environ.get("SWE_CA_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    members = get_members()
    print(f"{len(members)} SPX members")
    if smoke:
        members = ["KO UN", "AAPL UW", "ABNB UW", "APD UN", "CMI UN"]
        print(f"SMOKE: {members}")

    frames = []
    for i, mb in enumerate(members, 1):
        d = pull_one(mb)
        if d is not None and len(d):
            frames.append(d)
        if i % 50 == 0:
            print(f"  {i}/{len(members)}", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=OUT_COLS, keep="first")  # exact-dup removal only
    df = df.sort_values(["ticker", "effective_date"]).reset_index(drop=True)
    eff = df["effective_date"].dropna()
    print(f"\nrows={len(df):,}  tickers={df['ticker'].nunique()}  "
          f"effective_date {eff.min()} -> {eff.max()}")
    print("action_type breakdown:")
    print(df["action_type"].value_counts().head(20).to_string())

    if smoke:
        print(df[df.action_type.isin(['Stock Split','Spinoff','Special Cash'])].head(8).to_string(index=False))
        return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return
    df.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
