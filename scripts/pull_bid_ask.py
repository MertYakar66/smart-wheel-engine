"""
Pull underlying bid-ask spread -> data/bloomberg/sp500_bid_ask.csv (SIBLING file)

Cost-model input. BID_ASK_SPREAD is NOT a historical field at this tier (bdh/bdp
return empty), but PX_BID / PX_ASK ARE available daily back to ~1990. So the spread
is DERIVED. Written as a sibling (not a column on sp500_liquidity.csv) to keep the
connector-read liquidity monolith clean & <100 MB and avoid (date,ticker) merge
misalignment. Date floor aligned to liquidity (2015-01-01) -> today; PX_BID/PX_ASK
reach ~1990 if a deeper pull is later wanted.

Schema:
  date,ticker,px_bid,px_ask,bid_ask_spread,bid_ask_spread_bps
  bid_ask_spread     = px_ask - px_bid
  bid_ask_spread_bps = (px_ask - px_bid) / mid * 10000   (mid = (ask+bid)/2)
Ticker "A UN" (no " Equity").

Env knobs:
  SWE_BA_SMOKE=1    pull 3 tickers recent, print, NO write.
  SWE_PULL_NO_WRITE pull+print, skip write.
  SWE_PULL_FLOOR    floor (default 2015-01-01).  SWE_PULL_END (default today).
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
from xbbg import blp

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8", errors="replace")

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg")
OUT = os.path.join(DATA, "sp500_bid_ask.csv")
FIELDS = ["PX_BID", "PX_ASK"]
OUT_COLS = ["date", "ticker", "px_bid", "px_ask", "bid_ask_spread", "bid_ask_spread_bps"]
CHUNK = 30


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
    w = w.rename(columns={c: c.upper() for c in w.columns if c not in ("date", "ticker")})
    w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    bid = pd.to_numeric(w.get("PX_BID"), errors="coerce")
    ask = pd.to_numeric(w.get("PX_ASK"), errors="coerce")
    mid = (ask + bid) / 2.0
    out = pd.DataFrame({
        "date": w["date"], "ticker": w["ticker"],
        "px_bid": bid, "px_ask": ask,
        "bid_ask_spread": ask - bid,
        "bid_ask_spread_bps": np.where(mid > 0, (ask - bid) / mid * 10000.0, np.nan),
    })
    return out[OUT_COLS]


def main():
    smoke = bool(os.environ.get("SWE_BA_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    floor = os.environ.get("SWE_PULL_FLOOR", "2015-01-01")
    end = os.environ.get("SWE_PULL_END", pd.Timestamp.today().strftime("%Y-%m-%d"))
    members = get_members()
    print(f"{len(members)} SPX members; {floor} -> {end}")
    if smoke:
        members = ["AAPL UW", "A UN", "KO UN"]
        floor = "2026-05-20"
        print(f"SMOKE: {members} from {floor}")

    chunks = []
    for i in range(0, len(members), CHUNK):
        ck = [m + " Equity" for m in members[i:i + CHUNK]]
        print(f"  {i + 1}-{min(i + CHUNK, len(members))}/{len(members)}", flush=True)
        try:
            raw = to_native(blp.bdh(tickers=ck, flds=FIELDS, start_date=floor, end_date=end))
            if raw is not None and len(raw):
                chunks.append(reshape(raw))
        except Exception as e:
            print(f"    ERROR chunk {i}: {e}", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["px_bid", "px_ask"], how="all")
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    bps = df["bid_ask_spread_bps"]
    print(f"\nrows={len(df):,}  tickers={df['ticker'].nunique()}  {df['date'].min()} -> {df['date'].max()}")
    print(f"  spread_bps: median={bps.median():.2f}  p95={bps.quantile(0.95):.2f}  "
          f"neg={int((df['bid_ask_spread'] < 0).sum())}  nonpos_spread={int((df['bid_ask_spread'] <= 0).sum())}")

    if smoke:
        print(df.groupby("ticker").tail(2).to_string(index=False)); return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return
    df.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
