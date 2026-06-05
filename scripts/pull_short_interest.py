"""
Backfill short interest -> data/bloomberg/sp500_short_interest.csv

Sentiment input. Rewritten for xbbg 1.2.6 (the old df.stack(level=0) path breaks
on the long narwhals frames xbbg now returns). Backfills to the earliest
available short-interest history (~1990s) at MONTHLY cadence, refreshing the tail.

Schema preserved EXACTLY (committed order):
  date,ticker,short_interest,short_interest_pct_float,borrow_rate_net,shares_out,float_pct

Field map (verified against committed values):
  short_interest            <- SHORT_INT                (raw shares short)
  short_interest_pct_float  <- SI_PERCENT_EQUITY_FLOAT  (% of float)
  shares_out                <- EQY_SH_OUT               (millions)
  float_pct                 <- DERIVED EQY_FLOAT / EQY_SH_OUT * 100
  borrow_rate_net           <- EMPTY (EQUITY_SHORT_BORROW_RATE_NET not entitled)

bdh Per=M Fill=P -> month-end observations. Ticker "A UN" (no " Equity").

Env knobs:
  SWE_SI_SMOKE=1    pull 3 tickers, print recent + 2020-01, NO write (verify scale).
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
OUT = os.path.join(DATA, "sp500_short_interest.csv")
FIELDS = ["SHORT_INT", "SI_PERCENT_EQUITY_FLOAT", "EQY_SH_OUT", "EQY_FLOAT"]
OUT_COLS = ["date", "ticker", "short_interest", "short_interest_pct_float",
            "borrow_rate_net", "shares_out", "float_pct"]
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
    # field names come back upper-cased
    w = w.rename(columns={c: c.upper() for c in w.columns if c not in ("date", "ticker")})
    w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    out = pd.DataFrame({
        "date": w["date"], "ticker": w["ticker"],
        "short_interest": w.get("SHORT_INT"),
        "short_interest_pct_float": w.get("SI_PERCENT_EQUITY_FLOAT"),
        "borrow_rate_net": pd.NA,
        "shares_out": w.get("EQY_SH_OUT"),
        "float_pct": w.get("EQY_FLOAT") / w.get("EQY_SH_OUT") * 100,
    })
    return out[OUT_COLS]


def main():
    smoke = bool(os.environ.get("SWE_SI_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    floor = os.environ.get("SWE_PULL_FLOOR", "1990-01-01")
    end = os.environ.get("SWE_PULL_END", pd.Timestamp.today().strftime("%Y-%m-%d"))

    members = get_members()
    print(f"{len(members)} SPX members; {floor} -> {end} (Per=M)")
    if smoke:
        members = ["A UN", "ZTS UN", "AAPL UW"]
        print(f"SMOKE: {members}")

    chunks = []
    for i in range(0, len(members), CHUNK):
        ck = [m + " Equity" for m in members[i:i + CHUNK]]
        print(f"  {i + 1}-{min(i + CHUNK, len(members))}/{len(members)}", flush=True)
        try:
            raw = to_native(blp.bdh(tickers=ck, flds=FIELDS, start_date=floor,
                                    end_date=end, Per="M", Fill="P"))
            if raw is not None and len(raw):
                chunks.append(reshape(raw))
        except Exception as e:
            print(f"    ERROR chunk {i}: {e}", flush=True)

    df = pd.concat(chunks, ignore_index=True)
    df = df.dropna(subset=["short_interest", "short_interest_pct_float", "shares_out"], how="all")
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"\nrows={len(df):,}  tickers={df['ticker'].nunique()}  "
          f"{df['date'].min()} -> {df['date'].max()}")

    if smoke:
        for tk in ["A UN", "ZTS UN"]:
            s = df[df["ticker"] == tk]
            print(f"\n--- {tk} (first 2 + 2020-01 + last 2) ---")
            print(s.head(2).to_string(index=False))
            print(s[s["date"].str.startswith("2020-01")].to_string(index=False))
            print(s.tail(2).to_string(index=False))
        return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return
    df.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
