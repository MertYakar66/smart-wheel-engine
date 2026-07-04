"""T0-2 — Moneyness IV skew surface (BDH ...%MNY_DF). Bloomberg lab 2026-06-17.

Highest-value pull: reactivates the skew the ATM-only vol_iv can't express. The 100%MNY_DF
column == current ATM IV (brings ATM current to 06-17). Deep archive ABSENT in this clone → fresh.

FLDS-verified by NON-NULL VALUE count (not just field echo): the populated Bloomberg equity surface
is the documented 5x5 — tenors {30DAY,60DAY,3MTH,6MTH,12MTH} x moneyness {90,95,100,105,110}%MNY_DF.
Long tenors use MTH naming (90/180/365DAY return all-NaN); wings {80,120} unpopulated for equities.
25 fields => one bdh request per ticker-chunk. Wide output: date, ticker, iv_{tenor_d}_{mny}
(3MTH->90d, 6MTH->180d, 12MTH->365d). Ticker stored stripped ('AAPL UW'). xbbg 1.3.0 narwhals->pivot.

Usage:  python pull_iv_surface.py <start_idx> <end_idx>   (batch over the sorted ohlcv universe)
Appends to staging/iv_surface/sp500_iv_surface.csv (header on first batch).
"""
import os
import sys
import pandas as pd
from xbbg import blp

START, END = "2010-01-01", "2026-06-17"
OUT = os.path.dirname(__file__)
OUTFILE = os.path.join(OUT, "sp500_iv_surface.csv")
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
TENOR_MAP = {"30DAY": "30d", "60DAY": "60d", "3MTH": "90d", "6MTH": "180d", "12MTH": "365d"}
MNY = [90, 95, 100, 105, 110]
FIELDS = [f"{tok}_IMPVOL_{m}.0%MNY_DF" for tok in TENOR_MAP for m in MNY]  # 25
COLS = ["date", "ticker"] + [f"iv_{d}_{m}" for d in TENOR_MAP.values() for m in MNY]


def fld_to_col(f):
    tok = f.split("_IMPVOL_")[0]
    m = f.split("_IMPVOL_")[1].replace(".0%MNY_DF", "")
    return f"iv_{TENOR_MAP[tok]}_{m}"


def pull_chunk(bbg_tickers):
    nw = blp.bdh(bbg_tickers, FIELDS, START, END)
    d = nw.to_native() if hasattr(nw, "to_native") else nw
    if not {"ticker", "date", "field", "value"}.issubset(d.columns):
        return pd.DataFrame(columns=COLS)
    d = d[d["value"].notna()].copy()
    d["col"] = d["field"].map(fld_to_col)
    d["ticker"] = d["ticker"].str.replace(" Equity", "", regex=False)
    w = d.pivot_table(index=["date", "ticker"], columns="col", values="value", aggfunc="first").reset_index()
    for c in COLS:
        if c not in w.columns:
            w[c] = pd.NA
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    iv_cols = [c for c in COLS if c.startswith("iv_")]
    w[iv_cols] = w[iv_cols].round(2)  # 0.01% vol precision; keeps gzip < GitHub's 100MB limit
    return w[COLS].sort_values(["ticker", "date"])


if __name__ == "__main__":
    s, e = int(sys.argv[1]), int(sys.argv[2])
    uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
    batch = uni[s:e]
    print(f"universe {len(uni)}; batch [{s}:{e}] = {len(batch)} names")
    first = not os.path.exists(OUTFILE)
    total = 0
    for i in range(0, len(batch), 20):
        chunk = batch[i:i + 20]
        w = pull_chunk(chunk)
        if len(w):
            w.to_csv(OUTFILE, mode="a", header=first, index=False)
            first = False
            total += len(w)
        cov = w.drop(columns=["date", "ticker"]).notna().mean().mean() if len(w) else 0
        print(f"  [{s+i}:{s+i+len(chunk)}] {chunk[0]}..{chunk[-1]}: +{len(w)} rows, grid-coverage {cov:.0%} (total {total})")
