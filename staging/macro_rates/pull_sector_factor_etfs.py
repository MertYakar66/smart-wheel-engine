"""Bucket D — sector & factor ETF OHLCV. Bloomberg lab 2026-06-18.

Natural mapping (PX_OPEN->open ...) to match the committed sp500_sector_etfs.csv convention
(open==max ~10%, un-rotated; cols date,open,high,low,close,volume,etf). Refreshes to 06-18 and
extends breadth (adds SPY/QQQ/IWM/DIA + XLC/XLRE). Fresh staged file; monolith byte-untouched.
xbbg 1.3.0 narwhals tidy -> pivot. All 15 FLDS-verified entitled.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "1998-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)
ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC",
        "SPY", "QQQ", "IWM", "DIA"]
FIELDS = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]
MAP = {"PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low", "PX_LAST": "close", "PX_VOLUME": "volume"}


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


tickers = [f"{e} US Equity" for e in ETFS]
d = native(blp.bdh(tickers, FIELDS, START, END))
d = d[d["value"].notna()].copy()
w = d.pivot_table(index=["date", "ticker"], columns="field", values="value", aggfunc="first").reset_index()
w = w.rename(columns=MAP)
w["etf"] = w["ticker"].str.replace(" US Equity", "", regex=False)
w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
out = w[["date", "open", "high", "low", "close", "volume", "etf"]].sort_values(["etf", "date"])
out.to_csv(os.path.join(OUT, "sector_factor_etfs_ohlcv.csv"), index=False)

# validation: natural mapping (open should NOT be the max systematically), per-etf coverage
full = out.dropna(subset=["open", "high", "low", "close"])
omax = (full["open"] == full[["open", "high", "low", "close"]].max(axis=1)).mean()
hmax = (full["high"] == full[["open", "high", "low", "close"]].max(axis=1)).mean()
print(f"sector_factor_etfs_ohlcv.csv: {len(out)} rows, {out['etf'].nunique()} ETFs, {out['date'].min()}..{out['date'].max()}")
print(f"  natural-map check: high==max {hmax:.3f} (expect ~1.0), open==max {omax:.3f} (expect ~0.1)")
print("  per-etf rows + range:")
for e in ETFS:
    s = out[out["etf"] == e]
    if len(s):
        print(f"    {e:5s} {len(s):5d} rows {s['date'].min()}..{s['date'].max()} close[{pd.to_numeric(s['close']).min():.2f},{pd.to_numeric(s['close']).max():.2f}]")
