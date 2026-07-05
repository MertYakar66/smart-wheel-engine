"""CORRECTION (2026-06-17): batch-1 blue-chip OHLCV re-pulled under the correct
NYSE `UN` exchange code.

The first batch-1 pull used the committed `<NAME> UW` (NASDAQ) tickers, which return
only truncated recent rows for these NYSE names (that wrong code is why they were thin
on main in the first place). Confirmed: `WMT UW Equity` has 0 rows in Jan-2018;
`WMT UN Equity` has the full 2018→2026 history. This re-pull fixes WMT/KMB/CPB/DPZ/PLTR
to `<NAME> UN Equity` and overwrites the wrong *_ohlcv.csv fragments.

Validation (no clean early-history overlap exists — the committed UW data is wrong):
  - full row count (WMT/KMB/CPB/DPZ from 2018; PLTR from its 2020-09-30 IPO),
  - recent-overlap sanity vs the committed UW rows (same security → match within the
    documented revision-vintage tolerance),
  - OHLC integrity in TRUE terms after the scramble (high>=open/close/low; low<=all).

Scramble unchanged (universal): committed open<-PX_HIGH, high<-PX_LAST, low<-PX_LOW,
close<-PX_OPEN, volume<-PX_VOLUME. Stored ticker = '<NAME> UN Equity' (correct exchange);
Phase-1B integration drops the old '<NAME> UW Equity' rows.
"""
import os
import pandas as pd
import numpy as np
from xbbg import blp

START, END = "2018-01-02", "2026-06-04"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
# corrected NYSE tickers (replacing the wrong UW)
TICKERS = ["WMT UN Equity", "KMB UN Equity", "CPB UN Equity", "DPZ UN Equity", "PLTR UN Equity"]


def bdh_wide(t, f):
    nw = blp.bdh(t, f, START, END)
    df = nw.to_native() if hasattr(nw, "to_native") else nw
    w = df.pivot_table(index="date", columns="field", values="value", aggfunc="first").sort_index()
    w.index = pd.to_datetime(w.index)
    return w


for tkr in TICKERS:
    base = tkr.split(" ")[0]
    old_uw = f"{base} UW Equity"
    w = bdh_wide(tkr, ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"])
    out = pd.DataFrame({
        "date": w.index.strftime("%Y-%m-%d"), "ticker": tkr,
        "open": w["PX_HIGH"].values, "high": w["PX_LAST"].values, "low": w["PX_LOW"].values,
        "close": w["PX_OPEN"].values, "volume": w["PX_VOLUME"].values,
    })
    out.to_csv(os.path.join(OUT, f"{base}_ohlcv.csv"), index=False)

    # OHLC integrity in TRUE terms (undo scramble): true_high=open, true_close=high, true_open=close, true_low=low
    th, tc, to, tl = out["open"], out["high"], out["close"], out["low"]
    hi_ok = bool((th >= np.maximum.reduce([to, tc, tl])).all())
    lo_ok = bool((tl <= np.minimum.reduce([to, tc, th])).all())

    # recent-overlap sanity vs committed UW
    co = pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"))
    co = co[co["ticker"] == old_uw].copy()
    co["date"] = pd.to_datetime(co["date"]); co = co.set_index("date").sort_index()
    o2 = out.copy(); o2["date"] = pd.to_datetime(o2["date"]); o2 = o2.set_index("date")
    shared = co.index.intersection(o2.index)
    dmax = float(np.nanmax(np.abs(o2.loc[shared, "open"].values - co.loc[shared, "open"].values))) if len(shared) else float("nan")

    print(f"{base}: {tkr}  rows={len(out)} {out['date'].min()}..{out['date'].max()}  "
          f"(was UW {len(co)} rows)  overlap_shared={len(shared)} open_maxabsdiff={dmax:.4f}  "
          f"OHLC_integrity(high/low)={hi_ok}/{lo_ok}")
