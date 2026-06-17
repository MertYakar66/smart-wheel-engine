"""Blue-chip OHLCV (+vol_iv for IV-thin) backfill fragments (#355) — Phase 1A.

Same validated method as staging/casy/pull_casy.py. Pulls FRAGMENT CSVs to
staging/blue_chips/ (monoliths untouched; Phase 1B integration deferred).

Usage:  python pull_backfill.py WMT KMB CPB DPZ PLTR
        python pull_backfill.py VEEV COHR LITE SATS VRT

Per-name spec (ohlcv_ticker exactly as committed; vol_iv backfilled only for the
IV-thin names whose committed vol_iv == 52 rows).
"""
import sys
import os
import pandas as pd
import numpy as np
from xbbg import blp

START, END = "2018-01-02", "2026-06-04"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")

# name -> (ohlcv_ticker_exact, backfill_vol_iv?)
SPECS = {
    "WMT": ("WMT UW Equity", False),
    "KMB": ("KMB UW Equity", False),
    "CPB": ("CPB UW Equity", False),
    "DPZ": ("DPZ UW Equity", False),
    "PLTR": ("PLTR UW Equity", False),
    "VEEV": ("VEEV UN Equity", True),
    "COHR": ("COHR UN Equity", True),
    "LITE": ("LITE UW Equity", True),
    "SATS": ("SATS UW Equity", True),
    "VRT": ("VRT UN Equity", True),
}


def bdh_wide(ticker, fields, **kw):
    nw = blp.bdh(ticker, fields, START, END, **kw)
    df = nw.to_native() if hasattr(nw, "to_native") else nw
    if {"date", "field", "value"}.issubset(set(df.columns)):
        w = df.pivot_table(index="date", columns="field", values="value", aggfunc="first").sort_index()
        w.index = pd.to_datetime(w.index)
        return w
    return df


def _committed(fname, ticker):
    df = pd.read_csv(os.path.join(MONO, fname))
    df = df[df["ticker"] == ticker].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def pull_ohlcv(ohlcv_ticker):
    base = ohlcv_ticker.split(" ")[0]
    w = bdh_wide(ohlcv_ticker, ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"])
    if w is None or len(w) == 0 or "PX_HIGH" not in w.columns:
        return base, 0, ["EMPTY PULL — check ticker"]
    out = pd.DataFrame({
        "date": w.index.strftime("%Y-%m-%d"), "ticker": ohlcv_ticker,
        "open": w["PX_HIGH"].values, "high": w["PX_LAST"].values, "low": w["PX_LOW"].values,
        "close": w["PX_OPEN"].values, "volume": w["PX_VOLUME"].values,
    })
    out.to_csv(os.path.join(OUT, f"{base}_ohlcv.csv"), index=False)
    co = _committed("sp500_ohlcv.csv", ohlcv_ticker)
    o2 = out.copy(); o2["date"] = pd.to_datetime(o2["date"]); o2 = o2.set_index("date")
    ov = o2.reindex(co.index)  # validate against the committed (truncated) rows
    rep = [f"rows={len(out)} {out['date'].min()}..{out['date'].max()} (committed had {len(co)})"]
    for c in ["open", "high", "low", "close"]:
        rep.append(f"{c} maxabsdiff={np.nanmax(np.abs(ov[c].values - co[c].values)):.6f}")
    vd = np.abs(ov["volume"].values - co["volume"].values)
    rep.append(f"volume maxabsdiff={np.nanmax(vd):.0f} (n_mismatch={int(np.nansum(vd>0.5))}/{len(co)})")
    return base, len(out), rep


def pull_vol_iv(ohlcv_ticker):
    base = ohlcv_ticker.split(" ")[0]
    stripped = ohlcv_ticker.replace(" Equity", "")
    w = bdh_wide(ohlcv_ticker, ["30DAY_IMPVOL_100.0%MNY_DF", "VOLATILITY_30D", "VOLATILITY_60D",
                                "VOLATILITY_90D", "VOLATILITY_260D"])
    if w is None or len(w) == 0:
        return ["vol_iv EMPTY"]
    iv = w.get("30DAY_IMPVOL_100.0%MNY_DF")
    out = pd.DataFrame({
        "date": w.index.strftime("%Y-%m-%d"),
        "hist_put_imp_vol": iv.values if iv is not None else np.nan,
        "hist_call_imp_vol": iv.values if iv is not None else np.nan,
        "volatility_30d": w["VOLATILITY_30D"].values, "volatility_60d": w["VOLATILITY_60D"].values,
        "volatility_90d": w["VOLATILITY_90D"].values, "volatility_260d": w["VOLATILITY_260D"].values,
        "ticker": stripped,
    })
    out.to_csv(os.path.join(OUT, f"{base}_vol_iv.csv"), index=False)
    co = _committed("sp500_vol_iv_full.csv", stripped)
    o2 = out.copy(); o2["date"] = pd.to_datetime(o2["date"]); o2 = o2.set_index("date")
    ov = o2.reindex(co.index)
    rep = [f"vol_iv rows={len(out)} (committed {len(co)})"]
    for c in ["volatility_30d", "volatility_60d", "volatility_90d", "volatility_260d"]:
        rep.append(f"{c} maxabsdiff={np.nanmax(np.abs(ov[c].values - co[c].values)):.6f}")
    ivd = np.abs(ov["hist_put_imp_vol"].values - co["hist_put_imp_vol"].values)
    rep.append(f"IV maxabsdiff={np.nanmax(ivd):.4f} median={np.nanmedian(ivd):.4f} (surface revision)")
    return rep


if __name__ == "__main__":
    for name in sys.argv[1:]:
        tkr, do_iv = SPECS[name]
        base, n, rep = pull_ohlcv(tkr)
        print(f"=== {base}_ohlcv.csv ===")
        for r in rep:
            print("   ", r)
        if do_iv:
            for r in pull_vol_iv(tkr):
                print("   ", r)
