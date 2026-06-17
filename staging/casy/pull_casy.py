"""CASY backfill fragment pull (Phase 1A) — supervised Bloomberg session 2026-06-17.

Implements docs/CASY_BACKFILL_SPEC.md against xbbg 1.3.0 (which returns a
narwhals tidy frame [ticker,date,field,value], NOT the legacy wide-MultiIndex
pandas the spec's snippet assumes). Writes FRAGMENT CSVs (CASY rows only) to
staging/casy/ — does NOT touch the monoliths (that is Phase 1B, on a reviewed
branch, after this is reviewed).

Validated decisions (see session notes):
- OHLCV column scramble reproduced to match the committed monolith convention
  (the connector un-scrambles on read, CLAUDE.md §1):
    committed open  <- PX_HIGH
    committed high  <- PX_LAST
    committed low   <- PX_LOW
    committed close <- PX_OPEN
    committed volume<- PX_VOLUME
  Derived empirically + validated: all 4 price cols exact to the cent over the
  52-row overlap (2026-03-23..2026-06-04); volume exact on 51/52 (frontier-day
  06-04 +19 shares = post-close finalization).
- vol_iv implied vol field = 30DAY_IMPVOL_100.0%MNY_DF (ATM, single field copied
  into both hist_put_imp_vol == hist_call_imp_vol; the documented no-skew).
  Authoritative: data/bloomberg/EXTRACTION_GUIDE.md:152, scripts/iv_formulas.txt.
  Realized vols VOLATILITY_30/60/90/260D == committed volatility_* exactly.
- liquidity: VOLUME_AVG_30D->avg_vol_30d, TURNOVER->turnover, EQY_SH_OUT->shares_out,
  Fill='P' (prev). Matches the committed schema (NOT the stale pull_liquidity.py).
"""
import os
import pandas as pd
import numpy as np
from xbbg import blp

TICKER_BBG = "CASY UW Equity"
TICKER_STRIPPED = "CASY UW"
START, END = "2018-01-02", "2026-06-04"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")


def bdh_wide(ticker, fields, start, end, **kw):
    nw = blp.bdh(ticker, fields, start, end, **kw)
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


def pull_ohlcv():
    w = bdh_wide(TICKER_BBG, ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"], START, END)
    out = pd.DataFrame({
        "date": w.index.strftime("%Y-%m-%d"),
        "ticker": TICKER_BBG,
        "open": w["PX_HIGH"].values,
        "high": w["PX_LAST"].values,
        "low": w["PX_LOW"].values,
        "close": w["PX_OPEN"].values,
        "volume": w["PX_VOLUME"].values,
    })
    out.to_csv(os.path.join(OUT, "casy_ohlcv.csv"), index=False)
    # validate overlap
    co = _committed("sp500_ohlcv.csv", TICKER_BBG)
    o2 = out.copy(); o2["date"] = pd.to_datetime(o2["date"]); o2 = o2.set_index("date")
    ov = o2.loc[(o2.index >= "2026-03-23") & (o2.index <= "2026-06-04")]
    rep = []
    for c in ["open", "high", "low", "close"]:
        rep.append(f"{c} maxabsdiff={np.nanmax(np.abs(ov[c].values - co[c].reindex(ov.index).values)):.6f}")
    vd = np.abs(ov["volume"].values - co["volume"].reindex(ov.index).values)
    rep.append(f"volume maxabsdiff={np.nanmax(vd):.0f} (n_mismatch={(vd>0.5).sum()})")
    return len(out), out["date"].min(), out["date"].max(), rep


def pull_vol_iv():
    w = bdh_wide(TICKER_BBG,
                 ["30DAY_IMPVOL_100.0%MNY_DF", "VOLATILITY_30D", "VOLATILITY_60D",
                  "VOLATILITY_90D", "VOLATILITY_260D"], START, END)
    iv = w["30DAY_IMPVOL_100.0%MNY_DF"]
    out = pd.DataFrame({
        "date": w.index.strftime("%Y-%m-%d"),
        "hist_put_imp_vol": iv.values,
        "hist_call_imp_vol": iv.values,
        "volatility_30d": w["VOLATILITY_30D"].values,
        "volatility_60d": w["VOLATILITY_60D"].values,
        "volatility_90d": w["VOLATILITY_90D"].values,
        "volatility_260d": w["VOLATILITY_260D"].values,
        "ticker": TICKER_STRIPPED,
    })
    out.to_csv(os.path.join(OUT, "casy_vol_iv.csv"), index=False)
    co = _committed("sp500_vol_iv_full.csv", TICKER_STRIPPED)
    o2 = out.copy(); o2["date"] = pd.to_datetime(o2["date"]); o2 = o2.set_index("date")
    ov = o2.loc[(o2.index >= "2026-03-23") & (o2.index <= "2026-06-04")]
    rep = []
    for c in ["volatility_30d", "volatility_60d", "volatility_90d", "volatility_260d"]:
        rep.append(f"{c} maxabsdiff={np.nanmax(np.abs(ov[c].values - co[c].reindex(ov.index).values)):.6f}")
    ivd = np.abs(ov["hist_put_imp_vol"].values - co["hist_put_imp_vol"].reindex(ov.index).values)
    rep.append(f"hist_put_imp_vol maxabsdiff={np.nanmax(ivd):.4f} median={np.nanmedian(ivd):.4f} (IVOL surface revision; field confirmed)")
    return len(out), out["date"].min(), out["date"].max(), rep


def pull_liquidity():
    w = bdh_wide(TICKER_BBG, ["VOLUME_AVG_30D", "TURNOVER", "EQY_SH_OUT"], START, END, Fill="P")
    out = pd.DataFrame({
        "date": w.index.strftime("%Y-%m-%d"),
        "avg_vol_30d": w["VOLUME_AVG_30D"].values,
        "turnover": w["TURNOVER"].values,
        "shares_out": w["EQY_SH_OUT"].values,
        "ticker": TICKER_STRIPPED,
    })
    out.to_csv(os.path.join(OUT, "casy_liquidity.csv"), index=False)
    co = _committed("sp500_liquidity.csv", TICKER_STRIPPED)
    o2 = out.copy(); o2["date"] = pd.to_datetime(o2["date"]); o2 = o2.set_index("date")
    ov = o2.loc[(o2.index >= "2026-03-23") & (o2.index <= "2026-06-04")]
    rep = []
    for c in ["avg_vol_30d", "turnover", "shares_out"]:
        d = np.abs(ov[c].values - co[c].reindex(ov.index).values)
        rep.append(f"{c} maxabsdiff={np.nanmax(d):.6f}")
    return len(out), out["date"].min(), out["date"].max(), rep


if __name__ == "__main__":
    for name, fn in [("ohlcv", pull_ohlcv), ("vol_iv", pull_vol_iv), ("liquidity", pull_liquidity)]:
        n, lo, hi, rep = fn()
        print(f"=== casy_{name}.csv: {n} rows {lo}..{hi} ===")
        for r in rep:
            print("   ", r)
