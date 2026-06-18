"""Bucket A — currency refresh 2026-06-05 -> 06-18 (staged fragments; monoliths byte-untouched).

xbbg 1.3.0 narwhals tidy -> pivot. Pulls from the 06-04 overlap day so each fragment is validated
against the committed monolith TO THE CENT (rotation + field + scale check), then keeps 06-05+.
Fragments replicate each monolith's exact column set + ticker form so they are append-compatible.

OHLCV rotation (load-bearing, FRESH_LAB_BOX_SETUP): the committed monolith stores columns rotated;
a correct pull maps PX_HIGH->open, PX_LAST->high, PX_LOW->low, PX_OPEN->close, PX_VOLUME->volume and
every row must satisfy open==max(o,h,l,c) AND low==min(o,h,l,c).
"""
import os
import pandas as pd
from xbbg import blp

OVERLAP = "2026-06-04"          # last committed bar; pulled to validate, then dropped
START, END = "2026-06-04", "2026-06-18"
KEEP_AFTER = pd.Timestamp("2026-06-04")  # fragment keeps dates strictly after this
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
CHUNK = 30


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


def pull_long(tickers, fields):
    rows = []
    for i in range(0, len(tickers), CHUNK):
        ch = tickers[i:i + CHUNK]
        d = native(blp.bdh(ch, fields, START, END))
        if {"ticker", "date", "field", "value"}.issubset(d.columns):
            rows.append(d[d["value"].notna()])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ticker", "date", "field", "value"])


def pivot_wide(long_df):
    w = long_df.pivot_table(index=["date", "ticker"], columns="field", values="value", aggfunc="first").reset_index()
    w["date"] = pd.to_datetime(w["date"])
    return w


# ---------- OHLCV (rotated) ----------
def do_ohlcv():
    uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
    print(f"[ohlcv] universe {len(uni)}")
    w = pivot_wide(pull_long(uni, ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]))
    out = pd.DataFrame({
        "date": w["date"], "ticker": w["ticker"],
        "open": w["PX_HIGH"], "high": w["PX_LAST"], "low": w["PX_LOW"],
        "close": w["PX_OPEN"], "volume": w["PX_VOLUME"],
    })
    full = out.dropna(subset=["open", "high", "low", "close"])
    omax = (full["open"] == full[["open", "high", "low", "close"]].max(axis=1))
    lmin = (full["low"] == full[["open", "high", "low", "close"]].min(axis=1))
    print(f"[ohlcv] rotation gate: open==max {omax.mean():.4f}  low==min {lmin.mean():.4f}  (violations: {int((~omax).sum())}/{int((~lmin).sum())})")
    assert omax.all() and lmin.all(), "ROTATION GATE FAILED — do not stage"
    _overlap_check("ohlcv", out, ["open", "high", "low", "close", "volume"], "ticker")
    frag = out[out["date"] > KEEP_AFTER].sort_values(["ticker", "date"]).copy()
    frag["date"] = frag["date"].dt.strftime("%Y-%m-%d")
    frag.to_csv(os.path.join(OUT, "sp500_ohlcv__2026-06-05_2026-06-18.csv"), index=False)
    print(f"[ohlcv] fragment rows {len(frag)} dates {frag['date'].min()}..{frag['date'].max()} names {frag['ticker'].nunique()}")


# ---------- LIQUIDITY ----------
def do_liquidity():
    uni_full = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
    print(f"[liquidity] universe {len(uni_full)}")
    w = pivot_wide(pull_long(uni_full, ["VOLUME_AVG_30D", "TURNOVER", "EQY_SH_OUT"]))
    w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
    out = pd.DataFrame({
        "date": w["date"], "avg_vol_30d": w["VOLUME_AVG_30D"], "turnover": w["TURNOVER"],
        "shares_out": w["EQY_SH_OUT"], "ticker": w["ticker"],
    })
    _overlap_check("liquidity", out, ["avg_vol_30d", "turnover", "shares_out"], "ticker")
    frag = out[out["date"] > KEEP_AFTER].sort_values(["ticker", "date"]).copy()
    frag["date"] = frag["date"].dt.strftime("%Y-%m-%d")
    frag.to_csv(os.path.join(OUT, "sp500_liquidity__2026-06-05_2026-06-18.csv"), index=False)
    print(f"[liquidity] fragment rows {len(frag)} dates {frag['date'].min()}..{frag['date'].max()} names {frag['ticker'].nunique()}")


# ---------- VIX TERM STRUCTURE (constant-maturity) ----------
def do_vix_term():
    tmap = {"VIX Index": "vix", "VIX3M Index": "vix_3m", "VIX6M Index": "vix_6m"}
    w = pivot_wide(pull_long(list(tmap), ["PX_LAST"]))
    # one ticker per col -> reshape: rows are (date,ticker) with PX_LAST
    long_df = pull_long(list(tmap), ["PX_LAST"])
    p = long_df.pivot_table(index="date", columns="ticker", values="value", aggfunc="first").reset_index()
    p["date"] = pd.to_datetime(p["date"])
    out = pd.DataFrame({"date": p["date"]})
    for tkr, col in tmap.items():
        out[col] = p[tkr] if tkr in p.columns else pd.NA
    _overlap_check_indexed("vix_term", out, ["vix", "vix_3m", "vix_6m"])
    frag = out[out["date"] > KEEP_AFTER].sort_values("date").copy()
    frag["date"] = frag["date"].dt.strftime("%Y-%m-%d")
    frag.to_csv(os.path.join(OUT, "vix_term_structure__2026-06-05_2026-06-18.csv"), index=False)
    print(f"[vix_term] fragment rows {len(frag)} dates {frag['date'].min()}..{frag['date'].max()}")
    print(frag.to_string(index=False))


# ---------- overlap-to-the-cent validators ----------
def _overlap_check(name, frag_full, valcols, tickercol):
    mono = pd.read_csv(os.path.join(MONO, f"sp500_{name}.csv"))
    mono["date"] = pd.to_datetime(mono["date"])
    m0 = mono[mono["date"] == pd.Timestamp(OVERLAP)].set_index(tickercol)
    f0 = frag_full[frag_full["date"] == pd.Timestamp(OVERLAP)].set_index(tickercol)
    common = m0.index.intersection(f0.index)
    n_mismatch = 0
    for c in valcols:
        if c in m0.columns and c in f0.columns:
            a, b = m0.loc[common, c].astype(float), f0.loc[common, c].astype(float)
            diff = (a - b).abs()
            tol = a.abs() * 1e-4 + 1e-6   # 1bp relative tol for floats
            bad = int((diff > tol).sum())
            n_mismatch += bad
            print(f"[{name}] overlap {OVERLAP} {c}: {len(common)} names, max|Δ|={diff.max():.6g}, mismatches>{1e-4:.0e}rel={bad}")
    print(f"[{name}] OVERLAP {'OK' if n_mismatch == 0 else 'MISMATCH('+str(n_mismatch)+')'}")


def _overlap_check_indexed(name, frag_full, valcols):
    mono = pd.read_csv(os.path.join(MONO, "vix_term_structure.csv"))
    mono["date"] = pd.to_datetime(mono["date"])
    m0 = mono[mono["date"] == pd.Timestamp(OVERLAP)]
    f0 = frag_full[frag_full["date"] == pd.Timestamp(OVERLAP)]
    if len(m0) and len(f0):
        for c in valcols:
            a = float(m0[c].iloc[0]); b = float(f0[c].iloc[0])
            print(f"[{name}] overlap {OVERLAP} {c}: mono={a:.4f} frag={b:.4f} Δ={abs(a-b):.6g}")


if __name__ == "__main__":
    do_ohlcv()
    do_liquidity()
    do_vix_term()
    print("DONE bucket A")
