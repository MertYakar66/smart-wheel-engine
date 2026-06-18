"""Bucket C — forward consensus estimates via BEST_FPERIOD_OVERRIDE (1BF / 2BF). Lab 2026-06-18.

Fixes the ~21% coverage of default-period BEST_* (estimates_m): 1BF/2BF give ~100% monthly coverage.
xbbg 1.3.0 narwhals tidy -> pivot. Monthly 2010-2026. Staging only.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "2010-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
CHUNK = 30
BASE = ["BEST_EPS", "BEST_SALES", "BEST_EBITDA"]


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
frames = []
for period in ["1BF", "2BF"]:
    suf = period.lower()
    parts = []
    for i in range(0, len(uni), CHUNK):
        ch = uni[i:i + CHUNK]
        d = native(blp.bdh(ch, BASE, START, END, Per="M", BEST_FPERIOD_OVERRIDE=period))
        if {"ticker", "date", "field", "value"}.issubset(d.columns):
            parts.append(d[d["value"].notna()])
    long = pd.concat(parts, ignore_index=True)
    long["col"] = long["field"].map({f: f.lower().replace("best_", "best_") + f"_{suf}" for f in BASE})
    long["ticker"] = long["ticker"].str.replace(" Equity", "", regex=False)
    w = long.pivot_table(index=["date", "ticker"], columns="col", values="value", aggfunc="first")
    frames.append(w)
    print(f"period {period}: {long['ticker'].nunique()} names")

merged = pd.concat(frames, axis=1).reset_index()
merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
valcols = [c for c in merged.columns if c not in ("date", "ticker")]
merged = merged[["date", "ticker"] + valcols].sort_values(["ticker", "date"])
merged.to_csv(os.path.join(OUT, "estimates_fwd.csv"), index=False)
print(f"estimates_fwd.csv: {len(merged)} rows, {merged['ticker'].nunique()} names, {merged['date'].min()}..{merged['date'].max()}")
for c in valcols:
    v = pd.to_numeric(merged[c], errors="coerce"); nn = v.notna()
    print(f"  {c:18s}: nn {nn.mean():.0%} [{v[nn].min():.4g},{v[nn].max():.4g}] median {v[nn].median():.4g}")
