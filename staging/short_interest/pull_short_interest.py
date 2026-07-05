"""Bucket B / T0-12 — Short interest (biweekly). Bloomberg lab 2026-06-18.

FLDS-by-value-count entitlement check (this tier):
  ENTITLED:  SHORT_INTEREST (shares), SHORT_INT_RATIO (days-to-cover)  — biweekly settlement prints
  BLOCKED (all-NaN): EQY_SHORT_INTEREST, *_PCT_OF_FLOAT (every variant), all borrow-rate fields
                     (EQUITY_SHORT_BORROW_RATE_NET / COST_OF_BORROW / GC_RATE / ...) — no SLB entitlement.
So pct-of-float + borrow are can't-pull; they go to the manifest's bucket F. No Fill (keep true PIT
settlement dates, not forward-filled). xbbg 1.3.0 narwhals tidy -> pivot. Staging only.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "2015-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
FIELDS = ["SHORT_INTEREST", "SHORT_INT_RATIO"]
CHUNK = 30


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
print(f"universe {len(uni)}")
parts = []
for i in range(0, len(uni), CHUNK):
    ch = uni[i:i + CHUNK]
    d = native(blp.bdh(ch, FIELDS, START, END))
    if {"ticker", "date", "field", "value"}.issubset(d.columns):
        parts.append(d[d["value"].notna()])
    print(f"  [{i}:{i+len(ch)}] {ch[0]}..{ch[-1]}")
long = pd.concat(parts, ignore_index=True)
w = long.pivot_table(index=["date", "ticker"], columns="field", values="value", aggfunc="first").reset_index()
w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
w = w.rename(columns={"SHORT_INTEREST": "short_interest", "SHORT_INT_RATIO": "short_int_ratio"})
cols = ["date", "ticker", "short_interest", "short_int_ratio"]
w = w[[c for c in cols if c in w.columns]].sort_values(["ticker", "date"])
w.to_csv(os.path.join(OUT, "sp500_short_interest.csv"), index=False)

print(f"\nsp500_short_interest.csv: {len(w)} rows, {w['ticker'].nunique()} names, {w['date'].min()}..{w['date'].max()}")
for c in ["short_interest", "short_int_ratio"]:
    if c in w.columns:
        v = pd.to_numeric(w[c], errors="coerce"); nn = v.notna()
        print(f"  {c}: {int(nn.sum())} nn [{v[nn].min():.4g}, {v[nn].max():.4g}] median {v[nn].median():.4g}")
# cadence check on one name
aapl = w[w["ticker"] == "AAPL"]
print(f"  AAPL prints: {len(aapl)} (biweekly ~ 24/yr expected); last {aapl['date'].max() if len(aapl) else 'n/a'}")
