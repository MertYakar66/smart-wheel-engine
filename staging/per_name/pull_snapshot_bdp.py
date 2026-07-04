"""Bucket C — per-name BDP snapshot: ratings + watch/outlook + full GICS + ownership + next-earnings.
Bloomberg lab 2026-06-18. One row per name (snapshot as-of pull date). xbbg 1.3.0 bdp -> pivot wide.
"""
import os
import pandas as pd
from xbbg import blp

OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
ASOF = "2026-06-18"
CHUNK = 50
FMAP = {
    "RTG_SP_LT_LC_ISSUER_CREDIT": "rtg_sp", "RTG_MOODY_LONG_TERM": "rtg_moody",
    "RTG_FITCH_LT_ISSUER_DEFAULT": "rtg_fitch", "SP_LT_LC_ISSUER_OUTLOOK": "sp_outlook",
    "RATING_OUTLOOK": "rating_outlook", "RATING_WATCH": "rating_watch",
    "GICS_SECTOR_NAME": "gics_sector", "GICS_INDUSTRY_GROUP_NAME": "gics_ind_grp",
    "GICS_INDUSTRY_NAME": "gics_industry", "GICS_SUB_INDUSTRY_NAME": "gics_sub_ind",
    "EQY_INST_PCT_SH_OUT": "inst_pct", "EQY_FREE_FLOAT_PCT": "free_float_pct",
    "EQY_FLOAT": "float_shares", "EXPECTED_REPORT_DT": "next_earnings_dt",
    "CRNCY": "crncy",
}


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
fields = list(FMAP)
parts = []
for i in range(0, len(uni), CHUNK):
    ch = uni[i:i + CHUNK]
    d = native(blp.bdp(ch, fields))
    parts.append(d)
raw = pd.concat(parts, ignore_index=True)
# bdp returns long [ticker, field, value]; pivot wide
if {"ticker", "field", "value"}.issubset(raw.columns):
    w = raw.pivot_table(index="ticker", columns="field", values="value", aggfunc="first")
else:
    w = raw.set_index("ticker")
w = w.rename(columns=FMAP).reset_index()
w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
w.insert(0, "asof", ASOF)
cols = ["asof", "ticker"] + [c for c in FMAP.values() if c in w.columns]
w = w[cols].sort_values("ticker")
w.to_csv(os.path.join(OUT, "sp500_snapshot_bdp.csv"), index=False)
print(f"sp500_snapshot_bdp.csv: {len(w)} names")
for c in [c for c in FMAP.values() if c in w.columns]:
    nn = w[c].notna().mean()
    print(f"  {c:18s} cov {nn:.0%}", end="")
    if w[c].dtype == object and c in ("rtg_sp", "rating_outlook", "rating_watch", "gics_sector"):
        vc = w[c].dropna().value_counts().head(4).to_dict()
        print(f"  e.g. {vc}")
    else:
        print()
