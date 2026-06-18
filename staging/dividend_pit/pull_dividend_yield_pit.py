"""T0-3 / #354 — Dated PIT dividend-yield panel (BDH monthly). Bloomberg lab 2026-06-18.

Fixes the lookahead: get_fundamentals had a dateless 2026 snapshot of eqy_dvd_yld_12m; this is the
point-in-time monthly series so BSM q can be selected as-of. FLDS-verified: EQY_DVD_YLD_12M,
EQY_DVD_YLD_IND, DVD_SH_12M entitled; EQY_DVD_YLD_EST / DVD_SH_LAST all-NaN. NaN = non-payer at that
date (confirmed: GOOGL NaN until its 2024 initiation) -> maps to q=0; NOT missing data.
xbbg 1.3.0 narwhals tidy -> pivot. Staging only.
"""
import os
import pandas as pd
from xbbg import blp

START, END = "2010-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
FIELDS = ["EQY_DVD_YLD_12M", "EQY_DVD_YLD_IND", "DVD_SH_12M"]
CHUNK = 30


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
print(f"universe {len(uni)}")
parts = []
for i in range(0, len(uni), CHUNK):
    ch = uni[i:i + CHUNK]
    d = native(blp.bdh(ch, FIELDS, START, END, Per="M"))
    if {"ticker", "date", "field", "value"}.issubset(d.columns):
        parts.append(d[d["value"].notna()])
long = pd.concat(parts, ignore_index=True)
w = long.pivot_table(index=["date", "ticker"], columns="field", values="value", aggfunc="first").reset_index()
w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
w = w.rename(columns={"EQY_DVD_YLD_12M": "dvd_yld_12m", "EQY_DVD_YLD_IND": "dvd_yld_ind", "DVD_SH_12M": "dvd_sh_12m"})
cols = ["date", "ticker", "dvd_yld_12m", "dvd_yld_ind", "dvd_sh_12m"]
w = w[[c for c in cols if c in w.columns]].sort_values(["ticker", "date"])
w.to_csv(os.path.join(OUT, "sp500_dividend_yield_pit.csv"), index=False)
print(f"sp500_dividend_yield_pit.csv: {len(w)} rows, {w['ticker'].nunique()} names, {w['date'].min()}..{w['date'].max()}")
for c in ["dvd_yld_12m", "dvd_yld_ind", "dvd_sh_12m"]:
    if c in w.columns:
        v = pd.to_numeric(w[c], errors="coerce"); nn = v.notna()
        print(f"  {c}: {int(nn.sum())} nn [{v[nn].min():.4g},{v[nn].max():.4g}] median {v[nn].median():.4g}")

# ---- #354 diagnosis: prove NaN names are genuine non-payers, not missing ----
panel_names = set(w["ticker"].unique())
all_names = set(t.replace(" Equity", "") for t in uni)
# names with ANY non-null 12m yield ever in the panel
payers_panel = set(w[pd.to_numeric(w["dvd_yld_12m"], errors="coerce").notna()]["ticker"].unique())
never_yield = all_names - payers_panel
# cross-ref dividends.csv: do never-yield names have ANY dividend record?
dv = pd.read_csv(os.path.join(MONO, "sp500_dividends.csv"), usecols=["ticker", "dividend_amount"])
div_tickers = set(dv[pd.to_numeric(dv["dividend_amount"], errors="coerce").fillna(0) > 0]["ticker"].unique())
print(f"\n#354 diagnosis:")
print(f"  universe {len(all_names)}; names with >=1 non-null 12m yield in panel: {len(payers_panel)} ({len(payers_panel)/len(all_names):.1%})")
print(f"  never-yield names: {len(never_yield)}")
genuine_nonpayers = [n for n in never_yield if n not in div_tickers]
missing_suspects = [n for n in never_yield if n in div_tickers]
print(f"    of those, GENUINE non-payers (no >0 dividend record in sp500_dividends.csv): {len(genuine_nonpayers)}")
print(f"    SUSPECT (have dividend records but no panel yield -> investigate): {len(missing_suspects)}")
print(f"    suspects: {sorted(missing_suspects)[:30]}")
