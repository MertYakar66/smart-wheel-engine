"""Bucket C — per-name LOW-FREQUENCY panels (monthly/quarterly BDH). Bloomberg lab 2026-06-18.

Usage: python pull_lowfreq_panels.py <dataset>   (full universe in one run; chunked bdh)
  beta_shares (M), fundamentals_q (Q), estimates_m (M). All fields FLDS-verified entitled.
Caveat: BDH fundamentals are dated at PERIOD-END (not filing date — BEST_PERIOD_END_DT all-NaN at
this tier), and reflect current restatements; treat as period-end series, filing-lag PIT not captured.
xbbg 1.3.0 narwhals tidy -> pivot. Staging only.
"""
import os
import sys
import pandas as pd
from xbbg import blp

START, END = "2010-01-01", "2026-06-18"
OUT = os.path.dirname(__file__)
MONO = os.path.join(OUT, "..", "..", "data", "bloomberg")
CHUNK = 30

DATASETS = {
    "beta_shares": {"per": "M", "map": {"BETA_RAW_OVERRIDABLE": "beta_raw", "EQY_SH_OUT": "shares_out"}},
    "fundamentals_q": {"per": "Q", "map": {
        "SALES_REV_TURN": "revenue", "IS_OPER_INC": "oper_inc", "NET_INCOME": "net_income",
        "EBITDA": "ebitda", "IS_EPS": "eps", "BS_TOT_ASSET": "tot_asset", "BS_TOT_LIAB2": "tot_liab",
        "CF_FREE_CASH_FLOW": "fcf", "CF_CASH_FROM_OPER": "cfo", "RETURN_COM_EQY": "roe",
        "NET_DEBT_TO_EBITDA": "nd_to_ebitda", "GROSS_MARGIN": "gross_margin"}},
    "estimates_m": {"per": "M", "map": {
        "BEST_EPS": "best_eps", "BEST_SALES": "best_sales", "BEST_EBITDA": "best_ebitda",
        "BEST_TARGET_PRICE": "best_target", "BEST_PE_RATIO": "best_pe",
        "BEST_ANALYST_RATING": "best_rating", "TOT_ANALYST_REC": "analyst_count"}},
    "valuation_m": {"per": "M", "map": {
        "PX_TO_BOOK_RATIO": "px_to_book", "CURRENT_EV_TO_T12M_EBITDA": "ev_to_ebitda",
        "PX_TO_SALES_RATIO": "px_to_sales", "PE_RATIO": "pe", "BEST_PEG_RATIO": "peg"}},
    "fundamentals_ext_q": {"per": "Q", "map": {
        "RETURN_ON_INV_CAPITAL": "roic", "OPER_MARGIN": "oper_margin", "PROF_MARGIN": "net_margin",
        "EBITDA_MARGIN": "ebitda_margin", "TOT_DEBT_TO_TOT_EQY": "debt_to_equity",
        "INTEREST_COVERAGE_RATIO": "int_coverage", "DVD_PAYOUT_RATIO": "dvd_payout",
        "SALES_GROWTH": "sales_growth", "TRAIL_12M_FREE_CASH_FLOW": "trail_fcf"}},
}


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


def main():
    ds = sys.argv[1]
    cfg = DATASETS[ds]
    fields = list(cfg["map"])
    outfile = os.path.join(OUT, f"{ds}.csv")
    uni = sorted(pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique())
    cols = ["date", "ticker"] + list(cfg["map"].values())
    parts = []
    for i in range(0, len(uni), CHUNK):
        ch = uni[i:i + CHUNK]
        d = native(blp.bdh(ch, fields, START, END, Per=cfg["per"]))
        if {"ticker", "date", "field", "value"}.issubset(d.columns):
            parts.append(d[d["value"].notna()])
    long = pd.concat(parts, ignore_index=True)
    long["col"] = long["field"].map(cfg["map"])
    long = long[long["col"].notna()]
    long["ticker"] = long["ticker"].str.replace(" Equity", "", regex=False)
    w = long.pivot_table(index=["date", "ticker"], columns="col", values="value", aggfunc="first").reset_index()
    for c in cols:
        if c not in w.columns:
            w[c] = pd.NA
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    w = w[cols].sort_values(["ticker", "date"])
    w.to_csv(outfile, index=False)
    print(f"{ds}.csv: {len(w)} rows, {w['ticker'].nunique()} names, {w['date'].min()}..{w['date'].max()}")
    for c in list(cfg["map"].values()):
        v = pd.to_numeric(w[c], errors="coerce"); nn = v.notna()
        if nn.any():
            print(f"  {c:14s}: nn {nn.mean():.0%} [{v[nn].min():.4g},{v[nn].max():.4g}] median {v[nn].median():.4g}")


if __name__ == "__main__":
    main()
