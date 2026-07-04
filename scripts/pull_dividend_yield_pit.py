"""Dated PIT dividend-yield panel -> data/bloomberg/broad_pull/dividend_pit/sp500_dividend_yield_pit.csv

Feeds BSM carry-q as-of (EV-moving; wired #426/#428). Monthly (Per="M") point-in-time
series so q can be selected at the query date. FLDS-verified entitled: EQY_DVD_YLD_12M,
EQY_DVD_YLD_IND, DVD_SH_12M. NaN = non-payer at that date -> q=0 (not missing).

APPEND-BY-DESIGN: pulls [SWE_PULL_START, SWE_PULL_END] (default a recent tail), folds into
the existing panel, dedup (date,ticker) keep-last, sort (ticker,date). Origin:
staging/dividend_pit/pull_dividend_yield_pit.py on the broad-pull branch, promoted to a
live-path append producer. For a full rebuild set SWE_PULL_START=2010-01-01.
"""

import os

import pandas as pd
from xbbg import blp

HERE = os.path.dirname(__file__)
MONO = os.path.join(HERE, "..", "data", "bloomberg")
OUT = os.path.join(MONO, "broad_pull", "dividend_pit", "sp500_dividend_yield_pit.csv")
START = os.environ.get("SWE_PULL_START", "2026-06-01")
END = os.environ.get("SWE_PULL_END", "2026-07-02")
FIELDS = ["EQY_DVD_YLD_12M", "EQY_DVD_YLD_IND", "DVD_SH_12M"]
FMAP = {
    "EQY_DVD_YLD_12M": "dvd_yld_12m",
    "EQY_DVD_YLD_IND": "dvd_yld_ind",
    "DVD_SH_12M": "dvd_sh_12m",
}
COLS = ["date", "ticker", "dvd_yld_12m", "dvd_yld_ind", "dvd_sh_12m"]
CHUNK = 250


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


def main():
    uni = sorted(
        pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique()
    )
    parts = []
    for i in range(0, len(uni), CHUNK):
        ch = uni[i : i + CHUNK]
        print(f"  bdh(M) {i}-{min(i + CHUNK, len(uni))}/{len(uni)}", flush=True)
        d = native(blp.bdh(ch, FIELDS, START, END, Per="M"))
        if {"ticker", "date", "field", "value"}.issubset(d.columns):
            parts.append(d[d["value"].notna()])
    long = pd.concat(parts, ignore_index=True)
    w = long.pivot_table(
        index=["date", "ticker"], columns="field", values="value", aggfunc="first"
    ).reset_index()
    w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
    w["date"] = pd.to_datetime(w["date"]).dt.strftime("%Y-%m-%d")
    w = w.rename(columns=FMAP)
    for c in COLS:
        if c not in w.columns:
            w[c] = pd.NA
    w = w[COLS]
    prev = pd.read_csv(OUT, dtype=str) if os.path.exists(OUT) else None
    combined = pd.concat([prev, w], ignore_index=True) if prev is not None else w
    combined = (
        combined.drop_duplicates(["date", "ticker"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)[COLS]
    )
    combined.to_csv(OUT, index=False)
    new = w["date"].max()
    print(
        f"dividend_pit: pulled {START}..{END}; +{len(w)} rows; panel now "
        f"{len(combined)} rows, {combined['date'].min()}..{combined['date'].max()}; newest {new}"
    )


if __name__ == "__main__":
    main()
