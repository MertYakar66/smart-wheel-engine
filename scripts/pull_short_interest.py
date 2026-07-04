"""Short interest (biweekly settlement prints) -> data/bloomberg/broad_pull/short_interest/sp500_short_interest.csv

This is the CONSUMED panel (broad_pull loader). Supersedes the earlier naive producer that
wrote the non-consumed data/bloomberg/sp500_short_interest.csv with the entitlement-blocked
SI_PERCENT_FLOAT and Fill="P".

FLDS entitlement (this tier): SHORT_INTEREST (shares), SHORT_INT_RATIO (days-to-cover) ENTITLED;
*_PCT_OF_FLOAT and every borrow-rate field BLOCKED (bucket F, no SLB entitlement). No Fill —
keep true PIT settlement dates.

APPEND-BY-DESIGN: pulls [SWE_PULL_START, SWE_PULL_END] (default a recent tail), folds into the
existing panel, dedup (date,ticker) keep-last, sort (ticker,date). Full rebuild: SWE_PULL_START=2015-01-01.
"""

import os

import pandas as pd
from xbbg import blp

HERE = os.path.dirname(__file__)
MONO = os.path.join(HERE, "..", "data", "bloomberg")
OUT = os.path.join(MONO, "broad_pull", "short_interest", "sp500_short_interest.csv")
START = os.environ.get("SWE_PULL_START", "2026-06-01")
END = os.environ.get("SWE_PULL_END", "2026-07-02")
FIELDS = ["SHORT_INTEREST", "SHORT_INT_RATIO"]
FMAP = {"SHORT_INTEREST": "short_interest", "SHORT_INT_RATIO": "short_int_ratio"}
COLS = ["date", "ticker", "short_interest", "short_int_ratio"]
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
        print(f"  bdh {i}-{min(i + CHUNK, len(uni))}/{len(uni)}", flush=True)
        d = native(blp.bdh(ch, FIELDS, START, END))
        if {"ticker", "date", "field", "value"}.issubset(d.columns):
            parts.append(d[d["value"].notna()])
    long = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["ticker", "date", "field", "value"])
    )
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
    print(
        f"short_interest: pulled {START}..{END}; +{len(w)} rows (settle dates "
        f"{sorted(w['date'].unique())}); panel now {len(combined)} rows, "
        f"{combined['date'].min()}..{combined['date'].max()}"
    )


if __name__ == "__main__":
    main()
