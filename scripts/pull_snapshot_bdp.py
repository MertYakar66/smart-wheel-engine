"""Per-name BDP snapshot -> data/bloomberg/broad_pull/per_name/sp500_snapshot_bdp.csv

Refreshes the live per-name snapshot the connector overlays for the earnings
lockout (``next_earnings_dt``) + GICS/ratings/ownership. APPEND-ONLY PIT record:
one row per ``(asof, ticker)``; each run adds a new ``asof`` vintage (default the
pull date, ``SWE_SNAPSHOT_ASOF``) and NEVER replaces prior vintages — the
connector serves the newest snapshot with ``asof <= ref`` and refuses when none
qualifies (#465), so dropping a vintage silently un-arms the lockout for every
dated query in ``[old_asof, new_asof)``. Origin: staging/per_name/pull_snapshot_bdp.py on
origin/claude/bloomberg-broad-pull-2026-06-17 (2026-06-18 lab), promoted to a
real producer writing the LIVE path and pinned to the committed 14-col schema.

The three rating watch/outlook fields (sp_outlook/rating_outlook/rating_watch)
are entitlement-blocked on the lab tier (all-NaN) and are NOT in the committed
schema — we drop them explicitly so the schema never drifts on a re-pull.

If refreshed, bump EXPECTED_EARNINGS_CALENDAR_ASOF in
tests/test_preflight_environment.py in the SAME commit (guard-rot rule).
"""

import os

import pandas as pd
from xbbg import blp

HERE = os.path.dirname(__file__)
MONO = os.path.join(HERE, "..", "data", "bloomberg")
OUT = os.path.join(MONO, "broad_pull", "per_name", "sp500_snapshot_bdp.csv")
ASOF = os.environ.get("SWE_SNAPSHOT_ASOF", "2026-07-03")
CHUNK = 50

# bbg field -> committed output column. Order here IS the committed schema
# (asof, ticker are prepended). Entitlement-blocked outlook/watch omitted.
FMAP = {
    "RTG_SP_LT_LC_ISSUER_CREDIT": "rtg_sp",
    "RTG_MOODY_LONG_TERM": "rtg_moody",
    "RTG_FITCH_LT_ISSUER_DEFAULT": "rtg_fitch",
    "GICS_SECTOR_NAME": "gics_sector",
    "GICS_INDUSTRY_GROUP_NAME": "gics_ind_grp",
    "GICS_INDUSTRY_NAME": "gics_industry",
    "GICS_SUB_INDUSTRY_NAME": "gics_sub_ind",
    "EQY_INST_PCT_SH_OUT": "inst_pct",
    "EQY_FREE_FLOAT_PCT": "free_float_pct",
    "EQY_FLOAT": "float_shares",
    "EXPECTED_REPORT_DT": "next_earnings_dt",
    "CRNCY": "crncy",
}
COMMITTED_COLS = ["asof", "ticker"] + list(FMAP.values())


def native(nw):
    return nw.to_native() if hasattr(nw, "to_native") else nw


def main():
    uni = sorted(
        pd.read_csv(os.path.join(MONO, "sp500_ohlcv.csv"), usecols=["ticker"])["ticker"].unique()
    )
    tickers = [t if t.endswith(" Equity") else f"{t} Equity" for t in uni]
    fields = list(FMAP)
    parts = []
    for i in range(0, len(tickers), CHUNK):
        ch = tickers[i : i + CHUNK]
        print(f"  bdp {i + 1}-{min(i + CHUNK, len(tickers))}/{len(tickers)}", flush=True)
        parts.append(native(blp.bdp(ch, fields)))
    raw = pd.concat(parts, ignore_index=True)
    if {"ticker", "field", "value"}.issubset(raw.columns):
        w = raw.pivot_table(index="ticker", columns="field", values="value", aggfunc="first")
    else:
        w = raw.set_index("ticker")
    w = w.rename(columns=FMAP).reset_index()
    w["ticker"] = w["ticker"].str.replace(" Equity", "", regex=False)
    w.insert(0, "asof", ASOF)
    # Force the exact committed schema: add any missing col as NA, drop extras.
    for c in COMMITTED_COLS:
        if c not in w.columns:
            w[c] = pd.NA
    w = w[COMMITTED_COLS]
    # APPEND-ONLY: fold this vintage into the existing panel; dedup
    # (asof, ticker) keep-last (idempotent same-day re-run); sort (asof, ticker).
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT, dtype=str)
        combined = pd.concat([prev, w], ignore_index=True)
    else:
        combined = w
    combined = (
        combined.drop_duplicates(["asof", "ticker"], keep="last")
        .sort_values(["asof", "ticker"])
        .reset_index(drop=True)[COMMITTED_COLS]
    )
    combined.to_csv(OUT, index=False)
    cur = combined[combined["asof"] == ASOF]
    print(
        f"sp500_snapshot_bdp.csv: +{len(cur)} names @ asof {ASOF}; "
        f"{combined['asof'].nunique()} vintage(s), {len(combined)} rows total"
    )
    print(
        f"  new-vintage next_earnings_dt cov {cur['next_earnings_dt'].notna().mean():.0%}, "
        f"max {pd.to_datetime(cur['next_earnings_dt'], errors='coerce').max()}"
    )


if __name__ == "__main__":
    main()
