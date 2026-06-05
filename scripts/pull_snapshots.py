"""
Pull S&P 500 point-in-time Bloomberg snapshots (Phase 2c) via blp.bdp.

Refreshes the committed snapshot CSVs with TODAY's values. Each file's exact
committed schema, COLUMN ORDER, and per-file TICKER FORMAT are reproduced (these
differ per file and are load-bearing for the connector):

  sp500_fundamentals.csv       ticker "AAPL UW Equity" (suffix), ticker FIRST
  sp500_iv_snapshot_today.csv  ticker "AAPL UW Equity" (suffix), ticker FIRST
  sp500_credit_risk.csv        ticker "A UN" (no suffix),        ticker LAST
  sp500_institutional.csv      ticker "A UN" (no suffix),        ticker LAST
  sp500_analyst.csv            ticker "A UN" (no suffix),        ticker LAST

The field lists are the *committed headers* (= the prior session's verified
fields). bdp returns long (ticker, field, value) with field names exactly as
requested; we pivot -> wide, map field->committed column, reorder, set ticker
format, write.

Gotcha (2026-06-02 worklog): one invalid field can null a whole bdp batch ->
ALWAYS smoke-test first and watch the per-field non-null counts.

Env knobs:
  SWE_SNAP_SMOKE=1   pull only 3 tickers, print, DO NOT write (preview).
  SWE_SNAP_ONLY=<name>  restrict to one dataset (e.g. sp500_credit_risk).
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
from xbbg import blp

for _s in (sys.stdout, sys.stderr):
    if isinstance(_s, io.TextIOWrapper):
        _s.reconfigure(encoding="utf-8", errors="replace")

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "bloomberg")

# output_column (committed header, lowercase) -> Bloomberg field
SNAPSHOTS = {
    "sp500_fundamentals": dict(
        suffix=True, ticker_first=True,
        fields={
            "30day_impvol_100.0%mny_df": "30DAY_IMPVOL_100.0%MNY_DF",
            "best_pe_ratio": "BEST_PE_RATIO",
            "beta_raw_overridable": "BETA_RAW_OVERRIDABLE",
            "cur_mkt_cap": "CUR_MKT_CAP",
            "eqy_dvd_yld_12m": "EQY_DVD_YLD_12M",
            "free_cash_flow_yield": "FREE_CASH_FLOW_YIELD",
            "gics_industry_group_name": "GICS_INDUSTRY_GROUP_NAME",
            "gics_sector_name": "GICS_SECTOR_NAME",
            "pe_ratio": "PE_RATIO",
            "return_com_eqy": "RETURN_COM_EQY",
            "tot_debt_to_tot_eqy": "TOT_DEBT_TO_TOT_EQY",
            "volatility_30d": "VOLATILITY_30D",
        },
    ),
    "sp500_credit_risk": dict(
        suffix=False, ticker_first=False,
        fields={
            "altman_z_score": "ALTMAN_Z_SCORE",
            "interest_coverage_ratio": "INTEREST_COVERAGE_RATIO",
            "rtg_sp_lt_lc_issuer_credit": "RTG_SP_LT_LC_ISSUER_CREDIT",
        },
    ),
    "sp500_institutional": dict(
        suffix=False, ticker_first=False,
        fields={
            "eqy_free_float_pct": "EQY_FREE_FLOAT_PCT",
            "eqy_inst_pct_sh_out": "EQY_INST_PCT_SH_OUT",
            "eqy_sh_out": "EQY_SH_OUT",
        },
    ),
    "sp500_analyst": dict(
        suffix=False, ticker_first=False,
        fields={
            "best_analyst_rating": "BEST_ANALYST_RATING",
            "best_eps": "BEST_EPS",
            "best_sales": "BEST_SALES",
            "best_target_price": "BEST_TARGET_PRICE",
            "tot_analyst_rec": "TOT_ANALYST_REC",
        },
    ),
    "sp500_iv_snapshot_today": dict(
        suffix=True, ticker_first=True,
        fields={
            "30day_impvol_100.0%mny_df": "30DAY_IMPVOL_100.0%MNY_DF",
            "60day_impvol_100.0%mny_df": "60DAY_IMPVOL_100.0%MNY_DF",
            "volatility_30d": "VOLATILITY_30D",
        },
    ),
}


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def get_members():
    m = to_native(blp.bds("SPX Index", "INDX_MWEIGHT"))
    col = [c for c in m.columns if "member" in c.lower() and "ticker" in c.lower()][0]
    return [t + " Equity" for t in m[col].tolist()]


def pull_one(name, cfg, tickers, smoke):
    flds = list(cfg["fields"].values())
    bbg_to_col = {v.upper(): k for k, v in cfg["fields"].items()}
    use = tickers[:3] if smoke else tickers
    print(f"\n[{name}] bdp {len(use)} tickers x {len(flds)} fields", flush=True)
    raw = to_native(blp.bdp(tickers=use, flds=flds))
    if raw is None or len(raw) == 0:
        print("  EMPTY result -> skipping (check field validity!)")
        return None
    raw.columns = [c.lower() for c in raw.columns]
    wide = raw.pivot(index="ticker", columns="field", values="value").reset_index()
    wide.columns.name = None
    # field names come back exactly as requested (any case) -> map to committed col
    wide = wide.rename(columns={c: bbg_to_col.get(c.upper(), c) for c in wide.columns if c != "ticker"})
    for col in cfg["fields"]:
        if col not in wide.columns:
            wide[col] = pd.NA
    if not cfg["suffix"]:
        wide["ticker"] = wide["ticker"].str.replace(" Equity", "", regex=False)
    order = (["ticker"] + list(cfg["fields"])) if cfg["ticker_first"] else (list(cfg["fields"]) + ["ticker"])
    out = wide[order].sort_values("ticker").reset_index(drop=True)
    return out


def main():
    smoke = bool(os.environ.get("SWE_SNAP_SMOKE"))
    only = os.environ.get("SWE_SNAP_ONLY")
    tickers = get_members()
    print(f"{len(tickers)} SPX members (current INDX_MWEIGHT)")
    for name, cfg in SNAPSHOTS.items():
        if only and name != only:
            continue
        out = pull_one(name, cfg, tickers, smoke)
        if out is None:
            continue
        nn = {c: int(out[c].notna().sum()) for c in cfg["fields"]}
        print(f"  rows={len(out)} | non-null per field: {nn}")
        empty = [c for c, v in nn.items() if v == 0]
        if empty:
            print(f"  !! WARNING fields entirely NULL (possible invalid field nulling batch): {empty}")
        if smoke:
            print(out.head(3).to_string(index=False))
            continue
        path = os.path.join(DATA, name + ".csv")
        out.to_csv(path, index=False)
        print(f"  WROTE {os.path.normpath(path)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
