"""
Pull forward earnings dates + consensus estimate dispersion (point-in-time snapshot)
-> data/bloomberg/sp500_earnings_estimates.csv

ADDITIVE / NEW file. Does NOT touch sp500_earnings.csv (which the engine's event_gate
consumes via get_next_earnings/get_recent_earnings) -- that file is historical realized
earnings + a single estimate_eps. This snapshot adds, per current SPX member:
  - the FORWARD expected report date (EXPECTED_REPORT_DT) -- the next earnings date,
  - the last announced date (ANNOUNCEMENT_DT),
  - consensus EPS & SALES with DISPERSION (mean / median / high / low / #estimates).

Why: event_gate.py (the engine's largest tail-loss guard) keys off historical
realized earnings with no forward confirmed date and only a single point estimate.
A forward date + estimate dispersion (hi-lo spread, analyst count) lets the lockout
widen when the next print is uncertain. (BEST_EPS_SD / _STD_DEV are NOT populated at
this Bloomberg tier -> dispersion is captured via hi/lo/median/#est instead.)

Source: blp.bdp (point-in-time). Verified field mnemonics 2026-06-05 on AAPL/JPM.
Ticker "A UN" (SPX member code, no " Equity"). Dates YYYY-MM-DD.

Env knobs:
  SWE_EE_SMOKE=1    pull 5 tickers, print, NO write.
  SWE_PULL_NO_WRITE pull+print, skip write.
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
OUT = os.path.join(DATA, "sp500_earnings_estimates.csv")

# Bloomberg field -> output column (order defines CSV column order after ticker)
FIELD_MAP = {
    "EXPECTED_REPORT_DT": "expected_report_date",
    "ANNOUNCEMENT_DT": "last_announcement_date",
    "BEST_EPS": "best_eps",
    "BEST_EPS_MEDIAN": "best_eps_median",
    "BEST_EPS_HI": "best_eps_high",
    "BEST_EPS_LO": "best_eps_low",
    "BEST_EPS_NUMEST": "best_eps_numest",
    "BEST_SALES": "best_sales",
    "BEST_SALES_MEDIAN": "best_sales_median",
    "BEST_SALES_HI": "best_sales_high",
    "BEST_SALES_LO": "best_sales_low",
    "BEST_SALES_NUMEST": "best_sales_numest",
}
DATE_COLS = ["expected_report_date", "last_announcement_date"]
FIELDS = list(FIELD_MAP)
OUT_COLS = ["ticker"] + list(FIELD_MAP.values())


def to_native(o):
    return o.to_native() if hasattr(o, "to_native") else o


def get_members():
    m = to_native(blp.bds("SPX Index", "INDX_MWEIGHT"))
    col = [c for c in m.columns if "member" in c.lower() and "ticker" in c.lower()][0]
    return m[col].tolist()


def main():
    smoke = bool(os.environ.get("SWE_EE_SMOKE"))
    no_write = bool(os.environ.get("SWE_PULL_NO_WRITE"))
    members = get_members()
    print(f"{len(members)} SPX members")
    use = ["AAPL UW", "JPM UN", "NVDA UW", "KO UN", "MSFT UW"] if smoke else members
    tickers = [m + " Equity" for m in use]

    raw = to_native(blp.bdp(tickers=tickers, flds=FIELDS))
    raw.columns = [c.lower() for c in raw.columns]
    wide = raw.pivot_table(index="ticker", columns="field", values="value",
                           aggfunc="first").reset_index()
    wide.columns.name = None
    bbg_to_col = {k.upper(): v for k, v in FIELD_MAP.items()}
    wide = wide.rename(columns={c: bbg_to_col.get(c.upper(), c)
                                for c in wide.columns if c != "ticker"})
    for col in FIELD_MAP.values():
        if col not in wide.columns:
            wide[col] = pd.NA
    wide["ticker"] = wide["ticker"].str.replace(" Equity", "", regex=False)
    for c in DATE_COLS:
        wide[c] = pd.to_datetime(wide[c], errors="coerce").dt.strftime("%Y-%m-%d")
    out = wide[OUT_COLS].sort_values("ticker").reset_index(drop=True)

    nn = {c: int(out[c].notna().sum()) for c in FIELD_MAP.values()}
    print(f"rows={len(out)}  non-null per field: {nn}")
    empty = [c for c, v in nn.items() if v == 0]
    if empty:
        print(f"  !! WARNING entirely-NULL fields: {empty}")
    if smoke:
        print(out.to_string(index=False)); return
    if no_write:
        print("SWE_PULL_NO_WRITE -> not written."); return
    out.to_csv(OUT, index=False)
    print(f"WROTE {os.path.normpath(OUT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
