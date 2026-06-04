"""
#14 - Liquidity metrics (critical for execution)
Pull S&P 500 liquidity metrics from Bloomberg -> data/bloomberg/sp500_liquidity.csv

Contiguous-backfill aware (see scripts/_bbg_panel.py): fills the forward gap to
END_DATE and walks backward to the 1994 floor newest-first, rewriting the CSV
after every window so a metered-API cap never loses contiguous coverage.

Columns (committed schema, preserved EXACTLY): date,avg_vol_30d,turnover,shares_out,ticker
  - shares_out comes from EQY_SH_OUT and MUST NOT be dropped.
Ticker format: "AAPL UW" (exchange code, no " Equity" suffix).
Fill="P" carries the previous value across non-print days.

Env knobs: see scripts/_bbg_panel.py.
"""

from _bbg_panel import PanelConfig, run

run(PanelConfig(
    out_name="sp500_liquidity.csv",
    fields=["VOLUME_AVG_30D", "TURNOVER", "EQY_SH_OUT"],
    field_map={
        "VOLUME_AVG_30D": "avg_vol_30d",
        "TURNOVER": "turnover",
        "EQY_SH_OUT": "shares_out",
    },
    out_cols=["date", "avg_vol_30d", "turnover", "shares_out", "ticker"],
    start_date_full="2015-01-01",
    end_date="2026-06-04",
    floor="1994-01-01",
    strip_equity_suffix=True,
    bdh_kwargs={"Fill": "P"},
))
