"""
S&P 500 implied / historical volatility from Bloomberg -> data/bloomberg/sp500_vol_iv_full.csv

Contiguous-backfill aware (see scripts/_bbg_panel.py): fills the forward gap to
END_DATE and walks backward to the 1994 floor (IV's Bloomberg hard floor)
newest-first, rewriting the CSV after every window so a metered-API cap never
loses contiguous coverage.

Equivalent BQL:
  get(hist_put_imp_vol, hist_call_imp_vol, volatility_30d, volatility_60d,
      volatility_90d, volatility_260d) for(members('SPX Index'))
  with(dates=range(<floor>, <end>), fill=prev)

Columns (committed schema, preserved EXACTLY):
  date,hist_put_imp_vol,hist_call_imp_vol,volatility_30d,volatility_60d,volatility_90d,volatility_260d,ticker
Ticker format: "AAPL UW" (exchange code, no " Equity" suffix).

Env knobs: see scripts/_bbg_panel.py.
"""

from _bbg_panel import PanelConfig, run

run(PanelConfig(
    out_name="sp500_vol_iv_full.csv",
    fields=[
        "HIST_PUT_IMP_VOL",
        "HIST_CALL_IMP_VOL",
        "VOLATILITY_30D",
        "VOLATILITY_60D",
        "VOLATILITY_90D",
        "VOLATILITY_260D",
    ],
    field_map={
        "HIST_PUT_IMP_VOL": "hist_put_imp_vol",
        "HIST_CALL_IMP_VOL": "hist_call_imp_vol",
        "VOLATILITY_30D": "volatility_30d",
        "VOLATILITY_60D": "volatility_60d",
        "VOLATILITY_90D": "volatility_90d",
        "VOLATILITY_260D": "volatility_260d",
    },
    out_cols=[
        "date",
        "hist_put_imp_vol",
        "hist_call_imp_vol",
        "volatility_30d",
        "volatility_60d",
        "volatility_90d",
        "volatility_260d",
        "ticker",
    ],
    start_date_full="2015-01-01",
    end_date="2026-06-04",
    floor="1994-01-01",
    strip_equity_suffix=True,
    bdh_kwargs={"Fill": "P"},
))
