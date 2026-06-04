"""
Pull S&P 500 daily OHLCV from Bloomberg -> data/bloomberg/sp500_ohlcv.csv

Contiguous-backfill aware (see scripts/_bbg_panel.py): fills the forward gap
to END_DATE and walks backward to the 1994 floor newest-first, rewriting the
CSV after every window so a metered-API cap never loses contiguous coverage.

ROTATED LAYOUT (load-bearing — do not "tidy"):
  The committed CSV stores OHLC with column labels rotated one position so the
  connector's compensating rename (engine/data_connector.py: open->high,
  high->close, close->open) yields correct prices. We reproduce that exact
  on-disk layout by MAPPING fields accordingly:
      stored "open"  <- PX_HIGH   (true daily high; must be the row-max)
      stored "high"  <- PX_LAST   (true close)
      stored "low"   <- PX_LOW    (true low; must be the row-min)
      stored "close" <- PX_OPEN   (true open)
  Post-build gate asserts EVERY row has stored open == max(o,h,l,c) and stored
  low == min(o,h,l,c); any failure means the layout drifted -> STOP.

Columns (unchanged schema): date,ticker,open,high,low,close,volume
Ticker format: "AAPL UW Equity" (exchange code + " Equity").

Env knobs: see scripts/_bbg_panel.py.
"""

from _bbg_panel import PanelConfig, run


def ohlcv_rotation_gate(df):
    """Assert the rotated layout holds: stored open is the row-max, low the row-min.

    Exact float equality is correct here: the row-max is, by construction, one
    of the four stored values, so `open == rowmax` holds iff `open` IS that
    maximum. Ties (e.g. flat day) do not trip it. Raises SystemExit on any
    failing row so a wrong field map / regenerated-canonical CSV halts the run.
    """
    sub = df.dropna(subset=["open", "high", "low", "close"])
    n = len(sub)
    if n == 0:
        print("  OHLCV GATE: no non-NaN OHLC rows to check.")
        return
    ohlc = sub[["open", "high", "low", "close"]]
    bad_hi = int((sub["open"] != ohlc.max(axis=1)).sum())
    bad_lo = int((sub["low"] != ohlc.min(axis=1)).sum())
    print(f"  OHLCV GATE: {bad_hi} rows open!=max, {bad_lo} rows low!=min of {n:,} non-NaN rows")
    if bad_hi or bad_lo:
        raise SystemExit(
            f"OHLCV GATE FAILED ({bad_hi} open!=max, {bad_lo} low!=min): "
            "stored layout is not the rotated convention the connector expects. "
            "Field map is wrong or the CSV was regenerated canonical. STOP."
        )


run(PanelConfig(
    out_name="sp500_ohlcv.csv",
    fields=["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"],
    field_map={
        "PX_HIGH": "open",     # rotated: stored 'open' holds the true HIGH
        "PX_LAST": "high",     # rotated: stored 'high' holds the true CLOSE
        "PX_LOW": "low",       # stored 'low' holds the true LOW
        "PX_OPEN": "close",    # rotated: stored 'close' holds the true OPEN
        "PX_VOLUME": "volume",
    },
    out_cols=["date", "ticker", "open", "high", "low", "close", "volume"],
    start_date_full="2018-01-01",
    end_date="2026-06-04",
    floor="1994-01-01",
    strip_equity_suffix=False,
    bdh_kwargs={},
    validate=ohlcv_rotation_gate,
))
