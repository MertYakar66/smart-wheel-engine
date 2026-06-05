"""
S&P 500 implied-vol MONEYNESS/SKEW SURFACE (Phase 4) from Bloomberg.

Per-name panel of implied vol at a fixed moneyness x tenor grid -- the raw
material for put-side strike selection and skew/term-structure signals.

WIDE layout (operator decision 2026-06-05): one row per (date,ticker) with one
IV column per tenor x moneyness. ~4x smaller than the tidy-long alternative, so
the gz fits GitHub's 100 MB limit on the deep-history buffer branch (the tidy
long shape melts out of this in one line downstream). Like the other deep
per-name panels it is grown in an OFF-MONOLITH scratch via SWE_OUT_PATH and
carved to a gz on `deep-history/bloomberg-raw`; the connector read-path is
deferred (see worklog DEFER).

Grid (VERIFIED 2026-06-05 on AAPL -- the worklog's "90DAY/120DAY/180DAY" tokens
are INVALID; Bloomberg uses 30DAY/60DAY then 3MTH/6MTH/12MTH; moneyness is fixed
at 90/95/100/105/110, deeper wings unavailable; surface floor is ~2005-01-03):
  tenors  : 30DAY 60DAY 3MTH(=90d) 6MTH(=180d) 12MTH(=360d)
  moneyness: 90 95 100 105 110   (% of spot)
  field    : {tenor}_IMPVOL_{mny}.0%MNY_DF

Columns: date, iv_{30d,60d,90d,180d,360d}_{90,95,100,105,110} (25), ticker.
Ticker format "AAPL UW" (no " Equity" suffix), matching sp500_vol_iv_full.

Env knobs: see scripts/_bbg_panel.py (SWE_OUT_PATH, SWE_PULL_MODE,
SWE_BACKFILL_CHUNK_MONTHS, SWE_BACKFILL_MAX_WINDOWS, SWE_PULL_LIMIT, ...).
"""

from _bbg_panel import PanelConfig, run

# tenor BBG token -> day-equivalent label used in the column name
TENORS = {"30DAY": "30d", "60DAY": "60d", "3MTH": "90d", "6MTH": "180d", "12MTH": "360d"}
MONEYNESS = [90, 95, 100, 105, 110]

_fields = []
_field_map = {}
_iv_cols = []
for _tok, _lab in TENORS.items():
    for _m in MONEYNESS:
        _f = f"{_tok}_IMPVOL_{_m}.0%MNY_DF"
        _col = f"iv_{_lab}_{_m}"
        _fields.append(_f)
        _field_map[_f] = _col
        _iv_cols.append(_col)

run(PanelConfig(
    out_name="sp500_iv_surface.csv",
    fields=_fields,
    field_map=_field_map,
    out_cols=["date"] + _iv_cols + ["ticker"],
    start_date_full="2025-01-01",   # fresh-pull SEED window; backfill walks to floor
    end_date="2026-06-04",
    floor="2005-01-01",             # surface inception (pre-2005 is empty)
    strip_equity_suffix=True,
    bdh_kwargs={"Fill": "P"},
))
