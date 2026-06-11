# CASY backfill spec — the one Bloomberg-gated piece of #339

**Status:** the only true Bloomberg blocker for clearing S34's provisional flag.
Everything else in #339 (BK↔BNY entity collapse, the `CTRA/LW/MTCH/PAYC`
dividend gap, `UNIVERSE_100` re-derivation, the 4-snapshot re-baseline) is
reconstructable from data already in git and will be done **after** this pull
lands. See #339 and the 2026-06-07 investigation in
`autonomous-merge-campaign-2026-06-06` (memory).

## Why only CASY

On-the-bytes audit of `data/bloomberg/` (main) vs the
`data/bloomberg-refresh-2026-06-02` branch, 2026-06-07:

| Concern | Reality | Needs Bloomberg? |
|---|---|---|
| `BK`→`BNY` rebrand | `BK UN Equity` OHLCV full 2018→2026-03-20 (753 in-window); `BNY` re-tickers the rest. Dividends: `BNY UN` carries BK's full 1980→2026-04 history. | **No** — collapse from git |
| `CTRA/LW/MTCH/PAYC` dividends | dropped by the refresh; present in main (`MTCH` was missed by #339) | **No** — union from git |
| **`CASY` pre-2026 OHLCV / vol_iv / liquidity / earnings** | **only 2026-03-23→ exists in any committed file** — no pre-2026 history anywhere | **YES — this doc** |
| `CASY` dividends / fundamentals / credit_risk | refresh has full `CASY UW` dividends (1985→2026-05); fundamentals + credit_risk snapshot rows present | **No** — already covered |

So CASY needs **4 files** backfilled: `ohlcv`, `vol_iv`, `liquidity`,
`earnings`. Ticker on Bloomberg: **`CASY UW Equity`** (Casey's General Stores,
NASDAQ). Date range: **2018-01-02 → 2026-06-04** (the current end-date of every
other name; the existing 52 rows at 2026-03-23→2026-06-04 will be replaced, and
their overlap is your validation check).

## What to produce

Write **fragment CSVs** (CASY rows only) — do **not** edit the big monoliths on
the pull box; push/send the fragments and the integration + de-dup is done here
on a reviewed branch. Match each connector file's exact schema and ticker
format:

### 1. `ohlcv` — schema `date,ticker,open,high,low,close,volume`, ticker `CASY UW Equity`
Exact same method as `scripts/pull_ohlcv.py` (split-adjusted, the repo default):
```python
from xbbg import blp
import pandas as pd
df = blp.bdh("CASY UW Equity",
             ["PX_OPEN","PX_HIGH","PX_LOW","PX_LAST","PX_VOLUME"],
             "2018-01-02", "2026-06-04")
df.columns.names = ["ticker","field"]
long = df.stack(level=0).reset_index()
long.columns = ["date","ticker","open","high","low","close","volume"]
long.to_csv("casy_ohlcv.csv", index=False)   # ticker stays 'CASY UW Equity'
```

### 2. `vol_iv` — schema `date,hist_put_imp_vol,hist_call_imp_vol,volatility_30d,volatility_60d,volatility_90d,volatility_260d`, ticker `CASY UW`
- Realized vols (certain): `VOLATILITY_30D`, `VOLATILITY_60D`, `VOLATILITY_90D`,
  `VOLATILITY_260D` via `blp.bdh`.
- Implied vols: the committed file has **`hist_put_imp_vol == hist_call_imp_vol`
  exactly** for 100% of rows (no skew on Bloomberg — see memory
  `bloomberg-iv-no-skew`). Match whatever ATM measure the original used —
  candidates `3MO_PUT_IMP_VOL` / `3MO_CALL_IMP_VOL` (or `PUT_IMP_VOL_30D` /
  `CALL_IMP_VOL_30D`). **VERIFY** by pulling the same field for an existing name
  (e.g. `AAPL UW`) over a recent month and confirming the values match the
  committed `sp500_vol_iv_full.csv` to ~2 decimals before trusting CASY's.
- Strip ` Equity` so ticker = `CASY UW`.

### 3. `liquidity` — schema `date,avg_vol_30d,turnover,shares_out`, ticker `CASY UW`
- Fields: `VOLUME_AVG_30D` → `avg_vol_30d`, `TURNOVER` → `turnover`,
  `EQY_SH_OUT` → `shares_out`. **Note:** the committed file's 3rd column is
  `shares_out`, *not* the `bid_ask_spread` that `scripts/pull_liquidity.py`
  emits — use `EQY_SH_OUT`. Strip ` Equity`.

### 4. `earnings` — schema `year/period,announcement_date,announcement_time,earnings_eps,comparable_eps,estimate_eps`, ticker `CASY UW`
- Historical earnings announcements (feeds the event-lockout gate). Source from
  the earnings/EEO screen or `ERN_ANN_DT_AND_PER` + actual/comparable/estimate
  EPS. Lowest-priority of the four — the event gate degrades gracefully (CASY
  simply won't be earnings-gated) if this can't be produced, so ship the other
  three even if earnings lags.

## After you push/send the fragments — done here (no Bloomberg)

1. **Integrate CASY** fragments into the four monoliths (de-dup the 52 existing
   recent rows against the overlap).
2. **BK↔BNY collapse** — fold `BNY` into `BK`'s continuous history so the
   connector sees one entity (frees the phantom slot).
3. **Dividends union** — `data/bloomberg/sp500_dividends.csv` = refresh ∪ main's
   `CTRA UN / LW UN / MTCH UW / PAYC UN` rows (current **and** complete).
4. **Re-derive `UNIVERSE_100`** — `CMG`/`CMI` return; CASY is now a legitimate
   in-window member. `test_universes_match_connector` enforces the derivation.
5. **Re-baseline all four snapshots** (S27/S32/S34/S35, ~4 h) — the #340 guard
   `test_snapshot_data_fingerprint_matches_current` will (correctly) go red the
   moment the data changes, forcing this re-baseline before the markers run.
6. Clear S34's ⚠️ provisional flag; §2 review panel on the connector change
   (it shifts the backtest universe — `EVEngine.evaluate` untouched).

## Validation gates

- CASY OHLCV overlap (2026-03-23→2026-06-04) must match the existing 52 committed
  rows to the cent.
- vol_iv implied-vol field verified against an existing name (above).
- Post-integration: `test_universes_match_connector` green; the seam audit shows
  **zero** recent-only (0-in-window) names left in `UNIVERSE_100`.
