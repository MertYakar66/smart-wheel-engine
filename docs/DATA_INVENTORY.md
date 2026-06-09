# DATA INVENTORY — everything we hold (Bloomberg + Theta + derived)

_Verified 2026-06-08 by a direct file scan (`scripts/inventory_data.py`), not copied
from prior docs. Every date range below was read from the actual CSV date columns or
parquet footers/filenames on disk. Counts that required sampling tens of thousands of
parquet files are marked **(sampled)**._

**Provider note:** the engine's default provider is `bloomberg`. Bloomberg prices are
**split-adjusted**; Theta prices are **raw**. Never mix the two on one series.

**Where things live & what's on GitHub:**
- `data/bloomberg/*.csv` — committed to the repo (tracked on `main`).
- `data/bloomberg/deep/` — **gitignored**, ~365 MB deep-history archive. Lives on git
  branch `deep-history/bloomberg-raw` + Google Drive `swe-deep-history/`. Mirror locally with
  `git checkout origin/deep-history/bloomberg-raw -- data/bloomberg/deep`.
- `data_processed/` (all Theta + derived parquet) — **gitignored**, local-only (~3.4 GB).
- This inventory document **is** committed to GitHub.

---

## 1. Bloomberg — monolith CSVs (`data/bloomberg/`, tracked on `main`)

Universe ≈ 503 current S&P 500 names unless noted.

| File | Type / title | Date range (verified) | Rows | Keyed date col |
|---|---|---|---|---|
| `sp500_ohlcv.csv` | Daily equity **OHLCV** (split-adj) | **2018-01-02 → 2026-03-20** | 988,809 | `date` |
| `sp500_vol_iv_full.csv` | Daily **implied & realized vol** (put/call IV, RV 30/60/90/260d) | **2015-01-02 → 2026-03-20** | 1,361,615 | `date` |
| `sp500_vol_dvd.csv` | Daily **vol + dividend-yield** panel | **2018-01-02 → 2026-03-20** | 988,837 | `date` |
| `sp500_liquidity.csv` | Daily **liquidity** (avg vol / turnover / shares out) | **2015-01-02 → 2026-03-20** | 1,362,737 | `date` |
| `sp500_historical_fundamentals.csv` | **Fundamentals time series** | **2015-01-02 → 2026-02-28** | 30,347 | `date` |
| `sp500_macro.csv` | **Macro** indicators (daily) | **2015-01-01 → 2026-03-20** | 17,320 | `date` |
| `sp500_sector_etfs.csv` | **Sector-ETF** OHLC (daily) | **2015-01-02 → 2026-03-20** | 29,954 | `date` |
| `sp500_vix_full.csv` | **VIX** full history (daily) | **2015-01-02 → 2026-03-20** | 16,955 | `date` |
| `vix_term_structure.csv` | **VIX term-structure** (daily) | **2018-01-02 → 2026-03-20** | 2,094 | `date` |
| `treasury_yields.csv` | **Treasury yield** curve (daily) | **2021-05-07 → 2026-05-05** | 1,254 | `date` |
| `sp500_dividends.csv` | **Dividend events** (declared/ex/record/pay + amount) | ex-date **1962-05-31 → 2027-03-12** (fwd-declared) | 50,230 | `ex_date` |
| `sp500_earnings.csv` | **Earnings** (EPS actual/est, announce date) | announce **1980-01-31 → 2028-01-19** (fwd-est) | 49,379 | `announcement_date` |
| `sp500_earnings_yf.csv` | Earnings (yfinance backfill) | announce **2008-01-17 → 2026-08-24** | 12,242 | `announcement_date` |
| `sp500_index_membership.csv` | **Index membership / weights** (45 monthly snapshots) | as-of **2015-01-01 → 2026-01-01** | 22,690 | `as_of_date` |
| `sp500_analyst.csv` | **Analyst** ratings/targets (point-in-time snapshot) | snapshot (no date col) | 503 | — |
| `sp500_credit_risk.csv` | **Credit risk** (Altman-Z, S&P rating) snapshot | snapshot | 500 | — |
| `sp500_fundamentals.csv` | **Fundamentals** snapshot (GICS, PE, beta, dvd yld…) | snapshot | 503 | — |
| `sp500_fundamentals_yf.csv` | Fundamentals snapshot (yfinance) | snapshot | 503 | — |
| `sp500_institutional.csv` | **Institutional / float** snapshot | snapshot | 503 | — |
| `sp500_iv_snapshot_today.csv` | Single-day **IV snapshot** (30/60d ATM) | snapshot | 503 | — |
| `sp500_corporate_actions.csv` | Corporate actions | **EMPTY stub on `main`** (0 rows). Populated copy exists on `deep-history/bloomberg-raw` (873 KB). | 0 | — |
| `sp500_iv_history.csv` | (legacy) | **EMPTY** (0 rows, 20 bytes) — superseded by `sp500_vol_iv_full.csv` | 0 | — |

Side files present but not core: `sp500_short_interest.csv.xlsx` (Excel), `EXTRACTION_GUIDE.md`.

---

## 2. Bloomberg — deep-history archive (`data/bloomberg/deep/`, gitignored)

Restored from git branch `deep-history/bloomberg-raw` (`e7818f4`) — the same set the Google
Drive `swe-deep-history/` folder mirrors. Gzipped CSV. Two flavours: **dated slices** (current
S&P names, split-adj) and **`__delisted`** slices (survivorship-complete: ~1,000+ tickers incl.
dead names, back to 1990). All ranges below read from the gz date columns.

| File | Type | Date range (verified) | Rows | Tickers |
|---|---|---|---|---|
| `sp500_ohlcv__1994_2018.csv.gz` | Deep **OHLCV** | **1994-01-03 → 2017-12-29** | 2,083,270 | 449 |
| `sp500_ohlcv__delisted.csv.gz` | OHLCV incl. **delisted** | **1990-01-02 → 2026-06-05** | 2,383,622 | 1,015 |
| `sp500_vol_iv_full__1994_2012.csv.gz` | Deep **vol/IV** (≡ Drive copy, byte-identical) | **1994-01-03 → 2012-06-29** | 1,661,191 | 436 |
| `sp500_vol_iv_full__2012_2018.csv.gz` | Deep **vol/IV** | **2012-07-02 → 2017-12-29** | 630,743 | 476 |
| `sp500_vol_iv__delisted.csv.gz` | Vol/IV incl. **delisted** | **1990-01-02 → 2026-06-05** | 2,408,183 | 1,016 |
| `sp500_liquidity__1994_2015.csv.gz` | Deep **liquidity** | **1994-01-03 → 2014-12-31** | 1,987,751 | 457 |
| `sp500_liquidity__delisted.csv.gz` | Liquidity incl. **delisted** | **1990-01-01 → 2026-06-05** | 2,393,425 | 1,011 |
| `sp500_iv_surface__2005_2011.csv.gz` | **IV moneyness/skew surface** (5 tenor × 5 mny) | **2005-01-03 → 2011-12-30** | 685,310 | 430 |
| `sp500_iv_surface__2012_2018.csv.gz` | IV surface | **2012-01-03 → 2018-12-31** | 795,760 | 470 |
| `sp500_iv_surface__2019_2026.csv.gz` | IV surface | **2019-01-02 → 2026-06-04** | 912,802 | 501 |
| `delisted_status.csv` | Delisting status map | (no date col) | 1,016 | 1,016 |
| `ohlcv_dropped_ticks.csv` | Gate audit — dropped bad ticks | 1994-01-21 → 2009-01-22 | 97 | 32 |
| `ohlcv_dropped_ticks__delisted.csv` | Gate audit (delisted) | 1990-03-26 → 2006-12-18 | 51 | 37 |

> The Drive folder's standalone `sp500_vol_iv_full__1994_2026_FULL.csv.gz` (3,329,212 rows to
> 2026-06-04) is a **convenience concat** = the two dated vol/IV slices + the refreshed monolith.
> It is **not** in git and not byte-reproducible from the current on-disk monolith (which ends
> 2026-03-20, not 2026-06-04). The two dated slices above are byte-identical to their Drive copies.

---

## 3. Theta — option/market data (`data_processed/theta/`, gitignored, local-only)

All raw (unadjusted). Universe "A" ≈ the ~500 S&P names; small index/ETF universes noted inline.

| Dataset | Type / title | Symbols | Date coverage (verified) | Rows | Files |
|---|---|---|---|---|---|
| `option_history/` | **Full-depth EOD option chains** (all strikes, C+P, OI; no greeks/IV) — **LIVE pull** | 69 (Mag7-first, → 150) | expirations **2016-01-08 → 2026-06-05**; obs **2016-01-04 → 2026-06-01** (sampled) | **184,181,972** | 28,929 |
| `option_history_banded_backup_2026-06-01/` | EOD option chains, **Δ-banded** strikes + OI (backup) | 503 | exp **2016-01-15 → 2026-05-22**; obs **2017-06-23 → 2026-05-22** (sampled) | 66,574,386 | 51,729 |
| `chains/` | Full-chain **snapshots** w/ greeks+IV+quotes+OI | 495 | snapshots **2026-04-23, 05-24, 06-01, 06-05** | 116,214 | 1,521 |
| `iv_surface/` | Per-name **IV-surface snapshots** (strike×right×δ×iv×mid×dte) | 502 (incl. 8 ETFs) | snapshots **2026-04-23, 05-24, 06-01** | 364,192 | 558 |
| `iv_surface_history/` | **IV-surface daily time series** (pilot) | 4 (A, AAPL, ABBV, ABNB) | **2026-04-13 → 2026-06-03** | — | 108 |
| `iv_history/` | **ATM-IV daily time series** (`iv_atm`) — long history | 497 | **2015-01-02 → 2026-03-20** | 1,291,775 | 497 |
| `index_options_chains/` | Index-option full-chain snapshots | 8 indices* | snapshots **2026-04-23, 05-24, 06-01** | 8,508 | 21 |
| `index_options_surfaces/` | Index-option IV surfaces | 8 indices* | snapshots **2026-04-23, 05-24, 06-01** | 66,230 | 21 |
| `option_ohlc/` | **Per-contract daily OHLC + bid/ask** (4 near expirations) | 502 | bars **≈2026-02-23 → 2026-05-22** (sampled); exp 06-18/06-26/06-30/07-10 | 46,886 | 1,507 |
| `stocks_eod/` | **Equity EOD OHLCV** (underlying) | 493 | **2024-04-23 → 2026-03-20** | 233,912 | 493 |
| `vix_family/` | **VIX term-structure** index OHLC | VIX, VIX3M, VIX6M, VIX9D | **2023-04-24 → 2026-04-22** | 3,030 | 1 |

\* Index universe (8): SPX, SPXW, NDX, RUT, DJX, VIX, XSP, AAPL. iv_surface ETF add-ons (8): SPY,
QQQ, DIA, IWM, XLE, XLF, XLK, XLV.

**Field schemas:** `option_history`/banded (22 cols: OHLC + bid/ask w/ size·exchange·condition +
volume·count + OI; **no greeks/IV**); `chains` (greeks Δ,Θ,V,ρ,ε,λ + IV + quotes + OI);
`iv_surface`/`index_options_surfaces` (strike·right·delta·iv·mid·expiration·dte); `iv_history`
(`iv_atm,ticker,source,date`); `option_ohlc` (OHLC+bid/ask+date); `stocks_eod`
(OHLCV+date); `vix_family` (OHLC+symbol+date).

`corporate_actions/` is an **empty** dir (Theta corp-actions never produced output).
`_manifest.json` records pull runs; `index_reference/` holds only a manifest.

---

## 4. Derived / other data stores

| Path | Type | Date range (verified) | Rows | Git |
|---|---|---|---|---|
| `data_processed/vol_indices.parquet` | **Vol-index** long series (VIX/VVIX/SKEW/MOVE/GVZ/OVX/VXN/VIX3M/6M/9D) | **2011-05-31 → 2026-05-22** | 35,062 | ignored |
| `data_processed/vol_indices_wide.parquet` | Same, wide (per-index close cols) | **2011-05-31 → 2026-05-22** | 3,783 | ignored |
| `data_processed/ibkr/` | Real **IBKR portfolio** state: `wheel_ledger.json`, `portfolio_snapshot.json`, `portfolio_history.json`, `ev_calibration*.{json,csv}` | live account (acct U17853958) | — | ignored |
| `data_processed/trade_universe/2025-11-22_trade_universe.csv` | Ranked **trade universe** snapshot | 2025-11-22 | — | ignored |
| `data_raw/ohlcv/*.csv` (5), `data_raw/yfinance/options/*` (5) | yfinance raw OHLCV + option chains (smoke-test sample) | 2025-11-22 | — | ignored |
| `data_raw/sp500_constituents_current.csv` | Current S&P constituents | snapshot | — | tracked |

---

## 5. Google Drive `swe-deep-history/` — status

The Drive folder (`drive.google.com/drive/folders/1abi_CoTa8d-sgzJE0rkwBhHW8Lhv5o58`) contains:
`README.txt`, `MANIFEST.txt`, `t.txt` (junk), `sp500_vol_iv_full__1994_2012.csv.gz` (28,297,508 B),
and `sp500_vol_iv_full__1994_2026_FULL.csv.gz` (58,316,365 B) — i.e. **2 of the 12** planned
deep-history files (its own MANIFEST says "2 of 12 present").

**Placed on this desktop at `data/bloomberg/deep/`** (gitignored). Because the Drive folder is a
*partial* mirror of git branch `deep-history/bloomberg-raw`, the desktop copy was restored from
that branch — which is the **complete** set (all 13 files above), a superset of the Drive folder.
The one Drive-exclusive file (the `__1994_2026_FULL` convenience concat) is **not** reproduced: it
is derivable but not byte-identical from the current on-disk monolith (see §2 note). To pull the
exact Drive bytes instead, set the folder's Drive sharing to "anyone with the link" and run
`gdown --folder <url>`.

---

_Regenerate this inventory: `.venv/Scripts/python.exe scripts/inventory_data.py` (writes
`data_processed/_inventory_scan.json`)._
