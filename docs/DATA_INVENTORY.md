# DATA INVENTORY — every file we hold (Bloomberg + Theta + derived)

_Title (type) · column schema · date range · counts for every dataset on disk.
**Verified from disk 2026-06-25.**_

**How this was verified.** `scripts/inventory_data.py` reads every file directly — CSV
date columns are streamed; parquet date ranges and row counts come from footer/row-group
statistics (no value is taken from any doc) — and writes `data_processed/_inventory_scan.json`.
The **2026-06-25 pass** extended the scanner to cover three new Theta staging trees
(`option_history_deep365`, `option_history_delisted`, `index_reference`) that earlier passes
missed, and re-summed every footer. Column names + dtypes in §6 come from a companion
schema dump (`pyarrow` arrow-schema for parquet, `pandas` dtype inference for CSV). The
event-table date ranges that have no daily `date` column (dividends, earnings, membership)
were read from their PIT key columns directly.

**Provider note.** Bloomberg prices are **split-adjusted**; Theta prices are **raw**. Never mix them.

**Greeks/IV provenance (Theta).** Re-probed live 2026-06-17 (`docs/THETA_ENTITLEMENT_RETEST_2026-06-17.md`):
every Theta greeks/IV history route **404s — not entitled at this tier**. So the deep
`option_history*` trees carry **EOD OHLC + bid/ask + volume/count + open-interest only, no
greeks/IV**. Greeks/IV exist on disk only as the recent `chains`/`iv_surface` snapshots and the
back-solved `iv_history` / `iv_surface_history` series. Any per-strike IV surface the engine
needs must be back-solved from the chains we hold — an engine task, not a vendor pull.

**Footprint / where each lives:**
- `data/bloomberg/*.csv` — committed to the repo (`main`), ~254 MB.
- `data/bloomberg/deep/` — **gitignored**, ~357 MB. On branch `deep-history/bloomberg-raw` + Google
  Drive `swe-deep-history/`. Restore: `git checkout origin/deep-history/bloomberg-raw -- data/bloomberg/deep`.
- `data_processed/` (all Theta + derived parquet) — **gitignored**, local-only. Theta tree ≈ **12 GB**.
- This inventory document **is** committed to GitHub.

> **Regenerate:** `.venv/Scripts/python.exe scripts/inventory_data.py` → `data_processed/_inventory_scan.json`.

---

## 1. Bloomberg — monolith CSVs (`data/bloomberg/`, tracked on `main`)

Universe ≈ 503 current S&P 500 names. "Date field" names the column the range is read from
(event tables key on ex/announce/as-of dates, not a daily `date`). Full column lists + dtypes in §6.

| File name | Type / title | Date range (verified) | Rows | Names | Date field |
|---|---|---|---|---|---|
| `sp500_ohlcv.csv` | Daily equity **OHLCV** (split-adj) | 2018-01-02 → 2026-03-20 | 988,809 | 503 | `date` |
| `sp500_vol_iv_full.csv` | Daily **implied + realized vol** (put/call IV, RV 30/60/90/260d) | 2015-01-02 → 2026-03-20 | 1,361,615 | 503 | `date` |
| `sp500_vol_dvd.csv` | Daily **vol + dividend-yield** panel | 2018-01-02 → 2026-03-20 | 988,837 | 503 | `date` |
| `sp500_liquidity.csv` | Daily **liquidity** (avg vol / turnover / shares out) | 2015-01-02 → 2026-03-20 | 1,362,737 | 503 | `date` |
| `sp500_historical_fundamentals.csv` | **Fundamentals** time series | 2015-01-02 → 2026-02-28 | 30,347 | 503 | `date` |
| `sp500_macro.csv` | **Macro** indicators (daily) | 2015-01-01 → 2026-03-20 | 17,320 | — | `date` |
| `sp500_sector_etfs.csv` | **Sector-ETF** OHLC (daily) | 2015-01-02 → 2026-03-20 | 29,954 | — | `date` |
| `sp500_vix_full.csv` | **VIX** full history (daily) | 2015-01-02 → 2026-03-20 | 16,955 | — | `date` |
| `vix_term_structure.csv` | **VIX term structure** (daily) | 2018-01-02 → 2026-03-20 | 2,094 | — | `date` |
| `treasury_yields.csv` | **Treasury yield** curve (daily) | 2021-05-07 → 2026-05-05 | 1,254 | — | `date` |
| `sp500_dividends.csv` | **Dividend events** (declared/ex/record/pay + amount) | ex-date 1962-05-31 → 2027-03-12 (fwd-declared) | 50,230 | 427 | `ex_date` |
| `sp500_earnings.csv` | **Earnings** (EPS actual/est + announce date) | 1980-01-31 → 2028-01-19 (fwd-est) | 49,379 | 503 | `announcement_date` |
| `sp500_earnings_yf.csv` | Earnings (yfinance backfill) | 2008-01-17 → 2026-08-24 | 12,242 | 498 | `announcement_date` |
| `sp500_index_membership.csv` | **Index membership / weights** (45 monthly snapshots) | as-of 2015-01-01 → 2026-01-01 | 22,690 | — | `as_of_date` |
| `sp500_analyst.csv` | **Analyst** ratings / targets | point-in-time snapshot | 503 | 503 | — |
| `sp500_credit_risk.csv` | **Credit risk** (Altman-Z, S&P rating) | snapshot | 500 | 500 | — |
| `sp500_fundamentals.csv` | **Fundamentals** snapshot (GICS, PE, beta, dvd yld…) | snapshot | 503 | 503 | — |
| `sp500_fundamentals_yf.csv` | Fundamentals snapshot (yfinance) | snapshot | 503 | 503 | — |
| `sp500_institutional.csv` | **Institutional / float** snapshot | snapshot | 503 | 503 | — |
| `sp500_iv_snapshot_today.csv` | Single-day **IV snapshot** (30/60d ATM) | snapshot | 503 | 503 | — |
| `sp500_corporate_actions.csv` | Corporate actions | **EMPTY stub on `main`** (2 bytes, no header; populated copy is on `deep-history/bloomberg-raw`) | 0 | 0 | — |
| `sp500_iv_history.csv` | (legacy) | **EMPTY** (header only, `date,ticker,iv_30d`) — superseded by `sp500_vol_iv_full.csv` | 0 | 0 | — |

Side files (not core data): `sp500_short_interest.csv.xlsx` (Excel workbook), `EXTRACTION_GUIDE.md`.

---

## 2. Bloomberg — deep-history archive (`data/bloomberg/deep/`, gitignored, ~357 MB, 13 files)

Restored from git branch `deep-history/bloomberg-raw` (`e7818f4`) — the set Google Drive
`swe-deep-history/` mirrors. Gzipped CSV. **Dated slices** = current S&P names, split-adj;
**`__delisted`** slices = survivorship-complete (~1,000+ tickers incl. dead names, back to 1990).

| File name | Type / title | Date range (verified) | Rows | Names |
|---|---|---|---|---|
| `sp500_ohlcv__1994_2018.csv.gz` | Deep **OHLCV** | 1994-01-03 → 2017-12-29 | 2,083,270 | 449 |
| `sp500_ohlcv__delisted.csv.gz` | OHLCV incl. **delisted** | 1990-01-02 → 2026-06-05 | 2,383,622 | 1,015 |
| `sp500_vol_iv_full__1994_2012.csv.gz` | Deep **vol/IV** (≡ Drive copy, **byte-identical**, 28,297,508 B) | 1994-01-03 → 2012-06-29 | 1,661,191 | 436 |
| `sp500_vol_iv_full__2012_2018.csv.gz` | Deep **vol/IV** | 2012-07-02 → 2017-12-29 | 630,743 | 476 |
| `sp500_vol_iv__delisted.csv.gz` | Vol/IV incl. **delisted** | 1990-01-02 → 2026-06-05 | 2,408,183 | 1,016 |
| `sp500_liquidity__1994_2015.csv.gz` | Deep **liquidity** | 1994-01-03 → 2014-12-31 | 1,987,751 | 457 |
| `sp500_liquidity__delisted.csv.gz` | Liquidity incl. **delisted** | 1990-01-01 → 2026-06-05 | 2,393,425 | 1,011 |
| `sp500_iv_surface__2005_2011.csv.gz` | **IV moneyness/skew surface** (5 tenor × 5 mny) | 2005-01-03 → 2011-12-30 | 685,310 | 430 |
| `sp500_iv_surface__2012_2018.csv.gz` | IV surface | 2012-01-03 → 2018-12-31 | 795,760 | 470 |
| `sp500_iv_surface__2019_2026.csv.gz` | IV surface | 2019-01-02 → 2026-06-04 | 912,802 | 501 |
| `delisted_status.csv` | Delisting-status map | (no date col) | 1,016 | 1,016 |
| `ohlcv_dropped_ticks.csv` | Gate audit — dropped bad ticks | 1994-01-21 → 2009-01-22 | 97 | 32 |
| `ohlcv_dropped_ticks__delisted.csv` | Gate audit (delisted) | 1990-03-26 → 2006-12-18 | 51 | 37 |

> The Drive-only `sp500_vol_iv_full__1994_2026_FULL.csv.gz` (3,329,212 rows to 2026-06-04, 58 MB)
> is a **convenience concat** of the two dated vol/IV slices + the refreshed monolith. It is **not**
> in git and **not** byte-reproducible from the current on-disk monolith (which ends 2026-03-20).

---

## 3. Theta — option/market data (`data_processed/theta/`, gitignored, local-only ≈ 12 GB)

The unit is a directory (thousands of parquet shards); the **File / path** column gives the dir and
its shard-naming convention. All raw (unadjusted). Universe "A" ≈ ~500 S&P names. **None of the
`option_history*` trees carry greeks or IV** (EOD OHLC + quote + OI only — see the greeks/IV note
in the header). Full column lists + dtypes in §6.

| File / path | Type / title (shard naming) | Date coverage (verified) | Rows | Names |
|---|---|---|---|---|
| `option_history/` | **Full-depth EOD option chains** — all strikes, C+P, daily OI; no greeks/IV. The **wheel larder** (149-name Mag7-first + BRKB + 4 orphans). (`ticker=…/expiration=YYYYMMDD/data.parquet`) | expirations **2016-01-08 → 2026-08-21**; bars ≈ 2016-01-04 → 2026-06-17 | **390,119,692** | 154 |
| `option_history_banded_backup_2026-06-01/` | EOD option chains, **Δ-banded** strikes + OI (static backup of the prior full-universe pull). (`ticker=…/expiration=YYYYMMDD/data.parquet`) | expirations **2016-01-15 → 2026-05-22**; bars 2017-06-23 → 2026-05-22 (sampled) | 66,574,386 | 503 |
| `option_history_deep365/` ⭑NEW | **Top-mega-cap term-structure depth** — 0–365 DTE EOD chains (Phase B). **Staging — never auto-enters the ranker.** (`ticker=…/expiration=YYYYMMDD/data.parquet`) | expirations **2016-01-08 → 2026-06-18**; bars ≈ 2016-01-04 → 2026-06-17 | 17,528,832 | 8 |
| `option_history_delisted/` ⭑NEW | **Delisted/acquired-name survivor-bias chains** (Phase D — tail-risk calibration). **Staging — never auto-enters the ranker.** (`ticker=…/expiration=YYYYMMDD/data.parquet`) | expirations **2016-01-08 → 2024-02-16**; bars ≈ 2016-01-04 → 2023-12-13 | 9,707,709 | 10 |
| `index_reference/option_history/` ⭑NEW | **Index / ETF GEX-reference chains** (Phase C roots + the SPY/QQQ Phase-2 pull). Reference surface for dealer-GEX overlays — **never enters the ranker.** (`option_history/ticker=…/expiration=YYYYMMDD/data.parquet`) | expirations **2016-01-08 → 2026-07-31**; bars ≈ 2016-01-04 → 2026-06-18 | 38,854,575 | 6 |
| `chains/` | Full-chain **snapshots** w/ greeks+IV+quotes+OI. (`TICKER_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01, 06-05** | 116,214 | 495 |
| `iv_surface/` | Per-name **IV-surface snapshots** (strike×right×δ×iv×mid×dte). (`TICKER_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01** | 364,192 | 502 |
| `iv_surface_history/` | **IV-surface daily time series** (stalled back-solve pilot). (`ticker=…/year=…/date=YYYY-MM-DD.parquet`) | **2026-04-13 → 2026-06-03** | 53,725 | 4 (A, AAPL, ABBV, ABNB) |
| `iv_history/` | **ATM-IV daily time series** (back-solved `iv_atm`). (`TICKER.parquet`) | **2015-01-02 → 2026-03-20** | 1,291,775 | 497 |
| `index_options_chains/` | Index-option full-chain snapshots (greeks+IV+OI). (`SYM_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01** | 8,508 | 8 indices* |
| `index_options_surfaces/` | Index-option IV surfaces. (`SYM_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01** | 66,230 | 8 indices* |
| `option_ohlc/` | **Per-contract daily OHLC + bid/ask** (4 near expirations). (`TICKER_YYYYMMDD_STRIKE_right.parquet`) | bars ≈ **2026-02-23 → 2026-05-22** (sampled); expirations 06-18 / 06-26 / 06-30 / 07-10 | 46,886 | 502 |
| `stocks_eod/` | **Equity EOD OHLCV** (underlying). (`TICKER.parquet`) | **2024-04-23 → 2026-03-20** | 233,912 | 493 |
| `vix_family/vix_family.parquet` | **VIX term-structure** index OHLC | **2023-04-24 → 2026-04-22** | 3,030 | VIX, VIX3M, VIX6M, VIX9D |

\* **Snapshot index universe (8):** SPX, SPXW, NDX, RUT, DJX, VIX, XSP, AAPL. `iv_surface` ETF
add-ons (8): SPY, QQQ, DIA, IWM, XLE, XLF, XLK, XLV.

**The three ⭑NEW staging trees** were added by the 2026-06-17 Theta enrichment run (see
`docs/THETA_ENRICH_RUNBOOK_2026-06-17.md`). They are **out-of-ranker by design** — survivor-bias
and term-structure/GEX inputs that feed dormant subsystems (Nelson-Siegel skew, tail / forward-dist
calibration, dealer GEX), **never** `rank_candidates_by_ev`. On-disk ticker counts are partial vs
the planned rosters (deep365 8/20, delisted 10/45, index_reference 6 roots incl. SPY/QQQ; VIX root
not yet present). Actual names:
- `option_history_deep365/` (8): AAPL, AMZN, AVGO, GOOG, GOOGL, META, MSFT, NVDA
- `option_history_delisted/` (10): ABMD, ATVI, FRC, PXD, RE, SBNY, SGEN, SIVB, SPLK, TWTR
- `index_reference/option_history/` (6): NDX, QQQ, RUT, SPX, SPY, XSP

**Pull status:** the larder pull (`option_history/`) is **complete** — the `theta_full_2026-06-01_DONE`
flag was written 2026-06-17; BRKB landed via PR #413. Counts above are a **static 2026-06-25 disk
snapshot**, not a live-growing pull. `corporate_actions/` was never produced (Theta corp-actions 404).

---

## 4. Derived / other stores

| File / path | Type / title | Date range (verified) | Rows | Git |
|---|---|---|---|---|
| `data_processed/vol_indices.parquet` | **Vol-index** long series (VIX/VVIX/SKEW/MOVE/GVZ/OVX/VXN/VIX3M/6M/9D) | 2011-05-31 → 2026-05-22 | 35,062 | ignored |
| `data_processed/vol_indices_wide.parquet` | Same, wide (per-index close cols) | 2011-05-31 → 2026-05-22 | 3,783 | ignored |
| `data_processed/trade_universe/2025-11-22_trade_universe.csv` | Ranked **trade universe** snapshot (20-col candidate set) | 2025-11-22 | 1,066 | ignored |
| `data_processed/ibkr/wheel_ledger.json` + `portfolio_snapshot.json` + `portfolio_history.json` + `ev_calibration.json` + `ev_calibration_detail.csv` | Real **IBKR portfolio** state (acct U17853958) | live account | — | ignored |
| `data_raw/ohlcv/*.csv` (5), `data_raw/yfinance/options/*` (5) | yfinance raw OHLCV + option chains (smoke-test sample) | 2025-11-22 | — | ignored |
| `data_raw/sp500_constituents_current.csv` | Current S&P constituents | snapshot | — | tracked |

---

## 5. Google Drive `swe-deep-history/` — status

Folder `drive.google.com/drive/folders/1abi_CoTa8d-sgzJE0rkwBhHW8Lhv5o58` contains: `README.txt`,
`MANIFEST.txt`, `t.txt` (junk), `sp500_vol_iv_full__1994_2012.csv.gz` (28,297,508 B), and
`sp500_vol_iv_full__1994_2026_FULL.csv.gz` (58,316,365 B) — i.e. **2 of the 12** planned deep-history
files (its own MANIFEST says "2 of 12 present").

**Placed on this desktop at `data/bloomberg/deep/`** (gitignored). Because the Drive folder is a
*partial* mirror of git branch `deep-history/bloomberg-raw`, the desktop copy was restored from that
branch — the **complete** 13-file set in §2, a superset of the Drive folder. The
`__1994_2012` slice is **byte-identical** to its Drive copy (28,297,508 B, verified). The one
Drive-exclusive file (`__1994_2026_FULL` convenience concat) is **not** reproduced — derivable but not
byte-identical from the current on-disk monolith (see §2). To pull the exact Drive bytes, set the
folder's Drive sharing to "anyone with the link" and run `gdown --folder <url>`.

---

## 6. Column schemas & dtypes (every dataset)

Verified 2026-06-25. CSV dtypes are pandas-inferred (dates are stored as strings); parquet dtypes are
the arrow types. `volume` is `float64` in the Bloomberg/Theta-EOD CSVs (not int).

### 6.1 Bloomberg monolith CSVs

| File | Columns (dtype) |
|---|---|
| `sp500_ohlcv.csv` | `date`(str), `ticker`(str), `open` `high` `low` `close`(f64), `volume`(f64) |
| `sp500_vol_iv_full.csv` | `date`(str), `hist_put_imp_vol` `hist_call_imp_vol`(f64), `volatility_30d` `_60d` `_90d` `_260d`(f64), `ticker`(str) |
| `sp500_vol_dvd.csv` | `date`(str), `ticker`(str), `vol_30d` `dvd_yld` `turnover`(f64) |
| `sp500_liquidity.csv` | `date`(str), `avg_vol_30d` `turnover` `shares_out`(f64), `ticker`(str) |
| `sp500_historical_fundamentals.csv` | `date`(str), `ticker`(str), `pe_ratio` `eps` `revenue` `ebitda` `book_value_per_share`(f64) |
| `sp500_macro.csv` | `date`(str), `open` `high` `low` `close`(f64), `instrument`(str) |
| `sp500_sector_etfs.csv` | `date`(str), `open` `high` `low` `close`(f64), `volume`(f64), `etf`(str) |
| `sp500_vix_full.csv` | `date`(str), `close`(f64), `instrument`(str) |
| `vix_term_structure.csv` | `date`(str), `vix` `vix_3m` `vix_6m`(f64) |
| `treasury_yields.csv` | `date`(str), `rate_3m` `rate_6m` `rate_2y` `rate_10y`(f64) |
| `sp500_dividends.csv` | `declared_date` `ex_date` `record_date` `payable_date`(str), `dividend_amount`(f64), `dividend_frequency` `dividend_type`(str), `ticker`(str) |
| `sp500_earnings.csv` / `_yf.csv` | `year/period`(str), `announcement_date` `announcement_time`(str), `earnings_eps` `comparable_eps` `estimate_eps`(f64), `ticker`(str) |
| `sp500_index_membership.csv` | `member_ticker_and_exchange_code`(str), `percentage_weight`(f64), `as_of_date`(str) |
| `sp500_analyst.csv` | `best_analyst_rating` `best_eps` `best_sales` `best_target_price` `tot_analyst_rec`(f64), `ticker`(str) |
| `sp500_credit_risk.csv` | `altman_z_score` `interest_coverage_ratio`(f64), `rtg_sp_lt_lc_issuer_credit`(str), `ticker`(str) |
| `sp500_fundamentals.csv` / `_yf.csv` | `ticker`(str), `30day_impvol_100.0%mny_df` `best_pe_ratio` `beta_raw_overridable` `cur_mkt_cap` `eqy_dvd_yld_12m` `free_cash_flow_yield` `pe_ratio` `return_com_eqy` `tot_debt_to_tot_eqy` `volatility_30d`(f64), `gics_industry_group_name` `gics_sector_name`(str) |
| `sp500_institutional.csv` | `eqy_free_float_pct` `eqy_inst_pct_sh_out` `eqy_sh_out`(f64), `ticker`(str) |
| `sp500_iv_snapshot_today.csv` | `ticker`(str), `30day_impvol_100.0%mny_df` `60day_impvol_100.0%mny_df` `volatility_30d`(f64) |

### 6.2 Bloomberg deep-history gz

| File group | Columns (dtype) |
|---|---|
| `sp500_ohlcv__*.csv.gz` | `date`(str), `ticker`(str), `open` `high` `low` `close` `volume`(f64) |
| `sp500_vol_iv*__*.csv.gz` | `date`(str), `hist_put_imp_vol` `hist_call_imp_vol` `volatility_30d` `_60d` `_90d` `_260d`(f64), `ticker`(str) |
| `sp500_liquidity__*.csv.gz` | `date`(str), `avg_vol_30d` `turnover` `shares_out`(f64), `ticker`(str) |
| `sp500_iv_surface__*.csv.gz` | `date`(str), `iv_{30,60,90,180,360}d_{90,95,100,105,110}`(f64) — 25 tenor×moneyness cols, `ticker`(str) |
| `delisted_status.csv` | `ticker` `name` `window`(str), `ohlcv_rows` `voliv_rows` `liq_rows` `dropped`(i64), `status`(str) |
| `ohlcv_dropped_ticks*.csv` | `date` `ticker`(str), `open` `high` `low` `close` `volume`(f64), `check_failed`/`which` `window`(str) |

### 6.3 Theta parquet

| Dataset | n | Columns (arrow type) |
|---|---|---|
| `option_history` / `_deep365` / `_delisted` / `index_reference` / banded backup | 22 | `symbol`(str), `expiration`(str), `strike`(f64), `right`(str), `created`(str), `last_trade`(str), `open` `high` `low` `close`(f64), `volume`(i64), `count`(i64), `bid_size`(i64), `bid_exchange`(i64), `bid`(f64), `bid_condition`(i64), `ask_size`(i64), `ask_exchange`(i64), `ask`(f64), `ask_condition`(i64), `open_interest`(f64), `ticker`(str) — **no greeks / IV** |
| `chains` | 23 | `symbol` `expiration`(ts) `strike` `right` `delta` `theta` `vega` `rho` `epsilon` `lambda` `iv` `iv_error` `underlying_timestamp` `underlying_price` `bid` `ask` `bid_size` `ask_size` `open_interest` `mid` `ticker` `snapshot_date` (+ index) |
| `index_options_chains` | 21 | as `chains` minus `iv_error`, `underlying_timestamp` |
| `iv_surface` / `index_options_surfaces` | 10 | `strike`(f64) `right`(str) `delta`(f64) `iv`(f64) `mid`(f64) `expiration`(ts) `dte`(i64) `ticker`(str) `snapshot_date`(str) (+ index) |
| `iv_surface_history` | 8 | `date`(ts) `ticker`(str) `expiration`(ts) `dte`(i64) `strike`(f64) `right`(str) `iv`(f64) `mid`(f64) |
| `iv_history` | 4 | `iv_atm`(f64) `ticker`(str) `source`(str) `date`(ts) |
| `option_ohlc` | 12 | `open` `high` `low` `close`(f64) `volume`(i64) `bid` `ask`(f64) `ticker`(str) `expiration`(str) `strike`(f64) `right`(str) `date`(ts) |
| `stocks_eod` | 7 | `open` `high` `low` `close`(f64) `volume`(f64) `ticker`(str) `date`(ts) |
| `vix_family` | 7 | `open` `high` `low` `close`(f64) `symbol`(str) `source`(str) `date`(ts) |

### 6.4 Derived parquet

| File | Columns (arrow type) |
|---|---|
| `vol_indices.parquet` | `date`(ts) `open` `high` `low` `close`(f64) `symbol`(str) `source`(str) |
| `vol_indices_wide.parquet` | `date`(ts) + per-index closes: `gvz_close` `move_close` `ovx_close` `skew_close` `vix_close` `vix3m_close` `vix6m_close` `vix9d_close` `vvix_close` `vxn_close`(f64) |

---

_Regenerate: `.venv/Scripts/python.exe scripts/inventory_data.py` → `data_processed/_inventory_scan.json`._
