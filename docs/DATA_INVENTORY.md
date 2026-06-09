# DATA INVENTORY — every file we hold (Bloomberg + Theta + derived)

_Title (type) · date range · counts for every dataset on disk._

**How this was verified.** First pass: `scripts/inventory_data.py` read every file directly
(CSV date columns; parquet footers/filenames). Second pass (2026-06-08): an independent 6-agent
re-derivation recomputed every number by a **different method** (pandas / `pyarrow.dataset`) and an
adversarial diff compared it against this doc — **62 values confirmed exact, 0 hallucinations**. The
only moving numbers are the two **live, still-growing** Theta pulls (`option_history`, `iv_history`),
labelled below with an as-of date. Sampled observation-windows are marked **(sampled)**.

**Provider note.** Bloomberg prices are **split-adjusted**; Theta prices are **raw**. Never mix them.

**Where each lives / what is on GitHub:**
- `data/bloomberg/*.csv` — committed to the repo (`main`).
- `data/bloomberg/deep/` — **gitignored**, ~365 MB. On branch `deep-history/bloomberg-raw` + Google
  Drive `swe-deep-history/`. Restore: `git checkout origin/deep-history/bloomberg-raw -- data/bloomberg/deep`.
- `data_processed/` (all Theta + derived parquet) — **gitignored**, local-only (~3.4 GB).
- This inventory document **is** committed to GitHub.

---

## 1. Bloomberg — monolith CSVs (`data/bloomberg/`, tracked on `main`)

Universe ≈ 503 current S&P 500 names. "Date field" names the column the range is read from
(event tables key on ex/announce/as-of dates, not a daily `date`).

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
| `sp500_corporate_actions.csv` | Corporate actions | **EMPTY stub on `main`** (populated copy is on `deep-history/bloomberg-raw`, 873 KB) | 0 | 0 | — |
| `sp500_iv_history.csv` | (legacy) | **EMPTY** (20 bytes) — superseded by `sp500_vol_iv_full.csv` | 0 | 0 | — |

Side files (not core data): `sp500_short_interest.csv.xlsx` (Excel), `EXTRACTION_GUIDE.md`.

---

## 2. Bloomberg — deep-history archive (`data/bloomberg/deep/`, gitignored, 13 files)

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

## 3. Theta — option/market data (`data_processed/theta/`, gitignored, local-only ~3.4 GB)

The unit is a directory (thousands of parquet shards); the **File / path** column gives the dir and
its shard-naming convention. All raw (unadjusted). Universe "A" ≈ ~500 S&P names.

| File / path | Type / title (shard naming) | Date coverage (verified) | Rows | Names |
|---|---|---|---|---|
| `option_history/` | **Full-depth EOD option chains** — all strikes, C+P, OI; no greeks/IV. **LIVE pull, growing.** (`ticker=…/expiration=YYYYMMDD/data.parquet`) | expirations **2016-01-08 → 2026-06-05**; bars ≈2016 → present | **≈185,242,409** | 70 |
| `option_history_banded_backup_2026-06-01/` | EOD option chains, **Δ-banded** strikes + OI (static backup). (`ticker=…/expiration=YYYYMMDD/data.parquet`) | expirations **2016-01-15 → 2026-05-22**; bars 2017-06-23 → 2026-04/05 (sampled) | 66,574,386 | 503 |
| `chains/` | Full-chain **snapshots** w/ greeks+IV+quotes+OI. (`TICKER_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01, 06-05** | 116,214 | 495 |
| `iv_surface/` | Per-name **IV-surface snapshots** (strike×right×δ×iv×mid×dte). (`TICKER_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01** | 364,192 | 502 |
| `iv_surface_history/` | **IV-surface daily time series** (pilot). (`ticker=…/year=…/date=YYYY-MM-DD.parquet`) | **2026-04-13 → 2026-06-03** (31 dates) | 53,725 | 4 (A, AAPL, ABBV, ABNB) |
| `iv_history/` | **ATM-IV daily time series** (`iv_atm`). (`TICKER.parquet`) | **2015-01-02 → 2026-03-20** (493 names); +4 names (CBOE, LKQ, MHK, SOLS) to **2026-06-01** | 1,291,775 | 497 |
| `index_options_chains/` | Index-option full-chain snapshots. (`SYM_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01** | 8,508 | 8 indices* |
| `index_options_surfaces/` | Index-option IV surfaces. (`SYM_YYYYMMDD.parquet`) | snapshots **2026-04-23, 05-24, 06-01** | 66,230 | 8 indices* |
| `option_ohlc/` | **Per-contract daily OHLC + bid/ask** (4 near expirations). (`TICKER_YYYYMMDD_STRIKE_right.parquet`) | bars ≈**2026-02-23 → 2026-05-22** (sampled); expirations 06-18/06-26/06-30/07-10 | 46,886 | 502 |
| `stocks_eod/` | **Equity EOD OHLCV** (underlying). (`TICKER.parquet`) | **2024-04-23 → 2026-03-20** | 233,912 | 493 |
| `vix_family/vix_family.parquet` | **VIX term-structure** index OHLC | **2023-04-24 → 2026-04-22** | 3,030 | VIX, VIX3M, VIX6M, VIX9D |

\* Index universe (8): SPX, SPXW, NDX, RUT, DJX, VIX, XSP, AAPL. iv_surface ETF add-ons (8): SPY,
QQQ, DIA, IWM, XLE, XLF, XLK, XLV.

**`option_history` / `iv_history` are LIVE** — counts above are an as-of **2026-06-08** snapshot and
keep increasing while the puller runs (e.g. `option_history` was 28,929 files / 184.2M rows ~20 min
earlier in the same session). `corporate_actions/` and `index_reference/` hold no data
(empty / manifest-only).

**Field schemas:** `option_history`/banded (22 cols: OHLC + bid/ask w/ size·exchange·condition +
volume·count + OI; **no greeks/IV**); `chains` (greeks Δ,Θ,V,ρ,ε,λ + IV + quotes + OI);
`iv_surface`/`index_options_surfaces` (strike·right·delta·iv·mid·expiration·dte); `iv_history`
(`iv_atm,ticker,source,date`); `option_ohlc` (OHLC+bid/ask+date); `stocks_eod` (OHLCV+date);
`vix_family` (OHLC+symbol+date).

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

_Regenerate: `.venv/Scripts/python.exe scripts/inventory_data.py` → `data_processed/_inventory_scan.json`._
