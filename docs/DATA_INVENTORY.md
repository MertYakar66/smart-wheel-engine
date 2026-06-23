# DATA INVENTORY — every dataset we hold (Bloomberg + Theta + derived + staged)

_Title (type) · date range · counts for every dataset. **Regenerated 2026-06-22.**_

**How this was verified (2026-06-22).** The committed Bloomberg monoliths (§1) were
read with `scripts/inventory_data.py`'s streaming reader against the **`origin/main`**
bytes (CSV date columns, byte-true row/ticker counts). The gitignored deep archive (§2)
and the Theta corpus (§3) were read from the **local desktop** parquet/gz (local-only,
as-of 2026-06-22). The staged broad-pull data (§6) was byte-scanned on branch
`claude/bloomberg-broad-pull-2026-06-17`. Numbers below are from the actual bytes, not
from docs.

> **⚠️ Trust note — the prior inventory was stale, and a naïve rescan reproduces the
> staleness.** `scripts/inventory_data.py` has a hardcoded `ROOT` pointing at the local
> desktop checkout. That checkout currently sits on branch
> **`claude/weakness-review-fixes`**, whose `data/bloomberg/` is an **older snapshot**
> (OHLCV → 2026-03-20, treasury 2021-05+, corporate-actions empty). Running the script
> there **reproduces the old numbers** — the trap the previous version of this doc fell
> into. The byte-true monolith census in §1 is therefore taken from **`origin/main`**
> (blob-verified to differ from the local branch on `sp500_ohlcv`, `sp500_vol_iv_full`,
> `treasury_yields`, `sp500_corporate_actions`, …); the broad-pull branch descends from
> the same `origin/main`. **To rescan correctly, point `ROOT` at an `origin/main`
> checkout, not the working branch.**

**Provider note.** Bloomberg prices are **split-adjusted**; Theta prices are **raw**. Never mix them.

**Where each lives / what is on GitHub:**
- `data/bloomberg/*.csv` — committed to the repo (`main`). §1.
- `data/bloomberg/deep/` — **gitignored**, ~365 MB; restore from `origin/deep-history/bloomberg-raw`. §2.
- `data_processed/` (all Theta + derived parquet) — **gitignored**, local-only (~several GB). §3–§4.
- `staging/` — committed **on branch `claude/bloomberg-broad-pull-2026-06-17` only** (held, not on `main`). §6.
- This inventory document **is** committed to GitHub.

---

## 0. Reconciliation — what changed vs the prior inventory (stale → byte-true)

The previous doc claimed daily files end `2026-03-20`, treasury starts `2021-05`, and
corporate-actions is a 2-byte stub. `origin/main` has since been refreshed/backfilled.
Corrected deltas (all byte-verified 2026-06-22):

| File | Prior inventory (stale) | `origin/main` (byte-true) |
|---|---|---|
| `sp500_ohlcv.csv` | 988,809 · 2018→**2026-03-20** · 503 nm | **1,014,920 · 2018-01-02→2026-06-04 · 511 nm** |
| `sp500_vol_iv_full.csv` | 1,361,615 · **2015**→2026-03-20 · 503 | **1,037,278 · 2018-01-02→2026-06-04 · 510** (pre-2018 IV now in deep slices) |
| `sp500_liquidity.csv` | 1,362,737 · 2015→2026-03-20 · 503 | **1,388,848 · 2015-01-02→2026-06-04 · 511** |
| `treasury_yields.csv` | 1,254 · **2021-05-07**→2026-05-05 | **8,458 · 1994-01-03→2026-06-05** (full curve; `rate_1m` blank pre-2001) |
| `sp500_corporate_actions.csv` | **EMPTY 2-byte stub** (0 rows) | **52,442 rows · 1962-05-23→2026-06-05 · 481 nm** |
| `sp500_historical_fundamentals.csv` | 30,347 · 2015→2026-02-28 | **79,198 · 1990-01-01→2026-05-10** |
| `sp500_index_membership.csv` | 22,690 · 2015→2026-01-01 | **72,696 · 1990-04-01→2026-04-01** |
| `sp500_macro.csv` | 17,320 · 2015-01-01→2026-03-20 | **56,180 · 1990-01-02→2026-06-04** |
| `sp500_sector_etfs.csv` | 29,954 · 2015→2026-03-20 | **66,824 · 1998-12-22→2026-06-05** |
| `vix_term_structure.csv` | 2,094 · 2018→2026-03-20 | **9,200 · 1990-01-02→2026-06-04** |
| `sp500_vix_full.csv` | 16,955 · →2026-03-20 | **17,274 · 2015-01-02→2026-06-05** |
| `sp500_vol_dvd.csv` | 988,837 · →2026-03-20 | **988,837 · →2026-03-20 — UNCHANGED (laggard; not refreshed)** |
| Theta `option_history/` | ≈185.2M rows · 70 nm | **390,119,692 rows · 154 nm** (live pull grew) |

> The broad-pull **currency refresh** (§6) carries each refreshed daily series further to
> **2026-06-18** (latest bar = today, gate-confirmed) — staged, not yet integrated.

---

## 1. Bloomberg — monolith CSVs (`data/bloomberg/`, tracked on `origin/main`)

Universe ≈ 503–511 current S&P 500 names. "Date field" names the column the range is read
from (event tables key on ex/announce/as-of dates, not a daily `date`).

| File name | Type / title | Date range (verified) | Rows | Names | Date field |
|---|---|---|---|---|---|
| `sp500_ohlcv.csv` | Daily equity **OHLCV** (split-adj) | 2018-01-02 → 2026-06-04 | 1,014,920 | 511 | `date` |
| `sp500_vol_iv_full.csv` | Daily **implied + realized vol** (put/call IV, RV 30/60/90/260d) | 2018-01-02 → 2026-06-04 | 1,037,278 | 510 | `date` |
| `sp500_vol_dvd.csv` | Daily **vol + dividend-yield** panel | 2018-01-02 → **2026-03-20** ⚠️ | 988,837 | 503 | `date` |
| `sp500_liquidity.csv` | Daily **liquidity** (avg vol / turnover / shares out) | 2015-01-02 → 2026-06-04 | 1,388,848 | 511 | `date` |
| `sp500_historical_fundamentals.csv` | **Fundamentals** time series | 1990-01-01 → 2026-05-10 | 79,198 | 503 | `date` |
| `sp500_macro.csv` | **Macro** indicators (daily) | 1990-01-02 → 2026-06-04 | 56,180 | — | `date` |
| `sp500_sector_etfs.csv` | **Sector-ETF** OHLC (daily) | 1998-12-22 → 2026-06-05 | 66,824 | — | `date` |
| `sp500_vix_full.csv` | **VIX** full history (daily) | 2015-01-02 → 2026-06-05 | 17,274 | — | `date` |
| `vix_term_structure.csv` | **VIX term structure** (daily) | 1990-01-02 → 2026-06-04 | 9,200 | — | `date` |
| `treasury_yields.csv` | **Treasury yield** curve (daily, full tenor + sofr) | 1994-01-03 → 2026-06-05 | 8,458 | — | `date` |
| `sp500_dividends.csv` | **Dividend events** (declared/ex/record/pay + amount) | ex-date 1962-05-31 → 2027-03-12 (fwd-declared) | 50,230 | 427 | `ex_date` |
| `sp500_earnings.csv` | **Earnings** (EPS actual/est + announce date) | 1980-01-31 → 2028-01-19 (fwd-est) | 49,379 | 503 | `announcement_date` |
| `sp500_earnings_yf.csv` | Earnings (yfinance backfill) | 2008-01-17 → 2026-08-24 | 12,242 | 498 | `announcement_date` |
| `sp500_corporate_actions.csv` | **Corporate actions** (splits/M&A/rights) | announce 1962-05-23 → 2026-06-05 | 52,442 | 481 | `announcement_date` |
| `sp500_index_membership.csv` | **Index membership / weights** | as-of 1990-04-01 → 2026-04-01 | 72,696 | — | `as_of_date` |
| `sp500_analyst.csv` | **Analyst** ratings / targets | point-in-time snapshot | 503 | 503 | — |
| `sp500_credit_risk.csv` | **Credit risk** (Altman-Z, S&P rating) | snapshot | 503 | 503 | — |
| `sp500_fundamentals.csv` | **Fundamentals** snapshot (GICS, PE, beta, dvd yld…) | snapshot | 503 | 503 | — |
| `sp500_fundamentals_yf.csv` | Fundamentals snapshot (yfinance) | snapshot | 503 | 503 | — |
| `sp500_institutional.csv` | **Institutional / float** snapshot | snapshot | 503 | 503 | — |
| `sp500_iv_snapshot_today.csv` | Single-day **IV snapshot** (30/60d ATM) | snapshot | 503 | 503 | — |
| `sp500_iv_history.csv` | (legacy) | **EMPTY** (20 bytes) — superseded by `sp500_vol_iv_full.csv` | 0 | 0 | — |

> **Laggard flag:** `sp500_vol_dvd.csv` alone still ends **2026-03-20** — every other daily
> panel was refreshed to 06-04/06-05. The broad-pull currency refresh does **not** include a
> `vol_dvd` tail (manifest: `vol_iv` ATM refresh "N/A — current via skew-surface 100%MNY col"),
> so this remains the one un-refreshed daily monolith. Wiring/refresh consumers should treat it
> as the stale series.

---

## 2. Bloomberg — deep-history archive (`data/bloomberg/deep/`, gitignored, local-only)

Restored from git branch `deep-history/bloomberg-raw` — the set Google Drive
`swe-deep-history/` partially mirrors. Gzipped CSV. **Dated slices** = current S&P names,
split-adj; **`__delisted`** slices = survivorship-complete (~1,000+ tickers incl. dead
names, back to 1990). Byte-confirmed 2026-06-22 (unchanged from the prior pass).

| File name | Type / title | Date range (verified) | Rows | Names |
|---|---|---|---|---|
| `sp500_ohlcv__1994_2018.csv.gz` | Deep **OHLCV** | 1994-01-03 → 2017-12-29 | 2,083,270 | 449 |
| `sp500_ohlcv__delisted.csv.gz` | OHLCV incl. **delisted** | 1990-01-02 → 2026-06-05 | 2,383,622 | 1,015 |
| `sp500_vol_iv_full__1994_2012.csv.gz` | Deep **vol/IV** (byte-identical to Drive copy, 28,297,508 B) | 1994-01-03 → 2012-06-29 | 1,661,191 | 436 |
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

> The on-disk `sp500_iv_surface__*` deep slices (2005→2026) are the historical companion to
> the **staged fresh** 5×5 surface in §6 (`iv_surface/sp500_iv_surface.csv.gz`, 2010→06-17).

---

## 3. Theta — option/market data (`data_processed/theta/`, gitignored, local-only)

The unit is a directory of parquet shards; counts are byte-true as-of **2026-06-22**. All
raw (unadjusted). `option_history` is a **LIVE pull and still growing**.

| File / path | Type / title | Date coverage (verified) | Rows | Names |
|---|---|---|---|---|
| `option_history/` | **Full-depth EOD option chains** — all strikes, C+P, OI; no greeks/IV. **LIVE, growing.** | expirations 2016-01-08 → 2026-08-21; obs 2016-01-04 → 2026-06-17 | **390,119,692** (71,027 files) | 154 |
| `option_history_banded_backup_2026-06-01/` | EOD option chains, **Δ-banded** strikes + OI (static backup) | expirations 2016-01-15 → 2026-05-22; obs 2017-06-23 → 2026-05-22 (sampled) | 66,574,386 (51,729 files) | 503 |
| `chains/` | Full-chain **snapshots** w/ greeks+IV+quotes+OI | snapshots 2026-04-23 → 2026-06-05 | 116,214 (1,521 files) | 495 |
| `iv_surface/` | Per-name **IV-surface snapshots** (strike×right×δ×iv×mid×dte) | snapshots 2026-04-23 → 2026-06-01 | 364,192 (558 files) | 502 |
| `iv_surface_history/` | **IV-surface daily time series** (pilot) | 2026-04-13 → 2026-06-03 | 53,725 (108 files) | 4 (A, AAPL, ABBV, ABNB) |
| `iv_history/` | **ATM-IV daily time series** (`iv_atm`) | 2015-01-02 → 2026-03-20 | 1,291,775 | 497 |
| `index_options_chains/` | Index-option full-chain snapshots | snapshots 2026-04-23 → 2026-06-01 | 8,508 (21 files) | 8 indices* |
| `index_options_surfaces/` | Index-option IV surfaces | snapshots 2026-04-23 → 2026-06-01 | 66,230 (21 files) | 8 indices* |
| `option_ohlc/` | **Per-contract daily OHLC + bid/ask** (near expirations) | expirations 2026-06-18 → 2026-07-10 | 46,886 (1,507 files) | 502 |
| `stocks_eod/` | **Equity EOD OHLCV** (underlying) | 2024-04-23 → 2026-03-20 | 233,912 | 493 |
| `vix_family/vix_family.parquet` | **VIX term-structure** index OHLC | 2023-04-24 → 2026-04-22 | 3,030 | VIX, VIX3M, VIX6M, VIX9D |

\* Index universe (8): SPX, SPXW, NDX, RUT, DJX, VIX, XSP, AAPL. `iv_surface` ETF add-ons (8):
SPY, QQQ, DIA, IWM, XLE, XLF, XLK, XLV.

> **`option_history` grew sharply** since the 2026-06-08 pass (≈185M rows / 70 names →
> **390M rows / 154 names**) — the live puller has been running. `iv_history` is static at the
> prior numbers (still ends 2026-03-20). `corporate_actions/` and `index_reference/` hold no
> data (empty / manifest-only). **No Theta API was hit to produce this — only local parquet
> footers were read.**

---

## 4. Derived / other stores (`data_processed/`, gitignored)

> **Not re-verified this pass** — `scripts/inventory_data.py` does not scan these; carried
> forward from the 2026-06-08 pass. Treat counts as last-known, not 2026-06-22-fresh.

| File / path | Type / title | Date range (last known) | Rows | Git |
|---|---|---|---|---|
| `data_processed/vol_indices.parquet` | **Vol-index** long series (VIX/VVIX/SKEW/MOVE/GVZ/OVX/VXN/VIX3M/6M/9D) | 2011-05-31 → 2026-05-22 | 35,062 | ignored |
| `data_processed/vol_indices_wide.parquet` | Same, wide (per-index close cols) | 2011-05-31 → 2026-05-22 | 3,783 | ignored |
| `data_processed/trade_universe/2025-11-22_trade_universe.csv` | Ranked **trade universe** snapshot | 2025-11-22 | 1,066 | ignored |
| `data_processed/ibkr/wheel_ledger.json` (+ portfolio/ev_calibration files) | Real **IBKR portfolio** state (acct U17853958) | live account | — | ignored |
| `data_raw/sp500_constituents_current.csv` | Current S&P constituents | snapshot | — | tracked |

> Note: `data_processed/_inventory_scan.json` is (re)written by `scripts/inventory_data.py`
> on each run (gitignored).

---

## 5. Google Drive `swe-deep-history/` — status (not re-checked this pass, remote)

Folder contains a **partial** mirror (2 of 12 planned deep-history files): `README.txt`,
`MANIFEST.txt`, `sp500_vol_iv_full__1994_2012.csv.gz` (28,297,508 B, byte-identical to the
on-disk deep slice), and `sp500_vol_iv_full__1994_2026_FULL.csv.gz` (58,316,365 B — a
convenience concat, **not** in git, **not** byte-reproducible from the current monolith).
The complete 13-file deep set in §2 is the superset (restored from
`deep-history/bloomberg-raw`). _Remote not accessed during this regeneration._

---

## 6. Staged broad-pull data (branch `claude/bloomberg-broad-pull-2026-06-17`, PENDING INTEGRATION)

The **31 staged files (~25 logical datasets)** pulled in the 2026-06-17/18 broad Bloomberg
session. **Committed on the broad-pull branch only (held, not on `main`)**, under `staging/`. Byte-scanned 2026-06-22
(rows/ranges/columns from the actual CSV/gz bytes; cross-checked against
`staging/BROAD_PULL_MANIFEST.md`). This is the input map for `docs/WIRING_CAMPAIGN.md`.

### 6A. Currency refresh — `staging/currency_refresh/` (frontier 06-05 → 06-18)

| File | Columns | Range | Rows | Names |
|---|---|---|---|---|
| `sp500_ohlcv__2026-06-05_2026-06-18.csv` | date,ticker,open,high,low,close,volume (rotated; KLAC 10:1 seam flagged) | 2026-06-05→06-18 | 5,075 | 508 |
| `sp500_liquidity__2026-06-05_2026-06-18.csv` | date,avg_vol_30d,turnover,shares_out,ticker | 2026-06-05→06-18 | 5,080 | 508 |
| `treasury_yields__2026-06-06_2026-06-18.csv` | date,rate_1m…rate_30y,sofr | 2026-06-08→06-18 | 9 | — |
| `vix_term_structure__2026-06-05_2026-06-18.csv` | date,vix,vix_3m,vix_6m | 2026-06-05→06-18 | 10 | — |

### 6B. Vol / rates / macro — `staging/macro_vol/` + `staging/macro_rates/`

| File | Columns | Range | Rows |
|---|---|---|---|
| `macro_vol/sp500_vol_indices.csv` | vix,vvix,skew,vxn,rvx,ovx,gvz,move,vxeem,cvix | 2004-01-01→2026-06-17 | 5,847 |
| `macro_vol/spx_correlation.csv` | cor1m,cor3m,cor6m | 2006-01-03→2026-06-17 | 5,146 |
| `macro_vol/credit_spreads.csv` | ig_oas,hy_oas | 2004-01-02→2026-06-16 | 5,647 |
| `macro_vol/vix_futures_curve.csv` | ux1…ux7 | 2006-01-03→2026-06-18 | 5,150 |
| `macro_rates/ois_sofr_curve.csv` | ois_1m…ois_30y,sofr_on,sofr_1y…10y | 2001-12-04→2026-06-18 | 6,393 |
| `macro_rates/real_yields.csv` | tips_2/5/10/30y,infl_swap_2/5/10y | 2000-01-03→2026-06-18 | 6,900 |
| `macro_rates/fed_funds.csv` | fed_target,ff_fut_front | 2000-01-03→2026-06-18 | 6,850 |
| `macro_rates/macro_surprise.csv` | citi_surprise_usd,citi_surprise_g10 | 2003-01-01→2026-06-18 | 6,044 |
| `macro_rates/fx.csv` | dxy,eurusd,usdjpy,gbpusd | 2000-01-03→2026-06-18 | 6,904 |
| `macro_rates/commodities.csv` | wti,gold,copper,natgas | 2000-01-04→2026-06-18 | 6,652 |
| `macro_rates/global_vol.csv` | vstoxx,vhsi,vnky,vkospi,cdx_ig_5y,cdx_hy_5y | 2000-01-03→2026-06-18 | 6,880 |
| `macro_rates/sector_factor_etfs_ohlcv.csv` | date,open,high,low,close,volume,etf (15 ETFs) | 1998-01-02→2026-06-18 | 94,646 |

### 6C. Skew surface + macro calendar

| File | Columns | Range | Rows | Names |
|---|---|---|---|---|
| `iv_surface/sp500_iv_surface.csv.gz` | **5 tenor × 5 mny** `iv_{30,60,90,180,365}d_{90,95,100,105,110}` | 2010-01-04→2026-06-17 | **1,944,699** | 509 |
| `macro_calendar/sp500_macro_calendar.csv` | event,ticker,name,country,release_datetime/date/time (11 events) | sched 2025-01-02→2027-12-08 | 352 | 11 |
| `macro_calendar/sp500_macro_releases.csv` | event,ticker,date,actual (11 events) | 2015-01-01→2026-06-17 | 4,724 | 11 |

### 6D. Per-name panels — `staging/per_name/` + `staging/dividend_pit/` + `staging/short_interest/`

| File | Columns | Range | Rows | Names |
|---|---|---|---|---|
| `per_name/returns_micro.csv` | tot_return,px_bid,px_ask | 2010-01-04→2026-06-18 | 1,874,882 | 511 |
| `per_name/vol_term_rv.csv.gz` | atm_iv_{30,60,90,180,365,730}d, rv_{10,20,30,60,90,120,180,260}d | 2010-01-04→2026-06-18 | 1,963,364 | 510 |
| `per_name/options_sentiment.csv` | pc_oi_ratio,pc_vol_ratio,oi_call,oi_put,news_sent (**102.3 MB**) | 2010-01-01→2026-06-18 | 1,998,083 | 511 |
| `per_name/beta_shares.csv` | beta_raw,shares_out | 2010-01-29→2026-05-29 (M) | 93,605 | 510 |
| `per_name/fundamentals_q.csv` | revenue,oper_inc,net_income,ebitda,eps,tot_asset,tot_liab,fcf,cfo,roe,nd_to_ebitda,gross_margin | 2010-01-01→2026-05-31 (Q) | 31,479 | 511 |
| `per_name/fundamentals_ext_q.csv` | roic,oper/net/ebitda_margin,debt_to_equity,int_coverage,dvd_payout,sales_growth,trail_fcf | 2010-01-01→2026-05-31 (Q) | 31,470 | 511 |
| `per_name/estimates_m.csv` | best_eps,best_sales,best_ebitda,best_target,best_pe,best_rating,analyst_count | 2010-01-29→2026-05-29 (M) | 92,680 | 511 |
| `per_name/estimates_fwd.csv` | best_{ebitda,eps,sales}_{1bf,2bf} | 2010-01-29→2026-05-29 (M) | 93,169 | 511 |
| `per_name/valuation_m.csv` | px_to_book,ev_to_ebitda,px_to_sales,pe,peg | 2010-01-29→2026-05-29 (M) | 89,079 | 509 |
| `per_name/sp500_snapshot_bdp.csv` | rtg_sp/moody/fitch,gics_sector/ind_grp/industry/sub_ind,inst_pct,free_float_pct,float_shares,next_earnings_dt | as-of 2026-06-18 | 511 | 511 |
| `dividend_pit/sp500_dividend_yield_pit.csv` | dvd_yld_12m,dvd_yld_ind,dvd_sh_12m (**DATED**) | 2010-01-29→2026-05-29 (M) | 72,461 | 421 |
| `short_interest/sp500_short_interest.csv` | short_interest,short_int_ratio (biweekly) | 2015-01-15→2026-05-29 | 134,035 | 509 |

### 6E. Coverage / entitlement caveats (from `BROAD_PULL_MANIFEST.md`)

- **Storage:** `iv_surface` (96.8 MB) and `vol_term_rv` (58.7 MB) committed **gzipped** (raw CSVs exceed GitHub's 100 MB limit; round IV to 2 dp). Loaders must read `.gz`. *(Byte sizes here and in §6 are decimal MB = 10⁶ B, not MiB.)*
- **Manifest vs bytes:** the manifest reports `credit_spreads` ending `2026-06-17`; the actual staged bytes end **2026-06-16** (one trading day earlier) — the §6B value is byte-true; do not "correct" it to the manifest.
- **Winsorization flags:** `options_sentiment` `pc_vol`/`news_sent` and several per-name level series carry outliers flagged for winsorization — clamp at load.
- **Entitlement-blocked (manifest bucket F, all-NaN — NOT pulled):** short-interest `pct_of_float` + borrow rate; `CDS_SPREAD_*`; rating `WATCH`/`OUTLOOK`; ESG scores; per-strike OI/greeks (use-Theta); `NFCI`; long IV tenors `7/14d` & DAY-named; `BEST_PERIOD_END_DT`. Substitutes used where noted (e.g. `SHORT_INTEREST`+`SHORT_INT_RATIO` for SI; `VOLATILITY_nD` for `nDAY_HV`).
- **PIT shape:** per-name fundamentals/estimates are **period-end dated** (filing-lag PIT not captured); `snapshot_bdp` is a **single as-of (2026-06-18)** (ratings/GICS/ownership are current values, not history).
- **Skew surface:** only moneyness `{90,95,100,105,110}` populate (wings `{80,120}` empty); the `100%MNY` column **is** current ATM IV.

---

_Regenerate: against an `origin/main` checkout, `.venv/Scripts/python.exe scripts/inventory_data.py`
→ `data_processed/_inventory_scan.json` (monoliths + deep + Theta). For the staged §6, byte-scan
`staging/` on the broad-pull branch separately — `inventory_data.py` does not cover it._
