---
id: bloomberg-deep-history-2026-06-04
title: Bloomberg OHLCV rotation fix + make-current + deep-history backfill (2026-06-04)
kind: docs
status: in-flight
terminal: lab
pr:
decisions: []
date: 2026-06-04
headline: Fixed the OHLCV column-rotation defect, made all Tier-1 + context data current to 2026-06-04, deepened all single-series context to inception, generalized the pullers to contiguous backfill, COMPLETED the vol_iv deep-history backfill to the 1994 floor (captures the 2000-2002 and 2008 crises), and routed deep data to a gz buffer branch (rclone/Drive staged) to keep the connector monoliths <100 MB.
surface: [scripts/_bbg_panel.py, scripts/pull_ohlcv.py, scripts/pull_liquidity.py, scripts/pull_vol_iv.py, scripts/pull_vix_term_structure.py, scripts/pull_context_index.py, data/bloomberg/]
---

## Goal

Execute the deep-history + make-current pull from the transient lab terminal:
FIX the OHLCV rotation defect, make every Tier-1/context series current to
2026-06-04, deepen cheap context to inception, and backfill the expensive
per-name panels (OHLCV/IV/liquidity) toward the 1994 IV floor, newest-first and
contiguous, committing each chunk. Data only — **no engine/connector edits**.

## Environment

`C:\Anaconda3\python.exe` (3.13.9) venv at `C:\Users\mertmert\bbg-venv`;
`blpapi 3.26.4.2`, `xbbg 1.2.6`, `pandas 3.0.3`; `PYTHONUTF8=1`. API entitled.
git identity `Mert Yakar <mertyakar.my@gmail.com>`.

## STORAGE ARCHITECTURE (operator decision — important)

GitHub's HARD file limit is **100 MB** (the earlier 50 MB notes were soft
warnings). The deep per-name backfill pushes the monoliths past 100 MB, so:

- **Refresh branch `data/bloomberg-refresh-2026-06-02`** keeps the
  connector-read MONOLITHS (`sp500_ohlcv.csv`, `sp500_liquidity.csv`,
  `sp500_vol_iv_full.csv`) **FROZEN at their <100 MB pushed floors**. As of
  2026-06-05 all three are aligned at the **2018-01-02** floor (vol_iv was sharded
  from the 2012 floor / 94.5 MB down to **2018+ / 59.3 MB** — see the 2026-06-05
  vol_iv shard section). The engine baseline stays clean & pushable.
- **Deep history goes to SEPARATE gz files** under `data/bloomberg/deep/`, pushed
  to a transient **buffer branch `deep-history/bloomberg-raw`** (NEVER merged).
  gz keeps them well under 100 MB (the 2007–2012 vol_iv slice is 8.7 MB gz).
- NOT Git LFS (quota/cost). The permanent home is **Google Drive**, set up
  later from the laptop (see DEFER).

## What was done

### Phase 1 — OHLCV rotation defect FIXED (commit 8fdf612, pushed)
Connector (`engine/data_connector.py:202-208`) renames `open->high, high->close,
close->open`; the stored CSV must be ROTATED (stored `open` = true HIGH/row-max,
stored `low` = true LOW). The prior 2026-03-23+ delta was CANONICAL → inverted.
Fix: restored the clean rotated base from origin/main (->2026-03-20), re-pulled
2026-03-23->2026-06-04 with field map `{PX_HIGH->open, PX_LAST->high,
PX_LOW->low, PX_OPEN->close, PX_VOLUME->volume}`. Post-build GATE: **0 violations
across 1,014,916 non-NaN rows**. Verified offline (round-trip) + live.

### Phase 2 — make-current + cheap context to inception (all pushed)
- **2a Tier-1 current (8fdf612):** ohlcv re-pulled to 06-04; liquidity + vol_iv
  to 06-04.
- **2b vix_term_structure (e386a51):** 2018->03-20 → **1990-01-02 -> 2026-06-04**
  (9,200 rows; vix_3m 2002, vix_6m 2008).
- **2d context indices to inception (ccd821d):**
  - `vix_futures_curve` 20,111 -> **36,779** (UX1-7 from 2004-03-26)
  - `vol_indices` 22,968 -> **43,244** (SKEW 1990 … VXEEM/VIX9D 2011)
  - `spx_correlation` 8,605 -> **15,396** (COR1M/3M/6M from 2006)
  - `rates_fx_vol` 8,716 -> **24,842** (MOVE 1988 / JPMVXYG7 1992 / CVIX 2001)

### Phase 3a — per-name deep backfill (begun; newest-first; floor 1994)
- **window 1 (8dc6f59, pushed in the refresh monolith):** vol_iv 2012-07 ->
  2015-01 (+280,302). This is the frozen floor of the refresh-branch monolith
  (vol_iv now 2012-07-02 -> 2026-06-04, 1,668,021 rows, 94.5 MB).
- **windows 2-8 (gz buffer branch `deep-history/bloomberg-raw` @ 61c2183):**
  vol_iv backfilled 2012-06 all the way to the **1994 IV floor — COMPLETE**.
  `data/bloomberg/deep/sp500_vol_iv_full__1994_2012.csv.gz` = 1,661,191 rows,
  **1994-01-03 -> 2012-06-29**, 436 tickers, 27.0 MB gz. **Both the 2000-2002
  dot-com bear (+9/11) and the 2008 GFC IV are captured.** With the frozen
  monolith the vol_iv panel is contiguous 1994 -> 2026. Full uncompressed scratch
  (1994-2026, 3,329,212 rows) also at `C:\Users\mertmert\deep_scratch\` on the box.
  Pulled with **zero throttling** (~14M data points across the session).

## 2026-06-05 — Phase 3b: OHLCV + liquidity deep backfill COMPLETE (to 1994)

Resumed on a fresh lab terminal (cloned the repo, recreated `bbg-venv`
[blpapi 3.26.4.2 / xbbg 1.2.6 / pandas 3.0.3], live API gate). Completed the two
remaining per-name panels to the 1994 floor, newest-first, with the SAME proven
machinery (`scripts/pull_ohlcv.py` / `pull_liquidity.py` + `_bbg_panel.py`,
`SWE_PULL_MODE=backfill` `SWE_BACKFILL_MAX_WINDOWS=1`). Each window grew an
**off-monolith scratch** (`C:\Users\mertmert\deep_scratch\`); the deep slice
(date < monolith floor) was carved → gz → pushed to `deep-history/bloomberg-raw`
after every chunk (durable on a transient box). Monoliths stayed FROZEN (verified
unmodified on the refresh branch). Commits went through a dedicated **git worktree
pinned to the buffer branch** instead of checkout-switching the main tree — same
result, no 850-file churn, scripts always present for the next pull.

- **OHLCV** `data/bloomberg/deep/sp500_ohlcv__1994_2018.csv.gz` —
  **1994-01-03 → 2017-12-29**, 2,083,270 rows, 449 tickers, 31.7 MB gz.
  Joins the frozen monolith (2018-01-02 → 2026-06-04) → unbroken 1994 → 2026.
  Final buffer SHA `5296515`.
- **Liquidity** `data/bloomberg/deep/sp500_liquidity__1994_2015.csv.gz` —
  **1994-01-03 → 2014-12-31**, 1,987,751 rows, 457 tickers, 20.9 MB gz.
  Joins the frozen monolith (2015-01-02 → 2026-06-04). Final buffer SHA `c814bc5`.

All three deep panels (vol_iv, OHLCV, liquidity) are now on the buffer branch and
join their frozen monoliths for contiguous **1994-01-03 → 2026-06-04** coverage.
All three gz are < 100 MB → GitHub-resident, no Drive needed for these.

## 2026-06-05 — sp500_macro refresh + deep backfill to 1990 (COMPLETE)

Cross-asset OHLC panel (NOT SPX members), so it does NOT use `_bbg_panel.py`;
new reusable `scripts/pull_macro.py` pulls six fixed instruments via `blp.bdh`
and stacks them long with the `instrument` label the connector keys on
(`data/consolidated_loader.py get_macro`). Small file → refreshed **in place** on
the refresh branch (not a frozen monolith). Schema preserved EXACTLY:
`date,open,high,low,close,instrument` (straight PX_OPEN/HIGH/LOW/LAST → open/high/
low/close; NOT rotated, unlike sp500_ohlcv).

**Per-instrument BBG map — VERIFIED** by reproducing the committed 2026-03-19/20
OHLC exactly (overlap match, all six) before any write:

| instrument | BBG ticker | note |
|---|---|---|
| us_10y | `USGG10YR Index` | 10Y CMT yield OHLC |
| us_2y | `USGG2YR Index` | 2Y CMT yield OHLC |
| spx | `SPX Index` | S&P 500 price |
| dxy | `DXY Curncy` | US dollar index |
| gold | `XAU Curncy` | spot gold (GC1 futures differ — rejected) |
| wti_oil | `CL1 Comdty` | front WTI future (USCRWTIC spot is flat O=H=L=C — rejected) |

Result: **17,320 → 56,180 rows**; every instrument now **1990-01-02 → 2026-06-04**
(was 2015-01-01 → 2026-03-20, stale). Refresh is a clean fresh-authoritative
rebuild (fresh reproduces the committed overlap identically; merged keep-last so
nothing dropped). Verified: header byte-identical, no dup `(date,instrument)`,
overlap values intact, forward gap filled (+52–54 rows/instrument).

**OHLC completeness (honest — Bloomberg deep-history availability, not a defect):**
`close` (PX_LAST) is ~100% complete for all six back to 1990. Intraday O/H/L is
close-only in the deep tail for two CMT/spot series:
- `gold` — full OHLC from **1992-08-03** (6.9% open-null overall; pre-1992-08 close-only)
- `us_2y` — full OHLC from **1998-10-20** (24.1% open-null; pre-1998-10 close-only)
- dxy/spx/us_10y/wti_oil — full OHLC from 1990-01-02 (≤32 stray open-nulls each).

IG/HY OAS + financial-conditions proxy (the optional "if entitled" extras) NOT
added — out of the verified six-instrument scope; defer unless requested.

### OHLCV bad-tick policy (operator-approved 2026-06-05)
The rotation gate (`open == row-max` AND `low == row-min`) tripped on **97 rows**
across the 1994–2010 windows — genuine bad Bloomberg prints on illiquid names
(true open/close a few cents OUTSIDE the reported high/low), **not** a rotation
defect: the field map is identical to the clean 2010–2018 windows, which passed
100%, and the per-window failure rate stayed 0.000–0.014% (never trended up as
the universe thinned). Decision: **drop only the gate-failing rows**, audit each
to `data/bloomberg/deep/ohlcv_dropped_ticks.csv` (ticker, date, OHLCV, which
check failed, window), under a **0.5%-per-window safety valve** (exceed → STOP as
a systematic signal). Breakdown: 51 `open!=max`, 46 `low!=min`. The carved gz
independently RE-GATES clean (0 violations). Liquidity has no OHLC invariant →
no gate, no drops.

## 2026-06-05 — Phase 4: IV moneyness/skew surface COMPLETE (2005→2026)

New per-name **implied-vol moneyness/skew surface** — raw material for put-side
strike selection and skew/term-structure signals. New reusable
`scripts/pull_iv_surface.py` (built on `_bbg_panel.py`, same machinery as
`pull_vol_iv.py`).

**Field tokens VERIFIED on AAPL first (the worklog spec was partly wrong):**
- Tenor `90DAY`/`120DAY`/`180DAY`/`360DAY` tokens are **INVALID** (return empty).
  Bloomberg uses `30DAY`, `60DAY`, then **`3MTH`(=90d), `6MTH`(=180d),
  `12MTH`(=360d)**. Operator chose the **extended 5-tenor** set.
- Moneyness is **fixed at 90/95/100/105/110** (80/85/115/120 unavailable).
- Surface floor is **~2005-01-03** (empty before then; NOT the 1994 IV floor).
- Skew shape sane (put wing richest; term structure rises).

**Layout/storage (operator decision):** WIDE — `date, iv_{30d,60d,90d,180d,360d}_
{90,95,100,105,110}` (25 cols), `ticker` ("AAPL UW"), `bdh Fill=P`. Grown in an
off-monolith scratch (`SWE_OUT_PATH`); 1.4-yr seed then 30-month backfill windows
to floor, writing after each (cap-safe). The single wide gz came to **174.8 MB
(> the 100 MB git blob limit)**, so it was **partitioned by era into 3 gz**, each
< 100 MB, pushed to the buffer branch `deep-history/bloomberg-raw` (matches the
eventual year-partitioned-Parquet plan; avoids the blocked Drive):

| file | rows | span | gz |
|---|---|---|---|
| `deep/sp500_iv_surface__2005_2011.csv.gz` | 685,310 | 2005-01-03 → 2011-12-30 | 49.2 MB |
| `deep/sp500_iv_surface__2012_2018.csv.gz` | 795,760 | 2012-01-03 → 2018-12-31 | 56.6 MB |
| `deep/sp500_iv_surface__2019_2026.csv.gz` | 912,802 | 2019-01-02 → 2026-06-04 | 69.1 MB |

Total **2,393,872 rows, 501 tickers, 2005-01-03 → 2026-06-04**. Buffer SHA
`5beac7b`. Non-null: 30/60/90d ~100%, 180d 96.6%, 360d 88.6% (long tenors thinner
in deep history). Ticker count tapers back (498→383) — fewer of today's members
had a listed surface that far back. Connector read-path **deferred** (like the
other deep panels — see DEFER). `pull_iv_surface.py` committed to the refresh branch.

## 2026-06-05 — Treasury risk-free backfill to 1994 (COMPLETE)

`data/bloomberg/treasury_yields.csv` (connector risk-free source,
`get_risk_free_rate`) started 2021-05-07 (yfinance) → NaN risk-free pre-2021 →
BSM NaN → every pre-2021 backtest candidate R1a-blocked. Backfilled via new
reusable `scripts/pull_treasury_yields.py` (Bloomberg `USGG3M/6M/2YR/10YR Index`
`PX_LAST`): **1994-01-03 → 2026-06-05, 8,458 rows**, schema unchanged
(`date,rate_3m,rate_6m,rate_2y,rate_10y`). PX_LAST kept in **PERCENT, unchanged**
(D20: connector divides by 100 — a wrong scale silently 100×'s the rate).

Gate: percent-scale confirmed (rate_3m mean 5.64 in 1995, ~0.04 in 2011-15, 5.17
in 2023; every tenor max>1). **rate_2y/rate_10y match `sp500_macro` us_2y/us_10y
exactly** (max|diff|=0.0000, same tickers). vs the old yfinance file: rate_10y
mean diff 0.011pp, but rate_6m/rate_2y differ up to 0.80/1.20pp — the old yfinance
6m/2y were interpolated proxies; the new are true CMT yields (a correctness fix).
Commit `8ff460b` (refresh branch).

## 2026-06-05 — vol_iv monolith sharded to 2018+ (COMPLETE)

`sp500_vol_iv_full.csv` was 94.5 MB (~a month of daily refresh from GitHub's
100 MB reject). Sharded at **2018-01-02** (the OHLCV monolith floor) via a
byte-preserving raw-line split (no pandas reserialize):
- pre-2018 tail → `deep/sp500_vol_iv_full__2012_2018.csv.gz` (630,743 rows,
  2012-07-02 → 2017-12-29, 10.1 MB) on the buffer branch (`e7d7069`).
  `__1994_2012.csv.gz` left untouched.
- monolith trimmed to **2018-01-02 → 2026-06-04, 1,037,278 rows, 59.3 MB**
  (`ec50c5e`, refresh branch). git diff = 630,743 deletions / 0 insertions, and
  retained 2018+ rows md5-verified byte-identical to the prior monolith.

Gate: row conservation exact (630,743 + 1,037,278 == 1,668,021 original);
identical schema across all three pieces; seams adjacent (1994_2012 ends 2012-06-29
→ 2012_2018 starts 2012-07-02; 2012_2018 ends 2017-12-29 → monolith starts
2018-01-02); zero date overlap monolith↔deep. No backtestable data lost (connector
reads only the monolith and OHLCV is already 2018+). All three monoliths now align
at the 2018 floor.

## 2026-06-05 — Core dividends refresh + archival completeness sweep (COMPLETE)

Six data-layer datasets refreshed/backfilled. Each got a new or rewritten reusable
puller, was smoke-verified against committed values BEFORE the full pull, and gated
on bytes before push. All are small monoliths on the refresh branch.

| dataset | rows was→now | range | puller / commit |
|---|---|---|---|
| sp500_dividends | 50,230 → 52,064 | ex 1962-05-31 → 2027-03-12; declared→2026-06-05; 158 fwd ex-dates (was 10); 481 payers | `pull_dividends.py` (new) · `2bbe0c2` |
| sp500_short_interest | 38,099 → 172,565 | 1990-01-31 → 2026-05-29, monthly | `pull_short_interest.py` (rewrite) · `94466bb` |
| sp500_sector_etfs | 29,954 → 66,824 | 1998-12-22 → 2026-06-05, 11 ETFs | `pull_sector_etfs.py` (new) · `8e50a1a` |
| sp500_historical_fundamentals | 30,347 → 79,198 | 1990-01-01 → 2026-05-10, quarterly | `pull_historical_fundamentals.py` (rewrite) · `6494eac` |
| sp500_corporate_actions | 17,717 → 52,442 | eff 1962-05-31 → 2027-03-12 | `pull_corporate_actions.py` (new) · `4ea2537` |
| sp500_vix_full | 16,955 → 17,274 | 2015-01-02 → 2026-06-05, 6 instruments | `pull_vix_full.py` (new) · `8e6332a` |

Notes / gotchas for the next session:
- **dividends & corp_actions both come from `blp.bds(EQY_DVD_HIST_ALL)`.** corp_actions
  was VERIFIED to be a 2015+ window of exactly this field (99.6% of rows match a
  dividend event, 100% equal values, announcement_date==declared_date). Routing:
  `ratio` for Stock Split / Stock Dividend / Split-Off, `amount` otherwise. dividends
  dedupes (ticker,ex_date) keep-last; corp_actions keeps same-ex_date duplicates
  (exact-row dedup only) → +378 rows.
- **short_interest & historical_fundamentals: the old `df.stack(level=0)` pullers were
  BROKEN on xbbg 1.2.6** (long narwhals frames) → rewritten on the working reshape
  pattern. Field maps reproduce committed values exactly (A UN 2020-01 short interest
  = 5091993 / 1.65 / 310.18342 / 99.488877). short_interest `borrow_rate_net` stays
  empty (not entitled); `float_pct` is DERIVED `EQY_FLOAT/EQY_SH_OUT*100` (>100 in
  ~0.05% of rows is the pre-existing as-of-mismatch artifact — old file had 0.03%).
- **VIX futures yellow-key is `UX1 Index` / `UX2 Index`, NOT `UX*Comdty`** (Comdty
  returns empty). vix_full kept at its 2015 floor (deep VIX history is in
  vix_term_structure → 1990); it is redundant context, refreshed for completeness.
- treasury_yields was backfilled to 1994 earlier this session (`8ff460b`); its
  rate_2y/rate_10y == sp500_macro us_2y/us_10y exactly.

## 2026-06-05 — Forward earnings estimates (new, event-gate enrichment)

New ADDITIVE snapshot `data/bloomberg/sp500_earnings_estimates.csv` (503 rows) +
`scripts/pull_earnings_estimates.py`. Closes the highest clean gap from the data-gap
audit: `event_gate.py` (largest tail-loss guard) keyed off historical realized
earnings with no forward date and a single point estimate. Per current SPX member,
point-in-time `blp.bdp`:
`ticker, expected_report_date (EXPECTED_REPORT_DT), last_announcement_date
(ANNOUNCEMENT_DT), best_eps[/median/high/low/numest], best_sales[/median/high/low/numest]`.
Does NOT modify the engine-consumed `sp500_earnings.csv`.

Field mnemonics verified on AAPL/JPM first; **`BEST_EPS_SD`/`_STD_DEV` are NOT
populated at this tier** → dispersion captured via hi/lo/median/#estimates (range +
analyst count) instead of stddev. Also: this xbbg backend (`xbbg_async`) drops the
whole bdp response when invalid mnemonics are mixed in — probe fields one-at-a-time.

Gate: expected_report_date 503/503 forward (2026-06-08 → 2026-10-21); EPS/sales
consensus+dispersion 501/503 (2 names no consensus); 0 dup; 0 rows high<low; mean
within [low,high] for all; no all-NaN field. Commit on the refresh branch.

### Data-gap audit (2026-06-05) — also surfaced two CODE defects (NOT data gaps)
A read-only consumption audit found the engine silently ignores data we DO have:
1. `get_credit_risk` returns key `sp_rating` but `wheel_runner.py:482` reads
   `rtg_sp_lt_lc_issuer_credit` → S&P credit rating always `""` (dead). `sp500_credit_risk.csv` wasted.
2. R9/R10 sector cap uses a hard-coded `DEFAULT_SECTOR_MAP` (`risk_manager.py:1579`)
   instead of the pulled `gics_sector_name` → names absent from the map gate as "Unknown".
Both are one-line code fixes for a separate PR (out of the data-only mandate). Other
audit notes: `sp500_vol_dvd.csv` is stale (2026-03-20) but superseded by vol_iv_full
(deprecate, don't refresh); `sp500_iv_history.csv` is an empty stub; index_membership
`percentage_weight` is the all-zeros sentinel so the PIT `min_weight` filter is a no-op.
Still genuinely DEFERRED (channel/entitlement, not Desktop-API): option chains (Theta),
dealer GEX (BQL subscription), `borrow_rate_net` (not entitled), macro `importance` (BQL).

## 2026-06-05 — Phase 1: survivorship membership + two cost-model gaps (COMPLETE)

Data only, refresh branch. Smoke + gate each. **Phase 2 (delisted price pulls) NOT
started** — `_delisted_universe.csv` is the list to review first.

**A. Historical SPX membership** (`pull_index_membership.py`) — `sp500_index_membership.csv`
overwritten. `bds(INDX_MWEIGHT_HIST, END_DATE_OVERRIDE=YYYYMMDD)` quarterly.
**as_of floor = 1990-04-01** (verified: 1990-01-01 → 0 names, 1990-04-01 → 500;
nothing 1986-1989). 72,696 rows, 145 snapshots 1990-04-01 → 2026-04-01, **1,523
distinct names ever** (was 781), ~500/snapshot (499-506), 0 dup. `percentage_weight`
stays the all-zeros sentinel — historical index weights NOT entitled (names+as_of
are the value). Derived `data/bloomberg/_delisted_universe.csv` (NEW; drives Phase 2):
**1,016 historical members with no OHLCV, all NAME-resolved** = 926 true delistings
(root absent) + 90 relistings/code-changes of held names (e.g. WMT UN → now WMT UW).
Famous delistings correct: Lehman 2008-07, Enron 2001-10, Bear Stearns 2008-04,
Compaq 2002-04, Kodak 2010-10. Cols ticker,name,first_in_index,last_in_index,
n_snapshots. Commit `82cb381`.

**B. Underlying bid-ask** (`pull_bid_ask.py`) — `BID_ASK_SPREAD` is NOT a historical
field at this tier (bdh/bdp empty), so spread is DERIVED from `PX_BID`/`PX_ASK`
(daily, ~1990+). Written as a **sibling `sp500_bid_ask.csv`** (NOT a liquidity column)
to keep the connector monolith clean (67MB) + avoid merge misalignment. 1,332,740
rows, 503 names, 2015-01-02 → 2026-06-05, **90.3MB (<100MB)**. Schema date,ticker,
px_bid,px_ask,bid_ask_spread,bid_ask_spread_bps. Gate: 0 dup; no all-NaN; spread_bps
median 2.28 / p95 9.41 (AAPL 1.20, KO 2.01, NVDA 2.23 bps); 0 negative; 0.16%
zero-spread + rare stale-quote outliers kept faithful. **Nearing 100MB → shard like
vol_iv when it crosses.** Commit `574f543`.

**C. Fuller rate curve** (`pull_treasury_yields.py`) — added rate_1m (USGG1M, 2001+),
rate_5y (USGG5YR), rate_30y (USGG30YR), sofr (SOFRRATE Index, 2018-04+) →
`treasury_yields.csv` now 8 tenors, 1994-01-03 → 2026-06-05, 8,458 rows, logical
tenor order, PERCENT unchanged. Gate: every tenor max>1 (percent); existing
rate_3m/6m/2y/10y IDENTICAL to prior (max|diff|=0); curve monotonic up. SOFR source =
SOFRRATE Index (SOFRINDX = compounding index level, wrong; SOFR Index = BAD_SEC).
Commit `cf0cd75`.

## 2026-06-05 — Phase 2: delisted-constituent price/vol backfill COMPLETE (survivorship)

The big survivorship-completion pull. Driver `_delisted_universe.csv` (1,016 historical
SPX members with no current OHLCV), windowed per-name `[first_in_index-1Q,
last_in_index+1Q]` (cost-bounded ~5x vs full 1990-2026), PIT Bloomberg IDs (the
numeric/`Q`/`Z` codes resolve to history). New `scripts/pull_delisted.py`
(checkpoint+resume every 50 names; per-name try/except log+skip). Three deep panels on
the buffer branch (schemas match the current-universe monoliths -> assemble cleanly):

- `deep/sp500_ohlcv__delisted.csv.gz` — **2,383,622 rows, 1,015 names, 1990-01-02 →
  2026-06-05, 32.5 MB**. Rotation-fixed map (PX_HIGH→open, PX_LAST→high, PX_LOW→low,
  PX_OPEN→close). **Rotation gate 0 violations on 2,379,809 complete rows**; 51 bad
  ticks dropped+audited (`ohlcv_dropped_ticks__delisted.csv`; 0.0021%, under the 0.5%
  per-name valve).
- `deep/sp500_vol_iv__delisted.csv.gz` — 2,408,183 rows, 1,016 names, 38.2 MB. Implied
  IV 1.40M non-null (post-1994), realized vol 2.39M (incl. pre-1994 bonus).
- `deep/sp500_liquidity__delisted.csv.gz` — 2,393,425 rows, 1,011 names, 23.0 MB.
- `deep/delisted_status.csv` — per-name provenance. 0 no_data, 0 errors/skips. One name
  (Office Depot `9995522D UN`) resolved vol_iv+liquidity but not OHLCV under that PIT code.

Verification was paused after the first 50 (oldest leavers, all PIT IDs resolved,
0 rotation violations) + a 6-name recent-delisting demo (Lehman last bar 2008-09-12 @
$3.65) before the full run. Buffer SHA `e7818f4`. All three gz <100 MB → no sharding.
**Drive mirror (rclone) still BLOCKED by the exfil classifier** — buffer branch is the
durable store (operator can mirror later; command in the Google Drive section). These
delisted panels + the current-universe monoliths/deep slices give survivorship-bias-free
1990-2026 coverage.

## The machinery

`scripts/_bbg_panel.py` (NEW) — shared engine. Fills FORWARD gap [max+1 -> END]
then walks BACKWARD [min-1 -> FLOOR] **newest-first in ~30-month windows**,
rewriting after EVERY window (durable). Per-window: pull members in 30-ticker
`bdh` -> pivot long->wide -> merge+dedupe(date,ticker keep-last) -> sort -> write
-> optional validate() (the OHLCV gate). Pullers are thin configs.

Env knobs: `SWE_PULL_END`, `SWE_PULL_FLOOR`, `SWE_PULL_MODE`
(forward|backfill|both), `SWE_BACKFILL_CHUNK_MONTHS`, `SWE_BACKFILL_MAX_WINDOWS`,
`SWE_OUT_PATH` (write to a scratch path instead of the monolith — used for deep
backfill), `SWE_PULL_LIMIT`, `SWE_PULL_NO_WRITE`.

Cheap-context pullers: `scripts/pull_vix_term_structure.py`,
`scripts/pull_context_index.py <dataset>` (per-ticker pulls so one bad ticker
can't null the batch). Offline tests (`test_pullers_offline.py`, kept off-repo)
prove window tiling + the rotation round-trip/gate. All passed.

## Gotchas

- xbbg 1.2.6 → long narwhals; `.to_native()` + `pivot_table`; `PYTHONUTF8=1`.
- Historical windows use **today's** SPX members (post-IPO names absent in old
  windows; departed names keep old rows). Survivorship is inherent.
- Meter: ~3-6M points this session, **zero throttling** (one 30-month per-name
  window ≈ 1.9M points). Full 3-panel backfill to 1994 ≈ 20-40M → will hit the
  cap; pull-until-error, per-window write keeps what landed, resume.

## RESUME RUNBOOK

Terminal up + logged in; venv active; `cd` repo; `$env:PYTHONUTF8='1'`.

### Deep backfill — ALL THREE PANELS DONE to 1994 (vol_iv + ohlcv + liquidity) [2026-06-05]
vol_iv, ohlcv, AND liquidity are now fully backfilled to the 1994 floor (deep gz on
the buffer branch `deep-history/bloomberg-raw`; see the 2026-06-05 section above for
spans/SHAs + the OHLCV bad-tick audit). The loop below is retained for reference /
any future per-name panel: grow an off-monolith scratch with `SWE_OUT_PATH`, then carve
the gz and push to the buffer branch. The refresh-branch monolith is NEVER advanced.
The proven loop (shown for vol_iv; swap the script + scratch + deep filename for
ohlcv `sp500_ohlcv__1994_2018.csv.gz` / liquidity `sp500_liquidity__1994_2015.csv.gz`):
```
$env:SWE_OUT_PATH='C:\Users\mertmert\deep_scratch\sp500_vol_iv_full.csv'
$env:SWE_PULL_MODE='backfill'; $env:SWE_BACKFILL_MAX_WINDOWS='2'
python scripts/pull_vol_iv.py            # grows scratch 2004..2007, etc.
# re-carve deep slice (date < 2012-07-02) -> data/bloomberg/deep/*.csv.gz, then:
git checkout deep-history/bloomberg-raw
git add -f data/bloomberg/deep/sp500_vol_iv_full__1994_2012.csv.gz
git commit -m "deep: vol_iv backfill chunk" ; git push origin deep-history/bloomberg-raw
git checkout data/bloomberg-refresh-2026-06-02
```
For OHLCV/liquidity: first SEED the scratch from the frozen monolith
(`copy data\bloomberg\sp500_ohlcv.csv <scratch>`), then same loop with
`SWE_OUT_PATH=<scratch>` and a deep filename `sp500_ohlcv__1994_2018.csv.gz` /
`sp500_liquidity__1994_2015.csv.gz`. (OHLCV scratch keeps the rotated map; its
gate still applies.)

### Not done this session (lower urgency / bespoke)
- **Phase 2c snapshots** — **DONE 2026-06-05.** Refreshed all five point-in-time
  `bdp` snapshots (`sp500_fundamentals`, `credit_risk`, `institutional`,
  `analyst`, `iv_snapshot_today`) to today (503 members) and codified the verified
  field lists + per-file schema/ticker-format/column-order in the new reusable
  `scripts/pull_snapshots.py` (`SWE_SNAP_SMOKE` / `SWE_SNAP_ONLY` knobs;
  3-ticker smoke before the full run). Committed to the refresh branch.
- **sp500_macro:** **DONE 2026-06-05.** Refreshed + backfilled all six instruments
  to **1990-01-02 → 2026-06-04** (56,180 rows) via the new `scripts/pull_macro.py`;
  per-instrument BBG map verified against the committed overlap. See the
  2026-06-05 macro section above for the map + OHLC-completeness notes. (IG/HY OAS +
  financial-conditions proxy not added — defer unless requested.)
- **Phase 4 IV moneyness/skew surface:** **DONE 2026-06-05.** 25-pt wide surface
  (5 tenors × 5 moneyness), 501 names, 2005-01-03 → 2026-06-04 (2,393,872 rows),
  3 era-partitioned gz on the buffer branch (SHA `5beac7b`) via the new
  `scripts/pull_iv_surface.py`. See the 2026-06-05 Phase 4 section above (note the
  worklog's `90DAY` field token was invalid → `3MTH`; floor is 2005 not 1994).

## Google Drive (rclone) — STAGED; upload pending operator permission
rclone v1.74.2 at `C:\Users\mertmert\rclone\rclone-v1.74.2-windows-amd64\rclone.exe`.
Remote `gdrive` authorized via browser OAuth (token in `%APPDATA%\rclone\rclone.conf`);
`rclone about gdrive:` shows **4.79 TiB free**; `gdrive:swe-deep-history/` created and
the t.txt round-trip verified. The deep-DATA upload is BLOCKED by Claude Code's
auto-mode classifier (data-exfiltration hard block) and the agent cannot self-grant
the permission. **Nothing is lost** — the deep gz is on the buffer branch. To finish
the Drive copy, EITHER the operator runs it, OR adds this allow rule to
`~/.claude/settings.local.json` (then the agent runs it):
```
permissions.allow += "PowerShell(& 'C:/Users/mertmert/rclone/rclone-v1.74.2-windows-amd64/rclone.exe' *)"
```
Upload + verify (forward-slash paths to match the rule):
```
& 'C:/Users/mertmert/rclone/rclone-v1.74.2-windows-amd64/rclone.exe' copy 'C:/Users/mertmert/deep_scratch/sp500_vol_iv_full__1994_2012.csv.gz' gdrive:swe-deep-history/ -P
& 'C:/Users/mertmert/rclone/rclone-v1.74.2-windows-amd64/rclone.exe' lsl gdrive:swe-deep-history/
```
Confirm bytes on Drive before deleting any local copy (the gz is also on the buffer branch).

## DEFER to laptop/overnight (NOT the metered terminal)
Move the deep gz files from `deep-history/bloomberg-raw` into **Google Drive**
(permanent home); convert to Parquet partitioned by year; add the connector
read-path that assembles recent(git monolith) + deep(Drive); then delete the gz
buffer branch. That connector wiring is the real fix and is out of scope here.
Also deferred (wrong channel): option chains (Theta), dealer GEX (subscription),
`borrow_rate_net` + macro `importance`/deep-history (BQL/Excel). (`treasury_yields`
is no longer deferred — backfilled to 1994 via Bloomberg on 2026-06-05; see above.)
