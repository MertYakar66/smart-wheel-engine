# Data-Layer Activation Roadmap — survivorship-free 1990–2026 → live engine

> **Status: PLAN (2026-06-05).** This is the deliverable of STEP 2 of the
> data-layer activation task. The *safe prep* (verified inventory below, the
> credit-rating code fix, and the companion design doc) is done on this branch;
> the **consequential steps — the data merge, the snapshot re-baseline, and the
> connector deep-read itself — are NOT executed here.** They are scoped below
> and await operator review. Nothing in `engine/ev_engine.py`,
> `engine/wheel_runner.py` (beyond the one diagnostic fix in R0), or
> `engine/candidate_dossier.py` is changed by the executed work.

## 0. The one-paragraph problem

The multi-day Bloomberg campaign produced a **survivorship-bias-free 1990→2026
data layer** — deep per-name history to the 1994 IV floor, 1,015 delisted
constituents (Lehman, Enron, WaMu, Bear Stearns…), a 1990→2026 PIT membership
panel, a full rate curve, a 2005→2026 IV moneyness/skew surface, and refreshed
context. **None of it is live.** `engine/data_connector.py._FILES` reads only
the three *recent monoliths* (`sp500_ohlcv.csv` 2018+, `sp500_vol_iv_full.csv`
2018+, `sp500_liquidity.csv` 2015+). The engine today sees nothing before
~2018 and none of the delisted names. Activating the campaign's value =
(a) merge the data, (b) teach the connector to assemble monolith + deep +
delisted, (c) make the backtest harness pick the point-in-time universe. Each
touches what the authoritative ranker sees, so each needs a re-baseline.

## 1. Verified data inventory (on the bytes)

Profiled file-by-file (chunked, memory-safe) at the **current** branch SHAs —
`data/bloomberg-refresh-2026-06-02 @ 6bb3399` and
`deep-history/bloomberg-raw @ e7818f4` (both advanced since the 2026-06-05 QA:
refresh `3a271c9→6bb3399`, deep `61c2183→e7818f4`). Harness:
`C:\tmp\swe-exp\inv\inventory.py` (read-only; `inventory_results.json`).

### 1a. Refresh branch — connector + context monoliths (`data/bloomberg/`, 33 files)

| file | rows | cols | span | tickers | dup-keys | role / note |
|---|--:|--:|---|--:|--:|---|
| `sp500_ohlcv.csv` | 1,014,920 | 7 | 2018-01-02 → 2026-06-04 | 511 | 0 | **connector OHLCV** (rotated layout) |
| `sp500_vol_iv_full.csv` | 1,037,278 | 8 | 2018-01-02 → 2026-06-04 | 510 | 0 | **connector IV** — sharded to 2018+ (was 2012) |
| `sp500_liquidity.csv` | 1,388,848 | 5 | 2015-01-02 → 2026-06-04 | 511 | 0 | **connector liquidity** |
| `treasury_yields.csv` | 8,458 | 9 | **1994-01-03** → 2026-06-05 | — | — | **connector risk-free** — 8 tenors; **QA CONCERN-1 (2021+ only) RESOLVED** |
| `sp500_fundamentals.csv` | 503 | 13 | snapshot | 503 | — | **connector fundamentals** (real `gics_sector_name`) |
| `sp500_credit_risk.csv` | 503 | 4 | snapshot | 503 | — | **connector credit** (`rtg_sp_lt_lc_issuer_credit`) |
| `sp500_earnings.csv` | 49,379 | 7 | 1980-01-31 → 2028-01-19 | 503 | 76 | **connector earnings**; 76 dup (ex)dates — flag |
| `vix_term_structure.csv` | 9,200 | 4 | 1990-01-02 → 2026-06-04 | — | — | **connector VIX** |
| `sp500_index_membership.csv` | 72,696 | 3 | **1990-04-01** → 2026-04-01 | (PIT) | — | PIT membership; `percentage_weight` ≈ `-2.4e-14` **all-zeros sentinel** (min_weight filter dead) |
| `_delisted_universe.csv` | 1,016 | 5 | — | 1,016 | — | drives the delisted backfill; `ticker,name,first_in,last_in,n_snapshots` |
| `sp500_bid_ask.csv` | 1,332,740 | 6 | 2015-01-02 → 2026-06-05 | 503 | 0 | NEW underlying bid/ask sibling; **89 MB — nearing the 100 MB git cap** |
| `sp500_macro.csv` | 56,180 | 6 | 1990-01-02 → 2026-06-04 | 6 | 0 | cross-asset OHLC (6 instruments) |
| `sp500_short_interest.csv` | 172,565 | 7 | 1990-01-31 → 2026-05-29 | 502 | 0 | monthly |
| `sp500_historical_fundamentals.csv` | 79,198 | 7 | 1990-01-01 → 2026-05-10 | 503 | 0 | quarterly |
| `sp500_corporate_actions.csv` | 52,442 | 6 | 1962-05-31 → 2027-03-12 | 481 | 378 | dup = same-ex_date keep (expected) |
| `sp500_dividends.csv` | 52,064 | 8 | 1962-05-31 → 2027-03-12 | 481 | 0 | 158 fwd ex-dates |
| `sp500_earnings_estimates.csv` | 503 | 13 | 2026-06-08 → 2026-10-21 | 503 | 0 | **NEW** forward consensus + dispersion (additive; not yet connector-read) |
| `sp500_sector_etfs.csv` | 66,824 | 7 | 1998-12-22 → 2026-06-05 | (11) | — | |
| `vix_futures_curve.csv` | 36,779 | 3 | 2004-03-26 → 2026-06-04 | 7 | 0 | |
| `vol_indices.csv` | 43,244 | 3 | 1990-01-02 → 2026-06-04 | 8 | 0 | |
| `spx_correlation.csv` | 15,396 | 3 | 2006-01-03 → 2026-06-04 | 3 | 0 | |
| `rates_fx_vol.csv` | 24,842 | 3 | 1988-04-04 → 2026-06-04 | 3 | 0 | |
| `sp500_vix_full.csv` | 17,274 | 3 | 2015-01-02 → 2026-06-05 | 6 | 0 | redundant context (deep VIX is in vix_term_structure) |
| snapshots: `sp500_analyst`, `sp500_institutional`, `sp500_iv_snapshot_today`, `sp500_fundamentals_yf`, `sp500_earnings_yf`, `sp500_macro_calendar` | 282–503 | — | snapshot/small | — | — | context / yfinance siblings |
| `sp500_vol_dvd.csv` | 988,837 | 5 | 2018-01-02 → **2026-03-20 (stale)** | 503 | 0 | **superseded by vol_iv_full → deprecate** (still loaded by `data/consolidated_loader.py:352`) |
| `sp500_iv_history.csv` | **0** | 3 | — | 0 | — | **empty stub** |
| `sp500_short_interest.csv.xlsx` | (binary) | — | — | — | — | **drop** — binary dup of the `.csv` |

### 1b. Deep branch — gz slices + delisted panels (`data/bloomberg/deep/`, 13 files)

| file | rows | cols | span | tickers | role |
|---|--:|--:|---|--:|---|
| `sp500_ohlcv__1994_2018.csv.gz` | 2,083,270 | 7 | 1994-01-03 → 2017-12-29 | 449 | OHLCV deep (current names) |
| `sp500_ohlcv__delisted.csv.gz` | 2,383,622 | 7 | 1990-01-02 → 2026-06-05 | 1,015 | OHLCV delisted |
| `sp500_vol_iv_full__1994_2012.csv.gz` | 1,661,191 | 8 | 1994-01-03 → 2012-06-29 | 436 | IV deep #1 |
| `sp500_vol_iv_full__2012_2018.csv.gz` | 630,743 | 8 | 2012-07-02 → 2017-12-29 | 476 | IV deep #2 (the new 2018-align shard) |
| `sp500_vol_iv__delisted.csv.gz` | 2,408,183 | 8 | 1990-01-02 → 2026-06-05 | 1,016 | IV delisted |
| `sp500_liquidity__1994_2015.csv.gz` | 1,987,751 | 5 | 1994-01-03 → 2014-12-31 | 457 | liquidity deep |
| `sp500_liquidity__delisted.csv.gz` | 2,393,425 | 5 | 1990-01-01 → 2026-06-05 | 1,011 | liquidity delisted |
| `sp500_iv_surface__2005_2011.csv.gz` | 685,310 | 27 | 2005-01-03 → 2011-12-30 | 430 | IV moneyness/skew surface (5 tenor × 5 mny) |
| `sp500_iv_surface__2012_2018.csv.gz` | 795,760 | 27 | 2012-01-03 → 2018-12-31 | 470 | " |
| `sp500_iv_surface__2019_2026.csv.gz` | 912,802 | 27 | 2019-01-02 → 2026-06-04 | 501 | " |
| `delisted_status.csv` | 1,016 | 8 | — | 1,016 | per-name provenance (`ticker,name,window,*_rows,dropped,status`) |
| `ohlcv_dropped_ticks.csv` | 97 | 9 | 1994-01-21 → 2009-01-22 | 32 | bad-tick audit (current) |
| `ohlcv_dropped_ticks__delisted.csv` | 51 | 9 | 1990-03-26 → 2006-12-18 | 37 | bad-tick audit (delisted) |

### 1c. The assembled view the connector *could* serve

Concatenating recent ∪ deep ∪ delisted per series (the design in
`DATA_LAYER_DEEP_READ_DESIGN.md`):

| series | assembled span (current names) | + delisted | assembled rows (≈) |
|---|---|---|--:|
| OHLCV | **1994-01-03 → 2026-06-04** (contiguous, 2018 seam) | +1,015 dead names 1990–2026 | ~5.48 M |
| vol_iv | **1994-01-03 → 2026-06-04** (contiguous, 2012 + 2018 seams) | +1,016 dead names | ~5.74 M |
| liquidity | **1994-01-03 → 2026-06-04** (2015 seam) | +1,011 dead names | ~5.77 M |
| iv_surface (skew) | 2005-01-03 → 2026-06-04 (3 shards) | current names only | ~2.39 M |

Ticker-key fact (verified on the bytes): delisted panels, the membership file,
and `_delisted_universe.csv` **all key on the Bloomberg PIT code** (e.g.
`0111145D UN`, `1323Q US`); `normalize_ticker` maps these to opaque-but-stable
stubs (`0111145D`, `1323Q`) that join consistently across the three. Current
names use the real-symbol form (`A UN → A`). The only collision risk is the
~90 relistings/code-changes of *held* names — bounded by concat precedence
(see design §A4).

### 1d. Anomalies carried from / confirmed against the 2026-06-05 QA

- **OHLC rotation invariant** holds (0 / 1,014,916); the deep + delisted OHLCV
  panels re-gate clean (worklog: 0 violations; 97 + 51 bad ticks dropped+audited).
- **vol_iv seams contiguous** (1994-2012 → 2012-2018 → 2018+ monolith), 0 overlap.
- **deep-IV sentinel** `134217.7` (≈ 2²⁷/1000, not byte-equal), ~945 rows, 1994–95 →
  NULL by magnitude (IV > ~500%) before IV-rank/VRP, **don't drop the rows**.
- **deep usable put≠call skew exists only 1994–2004** in `hist_*_imp_vol`;
  the dedicated **`iv_surface` shards are the real 2005→2026 skew source**.
- **treasury** now reaches **1994** (CONCERN-1 closed); **vol_dvd** stale/superseded;
  **iv_history** empty stub.

## 2. Prioritized roadmap

Legend — **Effort:** S(≤½d) / M(1–2d) / L(3d+). **Risk:** low / med / high.
**§2:** does it change what the authoritative ranker sees / how it decides?

| # | Item | Effort | Risk | §2-touch | Re-baseline? |
|---|---|:--:|:--:|---|:--:|
| **R0a** | **Fix credit-rating dead-read** (`wheel_runner.py:503`) — DONE on this branch | S | low | trio file, but **off the EV path** (legacy `_calculate_wheel_score` + memo/API only) | no |
| **R0b** | **Fix sector-cap source** (route R9/R10 gate + ev_row tag to real `gics_sector_name`) — **PLANNED, not done** | S–M | **med** | **R9/R10 gate behaviour** — moves which names the sector cap aggregates/blocks | **yes** |
| **R1** | **Data-only merge** of refresh `data/` into `main` (NEVER the branch whole — it reverts the engine) + re-baseline S27/S32/S34/S35 | M | high | data bytes feed the ranker | **yes** |
| **R2** | **Connector deep-read path** — assemble monolith ∪ deep ∪ delisted in `_load` (design doc) | L | high | feeds `EVEngine.evaluate` inputs (longer/wider history); trio untouched | **yes** |
| **R3** | **Survivorship-aware backtest harness** — PIT universe from membership; read current-or-delisted names | M | med | `backtests/` only; routes through `rank_candidates_by_ev` (§2-clean) | n/a (new harness) |
| **R4** | **Theta option chains into the cost model** — real bid/ask/mid where available, BSM(iv)+`bid_ask`/`iv_surface` fallback | L | high | `wheel_runner` premium sourcing → `EVEngine` inputs (downgrade-only contract preserved) | **yes** |
| **R5** | **Extend the snapshot fingerprint** — pin `vol_iv` + `treasury` (+ deep gz hashes when R2 lands) | S | low | `backtests/regression/_common.py` only | n/a |
| **R6** | **Deep-history smoke backtest** — a 2008 run where Lehman/WaMu flow through and go to zero (proves R2+R3) | M | low | new backtest, no engine change | n/a |
| **R7** | **Cleanups** — drop `*.xlsx`; deprecate `sp500_vol_dvd.csv` (+ remove the `consolidated_loader:352` hook); null the deep-IV sentinel on intake; shard `bid_ask` before 100 MB | S | low | data/loader hygiene | no |

**Recommended order:** R0a (done) → R5 (cheap guard) → R1 (merge + re-baseline) →
R0b (fold its re-baseline into R1's, since both move snapshots) → R2 → R3 → R6 →
R4. R7 rides along with R1.

### Why each "re-baseline = yes" matters (what breaks if skipped)

The four regression snapshots (`backtests/regression/snapshots/s{27,32,34,35}.json`)
pin `fingerprint.data_csv_sha256` to the **stale** OHLCV hash `c3d5443…`; the
refreshed OHLCV is `7a3e77a4…`. Any of R0b/R1/R2/R4 moves the locked ρ /
quartile / per-year claims. The fingerprint currently pins **OHLCV only** —
`vol_iv` and `treasury` also feed the backtests and *also* changed silently;
R5 closes that blind spot **before** R1 so the guard actually trips on the next
refresh. Running the suite to "measure the data impact" on the refresh branch
is **confounded** (the refresh branch's engine predates the 2026-06-03 launch
campaign, and the suite is already RED on main from benign post-lock drift) —
re-baseline against **main's current engine + the merged data**, in one tagged
commit, per `git log --grep "^audit"` discipline.

### The R1 merge hazard (load-bearing)

`git diff main…data/bloomberg-refresh-2026-06-02` **reverts** `engine/ev_engine.py`
(−50), `engine/wheel_runner.py` (−118), `engine_api.py` (−395) and many
launch-campaign tests — the branch predates them. **Do not merge it whole.**
Bring over **`data/` only** (new branch off `main`:
`git checkout origin/data/bloomberg-refresh-2026-06-02 -- data/bloomberg`,
verify with `scripts/check_manifest_coverage.py`). The deep gz live on the
buffer branch `deep-history/bloomberg-raw` and need an assembly/loader step
(R2) before the engine sees pre-2018 / delisted history.

## 3. §2 invariant posture

Every item that changes engine inputs (R0b, R1, R2, R4) stays §2-compliant by
construction: **all candidate flow continues to route through
`EVEngine.evaluate` via `WheelRunner.rank_candidates_by_ev`.** The connector
deep-read (R2) assembles history **below** the `get_*` accessors — the
decision-layer trio (`ev_engine`, `wheel_runner`, `candidate_dossier`) is not
restructured; it simply receives longer series. R4 prefers a real Theta mid in
the cost model but never converts a negative-EV candidate to tradeable
(reviewers stay downgrade-only; the dealer clamp `[0.70,1.05]` is untouched).
R0b changes an R9/R10 *downgrade-only* soft-warn input, never a rescue.

## 4. What is done on this branch vs. awaiting review

**Done (safe prep):** the verified inventory above; R0a (credit-rating fix); the
design doc `docs/DATA_LAYER_DEEP_READ_DESIGN.md`; this roadmap; the worklog
fragment. **Awaiting operator greenlight:** R0b, R1, R2, R3, R4, R5, R6, R7 —
i.e. every consequential / decision-layer-touching step.

_Inventory + design figures: `C:\tmp\swe-exp\inv\inventory.py` against
refresh `6bb3399` / deep `e7818f4`. Companion: `DATA_LAYER_DEEP_READ_DESIGN.md`._
