# Data + Engine Audit — Phase 1 (discovery)

_Generated 2026-06-07T23:30:29+00:00 · provider `MarketDataConnector` · frontier `2026-06-04` · as_of `2026-06-04` · universe `full` (511 names)._

> Discovery pass — asserts nothing. Phase 2 turns these findings into tests. Every engine probe routes through `WheelRunner.rank_candidates_by_ev` (no §2 bypass). Decision trio untouched.

## 0. Frontier reconciliation

Auto-detected frontier (most-recent bar common to OHLCV & IV) = **2026-06-04**. Per-file max: {'ohlcv': '2026-06-04', 'vol_iv': '2026-06-04', 'liquidity': '2026-06-04', 'vix': '2026-06-04', 'treasury': '2026-06-05'}. The SessionStart staleness warning reflects the *checked-out branch*, not `main`: `main` carries data through ~2026-06-04, while `claude/suggest-rolls-defensive-surfacing` (the primary working tree) ends 2026-03-20. The probe pins to the auto-detected frontier so it is deterministic on either branch.

## 1. Connector data inventory

| key | file | rows | tickers | date_col | range | stale_d | sha256[:16] |
|---|---|---|---|---|---|---|---|
| ohlcv | sp500_ohlcv.csv | 1,014,920 | 511 | date | 2018-01-02..2026-06-04 | 0 | 7a3e77a4fab07f85 |
| vol_iv | sp500_vol_iv_full.csv | 1,037,278 | 510 | date | 2018-01-02..2026-06-04 | 0 | aab4eada1af5c9aa |
| dividends | sp500_dividends.csv | 50,230 | 427 | ex_date | 1962-05-31..2027-03-12 | - | 317e56c69615fdd8 |
| earnings | sp500_earnings.csv | 49,379 | 503 | announcement_date | 1980-01-31..2028-01-19 | - | a3548da18a31c848 |
| treasury | treasury_yields.csv | 8,458 | - | date | 1994-01-03..2026-06-05 | -1 | 48a20a883580b038 |
| vix | vix_term_structure.csv | 9,200 | - | date | 1990-01-02..2026-06-04 | 0 | 8054b746f5110d49 |
| fundamentals | sp500_fundamentals.csv | 503 | 503 | - | - | - | 2509f94da9b7da38 |
| credit_risk | sp500_credit_risk.csv | 503 | 503 | - | - | - | 6a3e31e0c94b6ba6 |
| liquidity | sp500_liquidity.csv | 1,388,848 | 511 | date | 2015-01-02..2026-06-04 | 0 | 08ca8f7ec7a64151 |

## 2. Capability map (file -> engine capability)

| key | accessor | engine capability |
|---|---|---|
| ohlcv | get_ohlcv | Spot/price history -> forward distribution (empirical NOS -> overlapping -> block bootstrap -> HAR-RV) + 504-day survivorship gate |
| vol_iv | get_iv_history | Implied vol (PIT) for BSM pricing/Greeks; realized-vol cols feed the F4 RV30/RV252 widening signal |
| dividends | get_dividends / get_next_dividend | Ex-dividend early-assignment EV for the COVERED-CALL ranker only (the short-put ranker never consults ex-div) |
| earnings | get_earnings / get_next_earnings | Earnings-date event lockout (first gate in EVEngine.evaluate; degrades gracefully -> name is simply un-gated if absent) |
| treasury | get_risk_free_rate | Risk-free rate for BSM (get_risk_free_rate returns decimal via unconditional /100; NaN if missing so callers detect absence) |
| vix | get_vix / get_vix_regime | VIX level + term-structure regime (contango/backwardation) + percentile |
| fundamentals | get_fundamentals / screen_universe | BSM dividend_yield (eqy_dvd_yld_12m), GICS sector (R9 sector cap), beta, market cap, screening |
| credit_risk | get_credit_risk | S&P rating / Altman-Z / interest-coverage credit gate |
| liquidity | get_liquidity | avg_vol_30d / turnover / shares_out liquidity context |

## 3. Coverage matrix

### 3a. Cross-file referential gaps (vs OHLCV spine)

_`in_ohlcv_not_dividends` is mostly non-dividend-payers (expected, not a defect). treasury/vix have no ticker column and are excluded._

| relation | n | tickers |
|---|---|---|
| in_ohlcv_not_vol_iv | 1 | FDXF |
| in_ohlcv_not_dividends | 84 | ABNB, AKAM, ALGN, AMZN, ANET, APP, AXON, AZO, BIIB, BLDR, BNY, BRK/B, BSX, CASY, CBRE, CDNS, CHTR, CIEN, CMG, CNC ... |
| in_ohlcv_not_earnings | 8 | BNY, CASY, COHR, FDXF, LITE, SATS, VEEV, VRT |
| in_ohlcv_not_fundamentals | 8 | BK, CTRA, EPAM, HOLX, LW, MOH, MTCH, PAYC |
| in_ohlcv_not_credit_risk | 8 | BK, CTRA, EPAM, HOLX, LW, MOH, MTCH, PAYC |

### 3b. Seam clusters (auto-detected)

- Joiner clusters (>=3 names' first bar): `{'2026-03-23': 7}`
- Leaver clusters (>=3 names' last bar): `{'2026-03-20': 8}`

### 3c. Name classes

- **post-seam-only** (8): `['BNY', 'CASY', 'COHR', 'FDXF', 'LITE', 'SATS', 'VEEV', 'VRT']`
- **thin-history <504 bars** (17): `['BNY', 'CASY', 'COHR', 'CPB', 'DPZ', 'FDXF', 'KMB', 'LITE', 'PLTR', 'PSKY', 'Q', 'SATS', 'SNDK', 'SW', 'VEEV', 'VRT', 'WMT']`
- **delisted/stale** (8): `['BK', 'CTRA', 'EPAM', 'HOLX', 'LW', 'MOH', 'MTCH', 'PAYC']`

### 3d. Per-universe completeness

_`complete` = present in ALL core ticker files (ohlcv, vol_iv, fundamentals, credit_risk, earnings, liquidity); dividends/treasury/vix excluded._

| universe | size | complete | partial | flags |
|---|---|---|---|---|
| UNIVERSE_24 | 24 | 24 | 0 | post-seam 0, thin 0, absent-ohlcv 0 |
| UNIVERSE_100 | 100 | 97 | 3 | post-seam 2, thin 2, absent-ohlcv 0 |
| connector_universe | 511 | 495 | 16 | post-seam 8, thin 17, absent-ohlcv 0 |

## 4. Engine probe (ranker at frontier)

- requested **511** · produced **480** · dropped **31** · vanished **0**
- drops_summary: `{'total_dropped': 31, 'by_gate': {'event': 6, 'data': 8, 'history': 17}}`
- forward-distribution tiers: `{'empirical_non_overlapping': 473, 'empirical_overlapping': 7}`
- non-finite outputs (ev_dollars/ev_raw/prob_profit): `none`

Drops by gate → reason:

| gate | reason | n |
|---|---|---|
| event | event_lockout:earnings@2026-07-14 (±5d buffer) | 5 |
| event | event_lockout:earnings@2026-07-13 (±5d buffer) | 1 |
| data | as_of 2026-06-04 is 76 days beyond latest data (2026-03-20); max_as_of_staleness_days=30 | 8 |
| history | history 52d < required 504d | 7 |
| history | history 450d < required 504d | 1 |
| history | history 356d < required 504d | 1 |
| history | history 7d < required 504d | 1 |
| history | history 255d < required 504d | 1 |
| history | history 380d < required 504d | 1 |
| history | history 208d < required 504d | 1 |
| history | history 152d < required 504d | 1 |
| history | history 328d < required 504d | 1 |
| history | history 480d < required 504d | 1 |
| history | history 122d < required 504d | 1 |

## 5. Ranked weakness report

Severity → `fixable_in` says whether it can be fixed additively (data/tests) or only in the trio (log-only this lane).

### [HIGH] W1 · Ranker IV uses a conditional `if iv>3.0: iv/100` heuristic (per-trio), diverging from the unconditional /100 used for treasury & dividend_yield
- **category:** unit-scale
- **evidence:** wheel_runner.py:198/1101/2418/2980; a genuine sub-3% IV is read as up to 300% and survives the 0<iv<=5.0 guard. vol_iv rows with 0<IV<=3.0 today: 17. Same class as the D20 treasury fix.
- **fixable in:** TRIO (wheel_runner) — log only, separate lane-claimed PR
- **Phase-2 test:** data->engine: assert ranker output iv is decimal (0<iv<3) for a name whose as_of IV is genuinely low; pin the heuristic boundary

### [HIGH] W2 · fundamentals.csv & credit_risk.csv are dateless single snapshots — get_fundamentals/get_credit_risk ignore as_of (structural lookahead)
- **category:** pit-lookahead
- **evidence:** no date column in either file (fundamentals_has_date=False, credit_has_date=False); a 2026 snapshot feeds BSM dividend_yield, GICS sector (R9 cap) and the credit gate at every historical as_of
- **fixable in:** additive test can pin the contract; a true fix needs PIT fundamentals (data-layer)
- **Phase-2 test:** data->engine + pit: assert the snapshot is documented as as_of-invariant; extend tests/test_pit_leaks.py

### [HIGH] W3 · Snapshot fingerprint pins ONLY sp500_ohlcv.csv — a silent IV / treasury / dividends refresh does not force a re-baseline (the dividends-incident class)
- **category:** fingerprint-blindspot
- **evidence:** backtests/regression/_common.py ohlcv_sha256(); unpinned connector files: sp500_vol_iv_full.csv, sp500_dividends.csv, sp500_earnings.csv, treasury_yields.csv, vix_term_structure.csv, sp500_fundamentals.csv, sp500_credit_risk.csv, sp500_liquidity.csv
- **fixable in:** additive: extend the fingerprint to all connector-read files (in _common, not trio)
- **Phase-2 test:** integrity: assert connector_data_sha256 pins exactly MarketDataConnector._FILES

### [MEDIUM] W4 · Cross-file ticker-universe inconsistency from the 2026-03-23 index reconstitution: earnings=pre-seam membership, fundamentals/credit=post-seam, OHLCV/vol_iv/liquidity span both
- **category:** referential
- **evidence:** in OHLCV not vol_iv: ['FDXF'] (unpriceable, no IV); in OHLCV not earnings (post-seam joiners): ['BNY', 'CASY', 'COHR', 'FDXF', 'LITE', 'SATS', 'VEEV', 'VRT']; in OHLCV not fundamentals (departed): ['BK', 'CTRA', 'EPAM', 'HOLX', 'LW', 'MOH', 'MTCH', 'PAYC']
- **fixable in:** additive test pins the contract; data-layer follow-up to reconcile membership
- **Phase-2 test:** integrity: every vol_iv/fundamentals/credit/liquidity ticker exists in OHLCV with overlapping date range

### [MEDIUM] W5 · BK->BNY re-ticker at the seam splits one continuous company into a 'delisted' name (BK, ends 2026-03-20) + a 52-bar 'thin' name (BNY, from 2026-03-23) with no linkage
- **category:** re-ticker
- **evidence:** BK in OHLCV/vol_iv/dividends but not fundamentals; BNY in OHLCV/vol_iv/fundamentals but not earnings/dividends; the engine sees two unrelated names
- **fixable in:** additive: a re-ticker map (data-layer); not fixable in tests
- **Phase-2 test:** integrity: flag known re-tickers; assert continuity or an explicit alias

### [MEDIUM] W6 · 17 names fail the 504-day history gate — several are blue-chips whose thinness is a data gap, not genuine newness
- **category:** survivorship
- **evidence:** thin: ['BNY', 'CASY', 'COHR', 'CPB', 'DPZ', 'FDXF', 'KMB', 'LITE', 'PLTR', 'PSKY', 'Q', 'SATS', 'SNDK', 'SW', 'VEEV', 'VRT', 'WMT']; e.g. WMT/KMB/CPB/DPZ have full real histories but short OHLCV on the dev box (see backtests/regression/universes.py WMT note)
- **fixable in:** additive test pins which names are gated; data-layer backfill is the fix
- **Phase-2 test:** data->engine: thin names degrade gracefully (gate=history), never emit a tradeable from <504 bars

### [MEDIUM] W8 · IV band has implausible extremes (min 0.01%, max 769%) with no sanity gate on the raw file
- **category:** unit-scale
- **evidence:** vol_iv hist_put_imp_vol min=0.01 max=769.273 (percent); rows >300%: 10
- **fixable in:** additive test (no engine change needed; ranker guard bounds it downstream)
- **Phase-2 test:** integrity: 0.1% <= IV <= ~500% on the served file; flag outliers

### [LOW] W9 · Zero put/call IV skew — put_iv == call_iv EXACTLY across 100% of rows (Nelson-Siegel skew tooling is fed a flat surface; skew dormant on Bloomberg)
- **category:** structural
- **evidence:** put==call exact 100.0% of 1031368 both-present rows
- **fixable in:** not a defect — a load-bearing data fact asserted nowhere
- **Phase-2 test:** integrity: pin put_iv==call_iv so a future skew-bearing refresh is noticed

### [LOW] W10 · Treasury: rate_3m has a negative print and rate_1m is missing pre-2001 (memory's 'treasury only 2021-05+' is STALE — file now covers 1994-01-03)
- **category:** unit-scale
- **evidence:** rate_3m min=-0.1372 negatives=60; rate_1m first_nonnull=2001-07-31 nan%=23.4
- **fixable in:** additive test; data hygiene note
- **Phase-2 test:** integrity: rate curve plausible band; as_of before coverage -> NaN (not 0)

### [LOW] W11 · 82 negative dividend_amount values are float-epsilon noise on Discontinued/Omitted rows (should clamp to 0)
- **category:** data-hygiene
- **evidence:** negative_count=82 all >= -0.001 (materially negative: 0), min=-2.4245362661989844e-14
- **fixable in:** additive: clamp in a producer (no producer script exists today)
- **Phase-2 test:** integrity: dividend_amount >= -1e-9 (tolerance) OR clamp Discontinued/Omitted to 0

### [INFO] W7 · Memory-flagged dividends truncation (CTRA/BK/LW/PAYC) is NOT present on current main — full 2018-24 ex-div history is intact (resolved, like the stale treasury memory)
- **category:** staleness
- **evidence:** 2018-24 row counts: {'CTRA': 30, 'BK': 28, 'LW': 28, 'PAYC': 7, 'MTCH': 0} (PAYC/MTCH legitimately started paying late); overall most-recent past ex-date 2026-06-01; residual: no in-repo producer script + future-declared rows to 2027-03-12 (dividends is reconstructable-only)
- **fixable in:** n/a (resolved on main); residual is the missing producer + reconstructable-only
- **Phase-2 test:** data->engine: a dividends->CC->cash test pinning an in-window ex-div changes CC selection (the R1 mechanism)

### [INFO] W12 · SessionStart 'OHLCV 79 days stale / 2026-03-20' is BRANCH-LOCAL to claude/suggest-rolls-defensive-surfacing; main is current
- **category:** staleness
- **evidence:** Auto-detected frontier (most-recent bar common to OHLCV & IV) = **2026-06-04**. Per-file max: {'ohlcv': '2026-06-04', 'vol_iv': '2026-06-04', 'liquidity': '2026-06-04', 'vix': '2026-06-04', 'treasury': '2026-06-05'}. The SessionStart staleness warning reflects the *checked-out branch*, not `main`: `main` carries data through ~2026-06-04, while `claude/suggest-rolls-defensive-surfacing` (the primary working tree) ends 2026-03-20. The probe pins to the auto-detected frontier so it is deterministic on either branch.
- **fixable in:** not a defect on main; pin engine probes to the auto-detected frontier
- **Phase-2 test:** data->engine: frontier auto-detection is deterministic across branches

### [INFO] W13 · No silent drops — every requested ticker either produced a row or carried an explicit drop reason (drops attrs are complete)
- **category:** silent-drop
- **evidence:** requested=511 produced=480 dropped=31 vanished=0
- **fixable in:** n/a (positive control)
- **Phase-2 test:** data->engine: assert n_produced + n_dropped == n_requested

## 6. Phase-2 plan (build after green-light)

**A) Database-integrity** — `tests/test_data_integrity_*.py` (NEW; no existing test asserts the bundled-CSV contracts — `test_data_connector.py` uses synthetic tmp_path fixtures). Reuse the `HAS_BLOOMBERG_DATA` skipif pattern from `tests/test_data_integration.py`.

**B) Data→engine** — extend the only real-CSV data→engine test (`tests/test_audit_viii_real_data_smoke.py`) into `tests/test_data_to_engine_*.py`: pinned frontier, adversarial ticker set, finite/banded outputs, tier-cascade correctness, no silent drops, graceful degradation, determinism.

**Buckets** — confirmed data defects go to xfail(strict=True)+issue; test bugs get fixed. Full-universe sweep behind a slow marker.

---
_Reproduce: `python scripts/audit_data_engine.py --universe full --as-of 2026-06-04`_