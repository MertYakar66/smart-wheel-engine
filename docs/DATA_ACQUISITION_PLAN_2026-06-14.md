# Bloomberg Data-Acquisition Plan — Smart Wheel Engine (University Terminal Session)

**Compiled: 2026-06-14 · Ground truth = `origin/main`, verified via the live connector. Field mnemonics are best-effort — confirm EACH via `FLDS` before scripting any pull.**

---

## Executive summary

The data layer is in far better shape than the last roadmap snapshot implied. **Treasury (full 1m–30y + SOFR, 1994→) and corporate actions (52k rows, populated) are DONE** — earlier "treasury starts 2021" / "corp-actions 2-byte empty" claims are stale. The remaining *true* Bloomberg pulls are narrow: **(1) refresh the stale `sp500_vol_dvd` dividend-yield panel** (stuck 2026-03-20, the one input that biases every live BSM valuation), **(2) the never-pulled skew/moneyness IV grid** (the single largest net-new correctness gap; entitlement-doubtful at university tier), **(3) dated/PIT fundamentals** (the current fundamentals/credit files are dateless snapshots — every "PIT" claim is net-new dated-pull work), **(4) the macro-event calendar** (FOMC/CPI/NFP — the event gate's headline motivation, with NO data source and NO live wiring today), and **(5) dividend-history completeness** (427/503 names; CTRA/BK/LW/PAYC = 0 rows).

The adversarial expansion sweep across all twelve subsystems produced a blunt finding: **most "new" ideas are NOT new Bloomberg pulls.** They are either (a) already on disk in the gitignored deep-history archive, (b) derivable from served OHLCV at zero pull cost, (c) already free via FRED / CBOE / the existing `vol_indices` puller, or (d) per-strike data that only Theta can supply. The genuinely-new Bloomberg pulls worth queuing are a short list; everything else is **wiring/code** (Part E) or **do-not-pull** (Part F).

**Storage is unconstrained and the terminal is free — so we pull broadly.** Every plausible incremental benefit is catalogued below with exact mnemonics, target files, FLDS/entitlement gates, and split-adjustment overrides. But "pull broadly" never relaxes the **§2 hard invariant: NO input can rescue a negative-EV trade.** Every dataset here enters in exactly one of three §2-safe roles — `evaluate-input-correctness`, `remove-only-gate`, or `downgrade-only/advisory` — and several proposals were rejected or reframed precisely because their naïve wiring would have created an upgrade path (see the §2 hazards flagged inline in Parts D and E).

---

## Part A — Current state (verified 2026-06-14)

### A.1 Inventory (ground truth = `origin/main`, confirmed via connector)

| File / dataset | Coverage | Status | Key caveat |
|---|---|---|---|
| `sp500_ohlcv` | 2018→2026-06-04, all 503 names, 1,014,921 rows | CURRENT | **SPLIT-ADJ** (never mix with Theta raw); 504-day survivorship gate rejects candidates pre-~2020 |
| `sp500_vol_iv_full` | 2018→2026-06-04 | CURRENT | **ATM-ONLY, ZERO-SKEW** (`put_iv == call_iv` for 100% of rows); carries `volatility_30d/90d/260d` daily |
| `sp500_liquidity` | →2026-06-04 | CURRENT | header `date,avg_vol_30d,turnover,shares_out,ticker` — **NO bid/ask columns** |
| `sp500_macro` | 1990→2026-06-04 | CURRENT | long format; instruments = `dxy,gold,spx,us_2y,us_10y,wti_oil` — **a PRICE series, NOT a FOMC/CPI/NFP calendar** |
| `sp500_vix_full`, `vix_term_structure` | CURRENT | CURRENT | **NO UX1–UX7 futures** (tier-blocked both providers) |
| `sp500_sector_etfs` | 1998→ | CURRENT | — |
| `sp500_historical_fundamentals` | 1990→ | CURRENT | columns ONLY `pe_ratio,eps,revenue,ebitda,book_value_per_share` — **no GICS, no Z-score, no cash-flow** |
| `treasury_yields` | 1994→2026-06-05 | **DONE (flip)** | full `rate_1m,3m,6m,2y,5y,10y,30y,sofr` — earlier "starts 2021" claim was stale |
| `sp500_corporate_actions` | 52k rows | **DONE (flip)** | POPULATED — earlier "2-byte empty" claim was stale |
| `sp500_dividends` | 427/503 names | **PARTIAL** | **CTRA/BK/LW/PAYC = 0 rows** + 72 other names missing; carries real `dividend_amount` + `dividend_type` incl. 375 "Special Cash" rows |
| `sp500_earnings` | →2028-01-19 forward | CURRENT | `announcement_time` populated 43,293/49,379 (6,086 NaN); only **110 forward rows** |
| `sp500_index_membership` | current only | **NO PIT HISTORY** | has `percentage_weight` + `as_of_date` (current membership only) |
| `sp500_fundamentals` (snapshot) | as-of snapshot | **DATELESS** | has `gics_sector_name` (503/503), `gics_industry_group_name` (503/503), `beta_raw_overridable` (502/503), `volatility_30d`, `free_cash_flow_yield`, `tot_debt_to_tot_eqy` |
| `sp500_credit_risk` (snapshot) | 503 rows | **DATELESS** | `altman_z_score`, `interest_coverage_ratio` (410/503), `rtg_sp_lt_lc_issuer_credit` (451/503) — **NO date column** |
| `sp500_vol_dvd` (BSM `q` panel) | stuck **2026-03-20** | **STALE** | the dividend-yield `q` that feeds live BSM + GBM drift — biases every live valuation |
| `sp500_iv_surface.csv` | — | **ABSENT** | the skew/moneyness grid has NEVER been pulled |
| snapshot files | analyst / credit_risk / fundamentals / institutional / iv_snapshot_today | CURRENT | point-in-time, no history |

### A.2 The key flips vs. older roadmaps (read before scripting)

- ✅ **Treasury — DONE.** Full 1m–30y curve + SOFR, 1994→2026-06-05. Do **not** re-pull "deep UST ≥2018."
- ✅ **Corporate actions — DONE.** 52k rows populated. Do **not** "restore from off-main" or re-pull the whole file.
- ⚠️ **`sp500_vol_dvd` — STALE at 2026-03-20.** This is now the #1 real pull (live BSM `q` consumer).
- ⚠️ **`sp500_iv_surface.csv` — ABSENT.** The skew grid is genuinely net-new; entitlement-doubtful (same ceiling as the ATM-only IV file).
- ⚠️ **Dividends — 427/503.** CTRA/BK/LW/PAYC = 0 rows + 72 others.
- ✅ **NFLX ~10× mis-scale — FIXED** (data-layer).
- ⚠️ **BKNG/CVNA 2026-03-23 split seam — REAL.** Unadjusted-split rows; targeted re-pull required (P1).
- ⚠️ **Fundamentals/credit files are DATELESS snapshots.** Every "PIT" item in Parts C/D is a net-new dated `BDH` pull, not a wire-in.

### A.3 Reconciliation with the existing data audit

The repo's data audit — **`docs/DATA_ENGINE_AUDIT_2026-06-07.md`** (Phase-1 data+engine, generated at frontier 2026-06-04) and **`docs/DATA_TEST_AUDIT_2026-06-09.md`** (test-coverage round, register W1–W37) — is **current and verified-consistent with this plan.** Independent connector re-checks reproduced its headline numbers: treasury **8,458 rows 1994-01-03→2026-06-05**, dividends **50,230 rows / 427 names**, OHLCV **~1,014,920 rows 2018→2026-06-04**, and the **2026-03-23 joiner / 2026-03-20 leaver** reconstitution seam. Its still-open, not-yet-grabbed items map 1:1 onto this plan: **#372** (R9 sector cap still on the hardcoded 132-name `DEFAULT_SECTOR_MAP`, ignoring the pulled `gics_sector_name` → D.11 / E-20), **#354** (PIT fundamentals → P0(3)), **#355** (blue-chip backfill → Part G), **#357** (dividend-producer clamp → P0(2)). The test-audit's W1–W37 (T) register is otherwise **closed** (PRs #370–#379 merged).

**The staleness is isolated to `docs/DATA_INVENTORY.md`** (2026-06-08), which still describes the pre-#338 monolith state (treasury-from-2021, empty corp-actions, 2026-03-20 frontier) — regenerate it via `scripts/inventory_data.py`. The audit docs themselves are accurate. (Housekeeping aside: `FILE_MANIFEST.md` correctly lists both audit docs; the connector inventory in the engine audit covers 9 keys and does **not** include `sp500_vol_dvd` or `sp500_corporate_actions` — neither is connector-loaded today, which is why the stale `vol_dvd` panel and the populated corp-actions file are easy to miss.)

---

## Part B — What Theta already covers (do NOT pull from Bloomberg)

`data_processed/theta/` (~8 GB local) is the authoritative per-strike source. Bloomberg cannot mass-pull chains at this tier (OMON is manual screen-only), so **per-strike OI / greeks / GEX / smile-by-strike all route through Theta**, not Bloomberg.

| Theta asset | Coverage | Replaces / supplies |
|---|---|---|
| Full-depth EOD option chains | 2016→2026-06-10, ALL strikes, both rights, per-strike **OI + bid/ask** | 119/150 names + 503-name banded backup; per-strike OI for `DealerPositioningAnalyzer`, cost-model OI tiers, the `skew_mult` slope |
| `option_history/.../data.parquet` | per-strike daily **volume + OI + bid/ask**, 2016→ | `adv_contracts` for the Almgren-Chriss sqrt impact term (the snapshot `chains/*.parquet` carry **NO volume** — use `option_history`) |
| ATM-IV daily history | 11 yr / 497 names | ATM IV term backfill |
| greeks+IV+quote snapshots | 3 dates | spot QA |
| index option chains / surfaces | — | SPX/SPY/NDX per-strike (Bloomberg can't) |
| vol-index history | — | see deep-archive parquet below |
| **DEEP ARCHIVE** (`deep-history/bloomberg-raw`, gitignored) | OHLCV 1994-2018 + delisted; vol_iv 1994-2018; liquidity 1994-2015; **IV 5×5 moneyness surface 2005-2026**; `vol_indices.parquet` (VVIX/SKEW/MOVE/GVZ/OVX/VXN/VIX3M/6M/9D 2011→) | wire-don't-pull for deep-history tail fits + the dormant skew surface + the vol-index family |

> ⚠️ **Theta `option_history` carries NO greeks/IV columns** — IV must be back-solved from the chain. Any "Theta back-solve" fallback below inherits that implementation cost.

---

## Part C — Baseline pull list (the verified P0/P1/P2 plan — carried verbatim)

These are the already-approved baseline pulls. **Do not re-propose any of these as "new" in Part D.** Formatted as terminal checkboxes with exact mnemonics, target files, and gates.

### P0 — critical correctness (do first; batch into ONE re-baseline session)

- [ ] **(1) Refresh `sp500_vol_dvd`** — `EQY_DVD_YLD_12M` / `EQY_DVD_YLD_IND`, range **2026-03-21 → now**, all 503 names · `BDH` → append to `data/bloomberg/sp500_vol_dvd.csv` · *live BSM `q` consumer (`ev_engine.py:378,678`).*
- [ ] **(2) Re-pull dividend history** — `DVD_HIST` for **CTRA, LW, MTCH, PAYC** (the source-regression payers; per #339's 2026-06-08 spec + `docs/NEXT_DATA_SESSION_RUNBOOK.md` this is a **git-local union restore, NOT a Bloomberg pull** — `BK`/`BNY`'s 0 rows are the separate BK↔BNY re-ticker thread; the broader 427/503 gap is mostly legitimate non-payers) · `BDS` → `data/bloomberg/sp500_dividends.csv` · *fixes 427/503 coverage; keep `dividend_amount` + `dividend_type`.*
- [ ] **(3) Dated/PIT fundamentals** — `EQY_DVD_YLD_12M`, `GICS_SECTOR_NAME`, `CUR_MKT_CAP`, `PE_RATIO`, `BEST_EPS` **+ `BEST_PERIOD_END_DT`** · `BDH` (dated) → new dated fundamentals file · *⚠ FLDS-verify; existing fundamentals file is dateless — this is the PIT fix.*
- [ ] **(4) Macro-event calendar** — `ECO_RELEASE_*` for **`FDTR`, `CPI YOY`, `NFP TCH`, `PCE`, `GDP CQOQ`, `NAPMPMI`** · `BQL eco_release_dt` (roadmap T0-5; §1 of `bloomberg_bql_pulls.md` — see footnote) → new `sp500_macro_events.csv` · *the event gate's headline input — currently NO data source on disk.*
- [ ] **(5) Skew/moneyness grid** — `30/60/90/180/365DAY_IMPVOL_{80..120}%MNY_DF` **put + call** · `BDH` → `data/bloomberg/sp500_iv_surface.csv` (ABSENT today) · *⚠⚠ ENTITLEMENT-TEST FIRST — same ATM-only ceiling that gives zero-skew; likely tier-blocked.*

### P1 — high-value

- [ ] **BKNG / CVNA split-seam re-pull** — targeted OHLCV re-pull across the 2026-03-23 split, SPLIT-ADJ · `BDH PX_OPEN/HIGH/LOW/LAST` → patch `sp500_ohlcv.csv`.
- [ ] **Equity EOD bid/ask** — `PX_BID` / `PX_ASK` (close) · `BDH` → `data/bloomberg/sp500_liquidity.csv` (add columns) · *⚠ daily bid/ask history DEPTH is the FLDS risk.*
- [ ] **ADV + turnover** — `EQY_AVG_VOL_30D` / `EQY_AVG_VOL_*D`, `EQY_TURNOVER` · `BDH`.
- [ ] **Short-interest** — `EQY_SHORT_INTEREST`, `SHORT_INT_RATIO`, `*_PCT_OF_FLOAT`, `EQUITY_SHORT_BORROW_RATE_NET` (land as `.csv`) · `BQL eqy_short_interest` (§3) · *⚠ borrow-rate is often a separate SLB/securities-finance entitlement.*
- [ ] **ATM IV term** — `7/14/30/60/90/180/365/730DAY_IMPVOL_100.0%MNY_DF` · `BDH` · *⚠ FLDS-verify 730D entitlement; 30d/60d already on disk.*
- [ ] **Total-return** — `TOT_RETURN_INDEX_NET_DVDS` · `BDH`.
- [ ] **CDS + ratings** — `CDS_SPREAD_5Y` (BDH history) + `RTG_SP_LT_LC_ISSUER_CREDIT` history · *⚠ CDS is a SEPARATE Bloomberg entitlement, frequently absent at university tier.*

### P2 — pull-broadly / diagnostic

- [ ] **PIT index membership** — `INDX_MWEIGHT_HIST` · `BDH`.
- [ ] Realized-vol cones; risk-reversals / butterflies; IV rank.
- [ ] Full PIT fundamentals / estimates suite; beta / shares-out / float / adj-factors.
- [ ] OIS / SOFR / real-yields / OAS / correlation / NFCI / macro-surprise.
- [ ] Sector-ETF tail; events / sentiment overlays; FX / commodities / global vol.

---

## Part D — NET-NEW additions (this expansion)

Only items the adversarial verifier marked **KEEP**, **KEEP-AS-NEEDS-CODE**, or **FLDS-GATE** appear here. **REDUNDANT and REJECT items are in Part F.** Priorities (P1/P2/P3) are calibrated to the Part-C baseline. Most "KEEPs" require engine code first — see Part E.

> **Reading guide:** *usable-today* = data exists/derivable AND a live engine hook consumes it. *needs-code* = engine work required (Part E). *FLDS-GATE* = mnemonic and/or entitlement unverified — probe at the terminal before committing.

### D.1 BSM pricing & Greeks

| Data | Mnemonics | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status |
|---|---|---|---|---|---|---|---|---|
| **Per-DTE risk-free interpolation** (top defensible KEEP) | none — `treasury_yields.csv` already has `rate_1m..30y,sofr` | n/a | on-disk | `ev_engine.py:375` (`r`→BS price), `:678` (drift) | All 3 wheel call sites (`wheel_runner.py:1192/2511/3083`) pass NO tenor → every DTE gets hardcoded 3M (`data_connector.py:753`, `data_integration.py:300-343`). 60bps tenor gap on a 365-DTE put ≈ $30/contract (~3%); 7-DTE @20bps ≈ $0.19 | evaluate-input-correctness (symmetric; re-runs evaluate, never bypasses) | **zero-pull, pure code** | **needs-code** · Medium (wheel is short-DTE-dominant — overstated as High) |
| **Hard-to-borrow → effective-`q`** | ⚠ `EQUITY_SHORT_BORROW_RATE_NET` / `SECURITY_BORROW_RATE` / `INDICATIVE_BORROW_RATE` — ALL FLDS-unverified; borrow-rate entitlement often a separate SLB tier | S&P 500 | daily | `q=trade.dividend_yield` `ev_engine.py:378`, drift `r−q−½σ²` `:678` | Borrow cost raises effective carry → lowers CC fair value. ZERO borrow logic in `engine/` today | downgrade-only on calls (never rescues) | **FLDS-GATE** (mnemonic + entitlement) | **needs-code** · L–M (only ~2–3% of S&P names ever HTB) |

### D.2 Forward distribution + POT-GPD tails + HAR-RV

| Data | Mnemonics | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status |
|---|---|---|---|---|---|---|---|---|
| **Deep clean OHLC 1994-2018 → served close series** (top subsystem KEEP) | none — already on `deep-history/bloomberg-raw` | all + delisted | 1994-2018 daily | GPD gate `ev_engine.py:495` (`len(pnls)>=200`); HAR `lookback_years` `forward_distribution.py:223` | Directly attacks the documented single-data-point CVaR weakness (`tail_risk.py:6-9`); extends GPD + HAR fit window | evaluate-input-correctness | **on-disk (promote, don't pull)** | **needs-code** · High · ⚠ heaviest-coordination item — couples with the D21 calendar→trading-bar re-baseline |
| **Yang-Zhang / Garman-Klass in the widening signal** | none — `PX_HIGH/LOW/OPEN/LAST` already in `sp500_ohlcv` | S&P 500 | on-disk | widening signal is close-to-close `forward_distribution.py:505`; estimators exist UNUSED `realized_vol.py:80/109` | More efficient RV estimator for the widening factor | downgrade-only (factor ≥ 1.0, mean/sign-preserving) | **zero-pull, pure code** | **needs-code** · M · ⚠ re-pointing the ratio shifts the 1.30 gate calibration → forces re-calibration + re-baseline |
| **Overnight-gap series** (close[t-1]→open[t]) | derive from served `PX_LAST`/`PX_OPEN` · ⚠ DROP `DAY_TO_DAY_TOT_RETURN_GROSS_DVDS` (total-return-contaminated) | S&P 500 | on-disk | `_build_terminal_prices` `ev_engine.py:635` | Wheel tail is overnight gap risk; distinct distribution component | evaluate-input-correctness (shape only) | **zero-pull, derivable** | **needs-code** · M · speculative magnitude (no probe) |
| **Short-dated event-implied vol / ATM-straddle across earnings** | ⚠ `7DAY_IMPVOL_100%MNY_DF` / `14DAY_IMPVOL_100%MNY_DF` — almost certainly tier-blocked (ATM-only ceiling); fallback = Theta back-solve | S&P 500 | per-tenor daily | `earnings_drift.py:235-242` returns only REALIZED abs-median move; no forward-looking event vol | Real correctness gap: forward event vol the layer lacks | evaluate-input-correctness across event bar | **FLDS-GATE** (source) | **needs-code** · M–H if obtainable |
| **Jump/discontinuity flag** (Lee-Mykland / BNS bipower → HAR-CJ) | derived (no native field) | S&P 500 | daily | HAR hook `forward_distribution.py:217` | Separate continuous/jump variance | downgrade-only | **derived; LOW priority** | **needs-code** · L · ⚠ HAR-RV uses daily squared-log-return RV (`:260`), not intraday — jump split severely under-powered on daily bars |

### D.3 4-state Gaussian HMM regime

> **§2 HAZARD (load-bearing, applies to EVERY item below):** `position_multiplier` returns up to **1.25** and CAN lift above 1.0 (`regime_hmm.py:292-312`); `combined_regime_mult` multiplies it into `ev_dollars` (`wheel_runner.py:1580`). **There is no ≤1.0 cap in the code today.** A stress feature added as a raw HMM obs column could shift posterior mass toward a *calm* state and RAISE the multiplier — an upgrade path. **Every stress feature below MUST be applied as a separate `min(mult, 1.0)` overlay** (the pattern `credit_mult` already uses: 0.80/0.92/1.0, never >1.0), NOT blended into the lifting `position_multiplier`.

**Net-new Bloomberg pulls required from this subsystem: ZERO.** Every KEEP is FRED-free or on-disk wiring.

| Data | Source (NOT Bloomberg) | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|
| **NFCI** (Chicago Fed financial conditions) — TOP KEEP | FRED `NFCI` (free CSV, no key) — NOT a Bloomberg pull | HMM is 1-D today (`wheel_runner.py:1411,1432`); `FREDAdapter.get_series` exists | Only genuinely-absent broad leading-edge signal (0 hits in repo) | downgrade-only — **must ship as `min(·,1.0)` stress overlay, NOT a raw obs column** | free FRED | needs-code | P2 |
| **3m10y slope** | `treasury_yields.csv` `rate_3m`,`rate_10y` (on disk) OR FRED `DGS3MO`/`DGS10` | HMM obs | front-end vs Fed-policy axis 2s10s lacks | correctness-input / cap ≤1.0 | zero-pull | needs-code | P2 |
| **DXY Δlog** | `sp500_macro.csv` `instrument='dxy'` (1990→) | HMM obs | stationary Δlog regime axis | downgrade-only; cap ≤1.0 | zero-pull | needs-code | P2 |
| **Breadth: % > 200-DMA** | derive from `sp500_ohlcv.csv` — NOT `S5TH` (⚠ unverified) | HMM obs | internal-health axis the SPX-return HMM lacks | downgrade-only; cap ≤1.0 | zero-pull | needs-code | P2 |
| **Macro-event LOCKOUT as HMM fit-mask** | reuse `EventGate` FOMC/CPI/NFP dates (`event_gate.py:28-29,80,100`) | masks the HMM **fit window** (distinct from evaluate-time lockout) | de-noises mean/cov estimation | remove-only (masks fit input; cannot lift any state) — cleanest §2 posture of the batch | no new pull | needs-code | P2 |
| **2s10s slope** | `sp500_macro.csv` `us_2y,us_10y` OR FRED `T10Y2Y` (already in pack) | ⚠ `yield_curve_signal()` `fred_adapter.py:177-191` ALREADY computes this — partly redundant | regime axis | correctness-input; cap ≤1.0 | zero-pull | needs-code (reuse existing) | P3 |

### D.4 Skew / vol surface (Nelson-Siegel; SVI dormant)

> The skew subsystem is **dormant on BOTH providers**, not just Bloomberg: `get_atm_term_structure`, `NelsonSiegelTermStructure`, `ivs_dislocation_score`, `create_empirical_surface` have ZERO callers in `ev_engine.py`/`wheel_runner.py`. The only live skew consumer is `wheel_runner.py:1538` and it is Theta-`chain_df`-only. Live BSM uses a single scalar `trade.iv`.

| Data | Mnemonics | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|---|---|
| **Dated `q` refresh** (TOP KEEP — only verified live consumer) | `EQY_DVD_YLD_12M` | S&P 500 | 2026-03-21→now daily | `q` feeds live BSM `ev_engine.py:378` AND GBM drift `r−q` `:678` | stale `sp500_vol_dvd` (2026-03-20) biases EVERY live valuation | evaluate-input-correctness | BDH | **already P0** (flag only) | P0 |
| **ATM IV term-structure grid** (NS level/slope/curv feed) | `30/60/90/180/360DAY_IMPVOL_100.0%MNY_DF` ⚠ FLDS-REQUIRED (esp. 730D) | S&P 500 | per-tenor daily | NS dormant on both providers — needs new code at puller AND consumer | true term structure for NS / forward dist | evaluate-input-correctness + downgrade-only (after wiring) | **FLDS-GATE** | needs-code | P2 |
| **25Δ/ATM/25Δ delta-bucket IV** (the live `skew_slope` triple) | put `30DAY_IMPVOL_25.0DELTA_DFLT` / call `..._25.0DELTA_CALL` ⚠⚠ FLDS-REQUIRED, entitlement-doubtful | S&P 500 | daily | `wheel_runner.py:1538` consumes a `chain_df` with `delta/iv/right`, NOT a precomputed triple → needs new `get_skew_snapshot` + inject path | feeds `skew_mult∈[0.85,1.08]` → clamped `[0,1.25]` → `ev_dollars=ev_raw×regime_mult` (sign-preserving) | downgrade-only (§2-safe; multiplication can't flip a sign) | **FLDS-GATE** | needs-code | P2 |

### D.5 Dealer positioning / GEX / walls / gamma-flip

> **§2 HAZARD (load-bearing):** `dealer_regime_multiplier` is **two-sided** — `confidence` scales the distance from 1.0 in BOTH directions (probed: `long_gamma_dampening conf=0.95 → 1.0475` BOOST; `short_gamma_amplifying conf=0.95 → 0.7150` cut). Feeding a raw `confidence` input that RAISES confidence in a long-gamma regime RAISES `ev_dollars` on a positive candidate. The hard invariant (never rescue a *negative* trade) holds because the multiplier is always positive, but the soft contract breaks. **Every item below MUST be wired as a strict min-combine (can only LOWER chain-derived confidence) and must NOT lift the long-gamma 1.05 boost.**

| Data | Mnemonics | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|---|---|
| **Index P/C OI/vol ratios** (TOP true KEEP — the one signal Theta can't supply per-strike) | CBOE ratio indices `PCRA` / `PCEQ` / `PCIN Index` `PX_LAST` (also FREE from CBOE) · ⚠ single-name `PUT_CALL_*_RATIO` NOT mass-pullable | SPX/SPY index | daily | `_regime_confidence` (self-referential net_gex ratio `dealer_positioning.py:707-728`) | index-level is the only non-Theta-overlapping angle | downgrade-only — **must be a strict shrink-only clamp** | KEEP (CBOE-free or BDH) | needs-code | P2 |
| **25Δ RR/BF (delta-anchored skew curvature)** | ⚠ `30DAY_IMPVOL_25DELTA_RR_DF` / `..._BF_DF` FLDS-REQUIRED — entitlement-test (same tier risk as P0 grid) | S&P 500 | per-tenor daily | `skew_dynamics.py:167` computes `risk_reversal`; `wheel_runner.py:1561` `skew_mult=clip(1−0.5·slope,0.85,1.08)` | PIT historical skew (Theta OI is snapshot-only); distinct from fixed-moneyness P0 grid | **correctness/pricing input** (two-sided clamp — relabeled; NOT "downgrade-only", it CAN boost to 1.08) | **FLDS-GATE** | needs-code | P3 |

### D.6 Event lockout / event gate

> **CROSS-CUTTING CORRECTION (load-bearing):** `EventGate.from_bloomberg_calendar` (`event_gate.py:180`) has **ZERO production callers** — it is test-only. The live gate is built bare via `EventGate(...)` at `wheel_runner.py:908/2530/3127`, populated **earnings-only, date-only**. Consequences: the **macro lockout (FOMC/CPI/NFP) is entirely unpopulated in production** (and `sp500_macro.csv` is a price series, not a calendar — no calendar file exists on disk), and the **dividend lockout is also unfed**. The single highest-leverage gap is that the gate's own headline events have no data source AND no live wiring — this is what P0(4) addresses.

| Data | Mnemonics | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|---|---|
| **Forward earnings-date extension** (TOP KEEP — lowest §2 risk, uses existing ingest unchanged) | `BEST_PERIOD_END_DT`, `EXPECTED_REPORT_DT`, `ANNOUNCEMENT_DT` (forward) — FLDS-verify | S&P 500 | forward dates | live gate only locks dates it holds (`wheel_runner.py:1219-1235`); verified exactly **110 forward rows, max 2028-01-19** | a 60–90d put on a name with no forward earnings row is silently un-gated | remove-only-gate | BDH | needs-code (light) | P1 |
| **Earnings BMO/AMC session timing** (`time_of_day` on `ScheduledEvent`) | none for core — `announcement_time` already on disk (43,293/49,379); ⚠ DROP `EARN_ANN_TIME`/`ANNOUNCEMENT_TIME_UTC` (unverified, unnecessary) | S&P 500 | on-disk | `ScheduledEvent` has no time field (`event_gate.py:60-72`); `get_next_earnings` drops `announcement_time` | ±1 DTE-day of block precision | remove-only-gate | on-disk (normalize in code) | needs-code | P2 (M, 12% NaN-timed) |
| **Corporate-action effective dates → corp-action gate** | ⚠ `EQY_SPIN_OFF_DT`, `STOCK_SPLIT_DT` unverified — on-disk file is the real asset | S&P 500 | effective dates | `split` EventKind + `split_buffer_days=3` (`event_gate.py:82,104`) fed by NOTHING; no `get_corporate_actions` accessor | gate structural jumps (M&A/spin/split) | remove-only-gate | on-disk (wire) | needs-code · ⚠ "126 future rows" claim is FALSE — actual = **3 structural future rows** (max 2026-07-02); value is future-proofing | P2 |
| **PIT index add/drop → gate** | `INDX_MWEIGHT_HIST`; ⚠ `INDX_MEMBER_CHG_DT` unverified | S&P 500 | reconstitution dates | gate not wired; underlying pull overlaps already-planned PIT membership | reconstitution forced-flow jump | remove-only-gate | BDH (REDUNDANT-adjacent — wire, don't double-pull) | needs-code | P3 |
| **Confirmed-vs-estimated earnings flag** | ⚠ `EARNINGS_ANNOUNCEMENT_DT_CONF_FLAG` likely NOT a real mnemonic | S&P 500 | per-event | `ScheduledEvent.note` (`event_gate.py:72`) could attach | distinguish confirmed vs TBC dates | remove-only-gate — **§2 caveat: only ever WIDEN for unconfirmed; never NARROW below default for confirmed** (narrowing a hard gate = upgrade-adjacent) | **FLDS-GATE** | needs-code | P3 |
| **Guidance / investor-day / analyst-day dates** | ⚠ `COMPANY_GUIDANCE_DT`, `INVESTOR_DAY_DT`, `EVT_*` — all unverified; `EVT_*` likely manual/EVTS-screen, not BDP | S&P 500 | event dates | `EventKind` has `"custom"` but `_buffer_for` returns **0** (`event_gate.py:105`) — needs `custom_buffer_days` | gate soft catalysts | remove-only-gate | **FLDS-GATE** | needs-code | P3 |

### D.7 Student-t copula CVaR / correlation

| Data | Mnemonics | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|---|---|
| **Realized return matrix → correlation → `PortfolioContext`** (TOP KEEP — single load-bearing fix) | none — derive log-returns from `close` in `sp500_ohlcv.csv` | S&P 500 | on-disk | `check_var` returns `passed=True reason="missing_data"` when correlation/returns are None (`portfolio_risk_gates.py:865-872`); both live ctors omit them (`engine_api.py:495`, `ibkr_portfolio_adapter.py:504`); copula prod callers = 0 | makes R7 covariance-VaR + the entire t-copula actually fire (today inert) | correctness-input | **zero-pull** | needs-code | P1 |
| **Sector / block-correlation** | none — extend A1 matrix, cluster by `gics_sector_name` (on disk) | S&P 500 | on-disk | `SectorExposureManager` `risk_manager.py:1742` | block structure for CVaR | correctness-input | zero-pull (downstream of A1) | needs-code | P2 |
| **EQY_BETA → single-factor correlation shrinkage** | beta already on disk as `beta_raw_overridable`; PIT history via `APPLIED_BETA` ⚠ secondary | S&P 500 | snapshot (on disk) | `beta_raw_overridable` feeds ONLY the `max_beta` screen (`data_connector.py:1017-1018`), nothing on the risk path | shrink noisy sample correlation | correctness-input — **§2 caveat: MUST be worst-of-two vs raw sample corr** (shrinkage can LOWER corr → shrink CVaR → upgrade path) | no new pull (consumer is new) | needs-code | P3 |
| **CBOE implied-correlation index** | **`COR1M` / `COR3M` / `COR6M Index`** via `BDH PX_LAST` — NOT `KCJ`/`ICJ` (stale, discontinued) | index | daily | correlation-regime input | tightens copula df in stress | downgrade-only (raise corr / lower df only) | **FLDS-GATE** (also partly redundant with roadmap P2/P3) | needs-code | P3 |
| **Cross-name 5Y CDS time-series → credit-correlation** | `CDS_SPREAD_5Y` (already P1) — daily-history angle is net-new | S&P 500 | daily | downstream of A1 | credit co-movement overlay | downgrade-only (raises tail CVaR only) | **FLDS-GATE** (separate entitlement; often absent at uni tier) | needs-code | P3 |

### D.8 Cost model / liquidity / fills

| Data | Source | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|---|---|
| **Option contract ADV → activate dormant sqrt impact** (TOP KEEP) | use-theta `option_history/.../data.parquet` per-strike daily `volume` (NOT snapshot `chains/` — no volume column) | wheel belt | 2016→ daily | `transaction_costs.py:149-153` sqrt term; `ev_engine.py:350-355` omits `adv_contracts` → dormant | activates the tested-but-dead Almgren-Chriss slippage | correctness-input (only ever RAISES cost — §2-safe) | use-theta | needs-code · ⚠ NOT one-line: `ShortOptionTrade` (`ev_engine.py:106-128`) needs a new `adv_contracts` field + plumbing | P1 |
| **Per-strike OI history (replace `strike_oi=1000`)** | use-theta `open_interest` (present in both chains/ + option_history) | wheel belt | 2016→ daily | `wheel_runner.py:1500-1514` reads OI, falls back to 1000; tiers fire below 500/100/50 (`transaction_costs.py:129-135`) | makes the existing tested OI penalty fire | correctness-input (only raises multiplier) | use-theta | needs-code (lowest risk) | P1 |
| **Amihud illiquidity → per-name `impact_coefficient`** | derive from on-disk `close`+`volume` (zero pull); `EQY_AVG_VOL_30D` confirm-only | S&P 500 | on-disk | `transaction_costs.py:86,153` global `impact_coefficient=0.10` — only inside the sqrt branch | per-name impact calibration | correctness-input | zero-pull | needs-code · ⚠ INERT until the ADV sqrt term (above) lands first | P2 |
| **Borrow / cost-of-carry on assigned-stock leg** | ⚠ `EQUITY_SHORT_BORROW_RATE_NET` entitlement-doubtful; for the LONG-stock leg the correct field is a broker financing/margin rate, NOT a Bloomberg borrow field — consider a flat carry assumption | S&P 500 | daily | assigned stock held in `wheel_tracker.handle_put_assignment:907-960`; NO carry logic in `engine/` | adds an omitted holding cost | correctness-input (only lowers EV) | **FLDS-GATE** (or skip the pull, use flat assumption) | needs-code | P3 |
| **Per-name realized equity spread%** | `PX_BID`/`PX_ASK` close ⚠ FLDS (daily history DEPTH is the risk); DROP `EQY_WEIGHTED_AVG_PX` (not a spread field) | S&P 500 | daily | `ev_engine.py:345-349` flat `premium*0.10`; `wheel_runner.py:1371-1372` synthetic bid | weak proxy anchor for the OPTION spread | correctness-input (weak — equity→option leap is unjustified) | BDH (also P1 baseline) | needs-code · L–M | P2 |

### D.9 Fundamentals / PIT quality (downgrade-only; new R12 distress branch)

> The reviewer is strictly downgrade-only (rules guard on `verdict=="proceed"` before downgrading to `"review"`). A new **R12 distress branch** fits this pattern and is §2-compliant. **Caveat:** `sp500_credit_risk.csv` and `sp500_fundamentals.csv` are **dateless snapshots** → every Group-B "PIT" item is net-new dated-pull work, and Z/coverage signals are **live-trading-only until a PIT-dated refresh exists** (not backtest-valid). `EventKind` is a **closed Literal** with buffer-0 fallback → Group-C "clean reuse of the gate" understates the work.

| Data | Mnemonics | Universe | Range·freq | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|---|---|
| **Altman Z-Score → new R12** (TOP KEEP — only zero-new-pull item) | `ALTMAN_Z_SCORE` (col `altman_z_score` on disk) | S&P 500 | snapshot (dateless) | served `data_connector.py:918` but `wheel_runner.py:519` reads only `sp_rating` → confirmed DEAD read; R12 branch is new | distress downgrade | downgrade-only (R12) | on-disk (wire) — **live-only** until PIT refresh | needs-code | P1 |
| **Interest-coverage → R12** | col `interest_coverage_ratio` on disk (410/503) | S&P 500 | snapshot | served `data_connector.py:919`, ZERO consumers | distress downgrade | downgrade-only (R12) | on-disk (wire) | needs-code · L–M (~30% blank) | P2 |
| **Credit-rating sub-IG tier → R12** | `RTG_SP_LT_LC_ISSUER_CREDIT` ✓ on disk; ⚠ Moody's add `RTG_MOODY_LONG_TERM` / `RTG_MDY_ISSUER_RATING` unverified | S&P 500 | snapshot | `wheel_runner.py:519` → legacy fund_score only (off EV path) | sub-IG → downgrade mapping | downgrade-only (R12) | S&P on-disk; Moody's **FLDS-GATE** | needs-code | P2 |
| **Net-debt/EBITDA (PIT)** | `NET_DEBT_TO_EBITDA` plausible; ⚠ `BS_TOT_DEBT_TO_EBITDA` verify | S&P 500 | dated quarterly | distinct from on-disk `tot_debt_to_tot_eqy` (balance-sheet D/E) | cash-flow leverage (distress-model standard) | downgrade-only (R12) | BDH (net-new dated) | needs-code | P2 |
| **FCF conversion (FCF/NI)** | `CF_FREE_CASH_FLOW`, `IS_NET_INCOME` (FLDS-verify casing) | S&P 500 | dated quarterly | distinct from on-disk `free_cash_flow_yield` (a yield, not a ratio) | earnings quality | downgrade-only (R12) | BDH | needs-code (overlaps accruals — don't ship both) | P3 |
| **Sloan accruals (derive)** | derive `CF_FREE_CASH_FLOW` + `NET_INCOME`; ⚠ `TOTAL_ACCRUALS`/`ACCRUALS_TO_ASSETS` doubtful as single fields | S&P 500 | dated quarterly | new `get_quality` + R12 | accrual-anomaly distress | downgrade-only (R12) | BDH derive | needs-code · L–M | P3 |
| **DRSK 1-yr default prob** | ⚠⚠ `DEFAULT_PROBABILITY_1YR` / `DRSK_PROB_DEFAULT` / BQL `dflt_prob_1yr` all doubtful — DRSK is a terminal-function/entitlement product | S&P 500 | daily | new R12 | strong distress signal IF entitled | downgrade-only (R12) | **FLDS-GATE** (entitlement-test before promising) | needs-code · M–H if feasible | P3 |
| **Piotroski F-score** | ⚠ `PIOTROSKI_F_SCORE` not standard — budget for component reconstruction | S&P 500 | dated quarterly | new R12 | composite quality | downgrade-only (R12) | **FLDS-GATE** | needs-code | P3 |
| **Going-concern / restatement / auditor-change** | ⚠ `IS_GOING_CONCERN_FLAG`, `FINANCIAL_RESTATEMENT_FLAG` doubtful; `AUDITOR_NAME` (diff) plausible | S&P 500 | event dates | hard-block via `is_blocked` (`ev_engine.py:310-311`) IS real, but `EventKind` Literal is closed + buffer-0 → needs Literal extension + buffer | block on terminal distress events | remove-only-gate | **FLDS-GATE** + needs Literal extension | needs-code · M if feasible | P3 |
| **Guidance-cut events (BEST_EPS revision-cross)** | `BEST_EPS` (BDH series); ⚠ `GUIDANCE_EPS_*` verify | S&P 500 | revision series | same closed-Literal/buffer-0 defect; `"guidance_cut"` needs Literal extension | time-boxed lockout on guidance cut | remove-only-gate | **FLDS-GATE** | needs-code · lowest confidence | P3 |

### D.10 Advisor committee (Buffett / Munger / Simons / Taleb)

> **Overarching §2 note:** the committee is `"authority":"heuristic_diagnostic"` running AFTER the EV ranker (`engine_api.py:1774-1775`); it never feeds `EVEngine.evaluate`, so **no §2 upgrade path is reachable** — all items are §2-safe advisory narrative. The real risk is over-claiming on-disk data: several "1990→ quarterly" claims are actually **dateless point-in-time snapshots** (history = new BDH pulls).

| Data | Mnemonics | Persona | Engine hook | Why it helps | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|
| **IV−RV VRP → Simons** (TOP KEEP — zero new data, best value/effort) | none — `get_vol_risk_premium()` already returns it (`data_connector.py:563-580`) | Simons | `simons.py:248-250` asks it as a rhetorical string | replaces a rhetorical placeholder with the real number | zero-pull | needs-code (wire) | P1 |
| **ROIC** (replace hardcoded allowlist) | `RETURN_ON_INV_CAPITAL` ⚠ FLDS (pick ONE of `RETURN_ON_CAP`/`RETURN_ON_CAPITAL`) | Buffett | `buffett.py:104-105` hardcoded `{AAPL...}` allowlist; ROIC absent on disk; no `roic` schema slot | genuine quality metric vs a static list | BDH (new pull + schema + advisor block) | needs-code · H | P2 |
| **Realized ρ-to-SPY/sector → Taleb** | none — compute from `PX_LAST` vs SPY / sector ETF | Taleb | `taleb.py:252` keys on `top_5_concentration`, hardwired to `0.0` (`engine_api.py:1687`) → fires on NOTHING today | restores a genuinely broken signal | zero-pull | needs-code | P2 |
| **Historical max-gap / jump frequency → Taleb** | none — compute from deep+delisted OHLCV | Taleb | `taleb.py:304-308` hardcoded "gaps down 30-50%" boilerplate | per-name tail realism | zero-pull (deep archive) | needs-code | P2 |
| **Realized-vol cone z-score → Simons** | none — compute from OHLCV | Simons | `simons.py:151-158` static regime dict; no `get_realized_vol_cone` | regime context | zero-pull | needs-code | P2 |
| **Put-skew steepness (25Δ−ATM) → Taleb** | `30DAY_IMPVOL_90.0%MNY_DF` − `..100.0%MNY_DF` | Taleb | `taleb.py:181-246` has no skew input; monolith IV is ATM-only → MUST use the 5×5 surface (deep-history branch) | fragility from skew | deep-branch + connector + schema | needs-code · H | P2 |
| **Owner earnings (FCF − maint capex) → Buffett** | `fcf_yield` on disk; `CF_FREE_CASH_FLOW`,`CAPITAL_EXPEND` ⚠ for maint-capex | Buffett | `fcf_yield` in `get_fundamentals()` (`data_connector.py:894`) unread by `buffett.py` | cheapest variant = surface existing `fcf_yield` | partial on-disk | needs-code | P3 |
| **Cross-sectional momentum + reversal → Simons** | compute from `PX_LAST`; `TOT_RETURN_INDEX_NET_DVDS` ⚠ optional | Simons | no `get_factor_scores` | factor context | zero-pull (PX fallback) | needs-code | P3 |
| **Net-debt/EBITDA + interest coverage → Buffett/Taleb** | `NET_DEBT_TO_EBITDA` ⚠; `interest_coverage_ratio` on disk; `BS_ST_BORROW` ⚠ (maturity wall) | Buffett/Taleb | `interest_coverage_ratio` on disk (410/503, dateless) | leverage fragility | partial (consolidate with D.9 credit pulls) | needs-code | P3 |
| **CDS 5Y + ratings trajectory → Taleb** | `CDS_SPREAD_5Y` ⚠ entitlement; `sp_rating` on disk (451/503) | Taleb | `taleb.py` | credit fragility | sp_rating on-disk (wire); CDS **FLDS-GATE** | needs-code | P3 |
| **ROIC − WACC spread → Buffett** | `RETURN_ON_INV_CAPITAL` − `WACC` ⚠⚠ both FLDS | Buffett | — | moat economics | **FLDS-GATE** (two unverified mnemonics) | needs-code | P3 |
| **Earnings-quality / accruals → Buffett** | `ACCRUALS_TO_AVG_ASSETS`, `CFO_TO_NET_INCOME` ⚠⚠ both FLDS | Buffett | no accruals column on disk | quality | **FLDS-GATE** | needs-code | P3 |

### D.11 Portfolio risk gates (R7–R10)

> **Headline finding: ZERO net-new Bloomberg pulls from this subsystem.** The proposal's "highest-value new pull" (`VOLATILITY_30D/90D/260D`) is the single biggest miss — that data already sits in `sp500_vol_iv_full.csv` as a 2018→2026 daily series.

| Data | Source | Engine hook | Why it helps | §2 role | Feasibility | Status | Pri |
|---|---|---|---|---|---|---|---|
| **Wire R7 covariance-VaR** (TOP KEEP — converts a tested gate from never-fires to functional) | on-disk `volatility_30d/90d/260d` (`sp500_vol_iv_full.csv`, 2018→) + OHLCV corr matrix | `calculate_var` `risk_manager.py:402`; `check_var` `portfolio_risk_gates.py:827`; kwargs exist on `PortfolioContext:97-99` but NO builder threads them (`wheel_tracker.py:1881-1890`, `ibkr_portfolio_adapter.py:504-509` leave None) | R7 VaR soft-warn currently fires on nothing | correctness-input (downgrade-only by construction) | **zero-pull** | needs-code | P1 |
| **GICS sector_map loader** | on-disk `gics_sector_name` (503/503) + `gics_industry_group_name` (503/503) | `sector_map` kwarg defaults to a static 132-name `DEFAULT_SECTOR_MAP` (`risk_manager.py:1579`), threaded by NO production caller | R9 sector cap uses the real, full map | downgrade-only | zero-pull (wire) | needs-code | P2 |
| **β-dollar cap gate + β-weighted R8 shock** | on-disk `beta_raw_overridable` (502/503); ⚠ `EQY_BETA`/`BETA_ADJ_OVERRIDABLE` FLDS (not on disk) | no β cap exists; `_C4_VOL_SPIKE_SCENARIO` uniform −0.10 (`portfolio_risk_gates.py:154-159`) | β-weights the stress shock | correctness-input + downgrade-only | zero-pull (raw beta); adjusted = FLDS-GATE | needs-code | P2 |
| **Per-name downside-beta + crisis-window drawdown for R8** | compute on-disk from `sp500_ohlcv` (2020/2022 covered) vs SPY; ⚠ `DOWNSIDE_BETA` unverified + absent | `check_stress_scenario` `portfolio_risk_gates.py:933` applies uniform −10% | per-name empirical crisis shocks | correctness-input | zero-pull (prefer on-disk over the unverified pull) | needs-code | P2 |
| **Per-name daily total-return for R7 historical-sim VaR** | ⚠ `DAY_TO_DAY_TOT_RETURN_GROSS_DVDS` FLDS; OR price-return on-disk | `_historical_var` `risk_manager.py:583` | minor refinement over price-return | correctness-input | price-return on-disk (free); total-return = FLDS-GATE, low value | needs-code · L | P3 |

---

## Part E — Requires-new-code appendix (KEEP-AS-NEEDS-CODE set)

Data is on hand (on-disk, FRED-free, or Theta) for everything here — **engine work is the blocker, not the terminal.** One-line code note each. Several are independent of the terminal session entirely and can proceed in parallel.

| # | Item | One-line code change | §2 guard required |
|---|---|---|---|
| E-1 | Per-DTE risk-free | add `tenor=` selection at `wheel_runner.py:1192/2511/3083` → `get_risk_free_rate` (drop the hardcoded `rate_3m`) | none (symmetric) |
| E-2 | Deep OHLC 1994-2018 promotion | promote `deep-history/bloomberg-raw` into the served close series + re-baseline (couples with D21) | none (correctness) |
| E-3 | YZ/GK in widening signal | re-point `forward_distribution.py:505` from close-to-close to `realized_vol.py:80/109`; **re-calibrate the 1.30 gate** | factor ≥ 1.0 (preserve) |
| E-4 | Overnight-gap component | gap-aware `_build_terminal_prices` (`ev_engine.py:635`) | shape-only |
| E-5 | HMM stress features (NFCI, 3m10y, DXY, breadth, fit-mask) | add a **separate `min(mult,1.0)` stress overlay** (mirror `credit_mult`); do NOT touch `position_multiplier` | **mandatory ≤1.0 overlay** |
| E-6 | ATM IV term-structure + delta-bucket skew | new `MarketDataConnector.get_skew_snapshot` + inject path to `wheel_runner.py:1538`; wire dormant NS hooks | sign-preserving multiplier |
| E-7 | Index P/C ratio confidence | strict shrink-only min-combine into `_regime_confidence` (`dealer_positioning.py:707`); must NOT lift the 1.05 long-gamma boost | **mandatory shrink-only** |
| E-8 | Forward earnings extension | extend `get_next_earnings` ingest to pull all 110+ forward rows into the bare `EventGate` | remove-only |
| E-9 | Earnings BMO/AMC timing | add `time_of_day` to `ScheduledEvent` (`event_gate.py:60`); normalize existing `announcement_time` free-text | remove-only |
| E-10 | Corp-action gate | add `get_corporate_actions` accessor; feed effective dates into the existing `split`/`split_buffer_days=3` machinery | remove-only |
| E-11 | Correlation matrix → `PortfolioContext` | build returns matrix from `get_ohlcv()`; thread into `wheel_tracker.snapshot_to_context` + `ibkr_portfolio_adapter.build_portfolio_context` (the load-bearing R7 fix) | correctness |
| E-12 | Beta shrinkage (A2/factor) | **worst-of-two vs raw sample correlation** | **mandatory worst-of-two** |
| E-13 | Theta ADV → sqrt impact | add `adv_contracts` field to `ShortOptionTrade` (`ev_engine.py:106`) + plumb from `option_history`; pass `num_contracts` from `trade.contracts` | cost-only (raises) |
| E-14 | Theta per-strike OI | replace the `strike_oi=1000` fallback at `wheel_runner.py:1500-1514` | cost-only |
| E-15 | Amihud per-name impact | derive from on-disk close+volume → per-name `impact_coefficient`; **inert until E-13 lands** | cost-only |
| E-16 | R12 distress branch | new reviewer rule consuming on-disk `altman_z_score` + `interest_coverage_ratio` + sub-IG rating; guard on `verdict=="proceed"` | downgrade-only |
| E-17 | `EventKind` Literal extension | extend the closed Literal + add buffers for `going_concern`/`guidance_cut`/`custom` (currently buffer-0) | remove-only |
| E-18 | Advisor wirings (VRP, Taleb ρ, gaps, vol-cone, ROIC) | wire `get_vol_risk_premium()` into `simons.py:248`; restore `taleb.py:252` ρ (currently fires on hardwired `0.0`); compute gaps/cones | advisory (committee non-tradeable) |
| E-19 | R7 covariance-VaR wiring | load on-disk `volatility_*` + thread the corr matrix into `PortfolioContext` (same builder as E-11) | downgrade-only |
| E-20 | GICS sector_map loader | thread the on-disk full GICS map (replace the static 132-name `DEFAULT_SECTOR_MAP`) into the R9 gate | downgrade-only |

---

## Part F — Do-NOT-pull / infeasible / Theta-routed

### F.1 Tier ceilings (hard — both providers / this tier)

- ❌ **VIX futures UX1–UX7** — tier-blocked on BOTH providers. No single-name VIX-futures carry either.
- ❌ **Per-strike option chains from Bloomberg** — OMON is manual screen-only; no `BDS`/`BQL` mass-pull. Per-strike OI/greeks/GEX → **Theta**.
- ❌ **Single-name implied forward / repo (`BFV`/`IMPLIED_REPO`)** — single-name not on tier; redundant with Theta parity back-solve; no consumer hook.

### F.2 REDUNDANT — already on disk / free elsewhere (wire, don't pull from Bloomberg)

| Item | Why redundant |
|---|---|
| `VOLATILITY_30D/90D/260D` | already in `sp500_vol_iv_full.csv` daily 2018→ |
| GICS sector / industry-group | already in `sp500_fundamentals.csv` (503/503 each) |
| Raw equity beta | already `beta_raw_overridable` in `sp500_fundamentals.csv` (502/503) |
| Index membership weights | `percentage_weight` already in `sp500_index_membership.csv` (current-only) |
| HY OAS | `FREDAdapter.credit_regime()` already LIVE in path (`fred_adapter.py:137-175` → `wheel_runner.py:945-953` → `credit_mult`); FRED `BAMLH0A0HYM2` already in pack |
| IG OAS | `credit_regime()` already fetches it; FRED `BAMLC0A0CM` in pack |
| MOVE / GVZ / OVX | `pull_vol_indices.py:30` writes `move_close/ovx_close/gvz_close` (Theta/Yahoo, free); on deep-archive parquet |
| VVIX / SKEW / VIX3M | free via `CBOEAdapter` + `pull_vol_indices.py:29-30`; on deep-archive parquet 2011→. **VVIX is the dealer-subsystem "highest-value" nominee — but it's already free + on disk; the blocker to activating `vanna_total` was never VVIX availability.** |
| SKEW index history | `cboe_adapter.py:109-121` fetches it free; deep-archive parquet |
| CBOE single-name P/C (per-name) | already produced by Theta `pull_theta_options_flow.py:228,239` |
| 2s10s | `yield_curve_signal()` `fred_adapter.py:177-191` already computes it |
| `sp_rating` (for advisor) | already on disk (451/503) — wire, don't pull |
| Multi-tenor IV rank | `iv_rank` already on `CandidateTrade` (`schema.py:104`) + consumed (`base.py:223-228`, `munger.py:225`) |
| Special/irregular dividend "split by magnitude" | FALSE premise — `dividend_amount` (true discrete cash, incl. 375 specials) already flows `wheel_runner.py:2576-2586` → `ev_engine.py:408-416`. Only residual = the 76-name completeness re-pull (already P0). |

### F.3 REJECT — infeasible, no hook, or §2 upgrade-path

| Item | Reason |
|---|---|
| Dividend-cut probability as a `q` input (`DVD_PROJ_*`) | lower expected div RAISES CC fair value → **upgrade path**. Allowed only reframed as a downgrade-only put-tail flag. |
| Index total call/put OI cross-check for R6 (`OPEN_INT_TOTAL_CALL/PUT`) | R6 keys off single-name put-wall only; no consumer for an index aggregate. Fold into the index P/C ratio item. |
| 25Δ/10Δ RR + butterfly TERM structure | `skew_term_structure()` doesn't exist; SVI `rho`/`b`/`sigma` live in `create_empirical_surface` which is NOT in the EV path; triple-unwired + entitlement-doubtful. |
| Single-name listed vol indices (VXAPL etc.) | CBOE coverage discontinued/sparse (~6 names); no live hook; QA-only. |
| Borrow/HTB "surface forward" fix | LIVE pricing uses `S=trade.spot` directly (`ev_engine.py:372`); the surface forward isn't in the EV path → corrects nothing consumed. |
| Multi-factor (size/value/mom/quality) betas / Barra loadings | tier-blocked (PORT/MAC-3 entitlement); offsetting loadings could LOWER correlation → shrink CVaR → **upgrade path** unless worst-of-two-guarded. |
| ANFCI | collinear second-order variant of NFCI; decide NFCI-vs-ANFCI empirically, don't pull both. |
| SOFR–OIS / FRA-OIS (`USOSFR`/`USFOSC3 CMPN`) | OIS leg not entitled; 2018→ too short for the 504d HMM window; unverified mnemonics; low ROI. |
| Trading-halt / LULD history (`TRADING_HALTED`) | Bloomberg coverage weak; daily-EOD wheel rarely transacts into a halt; near-zero-frequency. |
| Round-lot / min-tick (`OPT_TICK_SIZE`) | static contract-spec; dominated by the 10%-spread proxy error; no EV-ordering effect near threshold. |
| Shareholder-meeting / lockup-expiry | unverified mnemonics; AGM/lockup never moves a mature S&P 500 name materially. |
| FDA/PDUFA & legal catalyst dates | lives in EVTS/news, not structured BDP fields — out of structured-data scope. |
| M&A "trade-the-deal" sizing / expected-move buffer relaxation | any EV-rescue/gate-shrink framing → **§2 violation**. Keep only the M&A *date* as a remove-only lockout. |
| CFTC COT (E-mini dealer net) | additive signal but FREE from CFTC (Socrata/CSV), not a Bloomberg pull; weekly+3-day lag forbids gating; low priority. |
| Confirmed-flag buffer *narrowing* | narrowing a hard gate to admit blocked trades = upgrade-adjacent. Only ever widen for unconfirmed. |

---

## Part G — Terminal execution notes

### G.1 Pre-session — pull the exact missing-name lists from GitHub

The CASY-4 / 11-truncated / 10-blue-chip backfill lists live in the data-queue issues. Pull them before scripting:

```
gh issue view 339   # CASY 4-file Bloomberg blocker
gh issue view 355   # 11-truncated names
gh issue view 354   # PIT fundamentals (dated pull — the as_of fix)
gh issue view 357   # treasury rate_1m note + dividend epsilon-clamp (low-pri)
```

⚠ Per the data-queue decision: #339/#355/#354/#357 are all **Terminal-gated and re-baseline-coupled** → **BATCH them into ONE Terminal session.** Peeling them apart pays the ~4h re-baseline tax twice.

**Authoritative runbook + corrected scope (2026-06-15, verified vs current `main`):** the single ordered execution plan for #339/#355/#354/#357 is **`docs/NEXT_DATA_SESSION_RUNBOOK.md`** (phases 1A Bloomberg / 1B git-local / 2 (E)-fixes / 3 re-baseline / 5 producer). Most of the data queue is **NOT** Bloomberg-gated:
- **Bloomberg-gated (Terminal):** CASY 4 files (`ohlcv`/`vol_iv`/`liquidity`/`earnings`, spec `docs/CASY_BACKFILL_SPEC.md`) + pre-seam OHLCV for the **10 other truncated names** — WMT, KMB, CPB, DPZ, PLTR, VEEV, COHR, LITE, SATS, VRT (#355, verified 52–450 bars today) + #354 dated/PIT fundamentals.
- **Git-local (desk, NOT terminal):** BK↔BNY collapse, `sp500_dividends.csv` union (**CTRA/LW/MTCH/PAYC**, not BK), `UNIVERSE_100` re-derive (CMG/CMI return).
- **Genuinely-recent — do NOT backfill:** BNY, FDXF, SNDK, SW, PSKY, Q.

### G.2 FLDS discipline (mandatory)

- **Confirm EVERY mnemonic via `FLDS` before scripting a pull.** All ⚠-flagged fields in Parts C/D are best-effort and several are likely wrong or entitlement-gated.
- **Entitlement-test the highest-risk items FIRST**, before committing the session plan: the **P0(5) skew/moneyness grid** (`xxDAY_IMPVOL_yy%MNY_DF` put+call), **CDS** (`CDS_SPREAD_5Y` — separate credit entitlement), **borrow rates** (`EQUITY_SHORT_BORROW_RATE_NET` — separate SLB tier), **DRSK** (terminal-function product), and the **delta-bucket IV** (`..._25.0DELTA_*`). If the entitlement test returns no data, fall back to Theta back-solve (smile) or drop the item — do NOT promise downstream wiring on an unconfirmed field.
- **Land all borrow/short-interest pulls as `.csv`** (not parquet) to match the connector load path.

### G.3 Re-baseline tax — batch the heavy pulls

Any change to OHLCV / dividends / PIT fundamentals / the calendar-DTE basis triggers a **mandatory S27 / S32 / S34 / S35 re-baseline** (the `backtest-regression` lock). Therefore:

- **Batch ALL of: P0(1) vol_dvd refresh, P0(2) dividend completeness, P0(3) dated fundamentals, BKNG/CVNA seam re-pull, and any deep-OHLC promotion (E-2) into ONE session**, then run the re-baseline ONCE.
- **E-2 (deep OHLC 1994-2018) couples with the D21 calendar→trading-bar change** — it is the heaviest-coordination item, not a drop-in. Schedule it as its own coordinated re-baseline, not bundled with the routine refresh.
- Re-point/re-calibration items (E-3 YZ/GK widening) shift the 1.30 gate calibration → they also force a re-baseline; do not treat them as free swaps.

### G.4 Split-adjustment overrides (data-integrity hazard)

- `sp500_ohlcv` is **SPLIT-ADJUSTED**; Theta chains are **RAW**. **Never mix them.** When cross-checking a Bloomberg close against Theta, expect the split-adjustment delta around 2026-03-23 (BKNG/CVNA) and any other reconstitution seam.
- Deep-archive OHLCV is split-adjusted-vs-raw-sensitive — verify the adjustment basis before promoting (E-2).

### G.5 Branch & §2 discipline

- **Branch all data work off the LIVE `origin/main`** (the live pull may be running on the primary clone — don't touch that tree; do data work in a worktree off live origin/main).
- **Branch + PR for every change.** Never commit data or code to `main` directly.
- **§2 reminder — none of this rescues a negative-EV trade.** Every dataset enters as `evaluate-input-correctness`, `remove-only-gate`, or `downgrade-only/advisory`. Three items in this plan carry a MANDATORY §2 guard that must be in the same PR as the wiring: the **HMM stress overlay must be `min(mult,1.0)`** (E-5), the **dealer P/C confidence must be shrink-only and must not lift the 1.05 long-gamma boost** (E-7), and any **beta/factor shrinkage must be worst-of-two vs raw sample correlation** (E-12). Reviewers downgrade; they never upgrade. The dealer multiplier clamp stays `[0.70, 1.05]`.

---

*This document is the authoritative pull plan for the university Bloomberg session. Companion specs: the data audit `docs/DATA_ENGINE_AUDIT_2026-06-07.md` + `docs/DATA_TEST_AUDIT_2026-06-09.md` (verified prior-art, see A.3), `docs/DATA_ACQUISITION_ROADMAP.md` (full catalog), `docs/BLOOMBERG_PULL_LIST.md`, `docs/bloomberg_bql_pulls.md` (BQL query bodies §1 `eco_release_dt` / §2 `cac_*` / §3 `eqy_short_interest` / §7 correlation — **referenced by the roadmap but NOT currently in the tree; author/locate the BQL bodies at the terminal**). Verified state confirmed against `origin/main` @ `83eacdd`, 2026-06-14.*
