# HT-D — R10 strict-mode at $1M / 100t scale (2026-05-30)

> **STATUS — IN FLIGHT.** Sections marked _PENDING RUN_ are populated from
> the driver's `summary.json` after the full 5y backtest completes. Pilot
> run results (smaller window) are reported under §3.

**Question:** *Every prior canonical backtest (S27 / S32 / S34 / S35 / S38 /
S40 / S43 / S44) ran with `require_ev_authority=False` on the
`WheelTracker`. That meant the **R10 single-name notional cap**
(`engine.portfolio_risk_gates.check_single_name_cap`, the doc-designated
**load-bearing magnitude guard** for the engine's `prob_profit` top-bin
over-confidence per `docs/PROB_PROFIT_CALIBRATION_2026-05-28.md` +
`docs/F4_TAIL_RISK_DIAGNOSTIC.md` §10 + `docs/PRODUCTION_READINESS.md`
§3 B1) has **never** been exercised in a backtest. S44 explicitly flagged
this open question in its AI handoff. Does R10 actually bind at this
scale, and if it does, does it materially move the engine-vs-passive
deployment story?*

**Headline answer:** _PENDING RUN_ — populated once the full 5y backtest
(2020-01-02 → 2024-12-31, $1M, 100 first-alphanumeric SP500 tickers, full
friction) completes.

**Engine SHA at run:** `origin/main` @ `56c671d` (post #284 / #285 / #286
/ #287 / #248 / #249 / #251 / #275 / #288 — the cycle-1 close commit).

**Baseline:** the canonical S38 / S44 documents at the same setup, with
`require_ev_authority=False`. Critically, this driver runs BOTH the loose
(`require_ev_authority=False`) and strict (`require_ev_authority=True`)
trackers in parallel on a SHARED daily SP rank call, so the strict-vs-
loose delta is harness-drift-free.

---

## 1. Setup

- **Window:** 2020-01-02 → 2024-12-31 (1,258 trading days — identical to
  S38 / S44 / S43 W3)
- **Capital:** $1M
- **Universe:** `backtests.regression.universes.UNIVERSE_100` (100 first-
  alphanumeric SP500 tickers — same as S34 / S38 / S40 / S43 / S44)
- **Strategy:** 35-DTE / 25-delta short puts, wheel into 35-DTE / 25-
  delta covered call on assignment, hold to expiry
- **Friction:** `full` only (canonical S38 / S44 headline). Friction is
  an independent dimension already characterised by S38 / S44 across all
  3 levels; R10 cares about per-name notional aggregation, not entry-
  cost shape, so adding the other friction levels would 3× compute for
  no R10 insight.
- **Tracker config:**
  - `tracker_loose`: `require_ev_authority=False` (identical to S38 / S44
    baseline)
  - `tracker_strict`: `require_ev_authority=True`, `min_nav_for_trading=0`,
    `connector=MarketDataConnector(...)` — triggers `_evaluate_d17_hard_blocks`
    on every put + CC open attempt
- **R10 defaults (production):** `_DEFAULT_MAX_SINGLE_NAME_PCT = 0.10`
  (10% NAV per ticker, aggregating short-option notional across put + CC
  legs). `_DEFAULT_MAX_SECTOR_PCT = 0.25` (R9). `_DEFAULT_DELTA_CAP_PER_100K_NAV
  = 300.0`. `_DEFAULT_KELLY_FRACTION = 0.5` (preemptively reserved per
  docstring — won't fire under one-contract sizing).
- **Engine SHA under test:** `origin/main` @ `56c671d` (cycle-1-close).
- **Driver:** `docs/verification_artifacts/r10_strict_driver.py` (this PR).
- **Output artifacts:** `rank_log_{loose,strict}.csv`, `open_attempts_{loose,strict}.csv`,
  `cc_attempts_{loose,strict}.csv`, `equity_curve_{loose,strict}.csv`,
  `tracker_{loose,strict}_state.json`, `daily_state.csv`, `summary.json`.
  Generated under `%TEMP%/r10_full/` (NOT committed — the rank logs are
  GB-scale per the cycle conventions; the driver + the headline summary
  here ARE committed).

## 2. Headline results

_PENDING FULL RUN._

The table will reproduce S38 / S44's headline format with the strict /
loose / Δ columns:

| Metric | Loose (S38/S44 baseline) | Strict (D17 hard-blocks ON) | Δ strict − loose |
|---|---|---|---|
| Final NAV | _pending_ | _pending_ | _pending_ |
| Return (5y) | _pending_ | _pending_ | _pending_ |
| Executed puts | _pending_ | _pending_ | _pending_ |
| CCs opened | _pending_ | _pending_ | _pending_ |
| Put assignments | _pending_ | _pending_ | _pending_ |
| Spearman ρ (all candidates) | _pending_ | _pending_ | _pending_ |
| Mean realized (all candidates) | _pending_ | _pending_ | _pending_ |

### 2.1 R10 / R9 bind rates

_PENDING FULL RUN._

For the **strict tracker** only (loose tracker has no D17 hard-blocks),
the binding rate of each gate over the full 5y window:

| Gate | Refuse count | % of strict put attempts | % of strict CC attempts |
|---|---|---|---|
| R10 single_name_breach | _pending_ | _pending_ | _pending_ |
| R9 sector_cap_breach | _pending_ | _pending_ | _pending_ |
| portfolio_delta_breach | _pending_ | _pending_ | _pending_ |
| kelly_size_exceeded | _pending_ | _pending_ | _pending_ |

### 2.2 Per-name peak short-option notional

_PENDING FULL RUN._

For each tracker, the peak per-name notional as a % of NAV reached at
ANY point in the run — and which ticker / date hit that peak. This is
the direct R10 question: does the engine's natural deployment ever
push a single name above 10% NAV in the loose path? If yes, R10 binds
the strict path on that date; if no, R10 is observationally inert.

| Tracker | Peak per-name % NAV | Ticker | Date | R10 would have bound? |
|---|---|---|---|---|
| Loose | _pending_ | _pending_ | _pending_ | _pending_ |
| Strict | _pending_ | _pending_ | _pending_ | (R10 is the gate — by construction at most 10%) |

## 3. Pilot run (correctness validation)

_PENDING PILOT._

A short-window pilot (2020-01-02 → 2020-04-30, 84 trading days
including the COVID-March drawdown) was run first to validate:

  1. Both trackers complete the full window without exception.
  2. The strict tracker's `_evaluate_d17_hard_blocks` actually fires
     during the run (i.e., the gate path is observably exercised even
     if R10 specifically rarely binds).
  3. Loose-tracker final NAV is in the ballpark of S38's 2020 Q1+Apr
     reference (cross-validates the harness against the canonical
     `_common.run_backtest_multi_friction`).

Pilot output captured at `docs/verification_artifacts/r10_pilot_*_raw_output.txt`.

## 4. Per-year breakdown (matches S44 format)

_PENDING FULL RUN._

Per-year, full friction, all candidates (the "if I'd blindly executed
every ranked candidate" view that S44 uses for refusal-mechanism
comparison):

| Year | n_candidates | Loose mean realized | Strict mean realized | Loose executed | Strict executed | R10 refuse count |
|---|---|---|---|---|---|---|
| 2020 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 2021 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 2022 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 2023 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |
| 2024 | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

## 5. Concentration / sector analysis

_PENDING FULL RUN._

Tracks: which tickers MOST often triggered R10 in strict mode? S44's AI
handoff predicted "AAPL / BKNG / AZO when the engine ranks them heavily
for several consecutive days." Confirms or refutes.

## 6. §2 invariant scan (mirrors S44 §5)

_PENDING FULL RUN._

For both trackers, count of:
- Executed PUT trades with `ev_dollars ≤ 0` (must be 0)
- Executed PUT trades with non-finite `ev_dollars` (must be 0)
- CC opens with `ev_dollars ≤ 0` (informational — harness EV floor is 0)
- Any non-finite anywhere in rank logs

If strict tracker shows 0 for the first two and loose tracker shows 0
for the first two, §2 holds in both modes. (Expected: 0 in both — R1 is
the upstream gate and `consume_ranker_row` refuses non-positive at
issue.)

## 7. Findings

_PENDING FULL RUN._

## 8. Implications for the deployment matrix

_PENDING FULL RUN._

The relevant `docs/PRODUCTION_READINESS.md` §5 row is "$500k-$1M
supervised, universe ≥ 100 tickers" currently amended to **⚠ Conditional
with explicit underperformance acknowledgment** citing S38's −52pp /
S44's −51.5pp engine-vs-SPY. This doc updates that framing only if
strict-mode R10 materially binds (i.e., the strict NAV differs from the
loose NAV by more than friction-noise).

## 9. AI handoff

_PENDING FULL RUN._

## 10. Method appendix

- **Engine SHA:** `origin/main` @ `56c671d` (post-#288 cycle-1 close).
- **Data:**
  - OHLCV: `data/bloomberg/sp500_ohlcv.csv` SHA256 captured in
    `summary.json` `setup.ohlcv_csv_sha256`. Window: 2018-01-02 →
    2026-03-20.
  - IV: `data/bloomberg/sp500_vol_iv_full.csv`. Window: 2015-01-02 →
    2026-03-20.
- **Provider:** `MarketDataConnector` (Bloomberg-only). Theta MCP not
  required for this run.
- **Driver:** `docs/verification_artifacts/r10_strict_driver.py`. Imports
  the canonical harness helpers (`friction_*`, `_spot_on_or_after`,
  `_next_business_day`, `_forward_replay_realized_pnl`,
  `assert_data_window_available`, `ohlcv_sha256`) from
  `backtests/regression/_common.py` and the canonical `UNIVERSE_100`
  from `backtests/regression/universes.py` to keep the comparison
  apples-to-apples with S38 / S44 / S43.
- **Methodology — A/B harness:** Both trackers built inside ONE
  `run_strict_vs_loose` call. They share **one** daily
  `rank_candidates_by_ev` invocation (the multi-friction pattern from
  `_common`) so the rank input is byte-identical for both. The only
  divergence is the open-path: loose calls `tracker.open_short_put`
  directly; strict calls `tracker.consume_ranker_row` (which issues a
  D16 token, then opens with `current_ev_dollars`, which triggers
  `_evaluate_d17_hard_blocks` — R9 + R10 + portfolio-delta + Kelly).
  Per-attempt outcome (`opened` / `refused` + `reason`) captured for
  EVERY attempt from `tracker._ev_authority_log` introspection,
  immediately before/after the attempt to attribute log entries to the
  attempt.
- **Compute:** _PENDING FULL RUN; pilot timing reported in §3._

---

## Appendix A — D17 hard-block evaluation order (engine reference)

For documentation reference, the strict tracker evaluates D17 gates in
this order inside `_evaluate_d17_hard_blocks` (see
`engine/wheel_tracker.py:1768`):

  1. **Pre-gate**: `nav < min_nav_for_trading` → `nav_exhausted`. Default
     `min_nav_for_trading=0.0` so this only fires on a strictly negative
     NAV; we set 0 explicitly.
  2. **R9 sector cap** (`check_sector_cap`, GICS-strict, 25% NAV cap).
  3. **R10 single-name cap** (`check_single_name_cap`, 10% NAV cap).
  4. **Portfolio delta cap** (`check_portfolio_delta`, ±$300 / $100k NAV).
  5. **Kelly cap** (`check_kelly_size`, 50% NAV per trade — preemptively
     reserved per the docstring; structurally unreachable at any
     realistic NAV under the single-contract sizing path).

First-failure-wins: the gate returns at the first failed check. The
audit-log entry carries `reason` + the details bag of that gate. R7
(`check_var`) and R8 (`check_stress_scenario` + `check_dealer_regime`)
do NOT fire in the tracker open path — they are dossier-side
**soft-warns** that downgrade verdict (`proceed → review`) but never
block. This driver does NOT exercise the dossier path because the
question is about the tracker hard-block; if the dossier-side R7/R8
firing rate is also of interest, that's a follow-up card.
