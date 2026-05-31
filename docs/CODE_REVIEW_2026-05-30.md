# Code-Correctness Review & Remediation — 2026-05-30

> Multi-agent read-only review of the full Smart Wheel Engine, followed by a
> remediation pass on the **authoritative codebase (`main`)**. This document
> records what was found, how each finding was verified, and the fixes applied.
> It is the durable companion to `CHANGELOG.md` for this work.

## Method

A four-stage review was run, then every significant finding was adversarially
re-verified against both the working tree and `origin/main`:

1. **Contracts** — `CLAUDE.md`, `DECISIONS.md`, `docs/GREEKS_UNIT_CONTRACT.md`,
   `docs/MODEL_CARDS.md`, `PROJECT_STATE.md`, `MODULE_INDEX.md`, `TESTING.md`,
   and the prior `docs/optionsengine_audit_2026-05-17.md`.
2. **The brain** — one deep-read per `engine/` file (50 files) plus six
   cross-cutting traces (EV value flow, Greeks units, downgrade-only, PIT,
   R1–R6, percent/decimal).
3. **Everything else** — advisors, both news stacks, ml/backtests, `src/`,
   scripts, the 71-file test suite, dashboard, docs, configs.
4. **Adversarial verification** — every critical/high/medium finding was
   re-checked against the working tree *and* `git show origin/main:<file>` to
   separate true live bugs from artifacts of a stale review branch.

## Critical context: the review branch was stale

The review was originally run on `claude/docs-tradingview-windows-setup`, which
was **186 commits behind `origin/main`**. Verification showed **15 findings were
already fixed on `main`** (e.g. point-in-time ranker IV via PR #179, the
`analyze_ticker` as-of/staleness gate via PR #227, and the news-sentiment
EV-channel severing via D18/PR #249). Those are *not* bugs in the authoritative
codebase and are not actioned here. After verification the actionable set for
`main` was **3 high + 27 medium**, of which only ~14 touch the live EV path.

This remediation is implemented on a fresh branch off `origin/main`
(`claude/code-review-fixes`), so it targets the real codebase and skips the
already-fixed items.

---

## Confirmed live bugs on `main` — remediation targets

Severities below are corrected for the authoritative codebase. Status is updated
as fixes land (see the Remediation log at the bottom).

### EV decision path (highest value)

| # | Location | Defect | Sev |
|---|---|---|---|
| E1 | `engine/ev_engine.py` cost block (~316/319/375) | `exit_commission`+`exit_slippage` are computed into `total_transaction_cost` but **never subtracted from `ev_dollars`**. The class docstring and the inline comment promise a `P(early close)·exit_cost` penalty that is not implemented, so EV is systematically overstated by the exit leg. | High |
| E2 | `engine/data_integration.py` `get_current_risk_free_rate`; `engine/data_connector.py` `get_risk_free_rate` | Percent/decimal heuristic `rate/100 if rate > 1 else rate` mis-reads any sub-1% **percent** rate as already-decimal → **100× too high** for the entire 2011–2022 ZIRP era (2,096/3,767 CSV rows). The treasury CSV is uniformly percent. Contaminates historical backtests; current live rates (>1%) normalize correctly. | High |
| E3 | `engine/forward_distribution.py` (62–63, 100–103) | `horizon_days` is documented as **calendar days** but indexed as **trading-day bars**; callers pass calendar DTE → forward horizon ~18–45% too long → terminal distribution over-dispersed for every candidate. | High (impact) |
| E4 | `engine/dealer_positioning.py` (`analyze` T) | Time-to-expiry uses `datetime.now()` instead of the `as_of` the function already accepts → collapses to ~0 in historical (Theta-provider) backtests, corrupting GEX/walls/flip/regime/multiplier. | Medium |
| E5 | `engine/theta_connector.py` `get_vol_risk_premium` | Mixes decimal realized vol with a fundamentals IV whose unit (percent vs decimal) depends on whether the live-IV injection ran → ~100× wrong VRP on the CSV-fallback branch. | Medium |
| E6 | `engine/regime_hmm.py` | Degenerate/constant-return windows yield a confident-but-meaningless regime label and a wrongly-shrunk position multiplier (no degenerate-fit guard). | Medium |
| T1 | `tests/test_data_connector.py`, `tests/test_data_integration.py` | Tests enshrine the wrong percent/decimal assumption and only cover post-2022 dates → actively mask E2. | Medium |

### Off-decision-path correctness

| # | Location | Defect | Sev |
|---|---|---|---|
| O1 | `engine/portfolio_tracker.py` (~739) | Partial option close books realized P&L on the **full** original quantity and deletes the whole position. | Medium |
| O2 | `engine/portfolio_tracker.py` (~871) | Risk metrics computed from raw `total_value.pct_change()` → deposits/withdrawals counted as returns. | Medium |
| O3 | `engine/risk_manager.py` parametric VaR | Vega component omits the decimal→vol-point ×100 conversion (GREEKS_UNIT_CONTRACT vega drift). | Medium |
| O4 | `engine/stress_testing.py` `greeks_stress_ladder` | Full-repricing `total_pnl` ignores the IV shock entirely → wrong P&L when `iv_shock != 0`; residual gate can mark such a row reliable. | Medium |
| O5 | `engine/payoff_engine.py` (~259) | CSP `expected_value` is algebraically degenerate (`strike − breakeven ≡ put_price`) → loss term has zero dependence on assignment depth. | Medium |
| O6 | `engine_api.py` `_handle_committee` | Conflates `prob_profit` with `p_otm` and synthesizes an inverted `p_profit`, feeding advisors swapped probabilities (`prob_assignment` ignored). Advisory only. | Medium |
| O7 | `engine/model_validation.py` LSM tier | Calls `price_american_option` with two kwargs it doesn't accept → always `TypeError`, swallowed → tier is dead. | Medium |
| O8 | `engine/news_sentiment.py` `get_ticker_sentiment` | tz-aware store `as_of` vs tz-naive PIT window raises `TypeError`, swallowed → overlay silently no-ops on the production store. (The EV multiplier itself is already severed per D18.) | Medium |
| O9 | `dashboard/.../options-panel.tsx` | Decimal IV rendered with a bare `%` suffix → 28% shows as "0.3%". | Medium |
| S1 | `scripts/pull_theta_option_tape.py`, `scripts/pull_theta_corp_actions.py` | Per-ticker workers don't catch `PerEndpointFailure` from the fetch calls → crash the run (the inline D11 comment is wrong). | Medium |
| S2 | `scripts/pull_theta_iv_surface_history.py` (~132) | Operator-precedence bug in `load_universe()` silently drops all-alpha tickers ending in "D" (AMD, GOLD, HOOD…). | Medium |
| S3 | `scripts/pull_earnings_yf.py` | Full-overwrites the CSV on a partial fetch (no merge); its output is also not actually read by the event gate despite the docstring/`pull_all.py` claim. | Medium |

### Edge-case / robustness hardening (logic strengthening)

`engine/realized_vol.py` (+inf on non-positive bars), `engine/performance_metrics.py`
(NaN for n=1; Sharpe explodes on constant-nonzero series), `engine/contracts.py`
(`validate_greeks_*` admit infinite Greeks), `engine/ev_engine.py` (spread guard
treats a legitimate `bid==0` as missing; unfiltered NaN `price_scenarios`;
dividend penalty missing `days_to_ex_div >= 0`), `engine/regime_detector.py`
(NaN on flat series), `engine/dealer_positioning.py` (`stored_gamma` truthiness
discards a legitimate 0.0).

### Documentation drift (all new; none repeat the prior audit)

`docs/MODEL_CARDS.md` §6 documents the deprecated off-path regime model and a
parametric-VaR formula the code doesn't use; `docs/GREEKS_UNIT_CONTRACT.md`'s own
vega finite-difference validation example is off by 100×; `docs/DATA_SPECIFICATION.md`
specs a non-existent Parquet store and a per-bar-sqrt Garman-Klass; the dormant
R4 rule is presented as live in `docs/LAUNCH_READINESS.md`; `MODULE_INDEX.md`
lists 3 of `realized_vol`'s 5 estimators.

---

## Already fixed on `main` (branch-staleness — NOT actioned)

Ranker IV point-in-time (PR #179); `analyze_ticker` as-of + staleness gate
(PR #227); news-sentiment EV multiplier severed (D18 / PR #249); `theta_connector`
live-IV look-ahead gate; `wheel_tracker.open_covered_call` EV gate; `backtests/__init__`
import; `ib_insync` removed from runtime deps; `validate_and_normalize_iv` grey-zone;
the hollow EV-authority-token test; `test_transaction_costs` zero-collect; PIT-leakage
test columns; CONTRIBUTING vs CI; `pull_theta_option_history` failure sidecar.

## Refuted on verification (reviewer error — NOT actioned)

17 findings did not survive adversarial re-check, mostly in the off-path
`news_pipeline/security/*` layer (sanitizer/classifier "bypass" claims), plus
`risk_manager._parametric_var` "double-annualizes", the monte_carlo/binomial
discrete-dividend handling, and several "deprecated-path test" complaints.

## Clean — verified correct (a real result)

§2 EV authority (every tradeable row from a direct `EVEngine.evaluate`);
downgrade-only + dealer `[0.70,1.05]` clamp applied to `ev_dollars` only;
R1–R6 fire on exactly their conditions and never upgrade; Greeks units in
`option_pricer` match the contract to finite-difference precision; block
bootstrap / HAR-RV / POT-GPD / HMM are PIT-correct with sound numerics; BSM
reproduces Hull 15.6 (4.7594); the live-imported `src/features/{technical,volatility}`
match MODEL_CARDS with no look-ahead.

---

## Remediation log

Updated as fixes land on `claude/code-review-fixes` (off `origin/main`).

| Item | Status | Commit / note |
|---|---|---|
| _baseline_ | ✅ | 193 relevant tests green on `origin/main` before edits |
| E1 exit costs | ✅ | `ev_engine.py` applies expected exit cost; D19 added; regression test pins `mean_pnl == gross − total_cost` |
| E2 risk-free rate + T1 tests | ✅ | both accessors ÷100 unconditionally; D20 added; masking test rewritten + ZIRP-era regressions added (connector & integration) |
| E3 forward horizon | ✅ | orchestrator converts calendar DTE → trading bars (`calendar_days_to_trading_bars`); D21 added; 2 regression tests |
| E4 dealer as_of T | ✅ | `analyze` anchors T to `ms.as_of`; wheel_runner passes `as_of`; regression test |
| E5 theta VRP unit | ✅ | IV normalized percent→decimal (>3.0 ÷100) before VRP subtraction |
| E6 HMM degenerate | ✅ | `fit` raises on near-constant input → caller neutral 1.0; regression test |
| O1–O9, S1–S3 | ⏳ | |
| _note_ | ℹ | `test_theta_connector` `test_ohlcv_shape`/`test_iv_rank_in_range` fail pre-existing on this box (live-Terminal env; CI skips) |
| edge-guard hardening | ⏳ | |
| dead-code deletion (conservative) | ⏳ | |
| doc reconciliation | ⏳ | |
