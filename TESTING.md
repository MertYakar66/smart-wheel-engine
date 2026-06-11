# Testing — Smart Wheel Engine

The full suite is `pytest tests/ -v` (run `pytest --collect-only -q | tail -1`
for the live count — pinned numbers drift within a day).

> **The per-PR / launch-blocker gate is `pytest tests/ -m "not backtest_regression"`**
> — that is the exact command CI runs (`.github/workflows/ci.yml`), and the
> ~11-minute run that gates merges. A *bare* `pytest tests/` does **not**
> auto-deselect the `backtest_regression` marker: `conftest.py` only *registers*
> it and `addopts` is just `-v`, so with the S27/S32/S34/S35 snapshots present
> locally a bare run pulls the **~4–5 h slow lane inline** (the reproducers
> `skip` only when their snapshot JSON is absent — and all four are committed).
> Use the `-m "not backtest_regression"` form for the fast gate; run the slow
> lane deliberately via `.claude/commands/backtest-regression.md`. The data-drift
> guards (`test_snapshot_*fingerprint*`) are *not* behind the marker, so they run
> in the fast gate and pre-flag snapshot drift before any multi-hour run.

Markers and hypothesis profiles are wired in `conftest.py`.

## Two-axis verification

The pytest suite is one of two complementary verification surfaces;
they answer different questions and feed each other.

| Surface | Question it answers | Source of truth |
|---|---|---|
| `tests/` (this file) | "Does the engine still produce the right output as code evolves?" — continuous red-green regression | ~2,300+ pytest functions; CI gates merges on the full suite plus `--cov-fail-under=80` |
| `docs/worklog/` fragments (index: `docs/worklog/INDEX.md`) | "Did the engine produce realistic output on this scenario one time?" — one-shot trader-meaningful walkthroughs | `Sn` fragments S1–S46 (split verbatim from the now-frozen `USAGE_TEST_LEDGER.md`); Realism Check tables compare engine output to Bloomberg CSV / IV file / real market moves row by row |

**Promotion path: Sn finding → pytest test.** When an Sn walkthrough
surfaces a real invariant or a real gap, the finding gets promoted
into the regression suite so it can't silently regress. Recent examples:

| Sn finding | Promoted to | PR |
|---|---|---|
| F4 tail-risk gap (COST 2022-04-04 / UNH 2024-11-11 deep drawdowns; S22 + S27) | `tests/test_f4_tail_risk_gap.py` | #196 |
| Dealer multiplier `[0.70, 1.05]` clamp survives integration to `EVResult.dealer_multiplier` | `tests/test_dealer_multiplier_evengine_integration.py` | #193 |
| `consume_ranker_row` chain with real ranker output | `tests/test_consume_ranker_row_anchor.py` | #186 |
| `EVEngine.evaluate` event-lockout short-circuit | `tests/test_evengine_event_lockout.py` | #185 |

The `xfail` markers in `test_f4_tail_risk_gap.py` are the
regression-watch contract for findings that aren't fixed yet: if the
engine ever starts producing the correct value, the xfail flips to
pass and the suite tells us.

**Third surface: predictive validity.** Whether `ev_dollars` actually
correlates with realized P&L over many trades lives on the *backtest*
axis — `docs/ENGINE_BACKTEST_2022_2024_IV_PIT_RERUN.md` reports
Spearman ρ = 0.2183 post-PIT-fix (vs. the pre-fix ρ = 0.4838 that
was inflated by the snapshot-IV bug).

## Test taxonomy

Filenames don't always reveal intent — some `test_audit_*` files pin
launch-blocker invariants, others are smoke tests. Use this map.

### Decision-layer invariants (launch-blocker)

These pin the EV invariant from `CLAUDE.md` §2. Break any of them and
the ranker is unsafe. **Run before every decision-layer change.**

| File | Pins |
|---|---|
| `tests/test_audit_invariants.py` | EV is the only ranker; reviewers cannot upgrade |
| `tests/test_dossier_invariant.py` | `EnginePhaseReviewer` rules R1–R10; downgrade-only contract; `MCPChartProvider` import-guarded contract test |
| `tests/test_r11_elevated_vol.py` | `EnginePhaseReviewer` rule R11 — elevated-vol top-bin size-down (VIX level > 25 + `prob_profit` > 0.90); downgrade-only; `vix_level=None` no-op (`DECISIONS.md` D23) |
| `tests/test_authority_hardening.py` | TV webhook / analyze / strangle / strikes / wheel_tracker route through EV (audit-vi) |
| `tests/test_audit_viii_unit_invariants.py` | IV / risk-free-rate percent↔decimal normalisation; rolled-position P&L accumulator (audit-viii) |
| `tests/test_audit_viii_e2e.py` | Webhook → HMAC → enrich → EV → token chain; HMM cache reuse; OHLCV invariant guard (11 e2e tests) |
| `tests/test_audit_viii_real_data_smoke.py` | Real-Bloomberg smoke: `rank_candidates_by_ev` returns non-empty rows; connector rate is decimal |
| `tests/test_launch_blockers.py` | Top-level launch checks |
| `tests/test_dossier_r9_r10_audit.py` | S42 structural audit of R9 sector-cap + R10 single-name cap (32 tests: firing, downgrade-only, fail-closed on missing context, R7→R8→R9→R10 ordering, strict-`>` boundaries) |
| `tests/test_dossier_downgrade_property.py` | §2 downgrade-only lattice property — `review()` may hold or downgrade (proceed<review<skip<blocked), never upgrade, across the R1–R11 chain |
| `tests/test_dossier_cp1252.py` | Reviewer notes are cp1252-safe (ASCII-only) across every note-producing branch — Windows-console regression |
| `tests/test_ev_non_finite_defense.py` | R1a: non-finite (`±inf`/`NaN`) EV → blocked, in the reviewer AND the webhook `_enrich_alert` parity path |
| `tests/test_ev_authority_log_schema.py` | D16/D17 audit-log shape closure — the issue/refuse/revoke/consume + five D17 hard-block reject shapes in `_VALID_SHAPES` |
| `tests/test_token_param_binding.py` | brain-audit M1 parameter binding — cross-ticker/strike/dte/side mismatch refused; unbound-token fail-closed; legacy snapshot rebuild; escape-hatch (explicit expiration) preserved |
| `tests/test_portfolio_risk_gates.py` | D17 gate library — per-gate unit tests (VaR/Kelly/delta/sector/single-name/stress/dealer-regime) against the locked defaults + Q3 missing-data semantics |
| `tests/test_decision_layer_wiring.py` | Production-caller chain: `rank_candidates_by_ev` → token → `open_short_put`; `PortfolioContext` → R7/R8 firing |
| `tests/test_ranker_tracker_wire.py` | C3/C4 wires — `consume_into_tracker` with EV-authority + D17 caps; `consume_into_live_book` armed (R9/R10) end-to-end |
| `tests/test_production_tracker_caps.py` | D22 — `make_live_book_tracker` arms R9+R10 token-free; bare-library default stays OFF; strict mode arms all four |
| `tests/test_reviewer_eventgate_invariants.py` | W65–W67 — R5 inclusive threshold, R3 strict-`>` spot tolerance, EventGate earliest-event-first ordering |
| `tests/test_credit_rating_population.py` | R0a regression — `analyze_ticker` credit_rating reads the connector's `sp_rating` key (dead-read fix, #333) |

### Quant correctness

| File | What it covers |
|---|---|
| `test_option_pricer.py` | BSM, BAW, IV solver vs textbook references |
| `test_binomial_tree.py` | Binomial-tree pricing |
| `test_monte_carlo.py` | Block bootstrap, jump diffusion, LSM |
| `test_greeks_unit_invariants.py` | Greek units (see `docs/GREEKS_UNIT_CONTRACT.md`) |
| `test_realized_vol.py` | Close-to-close, Parkinson, Garman-Klass estimators |
| `test_advanced_quant.py` | Advanced quant building blocks |
| `test_quant_upgrades.py` | Upgrades shipped with audits |
| `test_risk_manager.py` | Position sizing, sector exposure, HRP |
| `test_stress_testing.py` | Scenario engine |
| `test_payoff_engine.py` | Payoff diagrams |
| `test_regime_detector.py` | Rule-based regime |
| `test_dealer_positioning.py` | GEX / walls / gamma flip / regime |
| `test_quant_fixtures.py` | Shared fixtures |
| `test_tail_risk.py` | POT-GPD tail estimation — threshold selection, GPD fit, `gpd_var_cvar`, `pot_gpd_cvar`, tail-regime flag |
| `test_transaction_costs.py` | Spread + slippage edge branches — bid/ask fallback order, OI liquidity tiers, direction impact (D10 F7) |
| `test_earnings_drift.py` | Post-earnings-drift analytics — lazy loaders, per-sector drift, temporal filtering |
| `test_portfolio_copula_coverage.py` | `portfolio_copula` edge paths — PSD repair, Cholesky→eigen fallback, empty arrays, verdict ladder |
| `test_pricing_evaluate_invariants.py` | W63–W64 — BSM Greek units vs binomial cross-check; `EVEngine.evaluate` stays finite on degenerate DTE |
| `test_f4_rv_widening.py` | F4 fix v2 (#260) — RV30/RV252 widening factor calibration pins (1.30 threshold, 1.5× cap), PIT safety, sign/mean preservation |
| `test_premium_correction_pilot.py` | `studies/premium_correction` split-adjustment layer — AAPL 4:1 raw↔adjusted mapping, bogus-join prevention |

### W-series quant-invariant pins (2026-06 audit round 2; register: `docs/DATA_TEST_AUDIT_2026-06-09.md`)

| File | Pins |
|---|---|
| `test_forward_distribution_invariants.py` | W38–W43 — empirical→overlapping→block-bootstrap→HAR-RV cascade order + RV estimator invariants |
| `test_tail_copula_stress_invariants.py` | W44–W49 — GPD CVaR ≥ VaR, copula CVaR ≥ VaR on both Gaussian/t legs, stress-scenario execution |
| `test_regime_hmm_invariants.py` | W50–W55 — HMM multiplier envelope [0.2, 1.25], one-hot label weights, same-seed determinism, degenerate-input guards |
| `test_skew_dynamics_invariants.py` | W56–W59 — Nelson-Siegel unfit raise-fast, degenerate-fit fallbacks, skew-momentum empty-history, dislocation bounds |
| `test_dealer_positioning_invariants.py` | W60–W62 — multiplier clamp [0.70, 1.05] for any confidence, regime boundaries, flip-distance consistency |
| `tests/quant_benchmarks.py` | Shared benchmark module — **not a `test_*` file**; imported by the suite |

### Property + numerical + point-in-time

| File | Purpose |
|---|---|
| `test_properties.py` | Hypothesis-based invariants |
| `test_extreme_numerics.py` | Boundary-value behaviour |
| `test_edge_cases.py` | Edge cases across modules |
| `test_point_in_time.py` | No lookahead bias (PIT) |
| `test_pit_leaks.py` | S10/S11 PIT-leak regressions — historical `as_of` never surfaces future-dated news/credit data |

### Ranker & EV-path surface

| File | Purpose |
|---|---|
| `test_covered_call_ranker.py` | `rank_covered_calls_by_ev` — grid enumeration, negative-EV never surfaces, event lockout, ex-dividend plumbing, drop accounting |
| `test_strangle_ev_ranker.py` | `rank_strangles_by_ev` — two evaluate calls per candidate, additive EV composition, timing gate downgrade-only, never rescues |
| `test_strangle_recommendation_gate.py` | S14 phase/confidence gate — downgrade-only `_apply_phase_gate` on both Layer-1 and IV paths |
| `test_ranker_iv_pit.py` | S23 F3 — ranker uses PIT IV from `get_iv_history`, not snapshot fundamentals; symmetric on CC + strangle paths |
| `test_ranker_transparency.py` | Drop-reason `.attrs["drops"]`, `hmm_regime` label, `ev_raw` + `regime_multiplier` columns, GICS sector, zero-extra-evaluate invariant |
| `test_explore_ticker.py` | `explore_ticker` delta×DTE grid sweep — shape, columns, sorting, drops |
| `test_ev_engine_percentiles.py` | `EVResult.pnl_p25/p50/p75` — monotone, median match, pre-multiplier, NaN on small samples / lockout (#248) |
| `test_prob_profit_ci.py` | Wilson 95% CI small-sample honesty — `n_scenarios` + CI fields, gated to the IID `empirical_non_overlapping` tier (#317) |
| `test_wheel_runner_select_book.py` | `select_book` capital-constrained knapsack selection |
| `test_wheel_runner_coverage.py` | `WheelRunner` edge-branch coverage lift (D10) |
| `test_diagnostic_column_honesty.py` | S28/S29 — `expected_dividend` mirrors the EV gate; `skew_source` provenance ("chain" vs "unavailable") |

### Data pipeline

| File | Purpose |
|---|---|
| `test_bloomberg_loader.py` | Bloomberg CSV loader |
| `test_theta_connector.py` | Theta v3 connector |
| `test_data_pipeline.py` | End-to-end pipeline |
| `test_data_validation.py` | Schema + quality checks |
| `test_data_integration.py` | Provider selection + integration |
| `test_features.py` | Feature store |
| `test_data_connector.py` | `MarketDataConnector` full query surface on synthetic tmp_path CSVs — present/absent/edge branches |
| `test_data_connector_ticker_filter.py` | `_filter_ticker` cache equivalence vs naive mask — build/reuse verified |
| `test_data_quality.py` | `data/quality.py` chain gate — IV substring-match false-positive regression + real invalid-IV detection |
| `test_data_integrity_bloomberg.py` | Phase-2A contract on the committed Bloomberg CSVs — NaN/duplicate/GICS/frontier checks |
| `test_data_to_engine.py` | Phase-2B — real CSVs through both rankers; cascade source, drop accounting, thin/garbage degradation |
| `test_deep_iv_sentinel.py` | R7 — deep-IV `134217.7` corruption sentinel nulled above floor; real distressed extremes preserved |
| `test_deep_read_assembly_synthetic.py` | R2 deep-read assembly on synthetic gz fixtures — dedup precedence, multi-slice concat, default-OFF gate (CI-runnable) |
| `test_deep_read_connector.py` | R2 deep-read flag plumbing + graceful degrade; 1994-assembly/delisted checks local-only (`SWE_DEEP_TEST_DATA`) |
| `test_survivorship_harness.py` | R3+R6 PIT universe — delisted names included/excluded correctly (deep-data gated) |
| `test_survivorship_r6_lehman.py` | R6 proof — Lehman delisting realizes the loss at delisting price in a 2008 backtest (deep-data gated) |
| `test_mark_to_market_iv.py` | #118 P4 — MTM IV staleness fallback chain (explicit → connector as-of ATM → entry IV) |
| `test_iv_surface_failloud.py` | D9/A2 — `SurfaceDataUnavailable` + `require_surface` fail-loud SVI contract; no silent flat IV |
| `test_preflight_environment.py` | Environment-invariant guard — silent provider selection + stale-tree OHLCV frontier (`EXPECTED_FRONTIER`) (#364) |
| `test_theta_connector_v3.py` | Theta v3 HTTP-mocked surface (~87 tests) incl. `PerEndpointFailure` contamination contract (D11) |
| `test_theta_connector_coverage.py` | Theta connector edge branches — cache eviction, padding, VIX-family fallbacks, surface edge cases (D10 lift) |

### External-data adapters

| File | Purpose |
|---|---|
| `test_external_data_cboe.py` | CBOE adapter — VIX/VVIX/SKEW, term structure, contango; HTTP-mocked, cache + 4xx/5xx resilience |
| `test_external_data_edgar.py` | EDGAR adapter — CIK lookup, Form-4, 8-K Item 2.02 earnings history + PIT-correct projection (#251) |
| `test_external_data_fred.py` | FRED adapter — treasury yields, HY-OAS credit regime, curve inversion; CSV + API paths |
| `test_external_data_yfinance.py` | yfinance adapter — OHLCV CSV parsing, latest close, caching, error handling |

### Theta pullers

| File | Purpose |
|---|---|
| `test_iv_surface_history_puller.py` | IV-surface history puller — historical endpoint, shared connector + expirations cache, per-bucket 472 fallback (#55/#58/#59) |
| `test_option_history_puller.py` | Option-history puller — 2016 history-floor clamp, atomic tmp→rename partition writes |
| `test_theta_indices_puller.py` | Indices puller — 365-day chunking, in-order concat, tier-gate messages, incremental up-to-date reporting |

### Wheel lifecycle

| File | Purpose |
|---|---|
| `test_wheel_lifecycle.py` | State transitions + cycle accounting |
| `test_wheel_backtest.py` | Backtest harness |
| `test_portfolio_tracker.py` | Portfolio bookkeeping |
| `test_available_buying_power.py` | `available_buying_power` — CSP collateral reservation across the SHORT_PUT→STOCK_OWNED→COVERED_CALL lifecycle |
| `test_wheel_tracker_persistence.py` | Tracker save/load round-trip incl. the D16 persisted-token consume |
| `test_wheel_tracker_suggest_rolls.py` | `suggest_rolls` — candidate generation, EV gating, per-row evaluate |
| `test_wheel_tracker_suggest_call_rolls.py` | `suggest_call_rolls` — covered-call leg roll suggestions |
| `test_suggest_rolls_drops.py` | S22 F1 — per-candidate drop logging `.attrs["drops"]` schema; survivors unchanged |
| `test_suggest_rolls_defensive.py` | S47 F1 — `include_defensive` surfaces debit rolls; `.attrs["defensive"]`; §2 path untouched |
| `test_event_calendar.py` | `event_calendar` — MarketEvent queries, FOMC/CPI/NFP loaders, risk filter, JSON ingestion + staleness |
| `test_event_gate.py` | `event_gate` hard lockouts — earnings/macro/dividend buffers, ticker matching, Bloomberg-calendar ingestion (NaT regression) |
| `test_event_gate_back_buffer.py` | S23 F1 — symmetric post-earnings back-buffer block via `get_recent_earnings` |

### IBKR live book (D24/D26 — read-only, observational)

| File | Purpose |
|---|---|
| `test_ibkr_portfolio_adapter.py` | D24 snapshot→engine-types fidelity, universe filter, FX/NAV, R9/R10 on adapter-built context, no-trio-import guard |
| `test_portfolio_api_endpoints.py` | D26 `GET /api/portfolio/*` shapes + the observational guard (no verdict / `ev_dollars` ever emitted) |
| `test_ibkr_import.py` | PortfolioAnalyst PDF importer — OCC parsing, p6 positions, FX derivation, null-safety |
| `test_ibkr_flex_ledger.py` | Phase-4 exact-fill ledger — long/short stock round-trips, ACAT seed, dedup, FX builder |
| `test_ibkr_live_snapshot.py` | Live-connector snapshot builder — contract-description parsing, FX normalization, `schema_version: 1` |
| `test_ibkr_gateway_pull.py` | Headless IB Gateway puller — description synthesis, shared-parser round-trip losslessness |
| `test_ibkr_ev_calibration.py` | Phase-3 calibration stats — Wilson CI / Brier / ECE math + universe loader |

### TradingView / dossier / API

| File | Purpose |
|---|---|
| `test_tv_signals.py` | Pine signal parity |
| `test_tv_api.py` | `/api/tv/*` endpoints |
| `test_tv_dossier.py` | Mode-B dossier (audit-iv) |
| `test_dossier_invariant.py` | Dossier contract (also under launch-blockers) |
| `test_tv_dossier_d17_wire.py` | D17 live-wire on `/api/tv/dossier` + `/api/tv/enrich` — nav/holdings params → `PortfolioContext` → R7–R10 |
| `test_tv_nonce_register_lock.py` | Webhook nonce-replay register thread-safety |
| `test_engine_api_port.py` | `_resolve_port()` — `SWE_API_PORT` override, 8787 default, bounds + whitespace (D15/C7) |
| `test_engine_api_hardening.py` | API hardening R3/R18–R21 — CORS, 400 on malformed params, no-exception-leak + correlation id, 404 semantics |
| `test_engine_api_concentration.py` | `/api/concentration_preview` — armed R9/R10 caps on the live path, refuse-only contract, unmocked gate math (#351) |
| `test_mcp_client.py` | `MCPCLIClient` tv-CLI transport — five-call capture, canonical `MCP_ERROR_MODES`, no-retry-except-quote (D12; all subprocess-mocked) |

### News / advisors / ML

| File | Purpose |
|---|---|
| `test_financial_news.py` | `financial_news/` platform |
| `test_news_pipeline.py` | `news_pipeline/` browser pipeline |
| `test_news_processing.py` | News processing primitives |
| `test_adversarial_news.py` | Adversarial robustness |
| `test_advisors.py` | Buffett/Munger/Simons/Taleb committee |
| `test_new_modules.py` | Modules added in recent audits |
| `test_audit_improvements.py` | Audit-line improvements |
| `test_ev_engine_upgrades.py` | EV engine specific upgrades |
| `test_news_sentiment.py` | `NewsSentimentReader` — store reads, staleness, neutral default; `sentiment_multiplier` constant-1.0 parity |
| `test_news_severance.py` | D18 invariant — `sentiment_multiplier` is a constant-1.0 stub across the full (sentiment, n_articles) grid |
| `test_recovery_checkpoints.py` | news-pipeline checkpoints — stage ordering, progress tracking, serialization round-trip |
| `test_recovery_fallbacks.py` | news-pipeline degraded modes — NORMAL/PARTIAL/LOCAL_ONLY/OFFLINE evaluation |
| `test_recovery_health.py` | news-pipeline provider health — availability, rate-limit expiry, success-rate tracking |

### Infrastructure

| File | Purpose |
|---|---|
| `test_infrastructure.py` | Repo-level infra |
| `test_contracts.py` | Dataclass contracts |
| `test_dashboard.py` | Legacy dashboard CLI surface |
| `test_signals.py` | Signal aggregator framework |
| `test_strangle_timing.py` | Strangle entry timing gate |
| `test_check_lane_claim.py` | The decision-layer lane-claim CI gate (`scripts/check_lane_claim.py`) |
| `test_check_manifest_coverage.py` | The FILE_MANIFEST coverage gate's conflict-marker detection |
| `test_testing_md_taxonomy.py` | This file's taxonomy stays complete — every `tests/test_*.py` must be named in TESTING.md |
| `test_observability.py` | `engine/observability` — TraceContext, DecisionJournal, AuditLogger, trace decorator |
| `test_policy_config.py` | `engine/policy_config` — load/save/validate, default sanity, section schema |
| `test_trade_memo_ci.py` | Memo honesty — prob_profit rendered with Wilson CI + N + small-sample caveat |

## Running tests

```bash
# Full suite
pytest tests/ -v

# Decision-layer launch blockers only (fast subset)
pytest tests/test_audit_invariants.py \
       tests/test_dossier_invariant.py \
       tests/test_r11_elevated_vol.py \
       tests/test_authority_hardening.py \
       tests/test_audit_viii_unit_invariants.py \
       tests/test_audit_viii_e2e.py \
       tests/test_audit_viii_real_data_smoke.py \
       tests/test_launch_blockers.py -v

# Skip integration + slow
pytest tests/ -m "not integration and not slow" -v

# Coverage (CI scope per .github/workflows/ci.yml; threshold 80%)
pytest tests/ --cov=src --cov=engine --cov=advisors --cov=financial_news \
       --cov=data --cov-fail-under=80

# Hypothesis profiles (configured in conftest.py)
pytest tests/ --hypothesis-profile=ci      # 200 examples (CI)
pytest tests/ --hypothesis-profile=dev     # 50 examples (default-ish)
pytest tests/ --hypothesis-profile=debug   # 10 examples, verbose

# Quant-validation marker
pytest tests/ -m quant -v
```

## Markers (`conftest.py`)

| Marker | Use |
|---|---|
| `@pytest.mark.integration` | Requires external services (Theta Terminal, Ollama, browser sessions). Skip in CI. |
| `@pytest.mark.slow` | Long-running. Deselect with `-m "not slow"`. |
| `@pytest.mark.quant` | Quantitative validation tests. |
| `@pytest.mark.backtest_regression` | Long-running ledger-backtest reproducers (S27/S32/S34/S35). Excluded from per-PR CI; run via `.claude/commands/backtest-regression.md` or the `Backtest Regression` workflow. |

## What to run when you change ___

| You touched | Run at minimum |
|---|---|
| `engine/ev_engine.py` | `pytest tests/test_audit_invariants.py tests/test_audit_viii_*.py tests/test_ev_engine_upgrades.py` then **the full suite** (invariants are cross-cutting) |
| `engine/wheel_runner.py` | Full suite |
| `engine/candidate_dossier.py` | `pytest tests/test_dossier_invariant.py tests/test_r11_elevated_vol.py tests/test_tv_dossier.py tests/test_authority_hardening.py` |
| `engine/option_pricer.py` | `pytest tests/test_option_pricer.py tests/test_greeks_unit_invariants.py tests/test_properties.py` |
| `engine/data_connector.py` or `theta_connector.py` | `pytest tests/test_bloomberg_loader.py tests/test_theta_connector.py tests/test_data_pipeline.py` then `python scripts/theta_health_check.py` if Terminal is up |
| `engine/dealer_positioning.py` | `pytest tests/test_dealer_positioning.py tests/test_audit_invariants.py` |
| `engine/regime_detector.py` or `regime_hmm.py` | `pytest tests/test_regime_detector.py tests/test_audit_viii_e2e.py::test_hmm_cache_reuse` |
| `engine/wheel_tracker.py` | `pytest tests/test_wheel_lifecycle.py tests/test_audit_viii_unit_invariants.py` (the audit-VIII tests pin the rolled-P&L accumulator) |
| `advisors/*` | `pytest tests/test_advisors.py tests/test_authority_hardening.py` |
| `engine_api.py` | `pytest tests/test_tv_api.py tests/test_tv_dossier.py tests/test_audit_viii_e2e.py` then `python scripts/audit_api_smoke.py` against a running `engine_api.py` |
| `financial_news/` or `news_pipeline/` | `pytest tests/test_financial_news.py tests/test_news_pipeline.py tests/test_news_processing.py tests/test_adversarial_news.py` |
| `engine/ev_engine.py`, `engine/wheel_runner.py`, `engine/forward_distribution.py`, `engine/dealer_positioning.py`, `engine/tail_risk.py` | **Backtest regression** in addition to the launch blockers — run `.claude/commands/backtest-regression.md` (~4–5 h). The four S27/S32/S34/S35 backtests are downstream of all five files. |

## Backtest regression — re-baseline workflow

When `pytest tests/test_backtest_regression.py -m backtest_regression`
fails, the response is **diagnose first, re-baseline second**:

1. **Do not regenerate snapshots reflexively.** A failure is a signal,
   not a chore.
2. `git log --oneline engine/ since <last successful snapshot date>` —
   identify the candidate PR. The snapshot fingerprint records
   `engine_sha_at_snapshot_lock`; compare against current `HEAD`.
3. Read the offending PR description and `CHANGELOG.md`. Is the engine
   change **deliberate** (a real methodology improvement) or
   **accidental** (a refactor that should have been numerically
   invariant)?
4. **Deliberate:** regenerate via
   `python -m backtests.regression.<id> --update-snapshot`, amend the
   relevant `docs/ENGINE_BACKTEST_*.md` with a `## Rebased <date>`
   section preserving the original numbers, file the snapshot-update
   PR linking back to the engine PR that caused the drift.
5. **Accidental:** revert the engine PR. The harness has done its job.

**Data-drift guards (fast — run in the normal per-PR lane, NOT behind the
`backtest_regression` marker):**

- `test_snapshot_fingerprints_have_required_keys` asserts every snapshot
  carries the required fingerprint keys, including `connector_data_sha256`
  (the dict of per-file SHAs for *every* connector input — #340).
- `test_snapshot_data_fingerprint_matches_current` asserts each snapshot's
  pinned `connector_data_sha256` still matches the **current committed data**.
  It fails fast the moment ANY connector input CSV drifts from what the
  snapshot was generated against — so a data refresh forces an explicit
  re-baseline instead of a silent drift caught only by a multi-hour marker run.
  (The 2026-06-06 `sp500_dividends.csv` revert-after-generation went undetected
  until a 3.5 h S34 run failed, because the legacy fingerprint pinned only
  OHLCV/vol_iv/treasury and nothing compared a snapshot's pins to the live
  data.) **This guard is EXPECTED to fail on every legitimate data change** —
  that is the signal to regenerate the four snapshots (step 4 above) and
  re-pin, not a regression to investigate.

The legacy scalar `data_csv_sha256` / `vol_iv_sha256` / `treasury_sha256`
fields remain for back-compat; `connector_data_sha256` supersedes them by
pinning the full connector set.

## Sandbox notes

Sandbox-vs-laptop capability differences (pip-install chunking, the
`pyarrow` failure mode, why full-universe `diagnose_candidates.py`
needs an explicit 5-ticker list in Cowork) live in
`docs/DATA_POLICY.md` §7 as the canonical reference. The
5-ticker shim itself is the bring-up smoke test in `CLAUDE.md`.

## CI

`.github/workflows/ci.yml` runs on push to `main` / `develop` and on
PRs (it `pip install -e ".[dev]"`). CI jobs include the lane-claim gate,
FILE_MANIFEST coverage, lint, security scan, the 3.11/3.12 test suites,
quantitative validation, and integration tests. (The old broken
`wheel = "src.cli:app"` console-script was removed under ROADMAP B5 — no
`[project.scripts]` table exists today.)
