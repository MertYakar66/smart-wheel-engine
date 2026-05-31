# Testing — Smart Wheel Engine

The full suite is `pytest tests/ -v` (~2,300 tests).

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
| `tests/test_dossier_invariant.py` | `EnginePhaseReviewer` rules R1–R6; downgrade-only contract; `MCPChartProvider` import-guarded contract test |
| `tests/test_authority_hardening.py` | TV webhook / analyze / strangle / strikes / wheel_tracker route through EV (audit-vi) |
| `tests/test_audit_viii_unit_invariants.py` | IV / risk-free-rate percent↔decimal normalisation; rolled-position P&L accumulator (audit-viii) |
| `tests/test_audit_viii_e2e.py` | Webhook → HMAC → enrich → EV → token chain; HMM cache reuse; OHLCV invariant guard (11 e2e tests) |
| `tests/test_audit_viii_real_data_smoke.py` | Real-Bloomberg smoke: `rank_candidates_by_ev` returns non-empty rows; connector rate is decimal |
| `tests/test_launch_blockers.py` | Top-level launch checks |

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
| `tests/quant_benchmarks.py` | Shared benchmark module — **not a `test_*` file**; imported by the suite |

### Property + numerical + point-in-time

| File | Purpose |
|---|---|
| `test_properties.py` | Hypothesis-based invariants |
| `test_extreme_numerics.py` | Boundary-value behaviour |
| `test_edge_cases.py` | Edge cases across modules |
| `test_point_in_time.py` | No lookahead bias (PIT) |

### Data pipeline

| File | Purpose |
|---|---|
| `test_bloomberg_loader.py` | Bloomberg CSV loader |
| `test_theta_connector.py` | Theta v3 connector |
| `test_data_pipeline.py` | End-to-end pipeline |
| `test_data_validation.py` | Schema + quality checks |
| `test_data_integration.py` | Provider selection + integration |
| `test_features.py` | Feature store |

### Wheel lifecycle

| File | Purpose |
|---|---|
| `test_wheel_lifecycle.py` | State transitions + cycle accounting |
| `test_wheel_backtest.py` | Backtest harness |
| `test_portfolio_tracker.py` | Portfolio bookkeeping |

### TradingView / dossier / API

| File | Purpose |
|---|---|
| `test_tv_signals.py` | Pine signal parity |
| `test_tv_api.py` | `/api/tv/*` endpoints |
| `test_tv_dossier.py` | Mode-B dossier (audit-iv) |
| `test_dossier_invariant.py` | Dossier contract (also under launch-blockers) |

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

### Infrastructure

| File | Purpose |
|---|---|
| `test_infrastructure.py` | Repo-level infra |
| `test_contracts.py` | Dataclass contracts |
| `test_dashboard.py` | Legacy dashboard CLI surface |
| `test_signals.py` | Signal aggregator framework |
| `test_strangle_timing.py` | Strangle entry timing gate |

## Running tests

```bash
# Full suite
pytest tests/ -v

# Decision-layer launch blockers only (fast subset)
pytest tests/test_audit_invariants.py \
       tests/test_dossier_invariant.py \
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
| `engine/candidate_dossier.py` | `pytest tests/test_dossier_invariant.py tests/test_tv_dossier.py tests/test_authority_hardening.py` |
| `engine/option_pricer.py` | `pytest tests/test_option_pricer.py tests/test_greeks_unit_invariants.py tests/test_properties.py` |
| `engine/data_connector.py` or `theta_connector.py` | `pytest tests/test_bloomberg_loader.py tests/test_theta_connector.py tests/test_data_pipeline.py` then `python scripts/theta_health_check.py` if Terminal is up |
| `engine/dealer_positioning.py` | `pytest tests/test_dealer_positioning.py tests/test_audit_invariants.py` |
| `engine/regime_detector.py` or `regime_hmm.py` | `pytest tests/test_regime_detector.py tests/test_audit_viii_e2e.py::test_hmm_cache_reuse` |
| `engine/wheel_tracker.py` | `pytest tests/test_wheel_lifecycle.py tests/test_audit_viii_unit_invariants.py` (the audit-VIII tests pin the rolled-P&L accumulator) |
| `advisors/*` | `pytest tests/test_advisors.py tests/test_authority_hardening.py` |
| `engine_api.py` | `pytest tests/test_tv_api.py tests/test_tv_dossier.py tests/test_audit_viii_e2e.py` then `python audit.py` against a running `engine_api.py` |
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

The `data_csv_sha256` field in each snapshot's fingerprint pins the
Bloomberg CSV; a CSV refresh that changes the SHA forces an explicit
re-baseline rather than a silent drift.

## Sandbox notes

Sandbox-vs-laptop capability differences (pip-install chunking, the
`pyarrow` failure mode, why full-universe `diagnose_candidates.py`
needs an explicit 5-ticker list in Cowork) live in
`docs/DATA_POLICY.md` §7 as the canonical reference. The
5-ticker shim itself is the bring-up smoke test in `CLAUDE.md`.

## CI

`.github/workflows/ci.yml` runs on push to `main` / `develop` and on
PRs. It currently `pip install -e ".[dev]"` — note that pyproject's
`wheel = "src.cli:app"` console-script entry references a missing
file (see `PROJECT_STATE.md` §5).
