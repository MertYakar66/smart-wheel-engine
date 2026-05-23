# Testing — Smart Wheel Engine

The full suite is `pytest tests/ -v` (~2,100 tests).

Markers and hypothesis profiles are wired in `conftest.py`.

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
| `test_wheel_lifecycle.py` | State transitions |
| `test_wheel_cycle.py` | Cycle accounting |
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
| `engine/wheel_tracker.py` | `pytest tests/test_wheel_lifecycle.py tests/test_wheel_cycle.py tests/test_audit_viii_unit_invariants.py` (the audit-VIII tests pin the rolled-P&L accumulator) |
| `advisors/*` | `pytest tests/test_advisors.py tests/test_authority_hardening.py` |
| `engine_api.py` | `pytest tests/test_tv_api.py tests/test_tv_dossier.py tests/test_audit_viii_e2e.py` then `python audit.py` against a running `engine_api.py` |
| `financial_news/` or `news_pipeline/` | `pytest tests/test_financial_news.py tests/test_news_pipeline.py tests/test_news_processing.py tests/test_adversarial_news.py` |

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
