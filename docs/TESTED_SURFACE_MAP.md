# Tested-surface map

_Generated 2026-05-29 from `coverage.json` (suite timestamp `2026-05-28T23:51:11.986500`) by `scripts/generate_tested_surface_map.py`. Regenerate after a meaningful coverage shift._

This file answers _what is and isn't covered by the test suite_ at a module granularity. The numbers come from coverage.py's branch-aware report; the module → test mapping is a static import grep of `tests/test_*.py` (not a runtime trace), so a test file is listed if it imports the module — not necessarily if it exercises every line.

**CI scope** (per `pyproject.toml [tool.coverage.run]`):
`src` · `engine` · `advisors` · `financial_news` · `data`.
Modules listed in `[tool.coverage.run] omit` (research-tier ETL, Ollama-dependent memo generator, UI, etc.) are excluded by design — see `DECISIONS.md` D10 for the rationale on the 80% floor.

## Suite totals

| Metric | Value |
|---|---|
| Total statements (CI scope) | 14,170 |
| Covered statements | 12,239 |
| Missing statements | 1,931 |
| Excluded statements | 20 |
| Total branches | 4,416 |
| Covered branches | 3,343 |
| Partial branches | 695 |
| Missing branches | 1,073 |
| **Suite % covered** | **83.8%** |
| Files in scope | 82 |

## Top 15 coverage gaps

Ranked by **uncovered statements** (raw count). These are where additional tests would buy the most coverage; review the untested-function column before adding a test to confirm the gap is on a path that warrants exercise rather than an `omit`-candidate research module.

| Rank | Module | Stmts | Missed | % | Notable untested |
|---:|---|---:|---:|---:|---|
| 1 | `engine/wheel_runner.py` | 1,168 | 267 | 75.3% | `WheelRunner.rank_candidates_by_ev` (L719, 86/1069); `WheelRunner.rank_strangles_by_ev` (L2603, 50/585); `WheelRunner.rank_covered_calls_by_ev` (L2068, 48/531); … +4 more |
| 2 | `engine/wheel_tracker.py` | 772 | 106 | 83.3% | `WheelTracker.suggest_rolls` (L2043, 19/423); `WheelTracker.suggest_call_rolls` (L2467, 19/359); `WheelTracker._compute_live_nav` (L1714, 14/53); … +2 more |
| 3 | `src/features/assignment.py` | 182 | 103 | 39.2% | `AssignmentFeatures.compute_for_chain` (L511, 26/80); `AssignmentFeatures._vectorized_prob_touch` (L403, 20/38); `AssignmentFeatures.roll_vs_assignment_score` (L238, 18/59); … +2 more |
| 4 | `src/features/labels.py` | 152 | 92 | 33.7% | `LabelGenerator.csp_outcome` (L64, 25/68); `LabelGenerator.generate_training_labels` (L403, 20/60); `LabelGenerator.multi_class_outcome` (L360, 5/15) |
| 5 | `engine/risk_manager.py` | 703 | 86 | 83.5% | `RiskManager.calculate_position_size` (L195, 16/104); `RiskManager.calculate_monte_carlo_var` (L898, 15/259); `RiskManager._get_drawdown_scalar` (L300, 9/19); … +1 more |
| 6 | `src/features/regime.py` | 125 | 83 | 31.1% | `RegimeDetector.trend_regime` (L67, 15/47); `RegimeDetector.compute_all` (L384, 13/52); `RegimeDetector.vol_regime` (L140, 11/36); … +5 more |
| 7 | `src/features/events.py` | 112 | 81 | 25.4% | `EventVolatility.surprise_direction_streak` (L285, 13/26); `EventVolatility.compute_earnings_features` (L330, 11/51); `EventVolatility.days_since_event` (L71, 10/40) |
| 8 | `src/features/technical.py` | 171 | 76 | 50.2% | `TechnicalFeatures.hurst_exponent` (L308, 34/66); `TechnicalFeatures.calc_hurst` (L328, 32/44); `TechnicalFeatures.compute_all` (L375, 24/68) |
| 9 | `data/quality.py` | 384 | 76 | 76.1% | `DataQualityFramework.check_freshness` (L861, 15/39); `DataQualityFramework.validate_contract` (L901, 4/12) |
| 10 | `engine/portfolio_tracker.py` | 524 | 66 | 83.0% | `PortfolioTracker._process_option` (L720, 14/31); `PortfolioTracker._calculate_benchmark_return` (L1038, 13/21); `PortfolioTracker._process_sell` (L617, 11/77); … +1 more |
| 11 | `engine/option_pricer.py` | 425 | 62 | 81.7% | `vectorized_bs_delta` (L1086, 25/48) |
| 12 | `src/backtest/wheel_backtest.py` | 234 | 60 | 73.1% | `run_backtest` (L521, 28/49); `WheelBacktest._process_expirations` (L215, 10/48); `WheelBacktest._try_sell_covered_call` (L363, 10/31) |
| 13 | `advisors/committee.py` | 435 | 56 | 82.3% | `CommitteeEngine._advisor_post_mortem` (L731, 18/104); `CommitteeEngine._advisor_portfolio_review` (L509, 14/120) |
| 14 | `src/features/vol_edge.py` | 79 | 53 | 29.9% | `VolatilityEdge.compute_all` (L267, 16/52); `VolatilityEdge.percentile_rank` (L108, 6/7) |
| 15 | `advisors/scorecard.py` | 267 | 53 | 74.2% | `AdvisorScorecard._calculate_metrics` (L462, 15/74); `AdvisorScorecard._load_data` (L574, 12/17); `AdvisorScorecard._save_data` (L592, 9/15) |

## Per-module coverage

One row per CI-scope file. The **Tests** column lists `tests/` files that statically import the module; this is a coverage proxy (static import) not a runtime exercise trace.

### `advisors/`

| Module | Stmts | % | Notable untested | Tests |
|---|---:|---:|---|---|
| `advisors/__init__.py` | 12 | 83.3% | — | `tests/test_advisors.py` |
| `advisors/base.py` | 98 | 80.3% | — | — |
| `advisors/buffett.py` | 115 | 83.2% | `BuffettAdvisor._analyze` (L82, 16/214) | — |
| `advisors/committee.py` | 435 | 82.3% | `CommitteeEngine._advisor_post_mortem` (L731, 18/104); `CommitteeEngine._advisor_portfolio_review` (L509, 14/120) | `tests/test_new_modules.py` |
| `advisors/integration.py` | 48 | 78.3% | — | — |
| `advisors/munger.py` | 102 | 88.2% | — | — |
| `advisors/schema.py` | 237 | 100.0% | — | `tests/test_new_modules.py` |
| `advisors/scorecard.py` | 267 | 74.2% | `AdvisorScorecard._calculate_metrics` (L462, 15/74); `AdvisorScorecard._load_data` (L574, 12/17); … +1 more | `tests/test_infrastructure.py` |
| `advisors/simons.py` | 134 | 83.2% | `SimonsAdvisor._analyze` (L85, 15/293) | — |
| `advisors/taleb.py` | 116 | 83.3% | `TalebAdvisor._analyze` (L106, 14/311) | `tests/test_new_modules.py` |

### `data/`

| Module | Stmts | % | Notable untested | Tests |
|---|---:|---:|---|---|
| `data/quality.py` | 384 | 76.1% | `DataQualityFramework.check_freshness` (L861, 15/39); `DataQualityFramework.validate_contract` (L901, 4/12) | `tests/test_audit_improvements.py`, `tests/test_data_pipeline.py`, `tests/test_data_quality.py` |

### `engine/`

| Module | Stmts | % | Notable untested | Tests |
|---|---:|---:|---|---|
| `engine/__init__.py` | 19 | 100.0% | — | `tests/test_audit_improvements.py`, `tests/test_audit_invariants.py`, `tests/test_authority_hardening.py`, … +3 more |
| `engine/binomial_tree.py` | 156 | 84.6% | — | `tests/test_binomial_tree.py` |
| `engine/candidate_dossier.py` | 189 | 91.4% | — | `tests/test_dealer_positioning.py`, `tests/test_decision_layer_wiring.py`, `tests/test_dossier_cp1252.py`, … +4 more |
| `engine/chart_context.py` | 24 | 96.2% | — | `tests/test_dealer_positioning.py`, `tests/test_decision_layer_wiring.py`, `tests/test_dossier_cp1252.py`, … +4 more |
| `engine/contracts.py` | 75 | 93.6% | — | `tests/test_contracts.py` |
| `engine/data_connector.py` | 299 | 93.1% | `MarketDataConnector._to_ts` (L134, 3/5) | `tests/test_audit_viii_e2e.py`, `tests/test_audit_viii_real_data_smoke.py`, `tests/test_backtest_regression.py`, … +4 more |
| `engine/data_integration.py` | 127 | 92.0% | — | `tests/test_data_integration.py`, `tests/test_new_modules.py` |
| `engine/dealer_positioning.py` | 266 | 89.4% | — | `tests/test_dealer_multiplier_evengine_integration.py`, `tests/test_dealer_positioning.py`, `tests/test_evengine_event_lockout.py` |
| `engine/earnings_drift.py` | 141 | 88.1% | — | `tests/test_earnings_drift.py` |
| `engine/ev_engine.py` | 214 | 96.2% | — | `tests/test_audit_invariants.py`, `tests/test_covered_call_ranker.py`, `tests/test_dealer_multiplier_evengine_integration.py`, … +9 more |
| `engine/event_calendar.py` | 368 | 87.9% | `build_default_calendar` (L947, 13/88) | `tests/test_event_calendar.py` |
| `engine/event_gate.py` | 103 | 97.2% | — | `tests/test_evengine_event_lockout.py`, `tests/test_event_gate.py`, `tests/test_quant_upgrades.py` |
| `engine/external_data/__init__.py` | 5 | 100.0% | — | — |
| `engine/external_data/cboe_adapter.py` | 59 | 97.0% | — | `tests/test_external_data_cboe.py` |
| `engine/external_data/edgar_adapter.py` | 65 | 97.5% | — | `tests/test_external_data_edgar.py` |
| `engine/external_data/fred_adapter.py` | 87 | 98.2% | — | `tests/test_external_data_fred.py`, `tests/test_pit_leaks.py` |
| `engine/external_data/yfinance_adapter.py` | 66 | 97.5% | — | `tests/test_external_data_yfinance.py` |
| `engine/forward_distribution.py` | 156 | 88.2% | — | `tests/test_audit_improvements.py`, `tests/test_f4_rv_widening.py`, `tests/test_f4_tail_risk_gap.py` |
| `engine/mcp_client.py` | 112 | 98.7% | — | `tests/test_mcp_client.py` |
| `engine/model_validation.py` | 170 | 80.9% | `CrossModelValidator.validate` (L170, 10/114) | `tests/test_binomial_tree.py` |
| `engine/monte_carlo.py` | 331 | 95.0% | — | `tests/test_monte_carlo.py` |
| `engine/news_sentiment.py` | 92 | 94.5% | — | `tests/test_news_sentiment.py`, `tests/test_pit_leaks.py` |
| `engine/observability.py` | 99 | 98.3% | — | `tests/test_observability.py` |
| `engine/option_pricer.py` | 425 | 81.7% | `vectorized_bs_delta` (L1086, 25/48) | `tests/test_advanced_quant.py`, `tests/test_binomial_tree.py`, `tests/test_edge_cases.py`, … +5 more |
| `engine/payoff_engine.py` | 131 | 98.8% | — | `tests/test_payoff_engine.py` |
| `engine/policy_config.py` | 93 | 100.0% | — | `tests/test_policy_config.py` |
| `engine/portfolio_copula.py` | 83 | 100.0% | — | `tests/test_portfolio_copula_coverage.py`, `tests/test_quant_upgrades.py` |
| `engine/portfolio_risk_gates.py` | 179 | 98.0% | — | `tests/test_authority_hardening.py`, `tests/test_decision_layer_wiring.py`, `tests/test_dossier_invariant.py`, … +4 more |
| `engine/portfolio_tracker.py` | 524 | 83.0% | `PortfolioTracker._process_option` (L720, 14/31); `PortfolioTracker._calculate_benchmark_return` (L1038, 13/21); … +2 more | `tests/test_portfolio_tracker.py` |
| `engine/realized_vol.py` | 64 | 98.7% | — | `tests/test_realized_vol.py` |
| `engine/regime_detector.py` | 212 | 84.2% | `RegimeDetector.get_strategy_adjustments` (L393, 17/55) | `tests/test_regime_detector.py` |
| `engine/regime_hmm.py` | 157 | 92.9% | — | `tests/test_quant_upgrades.py`, `tests/test_ranker_transparency.py` |
| `engine/risk_manager.py` | 703 | 83.5% | `RiskManager.calculate_position_size` (L195, 16/104); `RiskManager.calculate_monte_carlo_var` (L898, 15/259); … +2 more | `tests/test_advanced_quant.py`, `tests/test_properties.py`, `tests/test_quant_fixtures.py`, … +3 more |
| `engine/signals.py` | 339 | 82.4% | — | `tests/test_signals.py`, `tests/test_strangle_timing.py` |
| `engine/skew_dynamics.py` | 84 | 89.1% | — | `tests/test_quant_upgrades.py` |
| `engine/strangle_timing.py` | 346 | 99.0% | — | `tests/test_strangle_ev_ranker.py`, `tests/test_strangle_recommendation_gate.py`, `tests/test_strangle_timing.py` |
| `engine/stress_testing.py` | 341 | 93.8% | `StressTester.from_policy` (L254, 3/6) | `tests/test_extreme_numerics.py`, `tests/test_greeks_unit_invariants.py`, `tests/test_launch_blockers.py`, … +2 more |
| `engine/tail_risk.py` | 86 | 87.7% | `fit_gpd_tail` (L82, 11/79) | `tests/test_quant_upgrades.py`, `tests/test_tail_risk.py` |
| `engine/theta_connector.py` | 511 | 97.3% | — | `tests/test_theta_connector.py`, `tests/test_theta_connector_coverage.py`, `tests/test_theta_connector_v3.py` |
| `engine/tradingview_bridge.py` | 148 | 92.1% | — | `tests/test_dossier_invariant.py`, `tests/test_mcp_client.py`, `tests/test_tv_dossier.py` |
| `engine/transaction_costs.py` | 79 | 100.0% | — | `tests/test_audit_improvements.py`, `tests/test_edge_cases.py` |
| `engine/tv_signals.py` | 255 | 86.9% | `_rsi_state` (L193, 4/10) | `tests/test_audit_viii_e2e.py`, `tests/test_authority_hardening.py`, `tests/test_ev_non_finite_defense.py`, … +2 more |
| `engine/wheel_runner.py` | 1,168 | 75.3% | `WheelRunner.rank_candidates_by_ev` (L719, 86/1069); `WheelRunner.rank_strangles_by_ev` (L2603, 50/585); … +5 more | `tests/test_audit_improvements.py`, `tests/test_audit_viii_e2e.py`, `tests/test_audit_viii_real_data_smoke.py`, … +24 more |
| `engine/wheel_tracker.py` | 772 | 83.3% | `WheelTracker.suggest_rolls` (L2043, 19/423); `WheelTracker.suggest_call_rolls` (L2467, 19/359); … +3 more | `tests/test_audit_viii_e2e.py`, `tests/test_audit_viii_unit_invariants.py`, `tests/test_authority_hardening.py`, … +15 more |

### `financial_news/`

| Module | Stmts | % | Notable untested | Tests |
|---|---:|---:|---|---|
| `financial_news/__init__.py` | 6 | 100.0% | — | `tests/test_financial_news.py` |
| `financial_news/calendar/__init__.py` | 2 | 100.0% | — | — |
| `financial_news/calendar/macro_calendar.py` | 114 | 66.2% | `MacroCalendar.get_tomorrow_calendar_summary` (L449, 10/15); `MacroCalendar.get_today_calendar_summary` (L434, 9/14); … +3 more | — |
| `financial_news/models.py` | 151 | 91.4% | — | — |
| `financial_news/processing/__init__.py` | 8 | 100.0% | — | — |
| `financial_news/processing/classifier.py` | 99 | 98.1% | — | `tests/test_news_processing.py` |
| `financial_news/schema.py` | 271 | 92.7% | `RunLog.duration_seconds` (L651, 3/4) | `tests/test_news_processing.py` |
| `financial_news/utils/__init__.py` | 0 | 100.0% | — | — |
| `financial_news/verification_engine.py` | 196 | 76.6% | `print_verification_queries` (L666, 15/20); `print_stats` (L688, 11/15) | — |

### `src/`

| Module | Stmts | % | Notable untested | Tests |
|---|---:|---:|---|---|
| `src/__init__.py` | 1 | 100.0% | — | — |
| `src/backtest/__init__.py` | 2 | 100.0% | — | — |
| `src/backtest/wheel_backtest.py` | 234 | 73.1% | `run_backtest` (L521, 28/49); `WheelBacktest._process_expirations` (L215, 10/48); … +1 more | `tests/test_wheel_backtest.py` |
| `src/data/__init__.py` | 3 | 100.0% | — | — |
| `src/data/schemas.py` | 147 | 97.4% | — | — |
| `src/execution/__init__.py` | 0 | 100.0% | — | — |
| `src/features/__init__.py` | 10 | 100.0% | — | — |
| `src/features/assignment.py` | 182 | 39.2% | `AssignmentFeatures.compute_for_chain` (L511, 26/80); `AssignmentFeatures._vectorized_prob_touch` (L403, 20/38); … +3 more | `tests/test_audit_invariants.py` |
| `src/features/dynamics.py` | 73 | 49.4% | `OptionsDynamics.compute_all` (L166, 19/57) | `tests/test_features.py` |
| `src/features/events.py` | 112 | 25.4% | `EventVolatility.surprise_direction_streak` (L285, 13/26); `EventVolatility.compute_earnings_features` (L330, 11/51); … +1 more | — |
| `src/features/labels.py` | 152 | 33.7% | `LabelGenerator.csp_outcome` (L64, 25/68); `LabelGenerator.generate_training_labels` (L403, 20/60); … +1 more | `tests/test_point_in_time.py` |
| `src/features/options.py` | 79 | 51.8% | `OptionsFeatures.compute_flow_features` (L228, 13/36) | `tests/test_features.py` |
| `src/features/regime.py` | 125 | 31.1% | `RegimeDetector.trend_regime` (L67, 15/47); `RegimeDetector.compute_all` (L384, 13/52); … +6 more | — |
| `src/features/technical.py` | 171 | 50.2% | `TechnicalFeatures.hurst_exponent` (L308, 34/66); `TechnicalFeatures.calc_hurst` (L328, 32/44); … +1 more | `tests/test_features.py`, `tests/test_point_in_time.py`, `tests/test_properties.py`, … +2 more |
| `src/features/vol_edge.py` | 79 | 29.9% | `VolatilityEdge.compute_all` (L267, 16/52); `VolatilityEdge.percentile_rank` (L108, 6/7) | — |
| `src/features/volatility.py` | 82 | 74.5% | `VolatilityFeatures.compute_all` (L258, 12/46) | `tests/test_features.py`, `tests/test_point_in_time.py`, `tests/test_properties.py`, … +1 more |
| `src/models/__init__.py` | 0 | 100.0% | — | — |
| `src/risk/__init__.py` | 0 | 100.0% | — | — |

## Methodology notes

- **Numbers** come from `coverage.json` (`pytest-cov --cov-report=json`), which itself reflects `pyproject.toml [tool.coverage.run]` — `source`, `omit`, `branch = true`, and `[tool.coverage.report] exclude_lines`.
- **"Notable untested"** functions are functions with ≥3 line body where either ≥33% of body lines or ≥10 raw lines intersect `missing_lines`. Methods are namespaced as `Class.method`. Truncated to the top 2–3 per row by missed count. A `—` means no single function clears the threshold; the row's gap is fragmented across many small misses (still surfaced via the row's `Stmts` / % columns).
- **"Tests"** column is built from `tests/test_*.py` AST imports (`import X` / `from X import Y`). A test file appears if it imports the module — it does not assert exercise. Enabling `coverage.dynamic_context = "test_function"` would give a true runtime mapping; the trade-off is run time (~1.5–2× slower) and a much larger `.coverage` SQLite file. Out of scope for this first artifact.
- **`__init__.py`** files appear when they hold re-exports or logic; pure namespace inits show 100% with a small statement count.
- **Branches** are counted but not enumerated per row — the branch totals at the top of the file give the suite-level branch picture; per-line branch detail lives in `coverage.json[files][...].missing_branches` for anyone wanting a deeper dive.

