# Tested-surface map

_Generated 2026-09-17 from `coverage.json` (suite timestamp `2026-09-17T07:55:40.458631`) by `scripts/generate_tested_surface_map.py`. Regenerate after a meaningful coverage shift._

This file answers _what is and isn't covered by the test suite_ at a module granularity. The numbers come from coverage.py's branch-aware report; the module → test mapping is a static import grep of `tests/test_*.py` (not a runtime trace), so a test file is listed if it imports the module — not necessarily if it exercises every line.

**CI scope** (per `pyproject.toml [tool.coverage.run]`):
`engine` · `data`.
Modules listed in `[tool.coverage.run] omit` (research-tier ETL, UI, etc.) are excluded by design — see `DECISIONS.md` D10 for the rationale on the 80% floor.

## Suite totals

| Metric | Value |
|---|---|
| Total statements (CI scope) | 12,203 |
| Covered statements | 10,622 |
| Missing statements | 1,581 |
| Excluded statements | 12 |
| Total branches | 3,894 |
| Covered branches | 3,062 |
| Partial branches | 554 |
| Missing branches | 832 |
| **Suite % covered** | **85.0%** |
| Files in scope | 53 |

## Top 15 coverage gaps

Ranked by **uncovered statements** (raw count). These are where additional tests would buy the most coverage; review the untested-function column before adding a test to confirm the gap is on a path that warrants exercise rather than an `omit`-candidate research module.

| Rank | Module | Stmts | Missed | % | Notable untested |
|---:|---|---:|---:|---:|---|
| 1 | `engine/wheel_runner.py` | 1,426 | 280 | 79.7% | `WheelRunner.rank_candidates_by_ev` (L1286, 72/1246); `WheelRunner.rank_strangles_by_ev` (L3458, 56/722); `WheelRunner.rank_covered_calls_by_ev` (L2812, 47/642); … +5 more |
| 2 | `engine/wheel_tracker.py` | 850 | 115 | 83.6% | `WheelTracker.suggest_rolls` (L2343, 19/464); `WheelTracker.suggest_call_rolls` (L2808, 19/376); `WheelTracker._evaluate_d17_hard_blocks` (L2026, 16/158); … +2 more |
| 3 | `engine/features/assignment.py` | 182 | 103 | 39.2% | `AssignmentFeatures.compute_for_chain` (L511, 26/80); `AssignmentFeatures._vectorized_prob_touch` (L403, 20/38); `AssignmentFeatures.roll_vs_assignment_score` (L238, 18/59); … +2 more |
| 4 | `engine/features/labels.py` | 152 | 92 | 33.7% | `LabelGenerator.csp_outcome` (L64, 25/68); `LabelGenerator.generate_training_labels` (L403, 20/60); `LabelGenerator.multi_class_outcome` (L360, 5/15) |
| 5 | `engine/risk_manager.py` | 734 | 86 | 84.2% | `RiskManager.calculate_position_size` (L200, 16/104); `RiskManager.calculate_monte_carlo_var` (L909, 15/259); `RiskManager._get_drawdown_scalar` (L305, 9/19); … +1 more |
| 6 | `engine/features/regime.py` | 125 | 83 | 31.1% | `RegimeDetector.trend_regime` (L67, 15/47); `RegimeDetector.compute_all` (L384, 13/52); `RegimeDetector.vol_regime` (L140, 11/36); … +5 more |
| 7 | `engine/features/events.py` | 112 | 81 | 25.4% | `EventVolatility.surprise_direction_streak` (L285, 13/26); `EventVolatility.compute_earnings_features` (L330, 11/51); `EventVolatility.days_since_event` (L71, 10/40) |
| 8 | `engine/features/technical.py` | 171 | 76 | 50.2% | `TechnicalFeatures.hurst_exponent` (L308, 34/66); `TechnicalFeatures.calc_hurst` (L328, 32/44); `TechnicalFeatures.compute_all` (L375, 24/68) |
| 9 | `data/quality.py` | 384 | 76 | 76.1% | `DataQualityFramework.check_freshness` (L861, 15/39); `DataQualityFramework.validate_contract` (L901, 4/12) |
| 10 | `engine/features/vol_edge.py` | 79 | 53 | 29.9% | `VolatilityEdge.compute_all` (L267, 16/52); `VolatilityEdge.percentile_rank` (L108, 6/7) |
| 11 | `engine/portfolio_tracker.py` | 541 | 53 | 85.8% | `PortfolioTracker._calculate_benchmark_return` (L1071, 13/21); `PortfolioTracker._process_sell` (L618, 11/77); `PortfolioTracker.get_daily_values` (L1176, 4/7) |
| 12 | `engine/data_connector.py` | 631 | 52 | 90.1% | `MarketDataConnector._to_ts` (L429, 3/5) |
| 13 | `engine/option_pricer.py` | 437 | 39 | 86.7% | — |
| 14 | `engine/features/dynamics.py` | 73 | 35 | 49.4% | `OptionsDynamics.compute_all` (L166, 19/57) |
| 15 | `engine/features/options.py` | 79 | 35 | 51.8% | `OptionsFeatures.compute_flow_features` (L228, 13/36) |

## Per-module coverage

One row per CI-scope file. The **Tests** column lists `tests/` files that statically import the module; this is a coverage proxy (static import) not a runtime exercise trace.

### `data/`

| Module | Stmts | % | Notable untested | Tests |
|---|---:|---:|---|---|
| `data/quality.py` | 384 | 76.1% | `DataQualityFramework.check_freshness` (L861, 15/39); `DataQualityFramework.validate_contract` (L901, 4/12) | `tests/test_audit_improvements.py`, `tests/test_data_pipeline.py`, `tests/test_data_quality.py` |
| `data/schemas.py` | 147 | 97.4% | — | — |

### `engine/`

| Module | Stmts | % | Notable untested | Tests |
|---|---:|---:|---|---|
| `engine/__init__.py` | 16 | 100.0% | — | `tests/test_audit_improvements.py`, `tests/test_audit_invariants.py`, `tests/test_authority_hardening.py`, … +8 more |
| `engine/binomial_tree.py` | 156 | 84.6% | — | `tests/test_binomial_tree.py`, `tests/test_pricing_evaluate_invariants.py` |
| `engine/candidate_dossier.py` | 223 | 90.2% | `EnginePhaseReviewer.review` (L297, 15/315) | `tests/test_dealer_positioning.py`, `tests/test_decision_layer_wiring.py`, `tests/test_dossier_cp1252.py`, … +10 more |
| `engine/chart_context.py` | 24 | 96.2% | — | `tests/test_dealer_positioning.py`, `tests/test_decision_layer_wiring.py`, `tests/test_dossier_cp1252.py`, … +8 more |
| `engine/contracts.py` | 79 | 92.2% | — | `tests/test_contracts.py` |
| `engine/data_connector.py` | 631 | 90.1% | `MarketDataConnector._to_ts` (L429, 3/5) | `tests/test_audit_viii_e2e.py`, `tests/test_audit_viii_real_data_smoke.py`, `tests/test_backtest_regression.py`, … +20 more |
| `engine/data_integration.py` | 129 | 92.2% | — | `tests/test_data_integration.py`, `tests/test_data_integrity_bloomberg.py`, `tests/test_w4_risk_free_pit.py` |
| `engine/dealer_positioning.py` | 270 | 89.5% | — | `tests/test_dealer_multiplier_evengine_integration.py`, `tests/test_dealer_positioning.py`, `tests/test_dealer_positioning_invariants.py`, … +1 more |
| `engine/ev_engine.py` | 238 | 95.5% | — | `tests/test_audit_invariants.py`, `tests/test_covered_call_ranker.py`, `tests/test_data_to_engine.py`, … +14 more |
| `engine/event_calendar.py` | 368 | 87.9% | `build_default_calendar` (L947, 13/88) | `tests/test_event_calendar.py` |
| `engine/event_gate.py` | 106 | 97.3% | — | `tests/test_corp_action_gate.py`, `tests/test_ev_engine_percentiles.py`, `tests/test_evengine_event_lockout.py`, … +4 more |
| `engine/external_data/__init__.py` | 5 | 100.0% | — | — |
| `engine/external_data/cboe_adapter.py` | 59 | 97.0% | — | `tests/test_external_data_cboe.py` |
| `engine/external_data/edgar_adapter.py` | 134 | 94.5% | — | `tests/test_external_data_edgar.py` |
| `engine/external_data/fred_adapter.py` | 87 | 98.2% | — | `tests/test_external_data_fred.py`, `tests/test_pit_leaks.py` |
| `engine/external_data/yfinance_adapter.py` | 66 | 97.5% | — | `tests/test_external_data_yfinance.py` |
| `engine/features/__init__.py` | 10 | 100.0% | — | — |
| `engine/features/assignment.py` | 182 | 39.2% | `AssignmentFeatures.compute_for_chain` (L511, 26/80); `AssignmentFeatures._vectorized_prob_touch` (L403, 20/38); … +3 more | `tests/test_audit_invariants.py` |
| `engine/features/dynamics.py` | 73 | 49.4% | `OptionsDynamics.compute_all` (L166, 19/57) | `tests/test_features.py` |
| `engine/features/events.py` | 112 | 25.4% | `EventVolatility.surprise_direction_streak` (L285, 13/26); `EventVolatility.compute_earnings_features` (L330, 11/51); … +1 more | — |
| `engine/features/labels.py` | 152 | 33.7% | `LabelGenerator.csp_outcome` (L64, 25/68); `LabelGenerator.generate_training_labels` (L403, 20/60); … +1 more | `tests/test_point_in_time.py` |
| `engine/features/options.py` | 79 | 51.8% | `OptionsFeatures.compute_flow_features` (L228, 13/36) | `tests/test_features.py` |
| `engine/features/regime.py` | 125 | 31.1% | `RegimeDetector.trend_regime` (L67, 15/47); `RegimeDetector.compute_all` (L384, 13/52); … +6 more | — |
| `engine/features/technical.py` | 171 | 50.2% | `TechnicalFeatures.hurst_exponent` (L308, 34/66); `TechnicalFeatures.calc_hurst` (L328, 32/44); … +1 more | `tests/test_features.py`, `tests/test_point_in_time.py`, `tests/test_properties.py`, … +2 more |
| `engine/features/vol_edge.py` | 79 | 29.9% | `VolatilityEdge.compute_all` (L267, 16/52); `VolatilityEdge.percentile_rank` (L108, 6/7) | — |
| `engine/features/volatility.py` | 82 | 74.5% | `VolatilityFeatures.compute_all` (L258, 12/46) | `tests/test_features.py`, `tests/test_point_in_time.py`, `tests/test_properties.py`, … +1 more |
| `engine/forward_distribution.py` | 167 | 90.5% | — | `tests/test_audit_improvements.py`, `tests/test_f4_rv_widening.py`, `tests/test_f4_tail_risk_gap.py`, … +4 more |
| `engine/ibkr_portfolio_adapter.py` | 405 | 89.9% | `load_trades` (L162, 5/12) | `tests/test_ibkr_history_twr.py`, `tests/test_ibkr_trades.py` |
| `engine/model_validation.py` | 170 | 84.8% | — | `tests/test_binomial_tree.py` |
| `engine/monte_carlo.py` | 331 | 95.0% | — | `tests/test_monte_carlo.py` |
| `engine/option_pricer.py` | 437 | 86.7% | — | `tests/test_advanced_quant.py`, `tests/test_binomial_tree.py`, `tests/test_edge_cases.py`, … +9 more |
| `engine/paper_book.py` | 266 | 84.6% | `_coerce_date` (L487, 4/9); `PaperBookStore.load_forecast_ledger` (L733, 4/5) | — |
| `engine/payoff_engine.py` | 137 | 98.9% | — | `tests/test_payoff_engine.py` |
| `engine/policy_config.py` | 93 | 100.0% | — | `tests/test_policy_config.py` |
| `engine/portfolio_copula.py` | 92 | 100.0% | — | `tests/test_portfolio_copula_coverage.py`, `tests/test_quant_upgrades.py`, `tests/test_tail_copula_stress_invariants.py` |
| `engine/portfolio_risk_gates.py` | 196 | 98.2% | — | `tests/test_authority_hardening.py`, `tests/test_decision_layer_wiring.py`, `tests/test_dossier_downgrade_property.py`, … +6 more |
| `engine/portfolio_tracker.py` | 541 | 85.8% | `PortfolioTracker._calculate_benchmark_return` (L1071, 13/21); `PortfolioTracker._process_sell` (L618, 11/77); … +1 more | `tests/test_portfolio_tracker.py` |
| `engine/realized_vol.py` | 73 | 98.8% | — | `tests/test_forward_distribution_invariants.py`, `tests/test_realized_vol.py` |
| `engine/regime_hmm.py` | 161 | 97.1% | — | `tests/test_held_finding_hmm_bull_quiet.py`, `tests/test_quant_upgrades.py`, `tests/test_ranker_transparency.py`, … +1 more |
| `engine/risk_manager.py` | 734 | 84.2% | `RiskManager.calculate_position_size` (L200, 16/104); `RiskManager.calculate_monte_carlo_var` (L909, 15/259); … +2 more | `tests/test_advanced_quant.py`, `tests/test_broad_pull_wiring_xfail.py`, `tests/test_data_integrity_bloomberg.py`, … +4 more |
| `engine/sim_portfolio.py` | 137 | 98.7% | — | `tests/test_sim_portfolio.py` |
| `engine/skew_dynamics.py` | 84 | 95.7% | — | `tests/test_quant_upgrades.py`, `tests/test_skew_dynamics_invariants.py` |
| `engine/strangle_timing.py` | 346 | 99.0% | — | `tests/test_strangle_ev_ranker.py`, `tests/test_strangle_recommendation_gate.py`, `tests/test_strangle_timing.py` |
| `engine/stress_testing.py` | 345 | 93.9% | `StressTester.from_policy` (L254, 3/6) | `tests/test_extreme_numerics.py`, `tests/test_greeks_unit_invariants.py`, `tests/test_launch_blockers.py`, … +3 more |
| `engine/tail_risk.py` | 86 | 87.7% | `fit_gpd_tail` (L82, 11/79) | `tests/test_quant_upgrades.py`, `tests/test_tail_copula_stress_invariants.py`, `tests/test_tail_risk.py` |
| `engine/theta_connector.py` | 514 | 96.4% | — | `tests/test_theta_connector.py`, `tests/test_theta_connector_coverage.py`, `tests/test_theta_connector_v3.py` |
| `engine/tradingview_bridge.py` | 91 | 82.6% | `PlaywrightChartProvider.fetch` (L262, 15/60) | `tests/test_tv_dossier.py` |
| `engine/transaction_costs.py` | 79 | 100.0% | — | `tests/test_audit_improvements.py`, `tests/test_edge_cases.py` |
| `engine/tv_signals.py` | 255 | 86.9% | `_rsi_state` (L193, 4/10) | `tests/test_audit_viii_e2e.py`, `tests/test_authority_hardening.py`, `tests/test_engine_api_hardening.py`, … +3 more |
| `engine/wheel_runner.py` | 1,426 | 79.7% | `WheelRunner.rank_candidates_by_ev` (L1286, 72/1246); `WheelRunner.rank_strangles_by_ev` (L3458, 56/722); … +6 more | `tests/test_asof_none_staleness.py`, `tests/test_audit_improvements.py`, `tests/test_audit_viii_e2e.py`, … +45 more |
| `engine/wheel_tracker.py` | 850 | 83.6% | `WheelTracker.suggest_rolls` (L2343, 19/464); `WheelTracker.suggest_call_rolls` (L2808, 19/376); … +3 more | `tests/test_audit_viii_e2e.py`, `tests/test_audit_viii_unit_invariants.py`, `tests/test_authority_hardening.py`, … +18 more |

## Methodology notes

- **Numbers** come from `coverage.json` (`pytest-cov --cov-report=json`), which itself reflects `pyproject.toml [tool.coverage.run]` — `source`, `omit`, `branch = true`, and `[tool.coverage.report] exclude_lines`.
- **"Notable untested"** functions are functions with ≥3 line body where either ≥33% of body lines or ≥10 raw lines intersect `missing_lines`. Methods are namespaced as `Class.method`. Truncated to the top 2–3 per row by missed count. A `—` means no single function clears the threshold; the row's gap is fragmented across many small misses (still surfaced via the row's `Stmts` / % columns).
- **"Tests"** column is built from `tests/test_*.py` AST imports (`import X` / `from X import Y`). A test file appears if it imports the module — it does not assert exercise. Enabling `coverage.dynamic_context = "test_function"` would give a true runtime mapping; the trade-off is run time (~1.5–2× slower) and a much larger `.coverage` SQLite file. Out of scope for this first artifact.
- **`__init__.py`** files appear when they hold re-exports or logic; pure namespace inits show 100% with a small statement count.
- **Branches** are counted but not enumerated per row — the branch totals at the top of the file give the suite-level branch picture; per-line branch detail lives in `coverage.json[files][...].missing_branches` for anyone wanting a deeper dive.

